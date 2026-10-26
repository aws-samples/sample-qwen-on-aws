"""On-policy distillation trainer -- 9B teacher (vLLM, TP=2, bf16) -> 2B LoRA student.

    generate rollouts (vLLM student sampler)  ->  score every token (remote teacher)
        ->  per-token reverse KL  ->  whitened advantage  ->  importance-sampling update
        ->  push adapter to the sampler  ->  log / checkpoint

Every flag defaults from an environment variable (manifests/env_vars, via the opd-config
ConfigMap), so the Job passes no arguments. By hand::

    python3 train_distill.py \
        --teacher-url http://teacher-vllm.default.svc.cluster.local:8000/v1 \
        --teacher-model Qwen/Qwen3.5-9B \
        --student-model Qwen/Qwen3.5-2B \
        --sampler-url http://student-sampler.default.svc.cluster.local:8000/v1 \
        --task gsm8k --steps 50 --output-dir /fsx/opd/gsm8k-opd

GUARDS
------
* trainable parameters are fp32 (`model_setup.assert_trainable_dtype`): bf16 would round
  1e-5 updates away;
* prompts come from the full train split, never cycled (`data.PromptStream`);
* train and eval share one prompt format (`prompts.build_prompt`);
* null teacher log-probs raise (`teacher_client`), and step 0 aborts with exit 3 unless the
  reverse KL is positive;
* rollout staleness is bounded (`RolloutBuffer.max_age`), the importance ratio corrects the
  rest;
* advantages are clamped (`--rkl-clip`) and whitened, the final-answer line is masked
  (`--answer-span-policy`), and capped rollouts get no credit on their cut-off token.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch

from data import PromptStream, load_eval_questions, load_prompt_pool
from distill_step import (
    completion_slice,
    distill_loss,
    rollout_reverse_kl,
    sequence_logprobs_batched,
    whiten_advantages,
)
from model_setup import (
    build_optimizer,
    build_scheduler,
    load_student,
    load_tokenizer,
    memory_report,
    save_student,
    trainable_parameter_report,
)
from prompts import build_prompt, get_task_spec
from rollout_buffer import Rollout, RolloutBuffer
from sampler import make_sampler
from teacher_client import assert_shared_vocabulary, make_teacher


def env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # -- models / services -------------------------------------------------------------
    ap.add_argument("--teacher-url", default=env("TEACHER_URL", "http://teacher-vllm.default.svc.cluster.local:8000/v1"))
    ap.add_argument("--teacher-model", default=env("TEACHER_MODEL_ID", "Qwen/Qwen3.5-9B"))
    ap.add_argument("--student-model", default=env("STUDENT_MODEL", "Qwen/Qwen3.5-2B"))
    ap.add_argument("--sampler", choices=["auto", "hf", "vllm"], default=env("STUDENT_SAMPLER", "auto"),
                    help="auto = vLLM when --sampler-url is set, else batched HF generate")
    ap.add_argument("--sampler-url", default=env("STUDENT_SAMPLER_URL", "") or None)
    ap.add_argument("--sampler-sync-every", type=int, default=int(env("SAMPLER_SYNC_EVERY", "8")),
                    help="push the LoRA adapter to the vLLM sampler every N steps")

    # -- task / data -------------------------------------------------------------------
    ap.add_argument("--task", default=env("TASK", "math500"), help="prompts.TASK_SPECS key")
    ap.add_argument("--train-dataset", default=env("TRAIN_DATASET", "EleutherAI/hendrycks_math"))
    ap.add_argument("--train-dataset-config", default=env("TRAIN_DATASET_CONFIG",
                    "algebra,counting_and_probability,geometry,intermediate_algebra,"
                    "number_theory,prealgebra,precalculus"),
                    help="comma-separated list; MATH publishes one config per subject")
    ap.add_argument("--train-split", default=env("TRAIN_SPLIT", "train"))
    ap.add_argument("--train-question-field", default=env("TRAIN_QUESTION_FIELD", "") or None)
    ap.add_argument("--prompt-pool-limit", type=int, default=int(env("PROMPT_POOL_LIMIT", "0")) or None)
    ap.add_argument("--fewshot", type=int, default=int(env("TRAIN_FEWSHOT", "0")),
                    help="MUST match the eval's --fewshot; 0 for an instruct student")
    ap.add_argument("--chat-template", choices=["auto", "raw"], default=env("CHAT_TEMPLATE", "auto"),
                    help="auto = tokenizer chat template (instruct); raw = few-shot completion (-Base)")
    ap.add_argument("--enable-thinking", action="store_true",
                    default=env("ENABLE_THINKING", "0") == "1",
                    help="must match the teacher's conditioning or the per-token KL is meaningless")

    # -- loop shape --------------------------------------------------------------------
    ap.add_argument("--steps", type=int, default=int(env("STEPS", "3000")))
    ap.add_argument("--group-size", type=int, default=int(env("GROUP_SIZE", "32")),
                    help="rollouts generated per step (= prompts x --samples-per-prompt)")
    ap.add_argument("--samples-per-prompt", type=int, default=int(env("SAMPLES_PER_PROMPT", "1")),
                    help="rollouts sampled per prompt. The Thinking Machines recipe uses 64 "
                         "prompts x 4 samples; must divide --group-size")
    ap.add_argument("--batch-size", type=int, default=int(env("BATCH_SIZE", "32")),
                    help="rollouts per optimizer update")
    ap.add_argument("--micro-batch-size", type=int, default=int(env("MICRO_BATCH_SIZE", "4")),
                    help="rollouts per forward/backward; gradient accumulation does the rest")
    ap.add_argument("--gen-micro-batch", type=int, default=int(env("GEN_MICRO_BATCH", "8")),
                    help="rollouts per batched HF generate call")
    ap.add_argument("--buffer-capacity", type=int, default=int(env("BUFFER_CAPACITY", "256")))
    ap.add_argument("--buffer-max-age", type=int, default=int(env("BUFFER_MAX_AGE", "-1")),
                    help="max trainer steps a rollout may be replayed for; "
                         "-1 = match --sampler-sync-every, 0 = strictly on-policy")
    ap.add_argument("--max-new-tokens", type=int, default=int(env("MAX_NEW_TOKENS", "512")),
                    help="keep >= the eval's --max-tokens or rollouts truncate mid-reasoning")
    ap.add_argument("--max-model-len", type=int, default=int(env("TEACHER_MAX_MODEL_LEN", "2048")),
                    help="the vLLM servers' --max-model-len; prompts that cannot fit "
                         "prompt + max_new_tokens + 1 inside it are dropped from the pool")
    ap.add_argument("--max-question-tokens", type=int, default=int(env("MAX_QUESTION_TOKENS", "0")),
                    help="drop training questions longer than this many tokens (0 = off). "
                         "Counts the question alone: the fixed few-shot prefix is ~300+ "
                         "tokens by itself, so a whole-prompt cap this small would drop "
                         "every prompt")
    ap.add_argument("--temperature", type=float, default=float(env("TRAIN_TEMPERATURE", "1.0")),
                    help="1.0 lets logp_old come free from generation; see sampler.py")
    ap.add_argument("--truncated-policy", choices=["drop", "mask", "keep"],
                    default=env("TRUNCATED_POLICY", "drop"),
                    help="rollouts that hit max_new_tokens without EOS: drop (default), "
                         "mask the final token's credit, or keep")

    # -- objective ---------------------------------------------------------------------
    ap.add_argument("--kl-coef", type=float, default=float(env("KL_COEF", "1.0")))
    ap.add_argument("--rkl-clip", type=float, default=float(env("RKL_CLIP", "10.0")),
                    help="clamp per-token reverse KL (nats) so outlier tokens do not eat "
                         "the whole grad-norm budget; 0 disables")
    ap.add_argument("--clip-ratio", type=float, default=float(env("CLIP_RATIO", "0.2")))
    ap.add_argument("--answer-span-policy", choices=["none", "mask"], default=env("ANSWER_SPAN_POLICY", "none"),
                    help="mask: zero advantage on the final-answer line and after (incl. EOS), so "
                         "the reverse KL does not teach the student to avoid committing to an answer")
    ap.add_argument("--truncation-penalty", type=float, default=float(env("TRUNCATION_PENALTY", "0")),
                    help="soft overlong punishment: capped rollouts get up to -this (advantage-std "
                         "units) ramped over their last --truncation-penalty-tail tokens. Needs "
                         "--truncated-policy mask|keep (drop removes capped rollouts). 0 = off")
    ap.add_argument("--truncation-penalty-tail", type=int, default=int(env("TRUNCATION_PENALTY_TAIL", "512")))
    ap.add_argument("--loss-norm", choices=["token", "sequence"], default=env("LOSS_NORM", "token"),
                    help="token: every completion token in the batch weighs the same (long "
                         "rollouts dominate). sequence: every rollout weighs the same; "
                         "see optimize_batch")
    ap.add_argument("--no-whiten", dest="whiten", action="store_false",
                    default=env("WHITEN_ADVANTAGES", "1") == "1",
                    help="use the raw unnormalised -kl_coef*rkl advantage instead")

    # -- optimization ------------------------------------------------------------------
    ap.add_argument("--precision", choices=["lora", "full-fp32"], default=env("PRECISION", "lora"))
    ap.add_argument("--lora-r", type=int, default=int(env("LORA_R", "32")))
    ap.add_argument("--lora-alpha", type=int, default=int(env("LORA_ALPHA", "64")))
    ap.add_argument("--init-adapter", default=env("INIT_ADAPTER", "") or None,
                    help="start from an existing LoRA adapter instead of the pristine base "
                         "student. Empty (the default) trains from the base model. The main "
                         "use is RESUMING: point it at a ${RUN_DIR}/student-step<N> checkpoint "
                         "after a crash, so a 22-30 h run does not restart from zero.")
    ap.add_argument("--lr", type=float, default=float(env("LR", "1e-5")))
    ap.add_argument("--weight-decay", type=float, default=float(env("WEIGHT_DECAY", "0.0")))
    ap.add_argument("--warmup-ratio", type=float, default=float(env("WARMUP_RATIO", "0.05")))
    ap.add_argument("--grad-clip", type=float, default=float(env("GRAD_CLIP", "1.0")))
    ap.add_argument("--adam8bit", action="store_true", default=env("ADAM_8BIT", "0") == "1")
    ap.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing",
                    action="store_false", default=True)

    # -- bookkeeping -------------------------------------------------------------------
    ap.add_argument("--output-dir", default=env("RUN_DIR", "/fsx/opd/gsm8k-opd"))
    ap.add_argument("--save-every", type=int, default=int(env("SAVE_EVERY", "500")))
    ap.add_argument("--log-every", type=int, default=int(env("LOG_EVERY", "10")))
    ap.add_argument("--teacher-concurrency", type=int, default=int(env("TEACHER_CONCURRENCY", "16")))
    ap.add_argument("--seed", type=int, default=int(env("SEED", "0")))
    ap.add_argument("--no-sanity-check", dest="sanity_check", action="store_false", default=True,
                    help="skip the step-0 assertion that the teacher signal is real")
    ap.add_argument("--trust-remote-code", action="store_true",
                    default=env("TRUST_REMOTE_CODE", "0") == "1")
    args = ap.parse_args(argv)

    if args.buffer_max_age < 0:
        # A vLLM sampler lags the trainer by up to one sync interval, so rollouts are
        # legitimately that stale; the importance ratio is what corrects for it. With the
        # in-process HF sampler there is no lag, so this collapses to 0 (strictly on-policy).
        args.buffer_max_age = args.sampler_sync_every if _uses_vllm(args) else 0
    if args.samples_per_prompt < 1 or args.group_size % args.samples_per_prompt:
        ap.error(f"--samples-per-prompt ({args.samples_per_prompt}) must divide "
                 f"--group-size ({args.group_size})")
    if args.truncation_penalty > 0 and args.truncated_policy == "drop":
        ap.error("--truncation-penalty needs --truncated-policy mask or keep: drop removes "
                 "the capped rollouts before any penalty can reach them")
    return args


def _uses_vllm(args: argparse.Namespace) -> bool:
    if args.sampler == "vllm":
        return True
    if args.sampler == "hf":
        return False
    return bool(args.sampler_url or os.environ.get("STUDENT_SAMPLER_URL"))


# ---------------------------------------------------------------------------
# One training step
# ---------------------------------------------------------------------------


def build_rollouts(samples, teacher_logps, policy_step: int, truncated_policy: str) -> list[Rollout]:
    """Pair sampler output with teacher scores, applying the truncation policy."""
    rollouts: list[Rollout] = []
    for sample, teacher_logp in zip(samples, teacher_logps):
        if sample.truncated and truncated_policy == "drop":
            continue
        rollouts.append(
            Rollout(
                token_ids=sample.token_ids,
                prompt_len=sample.prompt_len,
                student_logp_old=sample.logp_old,
                teacher_logp=teacher_logp,
                produced_at_step=policy_step,
                truncated=sample.truncated,
                prompt_text=sample.prompt_text,
            )
        )
    return rollouts


ANSWER_MARKERS = {"marker": "####", "boxed": "\\boxed"}


def answer_span_start(tokenizer, completion_ids: list[int], marker: str) -> int | None:
    """Index of the first completion token on the line holding the LAST `marker`, or None.

    Offsets come from decoding token by token, so a multi-byte character split across
    tokens can shift a boundary by one token -- harmless for masking a whole line.
    """
    pieces = [tokenizer.decode([t], skip_special_tokens=False) for t in completion_ids]
    text = "".join(pieces)
    k = text.rfind(marker)
    if k < 0:
        return None
    line_start = text.rfind("\n", 0, k) + 1
    pos = 0
    for i, piece in enumerate(pieces):
        if pos + len(piece) > line_start:
            return i
        pos += len(piece)
    return None


def compute_advantages(batch: list[Rollout], args: argparse.Namespace, device: str,
                       tokenizer=None, answer_style: str | None = None):
    """Phase 1: per-token reverse KL and advantage for the whole batch, with no forward pass.

    Both inputs (`student_logp_old`, `teacher_logp`) were recorded at sampling time, so the
    entire batch's advantages -- and therefore the whitening statistics -- are available
    before any gradient work starts. That is what makes batch-level whitening possible.
    """
    rkls, advantages = [], []
    for r in batch:
        logp_old = torch.tensor(r.student_logp_old, dtype=torch.float32, device=device)
        rkl = rollout_reverse_kl(
            prompt_len=r.prompt_len,
            seq_len=len(r.token_ids),
            teacher_logp_full=r.teacher_logp,
            student_logp_old=logp_old,
            rkl_clip=args.rkl_clip,
            device=device,
        )
        if r.truncated and args.truncated_policy == "mask" and rkl.numel() > 0:
            # The final token of a truncated rollout is an artefact of the length cap, not a
            # choice the student made; give it no credit either way.
            rkl = rkl.clone()
            rkl[-1] = 0.0
        rkls.append(rkl)
        advantages.append(-args.kl_coef * rkl)

    stats: dict = {}
    if args.whiten:
        advantages, stats = whiten_advantages(advantages)

    # Soft overlong punishment (DAPO), token-level: a rollout that hit max_new_tokens gets a
    # linearly ramped negative credit on its LAST `truncation_penalty_tail` tokens, reaching
    # -truncation_penalty at the cap. With `drop`, hitting the cap costs nothing, which lets a
    # student turn wrong answers into unfinished ones (loops in the tail). The early reasoning
    # of a capped rollout is often sound, so only the tail is penalised. Applied after
    # whitening, so it is in advantage-std units and is not re-centred away. Off by default.
    if args.truncation_penalty > 0:
        penalised = 0
        for i, r in enumerate(batch):
            n = min(args.truncation_penalty_tail, int(advantages[i].numel()))
            if r.truncated and n > 0:
                ramp = torch.linspace(1.0 / n, 1.0, n, device=device) * args.truncation_penalty
                advantages[i] = advantages[i].clone()
                advantages[i][-n:] -= ramp
                penalised += 1
        stats["penalised_rollouts"] = penalised

    # Answer-span masking: zero credit on the final-answer line and everything after it
    # (incl. <|im_end|>). The "#### N" tokens carry the largest reverse KL of any region
    # (~+0.54 vs ~+0.33 for reasoning on GSM8K) because the teacher can tell when the number
    # is wrong, and the student's cheapest way to lower it is to stop writing answers.
    # Masking keeps the teacher's signal on the reasoning tokens -- the "forking tokens" that
    # carry it -- and removes the pressure to avoid answering.
    if args.answer_span_policy == "mask" and tokenizer is not None and answer_style:
        marker = ANSWER_MARKERS.get(answer_style)
        masked = 0
        for i, r in enumerate(batch):
            start = answer_span_start(tokenizer, r.token_ids[r.prompt_len:], marker) if marker else None
            if start is not None and start < advantages[i].numel():
                advantages[i] = advantages[i].clone()
                advantages[i][start:] = 0.0
                masked += 1
        stats["answer_masked_rollouts"] = masked
    return rkls, advantages, stats


def optimize_batch(
    model, tokenizer, batch: list[Rollout], advantages: list[torch.Tensor],
    args: argparse.Namespace, device: str,
) -> dict:
    """Phase 2: micro-batched forward/backward with gradient accumulation.

    `--loss-norm token` (default) normalises by the batch's *total completion tokens*, so
    splitting the batch into micro-batches cannot change the update, and a long rollout
    does not get quietly down-weighted relative to a short one. The flip side: a 2048-token
    rambling rollout outweighs a 400-token finished one ~5x. `--loss-norm sequence` uses the
    per-rollout token mean, averaged over the batch: every rollout counts once. Either
    way the update is independent of the micro-batch split.
    """
    total_tokens = max(1, sum(int(a.numel()) for a in advantages))
    pad_id = tokenizer.pad_token_id
    diags: list[dict] = []
    loss_total = 0.0

    for start in range(0, len(batch), args.micro_batch_size):
        chunk = batch[start : start + args.micro_batch_size]
        chunk_adv = advantages[start : start + args.micro_batch_size]
        max_len = max(len(r.token_ids) for r in chunk)
        input_ids = torch.full((len(chunk), max_len), pad_id, dtype=torch.long, device=device)
        attn = torch.zeros((len(chunk), max_len), dtype=torch.long, device=device)
        for i, r in enumerate(chunk):
            n = len(r.token_ids)
            input_ids[i, :n] = torch.tensor(r.token_ids, dtype=torch.long, device=device)
            attn[i, :n] = 1

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            logp_new_all = sequence_logprobs_batched(
                model, input_ids, attn, with_grad=True,
                start=min(r.prompt_len for r in chunk) - 1,  # completion span only: see docstring
            )

        micro_loss = None
        for i, (r, adv) in enumerate(zip(chunk, chunk_adv)):
            comp = completion_slice(r.prompt_len, len(r.token_ids))
            logp_new = logp_new_all[i, comp].float()
            logp_old = torch.tensor(
                r.student_logp_old, dtype=torch.float32, device=device
            )[comp]
            piece, diag = distill_loss(
                logp_new, logp_old, adv, clip_ratio=args.clip_ratio, reduction="sum"
            )
            if args.loss_norm == "sequence":
                # After the `/ total_tokens` below this is piece / (n_tokens_i * batch):
                # rollout i's token mean, averaged over the batch.
                piece = piece * (total_tokens / (max(1, int(adv.numel())) * len(batch)))
            diags.append(diag)
            micro_loss = piece if micro_loss is None else micro_loss + piece

        if micro_loss is not None:
            scaled = micro_loss / total_tokens
            scaled.backward()
            loss_total += float(scaled.detach())

    return {
        "loss": loss_total,
        "clip_frac": statistics.fmean(d["clip_frac"] for d in diags) if diags else 0.0,
        "ratio_mean": statistics.fmean(d["ratio_mean"] for d in diags) if diags else 1.0,
        "ratio_max": max((d["ratio_max"] for d in diags), default=1.0),
        "policy_drift": statistics.fmean(d["policy_drift"] for d in diags) if diags else 0.0,
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    torch.manual_seed(args.seed)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    metrics_path = out / f"metrics_rank{local_rank}.jsonl"
    spec = get_task_spec(args.task)

    # -- preflight: the two silent-failure modes, checked before we spend a GPU-hour ----
    print(f"[preflight] verifying teacher/student tokenizer identity ...", flush=True)
    vocab = assert_shared_vocabulary(args.teacher_model, args.student_model)
    print(f"[preflight] shared vocabulary of {vocab} tokens: OK", flush=True)

    tokenizer = load_tokenizer(args.student_model, args.trust_remote_code)
    use_chat = args.chat_template == "auto"
    # Fail here rather than silently falling back to a bare question, which is what made
    # the training and eval prompt formats diverge in the first run.
    probe = build_prompt(spec, "probe question", args.fewshot, tokenizer,
                         chat=use_chat, enable_thinking=args.enable_thinking)
    print(f"[preflight] prompt format ({'chat' if use_chat else 'raw'}, "
          f"fewshot={args.fewshot}, thinking={args.enable_thinking}), "
          f"{len(tokenizer(probe).input_ids)} tokens", flush=True)

    # -- student ----------------------------------------------------------------------
    print(f"[rank {local_rank}] loading student {args.student_model} ({args.precision}) on {device}", flush=True)
    student = load_student(
        args.student_model, device=device, precision=args.precision,
        lora_r=args.lora_r, lora_alpha=args.lora_alpha,
        gradient_checkpointing=args.gradient_checkpointing,
        trust_remote_code=args.trust_remote_code, adapter_path=args.init_adapter,
    )
    print(f"[rank {local_rank}] {trainable_parameter_report(student)} {memory_report(device)}", flush=True)
    if args.init_adapter:
        print(f"[rank {local_rank}] resuming from adapter {args.init_adapter}", flush=True)
    else:
        print(f"[rank 0] training from the pristine base student, fewshot={args.fewshot}. "
              f"On-policy distillation REWEIGHTS tokens the student already samples, so it "
              f"sharpens an existing policy rather than installing a new one -- which makes "
              f"the prompt's worked exemplars the only thing teaching the answer format. "
              f"{'That is the intended setup.' if args.fewshot else 'WARNING: fewshot=0 and no init adapter, so nothing is teaching the format. If the base student does not already emit the answer marker reliably, there will be no correctly-formatted rollout for the reverse KL to reinforce. Set TRAIN_FEWSHOT (and EVAL_FEWSHOT to match).'} "
              f"Watch truncated_frac here and the format metrics in the eval.", flush=True)

    optimizer = build_optimizer(student, args.lr, args.weight_decay, adam8bit=args.adam8bit)
    scheduler = build_scheduler(optimizer, args.steps, warmup_ratio=args.warmup_ratio)

    # -- sampler + teacher ------------------------------------------------------------
    sampler = make_sampler(
        args.sampler, model=student, tokenizer=tokenizer, device=device,
        base_url=args.sampler_url, base_model=args.student_model,
        max_new_tokens=args.max_new_tokens, temperature=args.temperature,
        micro_batch=args.gen_micro_batch, stop_strings=spec.stop_seqs,
    )
    print(f"[rank {local_rank}] sampler = {type(sampler).__name__}", flush=True)
    teacher = make_teacher("remote", base_url=args.teacher_url, model=args.teacher_model,
                           max_concurrency=args.teacher_concurrency)

    # -- prompts ----------------------------------------------------------------------
    excluded = load_eval_questions(spec)
    pool = load_prompt_pool(
        args.train_dataset, args.train_dataset_config, args.train_split,
        question_field=args.train_question_field, limit=args.prompt_pool_limit,
        seed=args.seed, exclude=excluded,
    )
    # A few MATH questions (long Asymptote figures) overflow the context once the few-shot
    # prefix is added; vLLM rejects those with a 400 hundreds of steps in. Drop them now.
    # Budget: prompt + max_new_tokens for the sampler, +1 for the teacher's scoring token.
    prompt_budget = args.max_model_len - args.max_new_tokens - 1
    fits = [
        q for q in pool
        if len(tokenizer(build_prompt(spec, q, args.fewshot, tokenizer, chat=use_chat,
                                      enable_thinking=args.enable_thinking),
                         add_special_tokens=False).input_ids) <= prompt_budget
    ]
    if len(fits) < len(pool):
        print(f"[rank {local_rank}] dropped {len(pool) - len(fits)} prompts longer than "
              f"{prompt_budget} tokens (max_model_len={args.max_model_len} - "
              f"max_new_tokens={args.max_new_tokens} - 1)", flush=True)
    pool = fits
    # Optional memory cap: long questions (mostly Asymptote figures) make the longest
    # training sequences. Training-side only -- the eval set is never filtered.
    if args.max_question_tokens > 0:
        short = [q for q in pool
                 if len(tokenizer(q, add_special_tokens=False).input_ids) <= args.max_question_tokens]
        print(f"[rank {local_rank}] dropped {len(pool) - len(short)} of {len(pool)} questions "
              f"longer than {args.max_question_tokens} tokens (MAX_QUESTION_TOKENS)", flush=True)
        pool = short
    stream = PromptStream(pool, seed=args.seed)
    print(f"[rank {local_rank}] {len(pool)} unique training prompts "
          f"({len(excluded)} eval questions excluded)", flush=True)
    if len(pool) < args.steps * args.group_size / 4:
        print(f"[rank {local_rank}] NOTE: {args.steps * args.group_size} rollouts over "
              f"{len(pool)} prompts = {args.steps * args.group_size / len(pool):.1f} "
              f"passes per prompt.", flush=True)

    buffer = RolloutBuffer(capacity=args.buffer_capacity, seed=args.seed,
                           max_age=args.buffer_max_age)
    adapter_dir = out / "adapter-live"
    last_sync_step = 0
    if _uses_vllm(args):  # serve the warm-started adapter, not the pristine base
        save_student(student, tokenizer, adapter_dir)
        sampler.sync_adapter(str(adapter_dir))

    print(f"[rank {local_rank}] teacher @ {args.teacher_url}; {args.steps} steps, "
          f"group={args.group_size}, batch={args.batch_size}, "
          f"buffer_max_age={args.buffer_max_age}", flush=True)

    for step in range(args.steps):
        t0 = time.time()

        # 1) generate rollouts from the current student
        # group_size rollouts = (group_size / samples_per_prompt) prompts, each sampled
        # samples_per_prompt times at TRAIN_TEMPERATURE (distinct rollouts, same prompt).
        questions = [q for q in stream.take(args.group_size // args.samples_per_prompt)
                     for _ in range(args.samples_per_prompt)]
        prompt_texts = [
            build_prompt(spec, q, args.fewshot, tokenizer, chat=use_chat,
                         enable_thinking=args.enable_thinking)
            for q in questions
        ]
        samples = sampler.generate(prompt_texts, step=step)
        t_gen = time.time()

        # 2) score every token with the remote teacher (concurrent requests)
        teacher_logps = teacher.score_batch([s.token_ids for s in samples])
        t_score = time.time()

        # 3) buffer, honouring sampler lag: a vLLM sampler's weights are those of the last
        #    adapter push, so that -- not `step` -- is the policy these rollouts came from.
        policy_step = last_sync_step if _uses_vllm(args) else step
        rollouts = build_rollouts(samples, teacher_logps, policy_step, args.truncated_policy)
        buffer.extend(rollouts)
        truncated_frac = (
            sum(s.truncated for s in samples) / len(samples) if samples else 0.0
        )

        batch = (
            buffer.drain() if args.buffer_max_age == 0
            else buffer.sample_batch(args.batch_size, current_step=step)
        )
        if not batch:
            print(f"[rank {local_rank}] step {step}: no usable rollouts "
                  f"(truncated_frac={truncated_frac:.2f}); skipping", flush=True)
            continue
        batch = batch[: args.batch_size]

        # 4) advantages (no forward pass), then the accumulated update
        rkls, advantages, adv_stats = compute_advantages(batch, args, device, tokenizer,
                                                         spec.answer_style)
        mean_rkl = float(torch.cat([r.flatten() for r in rkls]).mean())

        if step == 0 and args.sanity_check:
            teacher_mean = statistics.fmean(
                statistics.fmean(r.teacher_logp[r.prompt_len :]) for r in batch
                if len(r.teacher_logp) > r.prompt_len
            )
            student_mean = statistics.fmean(
                statistics.fmean(r.student_logp_old[r.prompt_len - 1 :]) for r in batch
            )
            print(f"[sanity] mean completion logp: teacher={teacher_mean:.4f} "
                  f"student={student_mean:.4f} reverse_kl={mean_rkl:.4f}", flush=True)
            if not mean_rkl > 0.01:
                print(
                    "[sanity] FATAL: reverse KL is not positive at step 0. The teacher is "
                    "not providing a stronger distribution than the student, which almost "
                    "always means the teacher scores are wrong (null logprobs mapped to "
                    "0.0, a served-model mismatch, or teacher/student conditioning that "
                    "differ -- e.g. --enable-thinking set on one side only). Training now "
                    "would optimise the student against noise. Re-run with "
                    "--no-sanity-check only if you know why this is expected.",
                    file=sys.stderr, flush=True,
                )
                return 3

        optimizer.zero_grad(set_to_none=True)
        train_diag = optimize_batch(student, tokenizer, batch, advantages, args, device)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in student.parameters() if p.requires_grad], args.grad_clip
        )
        optimizer.step()
        scheduler.step()
        t_train = time.time()

        # 5) keep the sampler's weights close to the trainer's
        if _uses_vllm(args) and (step + 1) % args.sampler_sync_every == 0:
            save_student(student, tokenizer, adapter_dir)
            sampler.sync_adapter(str(adapter_dir))
            last_sync_step = step + 1

        completion_lens = [r.num_completion_tokens() for r in batch]
        rec = {
            "step": step,
            # The headline curve: reverse KL should trend DOWN. It says the student is
            # matching the teacher's distribution -- not that accuracy moved. Only the eval
            # says that.
            "teacher_kl": round(mean_rkl, 5),
            "loss": round(train_diag["loss"], 5),
            "lr": scheduler.get_last_lr()[0],
            "grad_norm": round(float(grad_norm), 4),
            # Health checks: clip_frac climbing past ~0.2 means rollouts are too stale;
            # policy_drift is how far the live student has moved from the sampling policy.
            "clip_frac": round(train_diag["clip_frac"], 4),
            "ratio_mean": round(train_diag["ratio_mean"], 4),
            "policy_drift": round(train_diag["policy_drift"], 5),
            "truncated_frac": round(truncated_frac, 3),
            "completion_tokens_mean": round(statistics.fmean(completion_lens), 1),
            "batch": len(batch),
            "buffer": len(buffer),
            "staleness": round(buffer.mean_staleness(step), 2),
            "epochs_done": stream.epochs_done,
            "sec_gen": round(t_gen - t0, 2),
            "sec_score": round(t_score - t_gen, 2),
            "sec_train": round(t_train - t_score, 2),
            **{k: round(v, 4) for k, v in adv_stats.items()},
        }
        with metrics_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        if step % args.log_every == 0:
            print(f"[rank {local_rank}] {rec}", flush=True)

        if args.save_every and (step + 1) % args.save_every == 0 and local_rank == 0:
            ckpt = save_student(student, tokenizer, out / f"student-step{step+1}")
            print(f"[rank {local_rank}] saved adapter {ckpt}", flush=True)

    if local_rank == 0:
        # Merged weights: `eval_student.py` loads this as a standalone model so the
        # before/after comparison does not depend on adapter plumbing.
        final = save_student(student, tokenizer, out / "student-final", merge=True,
                             trust_remote_code=args.trust_remote_code)
        print(f"[rank {local_rank}] saved merged checkpoint {final}", flush=True)
    print(f"[rank {local_rank}] done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
