"""Before / after / teacher evaluation of the distillation student, using Inspect AI.

The companion to `train_distill.py`: it measures whether on-policy distillation actually
moved the student. Three phases, the *same* task, prompt and samples for each:

    before   ->  the pristine base student from the Hub   (EVAL_BASE_MODEL)
    after    ->  the checkpoint the Job wrote on FSx      (EVAL_FT_MODEL_DIR)
    teacher  ->  the served teacher, through its OpenAI-compatible endpoint

WHY THE TEACHER IS A PHASE, NOT A FOOTNOTE
------------------------------------------
On-policy distillation cannot take the student past the teacher, so `after - before` on its
own is uninterpretable: a +3 point move is excellent if the teacher is only 5 points ahead
and dismal if it is 40 ahead. The teacher phase turns the result into the quantity that
actually reflects how well distillation worked:

    headroom_recovered = (after - before) / (teacher - before)

CHOOSING THE TASK AND SAMPLE COUNT
----------------------------------
Pick a task with real headroom for the student: GSM8K for a 0.8B or 2B, MATH-500 for a 4B (GSM8K
leaves it little to gain). Always evaluate the whole test
set -- at n=100 the standard error is ~+-5pp, wider than most distillation gains; at
GSM8K's n=1319 it is ~+-1.4pp.

Prompt construction, answer extraction and answer equivalence all come from `prompts.py`,
the *same* module `train_distill.py` uses, so the student is evaluated in exactly the format
it was trained in.

Run all three phases in one pod (the Job takes every flag from the ConfigMap instead)::

    python3 eval_student.py --phase all \
        --base-model Qwen/Qwen3.5-2B \
        --ft-model-dir /fsx/opd/gsm8k-opd/student-step25 \
        --student-url http://student-sampler:8000/v1 \
        --teacher-url http://teacher-vllm:8000/v1 --teacher-model Qwen/Qwen3.5-9B \
        --task gsm8k --limit 1319 --fewshot 4 --votes 8 --output-dir /fsx/opd/eval-gsm8k-opd

Outputs, all under ``$EVAL_DIR``:

* ``summary.json``          -- per-phase accuracy/stderr, secondary metrics, the
                              before->after ``delta_accuracy``, ``transitions``
                              (fixed / broken / both_correct / both_wrong) and
                              ``headroom_recovered``.
* ``samples_<phase>.jsonl`` -- one row per eval sample (answer, correct, output tokens,
                              format compliance) -- input to the comparison dashboard.
* ``logs/<phase>/``         -- full Inspect logs (``inspect view --log-dir ...``).

The same payload is printed between ``DASHBOARD_BEGIN``/``DASHBOARD_END`` sentinels so the
notebook can rebuild the dashboard from ``kubectl logs job/opd-student-eval`` alone.

Scoring is by *semantic* answer equivalence, not string equality (see
`prompts.answers_equivalent`): spelling ('1,000' / '$1000'), notation ('3/4' / '0.75' /
'\\frac{3}{4}'), decoration ('50%', '\\left') and wording ('seventy-two' / '72') do not cost
a point. Each row's ``match`` field and the per-phase ``lenient_matches`` count say how much
accuracy came from the tolerant paths.

MAJORITY VOTE (self-consistency)
--------------------------------
``--votes k`` (EVAL_VOTES) samples every problem k times at ``--vote-temperature`` (> 0) and
scores the most common final answer, for the phases in ``--vote-phases`` (EVAL_VOTE_PHASES,
default ``after``); the other phases decode with ``--temperature`` (greedy by default). The
shipped headline votes on the distilled student only -- greedy base, voted distilled, greedy
teacher -- so every result records its ``decoding`` and the dashboards label it. Each voted
row keeps its k answers, so ``maj_at`` (majority accuracy over the first 1..k votes) and
``pass_at_k`` (any vote correct) come from the same run. A phase is only carried forward
when its recorded decoding matches what this run would use for it.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from inspect_ai import Epochs, Task, eval as inspect_eval, task
from inspect_ai.model import get_model
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    Score,
    Target,
    accuracy,
    score_reducer,
    scorer,
    stderr,
)
from inspect_ai.solver import TaskState, generate

from prompts import (
    DEFAULT_ANSWER_REL_TOL,
    DEFAULT_TEXT_SIMILARITY,
    TASK_SPECS,
    TaskSpec,
    answers_equivalent,
    build_completion_prompt,
    build_fewshot_prefix,
    build_user_message,
    clean_number,
    extract_answer,
    get_task_spec,
    has_answer_marker,
)

# math-verify (sympy-backed) is LightEval's MATH-500 scorer. It handles symbolic equivalence
# the hand-rolled `answers_equivalent` misses -- \sqrt{2}/2 == 1/\sqrt{2} == \frac{\sqrt{2}}{2},
# expanded/factored polynomials, interval notation, ordered pairs, sets, complex numbers.
# Optional dependency: fall back to the regex/tolerance scorer if the import fails.
try:
    from math_verify import parse as mv_parse, verify as mv_verify   # type: ignore
    _MATH_VERIFY_AVAILABLE = True
except Exception:  # pragma: no cover - only exercised when math-verify is absent
    mv_parse = mv_verify = None  # type: ignore
    _MATH_VERIFY_AVAILABLE = False

# Renders the message contents verbatim -- a completion-style prompt for a base model,
# instead of the provider's "user: ...\n" fallback.
RAW_CHAT_TEMPLATE = "{% for message in messages %}{{ message['content'] }}{% endfor %}"

# Sentinels around the JSON the notebook's dashboard parses out of the Job logs.
DASHBOARD_BEGIN = "===EVAL_DASHBOARD_JSON_BEGIN==="
DASHBOARD_END = "===EVAL_DASHBOARD_JSON_END==="

# Inspect's generic OpenAI-compatible provider reads <PREFIX>_BASE_URL / <PREFIX>_API_KEY.
TEACHER_PROVIDER = "teacher"
STUDENT_PROVIDER = "student"   # the student-sampler vLLM, when --student-url is set
CHILD_ENV = "OPD_EVAL_CHILD"   # set on the per-phase subprocesses main() spawns


def record_to_sample(spec: TaskSpec, fewshot: int, chat_template: str):
    """Dataset row -> Inspect Sample, with the prompt built by `prompts.py`.

    In `raw` mode the input has to be the whole completion prompt, because
    RAW_CHAT_TEMPLATE renders the message contents verbatim. In `auto` mode the input is
    just the user-message content and Inspect applies the tokenizer's chat template -- so
    we must NOT pre-render it here or the template would be applied twice.
    """
    prefix = build_fewshot_prefix(spec, fewshot)

    def _convert(record: dict[str, Any]) -> Sample:
        question = str(record[spec.question_field]).strip()
        gold = spec.gold_fn(record)
        if chat_template == "raw":
            prompt = build_completion_prompt(spec, question, prefix)
        else:
            prompt = build_user_message(spec, question, prefix)
        return Sample(
            input=prompt,
            target=clean_number(gold) or gold,
            metadata={"question": question, "gold_raw": gold},
        )

    return _convert


def _math_verify_equivalent(answer: str | None, gold: str) -> tuple[bool, str] | None:
    """LightEval's sympy-backed check. Returns (equivalent, how) or None on failure/absence.

    Handles the symbolic cases the regex/tolerance path in `prompts.answers_equivalent`
    cannot: expanded vs. factored polynomials, radical rationalisation, interval and set
    notation, ordered pairs and tuples. Returning None lets the caller fall back cleanly.
    """
    if not _MATH_VERIFY_AVAILABLE or answer is None:
        return None
    try:
        gold_parsed = mv_parse(gold)                # type: ignore[misc]
        pred_parsed = mv_parse(answer)              # type: ignore[misc]
        # math_verify.verify signature is verify(gold, target); returns True/False.
        equivalent = bool(mv_verify(gold_parsed, pred_parsed))   # type: ignore[misc]
    except Exception:
        return None
    return equivalent, ("math_verify" if equivalent else "math_verify_mismatch")


@scorer(metrics=[accuracy(), stderr()])
def final_answer_match(
    answer_style: str = "marker",
    rel_tol: float = DEFAULT_ANSWER_REL_TOL,
    text_similarity: float = DEFAULT_TEXT_SIMILARITY,
):
    """Semantic match on the extracted final answer, in the task's answer style.

    Two-tier scorer: try math-verify (sympy) first, which recovers symbolic equivalences
    the regex path misses; fall back to `answers_equivalent` when math-verify is not
    installed, cannot parse a side, or reports mismatch (the fallback is *tolerant* on the
    numeric path, so a math-verify "mismatch" is not treated as authoritative unless the
    fallback also disagrees). `Score.metadata["match"]` records which tier matched, so a
    run can be audited for how much of its accuracy comes from each path.
    """

    async def score(state: TaskState, target: Target) -> Score:
        completion = state.output.completion
        answer = extract_answer(completion, answer_style)  # type: ignore[arg-type]
        gold = target.text.strip()

        mv_result = _math_verify_equivalent(answer, gold)
        if mv_result is not None and mv_result[0]:
            equivalent, how = mv_result
        else:
            equivalent, how = answers_equivalent(answer, gold, rel_tol, text_similarity)
            if mv_result is not None and not equivalent:
                how = mv_result[1]  # both disagreed -- surface the math-verify tag

        return Score(
            value=CORRECT if equivalent else INCORRECT,
            answer=answer or "",
            explanation=completion,
            metadata={"match": how, "has_marker": has_answer_marker(completion, answer_style)},  # type: ignore[arg-type]
        )

    return score


def _vote_key(answer: str | None) -> str:
    """Normalised final answer for counting votes ('1,000' and '1000' are one vote)."""
    return (clean_number(answer) or answer or "").strip()


def majority_of(answers: list[str | None], correct: list[bool]) -> tuple[str | None, bool, float]:
    """(winning answer, its correctness, vote share) among the parsed answers.

    Unparsed answers abstain; ties go to the answer seen first. All votes for one normalised
    answer share its correctness, so the winner's first occurrence decides.
    """
    counts = Counter(_vote_key(a) for a in answers if a)
    if not counts:
        return None, False, 0.0
    top, n = counts.most_common(1)[0]
    i = next(i for i, a in enumerate(answers) if a and _vote_key(a) == top)
    return answers[i], correct[i], n / len(answers)


@score_reducer(name="majority_vote")
def majority_vote():
    """Inspect epoch reducer: the most common final answer across a sample's epochs."""

    def reduce(scores: list[Score]) -> Score:
        answers = [s.answer or None for s in scores]
        correct = [str(s.value) == CORRECT for s in scores]
        answer, ok, share = majority_of(answers, correct)
        return Score(value=CORRECT if ok else INCORRECT, answer=answer or "",
                     metadata={"vote_share": share})

    return reduce


@task
def math_reasoning(
    task_key: str = "math500",
    split: str | None = None,
    limit: int = 500,
    select: str = "all",
    fewshot: int = 0,
    chat_template: str = "auto",
    shuffle_seed: int = 42,
    answer_rel_tol: float = DEFAULT_ANSWER_REL_TOL,
    text_similarity: float = DEFAULT_TEXT_SIMILARITY,
    votes: int = 1,
) -> Task:
    """Final-answer accuracy on the task's held-out test split.

    `select` picks which `limit` rows of the split to use: ``all`` (default) = the whole
    split up to `limit`, ``tail`` = the LAST `limit` rows, ``head`` = the first,
    ``shuffle`` = a seeded random subset. For MATH-500 ``all`` is right: the set is already
    a held-out 500-row sample and the trainer draws from the MATH *train* split.

    `answer_rel_tol` / `text_similarity` set how tolerant the scorer is -- keep them
    identical across phases or the delta is not comparable.
    """
    spec = get_task_spec(task_key)
    samples = hf_dataset(
        path=spec.dataset,
        name=spec.dataset_name,
        split=split or spec.test_split,
        sample_fields=record_to_sample(spec, fewshot, chat_template),
        # `tail` needs the whole split in hand before slicing off the end.
        limit=None if select == "tail" else limit,
        shuffle=select == "shuffle",
        seed=shuffle_seed,
        trust=True,
    )
    if select == "tail":
        samples = samples[-limit:]  # type: ignore[assignment]
    return Task(
        dataset=samples,
        solver=generate(),
        scorer=final_answer_match(spec.answer_style, answer_rel_tol, text_similarity),
        epochs=Epochs(votes, majority_vote()) if votes > 1 else None,
    )


def _is_bare_adapter(path: Path) -> bool:
    """A PEFT adapter directory with no base weights -- Inspect's hf provider cannot load it."""
    return (path / "adapter_config.json").exists() and not any(
        (path / n).exists() for n in ("config.json", "model.safetensors.index.json")
    )


def model_args_for(model_ref: str, phase: str, args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    """Map a Hub id, local checkpoint dir or served endpoint onto Inspect provider args."""
    if phase == "teacher":
        # Route through Inspect's generic OpenAI-compatible provider. The served model is
        # only being *generated from* here, so nothing about the trainer's log-prob path
        # (echo, --max-logprobs) matters for this phase.
        os.environ[f"{TEACHER_PROVIDER.upper()}_BASE_URL"] = args.teacher_url
        os.environ.setdefault(f"{TEACHER_PROVIDER.upper()}_API_KEY", "EMPTY")
        return f"openai-api/{TEACHER_PROVIDER}/{model_ref}", {}

    if args.student_url:
        # SERVED STUDENT: the same OpenAI-compatible route as the teacher, against the
        # student-sampler vLLM. Two reasons over the local hf provider: minutes instead of
        # hours, and correct stopping -- Qwen3.5's config eos is <|endoftext|> while a chat
        # turn ends in <|im_end|>, so hf `generate` runs every sample to max_tokens and
        # Inspect only trims the text afterwards (hours per phase, and output-token counts
        # several times the real answer length).
        os.environ[f"{STUDENT_PROVIDER.upper()}_BASE_URL"] = args.student_url
        os.environ.setdefault(f"{STUDENT_PROVIDER.upper()}_API_KEY", "EMPTY")
        name = model_ref if phase == "before" else load_served_adapter(Path(model_ref), args)
        return f"openai-api/{STUDENT_PROVIDER}/{name}", {}

    model_args: dict[str, Any] = {
        # Pinned for every local phase: an unpinned dtype lets `before` load in fp32 and
        # `after` in bf16 (or vice versa), and a dtype difference alone can move accuracy
        # by a point or two -- enough to fake or mask the entire distillation effect.
        "device": args.device,
        "torch_dtype": args.torch_dtype,
        "trust_remote_code": args.trust_remote_code,
        # temperature=0 only means greedy if sampling is off.
        "do_sample": decoding_for(phase, args)[1] > 0,
    }
    if args.chat_template == "auto":
        # Qwen3.5's template turns thinking OFF only for an explicit `enable_thinking=False`;
        # Inspect passes None when unset, which leaves thinking ON. The model then opens with
        # "The user wants me to..." and hits max_tokens before any \boxed{} -- near-zero
        # accuracy for every phase. Pass the same flag train_distill.py builds its prompts with.
        model_args["enable_thinking"] = args.enable_thinking
    if args.chat_template == "raw":
        model_args["chat_template"] = RAW_CHAT_TEMPLATE
    elif args.chat_template == "none":
        model_args["use_chat_template"] = False

    path = Path(model_ref)
    if path.exists():
        if _is_bare_adapter(path):
            raise RuntimeError(
                f"{model_ref} holds only a LoRA adapter, which the Inspect hf provider "
                f"cannot load on its own. train_distill.py's final save merges the adapter "
                f"into the base weights (save_student(..., merge=True)); point --ft-model-dir "
                f"at that merged directory, or merge this one first."
            )
        # Local checkpoint: the name is a placeholder, `model_path` does the loading.
        model_args["model_path"] = model_ref
        return "hf/local", model_args
    return f"hf/{model_ref}", model_args


def load_served_adapter(adapter_dir: Path, args: argparse.Namespace) -> str:
    """Register a LoRA checkpoint with the student sampler; returns its served model name.

    The sampler serves the base student plus runtime-loaded adapters (the trainer hot-swaps
    `student-live-<N>` the same way), so `after` is a bare adapter dir -- ${RUN_DIR}/
    student-step<N> -- not the merged `student-final`, which vLLM cannot load as a LoRA.
    """
    import urllib.error
    import urllib.request

    if not _is_bare_adapter(adapter_dir):
        raise RuntimeError(
            f"{adapter_dir} is not a LoRA adapter dir. The served route (--student-url) loads "
            f"the checkpoint into the student sampler as a LoRA, so point --ft-model-dir at "
            f"a ${{RUN_DIR}}/student-step<N> adapter -- or unset EVAL_STUDENT_URL to evaluate "
            f"merged weights with the local hf provider."
        )
    from sampler import export_adapter_for_vllm

    name = f"eval-{adapter_dir.parent.name}-{adapter_dir.name}"
    # Same renaming the trainer's sync uses -- without it vLLM silently serves the base model.
    served_dir = export_adapter_for_vllm(str(adapter_dir), args.base_model)
    req = urllib.request.Request(
        args.student_url.rstrip("/") + "/load_lora_adapter",
        data=json.dumps({"lora_name": name, "lora_path": served_dir}).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        urllib.request.urlopen(req, timeout=300).read()
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        # Re-running an eval re-registers the same name; anything else is a real failure.
        if "already" not in body.lower():
            raise RuntimeError(f"load_lora_adapter({adapter_dir}) failed: {e.code} {body}") from e
    print(f"[served] adapter {adapter_dir} registered with the sampler as '{name}'", flush=True)
    return name


def per_sample_records(log: Any) -> list[dict[str, Any]]:
    """Flatten the Inspect log into one compact row per sample -- the dashboard's input.

    With --votes the log holds one entry per (sample, epoch): those are folded into one row
    whose answer is the majority, keeping every vote for the maj@k / pass@k curves.
    """
    rows = [_sample_row(s) for s in log.samples or []]
    by_id: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_id.setdefault(r["id"], []).append(r)
    if all(len(g) == 1 for g in by_id.values()):
        for r in rows:
            del r["epoch"]
        return rows
    voted = []
    for sid, group in by_id.items():
        group.sort(key=lambda r: r["epoch"])
        for r in group:
            del r["epoch"]
        answers, correct = [r["answer"] for r in group], [r["correct"] for r in group]
        answer, ok, share = majority_of(answers, correct)
        first = next((r for r in group if r["answer"] and _vote_key(r["answer"]) == _vote_key(answer)),
                     group[0])
        tokens = [r["output_tokens"] for r in group if r["output_tokens"] is not None]
        voted.append({
            **first, "id": sid, "answer": answer, "correct": ok,
            "output_tokens": round(sum(tokens) / len(tokens), 1) if tokens else None,
            "seconds": round(sum(r["seconds"] for r in group), 2),
            "votes": answers, "votes_correct": correct, "vote_share": round(share, 3),
        })
    return voted


def _sample_row(sample: Any) -> dict[str, Any]:
    score = next(iter((sample.scores or {}).values()), None)
    meta = (score.metadata or {}) if score else {}
    usage = getattr(sample.output, "usage", None)
    return {
        "id": str(sample.id),
        "target": sample.target if isinstance(sample.target, str) else str(sample.target),
        "answer": (score.answer if score else None) or None,
        "correct": bool(score and str(score.value) == CORRECT),
        # How the scorer matched: exact / tolerance / similarity / *_mismatch / missing.
        "match": meta.get("match"),
        "output_tokens": getattr(usage, "output_tokens", None),
        # Did the model follow the answer-format instruction, and did we parse anything?
        "has_marker": bool(meta.get("has_marker")),
        "parsed": bool(score and score.answer),
        "seconds": round(sample.total_time or 0.0, 2),
        "epoch": getattr(sample, "epoch", 1),
    }


def sample_transitions(before: list[dict[str, Any]], after: list[dict[str, Any]]) -> dict[str, int]:
    """Paired per-sample outcome: what distillation fixed vs. broke on the same samples.

    Both phases' decoding is recorded in their results: with a voted `after` the flips include
    what the vote adds.
    """
    after_by_id = {r["id"]: r for r in after}
    counts = {"fixed": 0, "broken": 0, "both_correct": 0, "both_wrong": 0}
    for b in before:
        a = after_by_id.get(b["id"])
        if a is None:
            continue
        if a["correct"] and not b["correct"]:
            counts["fixed"] += 1
        elif b["correct"] and not a["correct"]:
            counts["broken"] += 1
        elif b["correct"]:
            counts["both_correct"] += 1
        else:
            counts["both_wrong"] += 1
    return counts


def aggregate_extras(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Secondary metrics the dashboard charts next to accuracy."""
    if not rows:
        return {}
    n = len(rows)
    tokens = [r["output_tokens"] for r in rows if r["output_tokens"] is not None]
    return {
        "format_rate": sum(r["has_marker"] for r in rows) / n,
        "parse_rate": sum(r["parsed"] for r in rows) / n,
        "mean_output_tokens": round(sum(tokens) / len(tokens), 1) if tokens else None,
        "mean_sample_seconds": round(sum(r["seconds"] for r in rows) / n, 2),
        # Correct answers credited by a tolerant / symbolic path rather than plain
        # equality -- if this is large, the accuracy number is leaning on the loosened
        # matching. `math_verify` covers sympy-based symbolic equivalence (added when
        # switching from the hand-rolled scorer).
        "lenient_matches": sum(r["match"] in {"tolerance", "similarity", "math_verify"} for r in rows),
        **vote_metrics(rows),
    }


def vote_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """maj@1..k and pass@k from the per-row votes (empty for greedy phases)."""
    if not rows or "votes" not in rows[0]:
        return {}
    k = min(len(r["votes"]) for r in rows)
    maj = {j: sum(majority_of(r["votes"][:j], r["votes_correct"][:j])[1] for r in rows) / len(rows)
           for j in range(1, k + 1)}
    return {
        "votes": k,
        "maj_at": {str(j): round(a, 4) for j, a in maj.items()},
        "pass_at_k": round(sum(any(r["votes_correct"]) for r in rows) / len(rows), 4),
        "mean_vote_share": round(sum(r["vote_share"] for r in rows) / len(rows), 3),
    }


def run_phase(
    phase: str, model_ref: str, spec: TaskSpec, args: argparse.Namespace
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate one model; returns (aggregate result, per-sample rows)."""
    model_name, model_args = model_args_for(model_ref, phase, args)
    print(f"\n=== eval [{phase}] {model_ref}  (provider {model_name}) ===", flush=True)
    # A served model (the teacher, or the student with --student-url) is a remote service:
    # many more concurrent requests, and no local chat override (the server applies its
    # own template).
    served = model_name.startswith("openai-api/")
    connections = args.teacher_connections if served else args.batch_size
    chat_template = "auto" if served else args.chat_template
    # ONE MODEL ON THE GPU AT A TIME. Inspect memoizes models by default, so with
    # EVAL_PHASE=all the `before` model stays resident while `after` loads next to it
    # on the same A10G -- at 4B, two bf16 models plus generation buffers do not fit. Build each
    # local model un-memoized and free it when the phase ends. The teacher phase is only
    # an HTTP client of the existing teacher-vllm Service and holds no GPU memory here --
    # as is a served student phase.
    t0 = time.time()
    if served:
        logs = _run_inspect(phase, model_name, model_args, chat_template, connections, spec, args)
    else:
        # Exiting the context calls the hf provider's close(), which drops the weights.
        with get_model(model_name, memoize=False, **model_args) as model:
            logs = _run_inspect(phase, model, {}, chat_template, connections, spec, args)
        del model
        free_gpu()
    return summarise_phase(phase, model_ref, model_name, logs, args, t0)


def free_gpu() -> None:
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def decoding_for(phase: str, args: argparse.Namespace) -> tuple[int, float]:
    """(votes, temperature) for one phase: voting phases sample, the rest use --temperature."""
    if args.votes > 1 and phase in args.vote_phases:
        return args.votes, args.vote_temperature
    return 1, args.temperature


def _run_inspect(phase, model, model_args, chat_template, connections, spec, args):
    # A served model's chat template runs server-side in vLLM; chat_template_kwargs is how
    # an OpenAI-compatible request reaches it. Same thinking switch as the local phases.
    served = isinstance(model, str) and model.startswith("openai-api/")
    extra = ({"extra_body": {"chat_template_kwargs": {"enable_thinking": args.enable_thinking}}}
             if served else {})
    return inspect_eval(
        math_reasoning(
            task_key=args.task,
            split=args.split or None,
            limit=args.limit,
            select=args.select,
            fewshot=args.fewshot,
            chat_template=chat_template,
            shuffle_seed=args.shuffle_seed,
            answer_rel_tol=args.answer_rel_tol,
            text_similarity=args.text_similarity,
            votes=decoding_for(phase, args)[0],
        ),
        model=model,
        model_args=model_args,
        log_dir=str(Path(args.output_dir) / "logs" / phase),
        max_tokens=args.max_tokens,
        temperature=decoding_for(phase, args)[1],
        max_connections=connections,
        # From the task spec, so a few-shot completion cannot run on into a hallucinated
        # next problem -- and so train and eval agree on where a rollout ends.
        stop_seqs=list(spec.stop_seqs),
        display="plain",
        max_samples=connections,
        fail_on_error=0.1,
        **extra,
    )


def summarise_phase(
    phase: str, model_ref: str, model_name: str, logs: Any, args: argparse.Namespace, t0: float
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    log = logs[0]
    if log.status != "success":
        message = getattr(log.error, "message", log.status)
        raise RuntimeError(f"phase '{phase}' failed: {message}")

    scores = {
        s.name: {k: v.value for k, v in s.metrics.items()} for s in (log.results.scores or [])
    }
    metrics = next(iter(scores.values()), {})
    rows = per_sample_records(log)
    samples_path = Path(args.output_dir) / f"samples_{phase}.jsonl"
    with samples_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    result = {
        "phase": phase,
        "model": model_ref,
        "provider_model": model_name,
        "accuracy": metrics.get("accuracy"),
        "stderr": metrics.get("stderr"),
        "samples": log.results.completed_samples,
        "scores": scores,
        "log_file": log.location,
        "samples_file": str(samples_path),
        "seconds": round(time.time() - t0, 1),
        "decoding": dict(zip(("votes", "temperature"), decoding_for(phase, args))),
        **aggregate_extras(rows),
    }
    print(f"=== [{phase}] accuracy={result['accuracy']} "
          f"over {result['samples']} samples in {result['seconds']}s ===", flush=True)
    return result, rows


def carry_forward(
    summary: dict[str, Any], rows_by_phase: dict[str, list[dict[str, Any]]],
    summary_path: Path, args: argparse.Namespace,
) -> None:
    """Reuse phases from an earlier summary with the SAME config, so phases can run separately.

    E.g. EVAL_PHASE=both first, then EVAL_PHASE=teacher: the second run picks up before/after
    from summary.json and still reports headroom_recovered. A phase is only reused when the
    whole eval config (dataset, limit, select, few-shot, max_tokens, ...) matches -- otherwise
    the phases did not see the same samples and comparing them would be meaningless.
    """
    if not summary_path.exists():
        return
    try:
        prev = json.loads(summary_path.read_text())
    except (OSError, json.JSONDecodeError):
        return
    if prev.get("task") != summary["task"] or prev.get("config") != summary["config"]:
        print(f"[carry-forward] {summary_path} has a different task/config; not reusing it",
              flush=True)
        return
    for phase, result in (prev.get("results") or {}).items():
        if phase in summary["results"]:
            continue
        # `after` must be the same checkpoint, or the old number describes another model.
        if phase == "after" and result.get("model") != args.ft_model_dir:
            continue
        # ...and decoded the same way (results written before `decoding` existed were single
        # answers at the config temperature).
        prev_decoding = result.get("decoding") or {"votes": 1, "temperature": prev["config"]["temperature"]}
        if (prev_decoding["votes"], prev_decoding["temperature"]) != decoding_for(phase, args):
            continue
        samples_file = Path(result.get("samples_file", ""))
        if not samples_file.is_file():
            continue
        summary["results"][phase] = result
        rows_by_phase[phase] = [json.loads(l) for l in samples_file.read_text().splitlines()
                                if l.strip()]
        print(f"[carry-forward] reusing phase '{phase}' (accuracy={result.get('accuracy')}) "
              f"from {summary_path}", flush=True)


def main() -> int:
    env = os.environ.get
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", default=env("EVAL_PHASE", "all"),
                    help="comma-separated subset of before,after,teacher -- or 'all'/'both' "
                         "(default: all three, then diff)")
    ap.add_argument("--task", choices=sorted(TASK_SPECS), default=env("TASK", "math500"),
                    help="gsm8k suits a 0.8B-2B student; math500 has headroom for a 4B")
    ap.add_argument("--base-model", default=env("EVAL_BASE_MODEL", env("STUDENT_MODEL", "Qwen/Qwen3.5-2B")),
                    help="pre-distillation student: HF Hub id or local dir")
    ap.add_argument("--ft-model-dir", default=env("EVAL_FT_MODEL_DIR", "/fsx/opd/gsm8k-opd/student-final"),
                    help="post-distillation checkpoint (merged weights) from train_distill.py")
    ap.add_argument("--teacher-url", default=env("TEACHER_URL", "http://teacher-vllm.default.svc.cluster.local:8000/v1"),
                    help="OpenAI-compatible base URL of the served teacher")
    ap.add_argument("--teacher-model", default=env("TEACHER_MODEL_ID", "Qwen/Qwen3.5-9B"))
    ap.add_argument("--student-url", default=env("EVAL_STUDENT_URL", ""),
                    help="OpenAI-compatible URL of the student sampler. When set, before/after "
                         "are generated by vLLM (after = a LoRA adapter dir, registered at "
                         "runtime) instead of local hf; empty = local hf provider")
    ap.add_argument("--teacher-connections", type=int, default=int(env("EVAL_TEACHER_CONNECTIONS", "32")),
                    help="concurrent requests to the teacher service (it batches server-side)")
    ap.add_argument("--split", default=env("EVAL_SPLIT", ""),
                    help="override the task's test split (default: the spec's)")
    ap.add_argument("--limit", type=int, default=int(env("EVAL_LIMIT", "500")),
                    help="eval samples per phase, identical across phases. 500 keeps the "
                         "standard error near +-1.7pp; 100 gives +-3.6pp, wider than any "
                         "realistic gain")
    ap.add_argument("--select", choices=["all", "tail", "head", "shuffle"],
                    default=env("EVAL_SELECT", "all"),
                    help="which --limit rows of the split to use (default: all)")
    ap.add_argument("--fewshot", type=int, default=int(env("EVAL_FEWSHOT", "0")),
                    help="worked examples in the prompt -- MUST match TRAIN_FEWSHOT")
    ap.add_argument("--shuffle-seed", type=int, default=int(env("EVAL_SHUFFLE_SEED", "42")))
    ap.add_argument("--answer-rel-tol", type=float,
                    default=float(env("EVAL_ANSWER_REL_TOL", str(DEFAULT_ANSWER_REL_TOL))),
                    help="relative tolerance for treating two numeric answers as equal "
                         "(0 = require exact equality after normalisation)")
    ap.add_argument("--text-similarity", type=float,
                    default=float(env("EVAL_TEXT_SIMILARITY", str(DEFAULT_TEXT_SIMILARITY))),
                    help="string-similarity threshold used only when neither the answer "
                         "nor the gold parses as a number")
    ap.add_argument("--max-tokens", type=int, default=int(env("EVAL_MAX_TOKENS", "512")))
    ap.add_argument("--temperature", type=float, default=float(env("EVAL_TEMPERATURE", "0")),
                    help="0 = greedy decoding (deterministic before/after comparison)")
    ap.add_argument("--votes", type=int, default=int(env("EVAL_VOTES", "1")),
                    help="majority vote over this many samples per problem for the "
                         "--vote-phases; 1 = a single answer per problem")
    ap.add_argument("--vote-phases", default=env("EVAL_VOTE_PHASES", "after"),
                    help="comma-separated phases that vote (default: after), or 'all'")
    ap.add_argument("--vote-temperature", type=float,
                    default=float(env("EVAL_VOTE_TEMPERATURE", "0.7")),
                    help="sampling temperature of the voting phases (must be > 0)")
    ap.add_argument("--batch-size", type=int, default=int(env("EVAL_BATCH_SIZE", "8")),
                    help="samples generated concurrently (the HF provider batches them)")
    ap.add_argument("--device", default=env("EVAL_DEVICE", "cuda:0"))
    ap.add_argument("--torch-dtype", default=env("EVAL_TORCH_DTYPE", "bfloat16"),
                    help="pinned identically for the before and after phases")
    ap.add_argument("--trust-remote-code", action="store_true",
                    default=env("EVAL_TRUST_REMOTE_CODE", env("TRUST_REMOTE_CODE", "0")) == "1")
    ap.add_argument("--chat-template", choices=["auto", "raw", "none"],
                    default=env("EVAL_CHAT_TEMPLATE", "auto"),
                    help="auto: tokenizer template (instruct models, the default); "
                         "raw: plain completion prompt (-Base models). MUST match CHAT_TEMPLATE")
    ap.add_argument("--enable-thinking", type=int, choices=[0, 1],
                    default=int(env("ENABLE_THINKING", "0") or "0"),
                    help="Qwen3.5 thinking mode for every phase. MUST match ENABLE_THINKING "
                         "(the trainer's prompts); 1 also needs max_tokens ~4096")
    ap.add_argument("--output-dir", default=env("EVAL_DIR", "/fsx/opd/eval"))
    ap.add_argument("--summary-name", default=env("EVAL_SUMMARY_NAME", "summary.json"))
    args = ap.parse_args()
    args.enable_thinking = bool(args.enable_thinking)
    args.vote_phases = (["before", "after", "teacher"] if args.vote_phases == "all" else
                        [p.strip() for p in args.vote_phases.split(",") if p.strip()])
    if args.votes > 1 and args.vote_temperature <= 0:
        print("ERROR: --votes > 1 needs --vote-temperature > 0: greedy samples are all identical",
              file=sys.stderr)
        return 2
    if args.student_url and args.chat_template != "auto":
        print("ERROR: --student-url applies the server's chat template; it needs "
              "--chat-template auto", file=sys.stderr)
        return 2

    spec = get_task_spec(args.task)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    requested = (["before", "after", "teacher"] if args.phase in ("all", "") else
                 ["before", "after"] if args.phase == "both" else
                 [p.strip() for p in args.phase.split(",") if p.strip()])
    unknown = [p for p in requested if p not in ("before", "after", "teacher")]
    if unknown:
        print(f"ERROR: unknown phase(s) {unknown}; choose from before,after,teacher",
              file=sys.stderr)
        return 2

    phases: list[tuple[str, str]] = []
    for phase in requested:
        if phase == "before":
            phases.append(("before", args.base_model))
        elif phase == "after":
            ft = Path(args.ft_model_dir)
            if not ft.exists():
                print(f"ERROR: fine-tuned checkpoint {ft} not found -- has the student Job "
                      f"finished writing it?", file=sys.stderr)
                return 2
            phases.append(("after", args.ft_model_dir))
        else:
            phases.append(("teacher", args.teacher_model))

    summary: dict[str, Any] = {
        "task": args.task,
        "config": {
            "dataset": f"{spec.dataset}/{spec.dataset_name}:{args.split or spec.test_split}",
            "answer_style": spec.answer_style,
            "limit": args.limit, "select": args.select, "fewshot": args.fewshot,
            "max_tokens": args.max_tokens, "temperature": args.temperature,
            "chat_template": args.chat_template,
            "enable_thinking": bool(args.enable_thinking),
            "torch_dtype": args.torch_dtype,
            "answer_rel_tol": args.answer_rel_tol,
            "text_similarity": args.text_similarity,
            # vLLM and hf greedy decoding diverge, so phases from different backends are not
            # comparable: recording the backend keeps carry_forward from mixing them. Only
            # added for vllm, so existing hf summaries still carry forward unchanged.
            **({"student_backend": "vllm"} if args.student_url else {}),
        },
        "results": {},
    }
    rows_by_phase: dict[str, list[dict[str, Any]]] = {}
    is_child = os.environ.get(CHILD_ENV) == "1"
    if len(phases) > 1 and not is_child:
        # ONE PROCESS PER PHASE. Freeing a model in-process is not enough: Inspect's hf
        # provider generates on a module-level daemon thread whose locals keep the last
        # batch's `model.generate` alive, so `before`'s 8 GiB stays on the A10G and `after`
        # OOMs loading beside it. A child process per phase returns the GPU to the driver
        # on exit. Each child writes summary.json; carry_forward below merges them.
        for phase, _ in phases:
            print(f"\n>>> phase '{phase}' in a subprocess", flush=True)
            rc = subprocess.run(
                [sys.executable, os.path.abspath(__file__), *sys.argv[1:], "--phase", phase],
                env={**os.environ, CHILD_ENV: "1"},
            ).returncode
            if rc != 0:
                print(f"ERROR: phase '{phase}' exited with {rc}", file=sys.stderr)
                return rc
    else:
        for phase, model_ref in phases:
            summary["results"][phase], rows_by_phase[phase] = run_phase(phase, model_ref, spec, args)
            free_gpu()
    carry_forward(summary, rows_by_phase, out / args.summary_name, args)

    before = summary["results"].get("before")
    after = summary["results"].get("after")
    teacher = summary["results"].get("teacher")

    def acc(result: dict[str, Any] | None) -> float | None:
        return result["accuracy"] if result and result["accuracy"] is not None else None

    a_before, a_after, a_teacher = acc(before), acc(after), acc(teacher)
    if a_before is not None and a_after is not None:
        summary["delta_accuracy"] = a_after - a_before
        summary["transitions"] = sample_transitions(rows_by_phase["before"], rows_by_phase["after"])
        print(f"\n### {args.task} accuracy  before={a_before:.4f}  after={a_after:.4f}  "
              f"delta={a_after - a_before:+.4f}", flush=True)
        print(f"### per-sample flips {summary['transitions']}", flush=True)
    if a_before is not None and a_after is not None and a_teacher is not None:
        gap = a_teacher - a_before
        # The only interpretable summary of a distillation run: the student cannot exceed
        # the teacher, so what matters is the share of the available gap it closed.
        summary["teacher_gap"] = gap
        summary["headroom_recovered"] = (a_after - a_before) / gap if gap > 0 else None
        if gap <= 0:
            print(f"\n### teacher={a_teacher:.4f} is NOT ahead of the base student "
                  f"({a_before:.4f}). There is no headroom to distil; on-policy "
                  f"distillation cannot help here. Pick a stronger teacher or a harder "
                  f"task before tuning anything else.", flush=True)
        else:
            print(f"\n### teacher={a_teacher:.4f}  gap={gap:+.4f}  "
                  f"headroom recovered={summary['headroom_recovered']:.1%}", flush=True)

    summary_path = out / args.summary_name
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {summary_path}", flush=True)

    # Also emit the whole dashboard payload (aggregates + per-sample rows) between sentinels,
    # so the notebook can rebuild the charts from `kubectl logs job/opd-student-eval` alone --
    # no /fsx access needed once this Job's pod has Completed.
    if is_child:
        return 0   # the parent prints the one merged payload the notebook parses
    payload = dict(summary, samples={p: rows for p, rows in rows_by_phase.items()})
    print(DASHBOARD_BEGIN, flush=True)
    print(json.dumps(payload, separators=(",", ":")), flush=True)
    print(DASHBOARD_END, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
