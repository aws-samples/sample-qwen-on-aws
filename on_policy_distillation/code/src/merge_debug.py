"""Merge a LoRA adapter into the base student OUTSIDE the trainer, and measure what bf16 costs.

Debug tool for "did the merge lose the distillation gain?". Runs on CPU, so it can go in any
pod that mounts /fsx and has transformers + peft (the trainer image does).

    python3 merge_debug.py --adapter /fsx/opd/gsm8k-opd/student-step100 \
        --out /fsx/opd/gsm8k-opd/student-final-remerged

Steps:
  1. merge in fp32:  W = W_base + B @ A * scaling      (same maths as save_student)
  2. report how much of the LoRA delta survives rounding W to bf16, per layer and overall
  3. compare next-token predictions of the three models the pipeline actually uses:
       unmerged  -- bf16 base + LoRA adapter   (what the vLLM sampler served during training)
       fp32      -- merged, fp32               (the exact merge)
       bf16      -- merged, rounded to bf16    (what the eval loads)
  4. save the bf16 merge to --out (skip with --no-save)

If step 2 shows a large lost fraction or step 3 shows bf16 disagreeing with unmerged, the
bf16 merge is eating the gain. If they agree, look elsewhere (checkpoint path, eval size).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--adapter", required=True, help="LoRA dir: student-stepN or adapter-live")
    ap.add_argument("--out", default=None, help="where to save the bf16 merge")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--task", default="math500")
    ap.add_argument("--fewshot", type=int, default=4)
    ap.add_argument("--n-prompts", type=int, default=8)
    ap.add_argument("--trust-remote-code", action="store_true")
    return ap.parse_args()


@torch.no_grad()
def rounding_report(base_bf16: dict[str, torch.Tensor], merged_fp32) -> None:
    """How much of delta = W_merged - W_base is lost when W_merged is stored as bf16."""
    print("\n[2] bf16 rounding of the merged delta (only layers LoRA touched)")
    tot_d2 = tot_err2 = 0.0
    tot_n = tot_zeroed = 0
    rows = []
    for name, w in merged_fp32.state_dict().items():
        if name not in base_bf16 or not w.is_floating_point():
            continue
        w0 = base_bf16[name].float()
        delta = w - w0
        if not delta.any():
            continue  # layer untouched by LoRA
        kept = w.to(torch.bfloat16).float() - w0          # delta as the bf16 checkpoint stores it
        err2 = (kept - delta).pow(2).sum().item()
        d2 = delta.pow(2).sum().item()
        zeroed = ((kept == 0) & (delta != 0)).sum().item()  # updates rounded away entirely
        n = (delta != 0).sum().item()
        tot_d2 += d2; tot_err2 += err2; tot_n += n; tot_zeroed += zeroed
        rows.append((name, (err2 / d2) ** 0.5, zeroed / max(n, 1),
                     (d2 ** 0.5) / max(w0.norm().item(), 1e-12)))
    if not rows:
        print("    no layer differs from the base -- the adapter is empty or did not load")
        return
    rows.sort(key=lambda r: -r[1])
    print(f"    {'layer':70s} {'rel_err':>8s} {'zeroed':>7s} {'|dW|/|W|':>9s}")
    for name, rel, z, mag in rows[:10]:
        print(f"    {name[:70]:70s} {rel:8.3f} {z:7.1%} {mag:9.2e}")
    print(f"    ... {len(rows)} layers. OVERALL: relative error of the delta after bf16 = "
          f"{(tot_err2 / tot_d2) ** 0.5:.3f}, elements whose update rounded to zero = "
          f"{tot_zeroed / tot_n:.1%}")
    print("    rule of thumb: rel_err < ~0.1 is harmless; > ~0.5 means bf16 storage is "
          "discarding most of the update")


@torch.no_grad()
def compare_predictions(models: dict, tokenizer, prompts: list[str]) -> None:
    print(f"\n[3] next-token agreement on {len(prompts)} prompts (all positions)")
    ref_name = "unmerged"
    stats = {k: [0, 0, 0.0] for k in models if k != ref_name}   # agree, total, sum KL
    for p in prompts:
        ids = tokenizer(p, return_tensors="pt", add_special_tokens=False).input_ids
        logp = {}
        for k, (m, dev) in models.items():
            logp[k] = torch.log_softmax(m(ids.to(dev)).logits.float().cpu()[0], -1)
        ref = logp[ref_name]
        for k in stats:
            stats[k][0] += (logp[k].argmax(-1) == ref.argmax(-1)).sum().item()
            stats[k][1] += ref.shape[0]
            # KL(unmerged || k), per position
            stats[k][2] += (ref.exp() * (ref - logp[k])).sum(-1).sum().item()
    for k, (agree, n, kl) in stats.items():
        print(f"    {k:5s} vs unmerged: top-1 agreement {agree / n:.2%}, mean KL {kl / n:.2e} nats")
    print("    expect fp32 ~100% / ~0. If bf16 is also >~98% with KL ~1e-3 or less, the merge is "
          "faithful to what training produced.")


def main() -> int:
    args = parse_args()
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.adapter if (Path(args.adapter) / "tokenizer_config.json").exists()
                                        else args.base, trust_remote_code=args.trust_remote_code)
    kw = dict(low_cpu_mem_usage=True, trust_remote_code=args.trust_remote_code)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[1] loading base {args.base} (bf16) + adapter {args.adapter}")
    base = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16, **kw)
    base_bf16 = {k: v.clone() for k, v in base.state_dict().items()}
    unmerged = PeftModel.from_pretrained(base, args.adapter).eval()   # adapter stays separate

    print("[1] merging in fp32 on CPU (same maths as save_student)")
    base32 = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.float32, **kw)
    merged32 = PeftModel.from_pretrained(base32, args.adapter).to(torch.float32).merge_and_unload().eval()

    rounding_report(base_bf16, merged32)
    del base_bf16

    merged16 = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16, **kw)
    merged16.load_state_dict({k: v.to(torch.bfloat16) for k, v in merged32.state_dict().items()})
    merged16.eval()

    try:
        from data import load_eval_questions
        from prompts import build_prompt, get_task_spec
        spec = get_task_spec(args.task)
        qs = sorted(load_eval_questions(spec))[: args.n_prompts]
        prompts = [build_prompt(spec, q, args.fewshot, tok, chat=True) for q in qs]
    except Exception as exc:  # the comparison still works on generic text
        print(f"    (could not build task prompts: {exc}; using a generic prompt)")
        prompts = ["Solve: what is 17 * 23? Put the final answer in \\boxed{}."]

    # unmerged and bf16 on the GPU if there is one (that is how sampler/eval run them);
    # fp32 stays on CPU so the node does not need 16 GiB of spare VRAM.
    compare_predictions({"unmerged": (unmerged.to(dev), dev),
                         "fp32": (merged32, "cpu"),
                         "bf16": (merged16.to(dev), dev)}, tok, prompts)

    if args.out and not args.no_save:
        out = Path(args.out)
        merged16 = merged16.cpu()
        merged16.save_pretrained(out, safe_serialization=True)
        merged16.config.torch_dtype = torch.bfloat16
        merged16.config.save_pretrained(out)
        tok.save_pretrained(out)
        print(f"\n[4] wrote bf16 merge to {out}  (point EVAL_FT_MODEL_DIR here)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
