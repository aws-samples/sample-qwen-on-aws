"""Compare merged student checkpoints against the base model, tensor by tensor, on CPU.

Answers "what is actually in this directory?" -- e.g. whether an old `student-final-bf16`
is a real distilled model or effectively the base, and how far two runs moved the weights.

    python3 compare_ckpts.py --base Qwen/Qwen3.5-2B \
        /fsx/opd/gsm8k-opd/student-final /fsx/opd/other-run/student-final

Per checkpoint it prints: file timestamps, config (architectures, dtype), keys missing or
extra versus the base (a naming mismatch means the eval loads some weights randomly
initialised), and the relative weight change |W - W_base| / |W_base| overall and for the
most-changed tensors. Reads one tensor at a time via safetensors, so RAM stays small.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import torch
from safetensors import safe_open


def resolve(ref: str) -> Path:
    p = Path(ref)
    if p.exists():
        return p
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(ref, allow_patterns=["*.safetensors", "*.json"]))


def canon(key: str) -> str:
    # A VLM checkpoint nests the text model under `model.language_model.`; a CausalLM save
    # does not. Normalise so the two layouts line up.
    return key.replace("model.language_model.", "model.")


def index(path: Path) -> dict[str, Path]:
    out = {}
    for f in sorted(path.glob("*.safetensors")):
        with safe_open(str(f), "pt") as sf:
            for k in sf.keys():
                out[canon(k)] = (f, k)
    return out


def load(entry) -> torch.Tensor:
    f, k = entry
    with safe_open(str(f), "pt") as sf:
        return sf.get_tensor(k)


def describe(path: Path) -> None:
    files = sorted(path.glob("*.safetensors"))
    newest = max((f.stat().st_mtime for f in files), default=0)
    size = sum(f.stat().st_size for f in files) / 2**30
    cfg = {}
    if (path / "config.json").exists():
        cfg = json.loads((path / "config.json").read_text())
    print(f"  written   : {dt.datetime.fromtimestamp(newest):%Y-%m-%d %H:%M}  "
          f"({len(files)} shard(s), {size:.2f} GiB)")
    print(f"  config    : architectures={cfg.get('architectures')} "
          f"dtype={cfg.get('torch_dtype') or cfg.get('dtype')} model_type={cfg.get('model_type')}")
    if (path / "adapter_config.json").exists():
        print("  NOTE      : adapter_config.json present -- this is an adapter, not merged weights")


@torch.no_grad()
def compare(base_idx: dict, ck_idx: dict, top: int) -> None:
    # Vision tower and the multi-token-prediction head (`mtp.*`, speculative decoding only)
    # are not part of the text CausalLM, so a merged student legitimately omits them.
    text_base = {k for k in base_idx
                 if "visual" not in k and "vision" not in k and not k.startswith("mtp.")}
    missing = sorted(text_base - ck_idx.keys())
    extra = sorted(ck_idx.keys() - base_idx.keys())
    print(f"  keys      : {len(ck_idx)} tensors; missing vs base {len(missing)}, extra {len(extra)}")
    for k in missing[:5]:
        print(f"      missing: {k}")
    for k in extra[:5]:
        print(f"      extra  : {k}")

    num = den = 0.0
    changed = total = 0
    rows = []
    for k in sorted(ck_idx.keys() & base_idx.keys()):
        w, w0 = load(ck_idx[k]).float(), load(base_idx[k]).float()
        if w.shape != w0.shape:
            print(f"      SHAPE MISMATCH {k}: {tuple(w.shape)} vs base {tuple(w0.shape)}")
            continue
        d2, b2 = (w - w0).pow(2).sum().item(), w0.pow(2).sum().item()
        num += d2; den += b2; total += 1
        if d2 > 0:
            changed += 1
            rows.append((k, (d2 / max(b2, 1e-30)) ** 0.5))
    print(f"  change    : {changed}/{total} tensors differ from base; "
          f"overall |dW|/|W| = {(num / max(den, 1e-30)) ** 0.5:.3e}")
    rows.sort(key=lambda r: -r[1])
    for k, rel in rows[:top]:
        print(f"      {rel:.3e}  {k}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoints", nargs="+", help="merged checkpoint dirs to inspect")
    ap.add_argument("--base", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--top", type=int, default=8, help="most-changed tensors to list")
    args = ap.parse_args()

    base_idx = index(resolve(args.base))
    print(f"base {args.base}: {len(base_idx)} tensors")
    for ck in args.checkpoints:
        path = Path(ck)
        print(f"\n== {ck}")
        if not path.exists():
            print("  does not exist")
            continue
        describe(path)
        compare(base_idx, index(path), args.top)
    print("\nReading it: |dW|/|W| ~0 or only norm tensors changed => effectively the base model. "
          "Missing keys => the eval silently re-initialises those weights. A run trained at a "
          "higher LR moving the weights several times further is expected, not by itself damage.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
