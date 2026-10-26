"""Pre-flight checks -- run these before spending GPU-hours on a distillation run.

Every check here corresponds to a failure that is either silent (the run completes and the
metrics look plausible, but the result is meaningless) or expensive (the teacher will not
start at all, twenty minutes into loading its weights).

    python3 preflight.py                       # all checks
    python3 preflight.py --skip-teacher        # config/memory only, no cluster needed

Checks:
  1. tensor-parallel divisibility -- a teacher whose KV heads are not divisible by TP simply
     cannot be served at that TP, and vLLM only tells you after loading.
  2. teacher quantisation + memory fit at TP on the target GPU. Two gates before the
     arithmetic: the method must still be in the serving build's registry (vLLM deleted
     bitsandbytes), and the GPU must meet the scheme's minimum compute capability (an FP8
     checkpoint needs sm_89 and A10G is sm_86 -- both failures happen minutes into pod
     startup, which is the expensive place to find them).
  3. shared tokenizer -- the teacher is scored by raw token id, so a differing tokenizer
     means it scores different text than the student generated. Silent and fatal.
  4. GPU topology -- A10G has no NVLink, so TP all-reduce crosses PCIe; `SYS` links are
     far worse than `PHB`/`PXB` and worth knowing about before blaming the trainer.
  5. teacher echo-logprobs actually populated -- missing ones would turn the advantage into
     self-reinforcement while `teacher_kl` still falls.
  6. student training memory fit for the chosen precision.
  7. train/eval prompt-format agreement.
  8. the single-node GPU partition -- teacher TP x REPLICAS + trainer + sampler must fit
     NODE_GPUS. With teacher and student sharing one ml.g5.24xlarge this is the constraint
     that decides whether anything runs at all, and getting it wrong shows up as a pod stuck
     Pending forever rather than as an error. As shipped (TP=2 x 1) the partition is
     2 + 1 + 1 = 4 of 4, exactly full.
  9. the student sampler serves STUDENT_MODEL -- a sampler left on another model (e.g. by
     the baseline loop) rejects the trainer's adapter with a bare tensor-size mismatch.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field

BYTES_PER_PARAM = {"bfloat16": 2, "float16": 2, "float8": 1, "int8": 1, "int4": 0.5}

# When a quantisation method is set it -- not --dtype -- determines the weight footprint.
# The values are bytes/param for the packed weights; QUANT_OVERHEAD covers the per-group
# scales and zero-points, which are stored at higher precision and are not negligible at
# group_size=128 (roughly +10% for 4-bit).
#
# These are only a fallback, and a poor one for 4-bit: real 4-bit repos are PARTIALLY
# quantised (Qwen3.5-27B-GPTQ-Int4 leaves attention, embeddings, lm_head and the vision tower
# in bf16, so it weighs 28.2 GiB, not the 13.5 GiB that 0.5 bytes/param predicts -- a 2x error
# in the one number that decides whether the teacher fits). Set TEACHER_WEIGHTS_GIB to the
# checkpoint's measured on-disk size and this table is bypassed entirely. For an unquantised
# teacher the bytes/param estimate is exact, but setting the measured size is still preferred.
QUANT_BYTES_PER_PARAM = {
    "awq": 0.5, "awq_marlin": 0.5, "gptq": 0.5, "gptq_marlin": 0.5,
    "marlin": 0.5, "compressed-tensors": 0.5,
    "fp8": 1.0, "int8": 1.0, "bitsandbytes": 0.5,
}
QUANT_OVERHEAD = 1.10

# Methods vLLM used to accept and has since dropped from its quantization registry. A pod
# configured with one of these dies at startup on a pydantic "Unknown quantization method"
# ValueError, before it loads a single weight -- so catch it here, where the message can say
# what to do instead.
RETIRED_QUANT_METHODS = {
    "bitsandbytes": (
        "vLLM removed bitsandbytes from its quantization registry (present in v0.11.0, "
        "absent on main). Pinning an older image is NOT a way out for a Qwen3.5 teacher: "
        "Qwen3_5ForConditionalGeneration only exists in builds newer than that removal, so "
        "no single image has both. In-flight NF4 is simply gone -- use a pre-quantised "
        "W4A16 repo (TEACHER_MODEL=sanskar003/Qwen3.5-9B-AWQ, "
        "TEACHER_QUANTIZATION=compressed-tensors) instead."
    ),
    "deepspeedfp": "removed from vLLM's registry; use a pre-quantised W4A16 repo.",
    "gguf": "removed from vLLM's registry; use a pre-quantised W4A16 safetensors repo.",
    "hqq": "removed from vLLM's registry; use a pre-quantised W4A16 repo.",
    "bitblas": "removed from vLLM's registry; use gptq_marlin or awq_marlin.",
    "gptq_bitblas": "removed from vLLM's registry; use gptq_marlin.",
}

# A method can be in vLLM's registry and STILL be refused by the kernels at load time, because
# the scheme has a minimum CUDA compute capability. This bit us concretely: an FP8 9B teacher
# (RedHatAI/Qwen3.5-9B-FP8-dynamic) looks ideal on paper -- 12.6 GiB, fits TP=1 -- but
# CompressedTensorsW8A8Fp8.get_min_capability() is 89, so on A10G (sm_86) vLLM raises
# "Quantization scheme is not supported for the current GPU. Min capability: 89. Current
# capability: 86." after the image has pulled and the pod has started. The W4A16 schemes
# (CompressedTensorsWNA16, gptq_marlin, awq_marlin) are "Turing and up" = 75, which is why the
# 4-bit route is the only quantised one available on this hardware.
#
# These are FLOOR values per method name. A checkpoint's own config decides which scheme it
# actually selects, so this is a screen for the obvious mistake, not a proof of support.
QUANT_MIN_CAPABILITY = {
    "fp8": 89, "fbgemm_fp8": 89, "modelopt": 89, "modelopt_fp4": 100,
    "modelopt_mxfp8": 100, "mxfp4": 100, "mxfp8": 100, "nvfp4_per_token": 100,
    "fp8_per_tensor": 89, "fp8_per_block": 89, "fp8_per_channel": 89,
    "gptq_marlin": 80, "awq_marlin": 80, "marlin": 80,
    "gptq": 75, "awq": 75, "compressed-tensors": 75,
}


@dataclass
class Report:
    passed: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    def ok(self, msg: str) -> None:
        self.passed.append(msg)
        print(f"  [ok]   {msg}", flush=True)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)
        print(f"  [warn] {msg}", flush=True)

    def fail(self, msg: str) -> None:
        self.failures.append(msg)
        print(f"  [FAIL] {msg}", flush=True)


def env(name: str, default: str) -> str:
    return os.environ.get(name, default)


# ---------------------------------------------------------------------------


#: Dimensions that decide TP divisibility and cache size. On a multimodal Qwen3.5 config these
#: live under `text_config`, NOT at the top level -- see _text_config.
_TEXT_KEYS = ("num_hidden_layers", "hidden_size", "num_attention_heads", "num_key_value_heads",
              "intermediate_size", "vocab_size", "head_dim", "num_experts", "num_local_experts",
              "layer_types", "full_attention_interval", "linear_num_key_heads",
              "linear_num_value_heads", "linear_key_head_dim", "linear_value_head_dim",
              "linear_conv_kernel_dim")


def _text_config(cfg):
    """Return the sub-config that actually holds the transformer dimensions.

    Qwen3.5 is a Qwen3_5ForConditionalGeneration: `config.json` has a `vision_config` and puts
    every text dimension under `text_config`, and HF composite configs do NOT forward attribute
    lookups to their children. So `getattr(cfg, "num_hidden_layers")` is None for this whole
    model family -- which made check [1] report "num_attention_heads=None divisible by 2" and
    check [2] compute a KV cost of exactly 0 KiB/token. Both silently passed while telling us
    nothing, on the one model we actually serve.
    """
    inner = getattr(cfg, "text_config", None)
    if inner is not None and getattr(inner, "num_hidden_layers", None) is not None:
        return inner
    return cfg


def check_tensor_parallel(rep: Report, model: str, tp: int, trust: bool) -> dict:
    """vLLM shards attention heads and the MLP across TP ranks; both must divide evenly."""
    from transformers import AutoConfig

    print(f"\n[1] tensor-parallel divisibility: {model} at TP={tp}", flush=True)
    cfg = AutoConfig.from_pretrained(model, trust_remote_code=trust)
    text = _text_config(cfg)
    facts = {k: getattr(text, k, None) for k in _TEXT_KEYS}
    if text is not cfg:
        facts["_multimodal"] = True
    shown = {k: v for k, v in facts.items()
             if v is not None and k not in ("layer_types", "_multimodal")}
    print(f"        {json.dumps(shown)}", flush=True)

    types = facts.get("layer_types") or []
    if types:
        full = sum(1 for t in types if t == "full_attention")
        facts["_full_attention_layers"] = full
        facts["_linear_attention_layers"] = len(types) - full
        print(f"        hybrid attention: {full} full-attention + {len(types) - full} "
              f"linear-attention layers (1 in {facts.get('full_attention_interval') or '?'})",
              flush=True)
    if facts.get("_multimodal"):
        rep.warn("this is a multimodal checkpoint (Qwen3_5ForConditionalGeneration): vLLM loads "
                 "the vision tower even though this pipeline sends text only, so it costs "
                 "weights but no cache. Its dimensions are under config.text_config.")

    heads = facts["num_attention_heads"]
    kv_heads = facts["num_key_value_heads"] or heads
    inter = facts["intermediate_size"]
    if heads and heads % tp:
        rep.fail(f"num_attention_heads={heads} is not divisible by TP={tp}")
    else:
        rep.ok(f"num_attention_heads={heads} divisible by {tp}")
    if kv_heads and kv_heads % tp:
        rep.fail(f"num_key_value_heads={kv_heads} is not divisible by TP={tp}; "
                 f"TP={tp} is impossible for this model. Use TP={_largest_divisor(kv_heads, tp)}.")
    else:
        rep.ok(f"num_key_value_heads={kv_heads} divisible by {tp}")
    if inter and inter % tp:
        rep.fail(f"intermediate_size={inter} is not divisible by TP={tp}")
    else:
        rep.ok(f"intermediate_size={inter} divisible by {tp}")
    # The linear-attention (gated delta-net) layers are sharded on their own head counts, which
    # are separate numbers from the attention ones and can be the binding constraint at high TP.
    for key in ("linear_num_key_heads", "linear_num_value_heads"):
        n = facts.get(key)
        if n and n % tp:
            rep.fail(f"{key}={n} is not divisible by TP={tp}")
        elif n:
            rep.ok(f"{key}={n} divisible by {tp}")
    if facts.get("num_experts") or facts.get("num_local_experts"):
        rep.warn("this is an MoE: memory is unchanged but compute is much cheaper. "
                 "Add --enable-expert-parallel to the vLLM args.")
    return facts


def _largest_divisor(n: int, upper: int) -> int:
    for candidate in range(min(n, upper), 0, -1):
        if n % candidate == 0:
            return candidate
    return 1


def check_quantization_available(rep: Report, quant: str) -> None:
    """Verify the serving vLLM build actually offers this quantisation method.

    The trainer pod runs the same image as the teacher (TRAIN_IMAGE == VLLM_IMAGE), so
    importing vllm here probes the *real* registry rather than a hardcoded list. A removed
    method (vLLM deleted bitsandbytes) otherwise fails at teacher startup with a pydantic
    ValueError, which looks nothing like a configuration problem.
    """
    if not quant:
        rep.ok("no quantisation requested (teacher served at --dtype)")
        return
    known: set[str] | None = None
    version = "unknown"
    try:
        import vllm
        from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS

        known = set(QUANTIZATION_METHODS)
        version = getattr(vllm, "__version__", "unknown")
    except Exception:  # noqa: BLE001  -- vllm absent (running preflight on a laptop)
        pass

    if known is not None and quant not in known:
        hint = RETIRED_QUANT_METHODS.get(quant, "")
        rep.fail(f"vLLM {version} has no quantization method {quant!r}. "
                 f"{hint or 'It offers: ' + ', '.join(sorted(known))}")
    elif known is None and quant in RETIRED_QUANT_METHODS:
        rep.fail(f"TEACHER_QUANTIZATION={quant}: {RETIRED_QUANT_METHODS[quant]}")
    elif known is None:
        rep.warn(f"cannot import vllm here, so {quant!r} was not verified against the "
                 f"serving build's registry; the teacher pod is where this fails.")
    else:
        rep.ok(f"vLLM {version} supports --quantization {quant}")
    _check_quant_capability(rep, quant)


def _check_quant_capability(rep: Report, quant: str) -> None:
    """Second gate: the method exists, but do THESE GPUs have the silicon for it?

    Being in the registry is necessary and not sufficient -- every scheme declares a minimum
    compute capability and vLLM raises after the pod is already up. See QUANT_MIN_CAPABILITY.
    """
    floor = QUANT_MIN_CAPABILITY.get(quant)
    if floor is None:
        return
    try:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("no CUDA")
        major, minor = torch.cuda.get_device_capability(0)
    except Exception:  # noqa: BLE001  -- no GPU here (laptop / CPU-only pod)
        if floor >= 89:
            rep.warn(f"{quant!r} needs compute capability >= {floor}; no GPU is visible here to "
                     f"check against. A10G is sm_86, so if that is the node, this WILL fail at "
                     f"load with 'Quantization scheme is not supported for the current GPU'.")
        return
    cap = major * 10 + minor
    if cap < floor:
        rep.fail(f"{quant!r} needs compute capability >= {floor} but this GPU is sm_{major}{minor} "
                 f"(={cap}). vLLM raises 'Quantization scheme is not supported for the current "
                 f"GPU' at load. On sm_86 the only quantised route is W4A16: "
                 f"compressed-tensors / gptq_marlin / awq_marlin over a 4-bit checkpoint.")
    else:
        rep.ok(f"{quant!r} is supported on sm_{major}{minor} (needs >= {floor})")
    if quant == "compressed-tensors":
        rep.warn("compressed-tensors is a container format, so the real floor comes from the "
                 "CHECKPOINT: a W4A16 'pack-quantized' repo needs sm_75, but a W8A8 "
                 "'float-quantized' (FP8) one needs sm_89 and will be refused on A10G. Confirm "
                 "config.json's quantization_config.format before trusting this check.")


def check_teacher_memory(rep: Report, facts: dict, params_b: float, tp: int,
                         gpu_gb: float, util: float, max_len: int, dtype: str,
                         quantization: str = "", weights_gib: float = 0.0) -> None:
    print(f"\n[2] teacher quantisation + memory at TP={tp} on {gpu_gb:.1f} GB GPUs", flush=True)
    quant = (quantization or "").strip().lower()
    check_quantization_available(rep, quant)
    if weights_gib > 0:
        # Measured on-disk size of the actual checkpoint. Preferred, because real 4-bit repos
        # quantise only some layers and no bytes/param figure describes them.
        weights_gb = weights_gib
        how = f"{quant or dtype}, measured TEACHER_WEIGHTS_GIB={weights_gib:g}"
    elif quant:
        bytes_per = QUANT_BYTES_PER_PARAM.get(quant, 0.5) * QUANT_OVERHEAD
        how = f"{quant} (~{bytes_per:.2f} bytes/param incl. scales, ESTIMATED)"
        weights_gb = params_b * 1e9 * bytes_per / 2**30
    else:
        bytes_per = BYTES_PER_PARAM.get(dtype, 2)
        how = f"{dtype} ({bytes_per} bytes/param)"
        weights_gb = params_b * 1e9 * bytes_per / 2**30
    per_gpu_weights = weights_gb / tp
    budget = gpu_gb * util
    # CUDA context + NCCL buffers + vLLM's own allocations, measured empirically at TP=4.
    # At TP=1 there is no NCCL communicator at all, so the fixed cost is materially lower.
    # At the shipped TP=2 the NCCL cost is charged.
    overhead = 2.0 if tp > 1 else 1.0
    activations = 1.5
    cache_gb = budget - per_gpu_weights - overhead - activations

    per_token_kb, per_seq_mb, detail = _cache_shape(facts, tp)

    print(f"        weights {weights_gb:.1f} GB total as {how}, "
          f"{per_gpu_weights:.1f} GB/GPU; budget {budget:.1f} GB/GPU at util={util}",
          flush=True)
    if cache_gb <= 0:
        rep.fail(f"no room for the KV cache: weights+overhead = "
                 f"{per_gpu_weights + overhead + activations:.1f} GB exceeds the "
                 f"{budget:.1f} GB budget. Raise TP (but on a shared node every extra rank "
                 f"is a GPU the student loses -- see check [8]), or quantise further, or "
                 f"serve a smaller teacher. Note bf16 Qwen3.5-9B is 18.0 GiB, which is "
                 f"exactly the case that cannot be served at TP=1 on a 22.5 GiB A10G.")
        return

    # Capacity is NOT tokens/kv_per_token when the model is hybrid: each sequence also pins a
    # fixed-size recurrent state in the same pool, which does not grow with length. Report the
    # sequence count both ways so max_num_seqs can be set against the real constraint.
    print(f"        cache pool {cache_gb:.1f} GB/GPU; {detail}", flush=True)
    tokens = int(cache_gb * 2**30 / (per_token_kb * 1024)) if per_token_kb else 0
    per_seq_full = per_seq_mb + (max_len * per_token_kb / 1024 if max_len else 0)
    concurrent = int(cache_gb * 1024 / per_seq_full) if per_seq_full else 0
    print(f"        ~= {tokens:,} tokens of KV, or ~{concurrent} concurrent seqs at "
          f"max_model_len={max_len} ({per_seq_full:.0f} MB each incl. recurrent state)",
          flush=True)
    if cache_gb < 1.0:
        rep.warn(f"only {cache_gb:.1f} GB/GPU left for the cache; lower --max-model-len or "
                 f"--max-num-seqs")
    elif concurrent < 8:
        rep.warn(f"only ~{concurrent} concurrent sequences fit; teacher scoring will "
                 f"serialise. Lower --max-model-len (scoring needs prompt+completion only).")
    else:
        rep.ok(f"{cache_gb:.1f} GB/GPU of cache (~{concurrent} concurrent sequences at "
               f"{max_len} tokens)")
    if quant or dtype not in ("bfloat16", "float16"):
        rep.warn(f"teacher is quantised ({quant or dtype}). Its per-token log-probs ARE the "
                 f"supervision signal, so quantisation error lands directly in the KL target "
                 f"(order 0.01-0.05 nats/token against a 0.3-1.0 nats/token signal). W4A16 of "
                 f"a 9B perturbs more than partial W4A16 of a 27B did, and the teacher is "
                 f"already the accuracy CEILING -- run EVAL_PHASE=before,teacher and check "
                 f"that teacher-before is a wide gap before committing to a long run.")
    if weights_gib <= 0 and quant:
        rep.warn(f"TEACHER_WEIGHTS_GIB is unset, so the {weights_gb:.1f} GB above is an "
                 f"estimate from {params_b:g}B x bytes/param. Real 4-bit repos quantise only "
                 f"some layers -- sanskar003/Qwen3.5-9B-AWQ leaves lm_head, the 24 "
                 f"linear-attention layers and the vision tower in bf16 and is 7.96 GiB, not "
                 f"the ~5.0 GiB this estimate predicts. Set TEACHER_WEIGHTS_GIB to the "
                 f"checkpoint's measured size.")


def _cache_shape(facts: dict, tp: int) -> tuple[float, float, str]:
    """Per-rank cache cost of one token (KiB) and of one sequence's fixed state (MiB).

    Qwen3.5 is a HYBRID: `layer_types` alternates 3 linear-attention layers to 1 full-attention
    layer (full_attention_interval=4). That changes both terms of the arithmetic compared with
    a uniform transformer:

      * only the full-attention layers keep a growing KV cache. For Qwen3.5-27B that is 16 of
        64 layers, so 64 KiB/token, not the 256 KiB/token a per-layer count gives.
      * each linear-attention layer instead holds a fixed recurrent (gated delta-net) state per
        SEQUENCE, sized num_value_heads x value_head_dim x key_head_dim in fp32
        (mamba_ssm_dtype). That is 48 MiB/seq for the 9B and 144 MiB/seq for the 27B, and at
        24+ sequences it dominates.

    Both terms are divided by TP: vLLM shards KV heads and linear-attention heads across ranks,
    so a rank stores only its slice. Comparing an unsharded cost against per-rank free memory
    understates capacity by a factor of TP.
    """
    layers = facts.get("num_hidden_layers") or 0
    kv_heads = facts.get("num_key_value_heads") or facts.get("num_attention_heads") or 0
    head_dim = facts.get("head_dim") or (
        (facts.get("hidden_size") or 0) // max(1, facts.get("num_attention_heads") or 1)
    )
    full = facts.get("_full_attention_layers")
    n_linear = facts.get("_linear_attention_layers") or 0
    if full is None:
        full = layers                      # uniform transformer: every layer caches K and V
    # K and V, 2 bytes each, per full-attention layer.
    per_token_kb = 2 * full * kv_heads * head_dim * 2 / 1024 / tp if full else 0.0
    state_elems = ((facts.get("linear_num_value_heads") or 0)
                   * (facts.get("linear_value_head_dim") or 0)
                   * (facts.get("linear_key_head_dim") or 0))
    per_seq_mb = n_linear * state_elems * 4 / 2**20 / tp
    detail = (f"{per_token_kb:.0f} KiB/token/rank over {full} full-attention layers"
              + (f", + {per_seq_mb:.0f} MB/seq/rank of recurrent state over {n_linear} "
                 f"linear-attention layers" if per_seq_mb else ""))
    return per_token_kb, per_seq_mb, detail


def check_shared_vocabulary(rep: Report, teacher: str, student: str) -> None:
    print("\n[3] teacher/student tokenizer identity", flush=True)
    from teacher_client import assert_shared_vocabulary

    try:
        vocab = assert_shared_vocabulary(teacher, student)
        rep.ok(f"identical tokenization, vocab_size={vocab}")
    except Exception as exc:  # noqa: BLE001
        rep.fail(str(exc))


def check_gpu_topology(rep: Report, tp: int, need: int = 1) -> None:
    """`need` is what THIS pod requires, which is not the same as the teacher's TP.

    Since the teacher and the student share one node, each pod is allocated only its own
    slice of the four devices. A trainer pod with one GPU is CORRECT even though TEACHER_TP=2
    on the shipped route -- the teacher's ranks live in a different container -- so only
    `need` is a hard requirement here. Failing on `count < tp` would make preflight reject
    every valid trainer.
    """
    print("\n[4] GPU count and interconnect", flush=True)
    try:
        import torch

        count = torch.cuda.device_count()
        if count == 0:
            rep.warn("no CUDA devices visible from this pod (fine if you are only "
                     "validating config)")
            return
        name = torch.cuda.get_device_name(0)
        major, minor = torch.cuda.get_device_capability(0)
        total = torch.cuda.get_device_properties(0).total_memory / 2**30
        print(f"        {count}x {name} ({total:.0f} GB, sm_{major}{minor})", flush=True)
        if count < need:
            rep.fail(f"this pod needs {need} GPU(s) but only {count} are visible; check its "
                     f"resources.limits['nvidia.com/gpu']")
        else:
            rep.ok(f"{count} GPU(s) visible, enough for this pod's {need}")
        if count < tp:
            print(f"        (TEACHER_TP={tp} > {count} visible here, as expected: the "
                  f"teacher's ranks run in the teacher-vllm pod, not this one)", flush=True)
        if (major, minor) < (8, 9):
            rep.warn(f"sm_{major}{minor} has no FP8 support (A10G is sm_86). Any fp8 or "
                     f"W8A8-float teacher checkpoint will be REFUSED at load, not silently "
                     f"dequantised -- see QUANT_MIN_CAPABILITY. W4A16 is the quantised route "
                     f"here.")
    except Exception as exc:  # noqa: BLE001
        rep.warn(f"could not query CUDA: {exc}")

    if tp == 1:
        # Nothing to all-reduce, so the PCIe topology below cannot affect the teacher. Only
        # relevant if you have switched to the W4A16 TP=1 fallback in env_vars; the shipped
        # route is TP=2 and drops through to the topology probe below.
        rep.ok("TEACHER_TP=1: no tensor-parallel all-reduce, so GPU interconnect is irrelevant "
               "to the teacher")
        return
    try:
        topo = subprocess.run(["nvidia-smi", "topo", "-m"], capture_output=True,
                              text=True, timeout=30).stdout
    except Exception:  # noqa: BLE001
        rep.warn("nvidia-smi topo unavailable; cannot verify the interconnect")
        return
    if "NV" in topo and any(f"NV{i}" in topo for i in range(1, 13)):
        rep.ok("NVLink present between GPUs")
    elif "SYS" in topo:
        rep.warn("GPU pairs are connected via SYS (across the CPU root complex) -- the "
                 "slowest path for TP all-reduce. PHB/PXB would be better; check that all "
                 "TP ranks land on GPUs under the same root complex.")
    else:
        rep.warn("no NVLink (expected on A10G/L4): TP all-reduce crosses PCIe. Acceptable "
                 "here because teacher scoring is prefill-bound, not decode-bound.")


def check_teacher_logprobs(rep: Report, url: str, model: str, student_model: str) -> None:
    """Echoed prompt log-probs must be real numbers: nulls would invert the advantage."""
    print("\n[5] teacher echo-logprobs are actually populated", flush=True)
    try:
        from transformers import AutoTokenizer

        from teacher_client import RemoteTeacher

        teacher = RemoteTeacher(base_url=url, model=model)
        served = teacher.served_models()
        if model not in served:
            rep.fail(f"--teacher-model {model!r} is not served; the server offers {served}. "
                     f"A name mismatch means every request 404s or hits the wrong weights.")
            return
        rep.ok(f"{model!r} is served at {url}")

        tok = AutoTokenizer.from_pretrained(student_model)
        text = "Solve: if 3x + 7 = 22, what is x? The answer is 5."
        ids = tok(text, add_special_tokens=False).input_ids
        logps = teacher.score(ids)
        if len(logps) != len(ids):
            rep.fail(f"got {len(logps)} logprobs for {len(ids)} tokens")
            return
        body = logps[1:]
        mean = sum(body) / len(body)
        if all(v == 0.0 for v in body):
            rep.fail("every teacher logprob is 0.0. This is the silent failure: the "
                     "advantage becomes -log pi_student, i.e. self-reinforcement, and "
                     "`teacher_kl` still prints a falling curve. Start vLLM with "
                     "--max-logprobs 1 and confirm echo=True is honoured.")
        elif mean > -0.02:
            rep.warn(f"mean teacher logprob {mean:.4f} is suspiciously close to 0 "
                     f"(perfect confidence on ordinary text is implausible)")
        else:
            rep.ok(f"teacher logprobs look real (mean {mean:.3f} nats/token over "
                   f"{len(body)} tokens)")
    except Exception as exc:  # noqa: BLE001
        rep.fail(f"teacher scoring smoke test failed: {type(exc).__name__}: {exc}")


def check_student_memory(rep: Report, params_b: float, precision: str, gpu_gb: float,
                         lora_r: int) -> None:
    print(f"\n[6] student training memory ({precision})", flush=True)
    p = params_b * 1e9
    if precision == "lora":
        base = p * 2 / 2**30                 # frozen bf16
        # LoRA on 7 projections, r=lora_r: a small multiple of r * hidden per layer. The
        # 0.4% figure is empirical for r=32 on a 4B Qwen-shaped model.
        adapter = p * 0.004 * 4 / 2**30      # fp32 adapter
        optimizer = adapter * 2              # Adam m + v
        grads = adapter
        activations = 4.0                    # with gradient checkpointing
        total = base + adapter + optimizer + grads + activations
    else:
        params = p * 4 / 2**30
        total = params * 3 + params + 4.0    # params + grads + Adam(m,v) + activations
    print(f"        estimated peak {total:.1f} GB vs {gpu_gb:.0f} GB per GPU", flush=True)
    if total > gpu_gb * 0.95:
        need = int(-(-total // (gpu_gb * 0.9)))
        rep.fail(f"{precision} needs ~{total:.0f} GB, more than one {gpu_gb:.0f} GB GPU. "
                 f"Use --precision lora (fits one GPU) or shard across >= {need} GPUs with "
                 f"FSDP (which also needs summon_full_params around generation).")
    else:
        rep.ok(f"~{total:.1f} GB fits one {gpu_gb:.0f} GB GPU")


def check_format_agreement(rep: Report, task: str, fewshot_train: int, fewshot_eval: int,
                           chat_train: str, chat_eval: str, max_new: int,
                           eval_max: int, thinking: bool) -> None:
    print("\n[7] train/eval prompt-format agreement", flush=True)
    if fewshot_train != fewshot_eval:
        rep.fail(f"TRAIN_FEWSHOT={fewshot_train} != EVAL_FEWSHOT={fewshot_eval}. The "
                 f"student would be optimised in a format it is not evaluated in.")
    else:
        rep.ok(f"fewshot matches ({fewshot_train})")
    norm = {"auto": "auto", "raw": "raw", "none": "raw"}
    if norm.get(chat_train) != norm.get(chat_eval):
        rep.fail(f"CHAT_TEMPLATE={chat_train} != EVAL_CHAT_TEMPLATE={chat_eval}")
    else:
        rep.ok(f"chat template matches ({chat_train})")
    if max_new < eval_max:
        rep.warn(f"MAX_NEW_TOKENS={max_new} < EVAL_MAX_TOKENS={eval_max}: training rollouts "
                 f"truncate earlier than the eval allows, so the student is trained on "
                 f"unfinished reasoning")
    else:
        rep.ok(f"MAX_NEW_TOKENS={max_new} >= EVAL_MAX_TOKENS={eval_max}")
    if thinking:
        rep.warn("ENABLE_THINKING=1: reasoning traces run 1000-4000 tokens. Confirm "
                 "MAX_NEW_TOKENS and the teacher's --max-model-len accommodate that, and "
                 "that the teacher is conditioned the same way -- otherwise the two token "
                 "streams diverge structurally and the per-token KL is meaningless.")
    print(f"        task={task}", flush=True)


def check_sampler_model(rep: Report, url: str, student_model: str) -> None:
    """The rollout sampler must serve the trainer's student, or the first adapter push fails."""
    print("\n[9] the student sampler serves the trainer's student", flush=True)
    if not url:
        rep.ok("no sampler (in-process sampling)")
        return
    try:
        import urllib.request
        served = [m["id"] for m in json.load(urllib.request.urlopen(f"{url.rstrip('/')}/models",
                                                                    timeout=30))["data"]]
    except Exception as exc:  # noqa: BLE001
        rep.fail(f"cannot list the sampler's models at {url}: {type(exc).__name__}: {exc}")
        return
    if student_model in served:
        rep.ok(f"{student_model!r} is served at {url}")
    else:
        rep.fail(f"the sampler serves {served}, not STUDENT_MODEL={student_model!r}: the "
                 f"trainer's adapter would not fit it. Re-apply "
                 f"manifests/student-sampler.yaml-template from the same env_vars and wait for it.")


def check_node_partition(rep: Report, node_gpus: int, teacher_tp: int, trainer_gpus: int,
                         sampler_gpus: int, sampler_url: str, teacher_replicas: int = 1) -> None:
    """The single-node constraint, as arithmetic.

    Teacher and student share one ml.g5.24xlarge. K8s allocates GPUs to a pod exclusively --
    there is no MPS or time-slicing here -- so the four devices are partitioned, and if the
    partition oversubscribes them the losing pod sits `Pending` indefinitely with no error in
    any log. That is a slow, silent failure mode, so it is worth one subtraction up front.

    The teacher's claim is TP x REPLICAS, not TP. That distinction is load-bearing: the shipped
    layout is TP=2 with replicas=1, but a bf16 9B could also run at TP=1 with replicas=2 (via
    the W4A16 fallback), and either way counting TP alone would silently miss the second axis.
    """
    print(f"\n[8] single-node GPU partition ({node_gpus} devices)", flush=True)
    if sampler_url and sampler_gpus == 0:
        rep.warn("STUDENT_SAMPLER_URL is set but SAMPLER_GPUS=0, so this check is ignoring a "
                 "pod that really does hold a GPU. Set SAMPLER_GPUS=1.")
        sampler_gpus = 1
    elif sampler_gpus and not sampler_url:
        rep.warn(f"SAMPLER_GPUS={sampler_gpus} but STUDENT_SAMPLER_URL is empty, so the "
                 f"trainer will sample in-process and those GPUs are reserved for nothing. "
                 f"Set SAMPLER_GPUS=0, or apply manifests/student-sampler.yaml-template.")

    teacher_claim = teacher_tp * max(1, teacher_replicas)
    claimed = teacher_claim + trainer_gpus + sampler_gpus
    print(f"        teacher TP={teacher_tp} x {teacher_replicas} replica(s) = {teacher_claim} + "
          f"trainer={trainer_gpus} + sampler={sampler_gpus} = {claimed}", flush=True)
    if claimed > node_gpus:
        rep.fail(f"the partition claims {claimed} GPUs but the node has {node_gpus}. The "
                 f"last pod scheduled will stay Pending forever. Lower TEACHER_REPLICAS or "
                 f"TEACHER_TP (which means quantising the teacher harder or serving a smaller "
                 f"one), or drop the sampler (STUDENT_SAMPLER_URL= and SAMPLER_GPUS=0).")
    elif claimed == node_gpus:
        rep.ok(f"{claimed}/{node_gpus} GPUs claimed -- exactly full")
        # The served eval (EVAL_STUDENT_URL, EVAL_GPUS=0) is an HTTP client and needs no GPU;
        # the local HF eval does, and cannot schedule until the trainer has freed its GPU.
        if int(os.environ.get("EVAL_GPUS", "0") or 0) > 0:
            rep.warn("the node is fully claimed and EVAL_GPUS > 0, so the eval Job can only "
                     "schedule after the trainer pod has COMPLETED and freed its GPU.")
    else:
        rep.ok(f"{claimed}/{node_gpus} GPUs claimed, {node_gpus - claimed} free")


# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teacher-model", default=env("TEACHER_MODEL_ID", "Qwen/Qwen3.5-9B"))
    ap.add_argument("--student-model", default=env("STUDENT_MODEL", "Qwen/Qwen3.5-2B"))
    ap.add_argument("--teacher-url", default=env("TEACHER_URL", "http://teacher-vllm.default.svc.cluster.local:8000/v1"))
    ap.add_argument("--tp", type=int, default=int(env("TEACHER_TP", "1")))
    ap.add_argument("--teacher-replicas", type=int, default=int(env("TEACHER_REPLICAS", "1")),
                    help="teacher Deployment replicas; each claims TEACHER_TP GPUs, so the "
                         "partition check [8] needs TP x REPLICAS")
    ap.add_argument("--node-gpus", type=int, default=int(env("NODE_GPUS", "4")))
    ap.add_argument("--trainer-gpus", type=int, default=int(env("GPU_PER_NODE", "1")))
    ap.add_argument("--sampler-gpus", type=int, default=int(env("SAMPLER_GPUS", "0")))
    ap.add_argument("--sampler-url", default=env("STUDENT_SAMPLER_URL", ""))
    ap.add_argument("--teacher-quantization",
                    default=env("TEACHER_QUANTIZATION", "compressed-tensors"))
    ap.add_argument("--teacher-params-b", type=float,
                    default=float(env("TEACHER_PARAMS_B", "9.65")))
    ap.add_argument("--teacher-weights-gib", type=float,
                    default=float(env("TEACHER_WEIGHTS_GIB", "0") or "0"),
                    help="measured on-disk size of the teacher checkpoint; overrides the "
                         "bytes/param estimate, which partially-quantised repos break")
    ap.add_argument("--student-params-b", type=float, default=float(env("STUDENT_PARAMS_B", "4")))
    ap.add_argument("--gpu-gb", type=float, default=float(env("GPU_MEM_GB", "22.5")))
    ap.add_argument("--gpu-util", type=float, default=float(env("TEACHER_GPU_UTIL", "0.90")))
    ap.add_argument("--max-model-len", type=int, default=int(env("TEACHER_MAX_MODEL_LEN", "2048")))
    ap.add_argument("--teacher-dtype", default=env("TEACHER_DTYPE", "bfloat16"))
    ap.add_argument("--precision", default=env("PRECISION", "lora"))
    ap.add_argument("--lora-r", type=int, default=int(env("LORA_R", "32")))
    ap.add_argument("--task", default=env("TASK", "math500"))
    ap.add_argument("--train-fewshot", type=int, default=int(env("TRAIN_FEWSHOT", "0")))
    ap.add_argument("--eval-fewshot", type=int, default=int(env("EVAL_FEWSHOT", "0")))
    ap.add_argument("--chat-template", default=env("CHAT_TEMPLATE", "auto"))
    ap.add_argument("--eval-chat-template", default=env("EVAL_CHAT_TEMPLATE", "auto"))
    ap.add_argument("--max-new-tokens", type=int, default=int(env("MAX_NEW_TOKENS", "512")))
    ap.add_argument("--eval-max-tokens", type=int, default=int(env("EVAL_MAX_TOKENS", "512")))
    ap.add_argument("--enable-thinking", action="store_true",
                    default=env("ENABLE_THINKING", "0") == "1")
    ap.add_argument("--trust-remote-code", action="store_true",
                    default=env("TRUST_REMOTE_CODE", "0") == "1")
    ap.add_argument("--skip-teacher", action="store_true",
                    help="skip checks that need the teacher Service to be up")
    args = ap.parse_args(argv)

    rep = Report()
    print("=== on-policy distillation pre-flight ===", flush=True)
    try:
        facts = check_tensor_parallel(rep, args.teacher_model, args.tp, args.trust_remote_code)
        check_teacher_memory(rep, facts, args.teacher_params_b, args.tp, args.gpu_gb,
                            args.gpu_util, args.max_model_len, args.teacher_dtype,
                            args.teacher_quantization, args.teacher_weights_gib)
    except Exception as exc:  # noqa: BLE001
        rep.fail(f"could not read the teacher config: {type(exc).__name__}: {exc}")
    check_shared_vocabulary(rep, args.teacher_model, args.student_model)
    check_gpu_topology(rep, args.tp, need=args.trainer_gpus)
    if args.skip_teacher:
        print("\n[5] skipped (--skip-teacher)", flush=True)
    else:
        check_teacher_logprobs(rep, args.teacher_url, args.teacher_model, args.student_model)
    check_student_memory(rep, args.student_params_b, args.precision, args.gpu_gb, args.lora_r)
    check_format_agreement(rep, args.task, args.train_fewshot, args.eval_fewshot,
                           args.chat_template, args.eval_chat_template,
                           args.max_new_tokens, args.eval_max_tokens, args.enable_thinking)
    check_node_partition(rep, args.node_gpus, args.tp, args.trainer_gpus,
                         args.sampler_gpus, args.sampler_url, args.teacher_replicas)
    if not args.skip_teacher:
        check_sampler_model(rep, args.sampler_url, args.student_model)

    print(f"\n=== {len(rep.passed)} passed, {len(rep.warnings)} warnings, "
          f"{len(rep.failures)} failures ===", flush=True)
    for msg in rep.failures:
        print(f"FAIL: {msg}", file=sys.stderr, flush=True)
    return 1 if rep.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
