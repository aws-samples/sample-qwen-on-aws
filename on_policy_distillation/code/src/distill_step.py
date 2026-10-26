"""The core on-policy distillation math: per-token reverse KL -> advantage -> loss.

Faithful to the Thinking Machines / Tinker `incorporate_kl_penalty` recipe:

    reverse_kl_t = log pi_student(y_t | y_<t) - log pi_teacher(y_t | y_<t)
    advantage_t  = - kl_coef * reverse_kl_t                     # negative reverse KL
    loss         = - mean_t [ stop_grad(advantage_t) * ratio_t ]
    ratio_t      = exp( log pi_student_new(y_t) - stop_grad(log pi_student_old(y_t)) )

`ratio_t` is the importance weight of the *current* student against the student that
generated the rollout. On fresh (on-policy) rollouts ratio ~= 1 and the loss reduces to the
intuitive `-(advantage * logp_new).mean()`; the ratio is what lets us reuse slightly stale
rollouts from the buffer without bias (PPO-style).

Alignment (see teacher_client): `teacher_logp` has length T and is aligned to `token_ids`
(element t = teacher log-prob of token t). A model's own per-token log-probs have length
T-1 (index i = log-prob of token i+1), so we compare against `teacher_logp[1:]`.
Completion tokens y_t live at sequence positions [prompt_len, T-1], i.e. log-prob indices
[prompt_len-1, T-2].

CONDITIONING
------------
1. **Outlier clamping.** Per-token reverse KL between a small student and a 9B teacher is
   heavy-tailed: a handful of tokens carry tens of nats, and under a global
   `clip_grad_norm_(1.0)` they would consume the whole step. `rkl_clip` bounds each token's
   credit.
2. **Batch whitening.** Advantages are centred and scaled across the whole batch
   (`whiten_advantages`). The batch mean acts as a baseline -- standard practice in policy
   gradient -- so the update is well conditioned instead of a uniformly negative push
   (the rollouts' reverse KL averages ~+0.3). This departs from the raw Tinker objective:
   it changes the *scale* and adds a baseline, so `kl_coef` becomes a pure temperature on
   an already normalised signal. Set `whiten=False` to recover the unnormalised recipe.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple
import torch


def completion_slice(prompt_len: int, seq_len: int) -> slice:
    """Indices into a length-(T-1) per-token log-prob array selecting completion tokens."""
    return slice(prompt_len - 1, seq_len - 1)


def sequence_logprobs(model, seq_ids: torch.Tensor, with_grad: bool) -> torch.Tensor:
    """Per-token log-probs of a single sequence: returns [T-1], index i = log p(token i+1)."""
    ctx = torch.enable_grad() if with_grad else torch.no_grad()
    with ctx:
        logits = model(seq_ids.unsqueeze(0)).logits[0]              # [T, V]
        logp = logits[:-1].log_softmax(-1)                          # [T-1, V]
        tok_logp = logp.gather(-1, seq_ids[1:, None]).squeeze(-1)   # [T-1]
    return tok_logp


def sequence_logprobs_batched(
    model,
    input_ids: torch.Tensor,        # [B, T] right-padded
    attention_mask: torch.Tensor,   # [B, T]
    with_grad: bool = True,
    start: int = 0,
) -> torch.Tensor:
    """Per-token log-probs for a right-padded batch: returns [B, T-1].

    Right padding (not left) because we are scoring existing sequences, not generating:
    every real token then sits at its true position and the causal mask does the rest.
    Positions beyond a sequence's length are garbage and must be masked by the caller.

    `start` skips the vocab projection for log-prob indices < start (returned as 0.0).
    The [B, T, V] logits over Qwen's ~248k vocab are the trainer's largest allocation, and
    with a few-shot prompt most positions are prompt tokens whose log-probs are never read.
    Pass `min(prompt_len) - 1` to project only the completion span.
    """
    T = input_ids.shape[1]
    start = max(0, min(start, T - 1))
    keep = T - start                               # logit positions [start, T-1]
    ctx = torch.enable_grad() if with_grad else torch.no_grad()
    with ctx:
        try:
            logits = model(input_ids=input_ids, attention_mask=attention_mask,
                           logits_to_keep=keep).logits                            # [B,keep,V]
        except TypeError:  # a model without `logits_to_keep`: project everything
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        logits = logits[:, -keep:]
        logp = logits[:, :-1].log_softmax(-1)                                      # [B,keep-1,V]
        tok_logp = logp.gather(-1, input_ids[:, start + 1:, None]).squeeze(-1)     # [B,keep-1]
        if start:
            tok_logp = torch.cat([tok_logp.new_zeros(tok_logp.shape[0], start), tok_logp], 1)
    return tok_logp


def reverse_kl(student_logp: torch.Tensor, teacher_logp_full: torch.Tensor) -> torch.Tensor:
    """Per-token reverse KL estimate over the completion span. Inputs already sliced/aligned."""
    return student_logp - teacher_logp_full  # log pi_student - log pi_teacher


def rollout_reverse_kl(
    prompt_len: int,
    seq_len: int,
    teacher_logp_full: Sequence[float] | torch.Tensor,
    student_logp_old: torch.Tensor,   # [T-1], from sampling time (detached)
    rkl_clip: float = 0.0,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Per-token reverse KL for one rollout's completion span. **No forward pass needed.**

    Both inputs were recorded at sampling time, so the whole batch's advantages can be
    computed (and whitened) before any gradient work begins.
    """
    comp = completion_slice(prompt_len, seq_len)
    if not torch.is_tensor(teacher_logp_full):
        teacher_logp_full = torch.tensor(teacher_logp_full, dtype=torch.float32, device=device)
    teacher_aligned = teacher_logp_full.to(device)[1:]            # [T-1]
    rkl = reverse_kl(student_logp_old.to(device)[comp], teacher_aligned[comp]).float()
    if rkl_clip and rkl_clip > 0:
        rkl = rkl.clamp(-rkl_clip, rkl_clip)
    return rkl


def whiten_advantages(
    per_rollout: List[torch.Tensor], eps: float = 1e-6
) -> Tuple[List[torch.Tensor], dict]:
    """Centre and scale advantages across the *whole batch* of rollouts.

    Pooling across rollouts (rather than per sequence) keeps the relative credit between
    an easy and a hard prompt intact, which a per-sequence normalisation would destroy.
    Returns the whitened list plus the statistics, for logging.
    """
    if not per_rollout:
        return [], {}
    flat = torch.cat([a.flatten() for a in per_rollout])
    mean, std = flat.mean(), flat.std(unbiased=False)
    scale = std.clamp_min(eps)
    stats = {
        "adv_mean_raw": float(mean),
        "adv_std_raw": float(std),
        "adv_abs_max_raw": float(flat.abs().max()),
    }
    return [(a - mean) / scale for a in per_rollout], stats


def distill_loss(
    logp_new: torch.Tensor,        # [C] current student log-probs (requires grad)
    logp_old: torch.Tensor,        # [C] student log-probs at sampling time (detached)
    advantage: torch.Tensor,       # [C] per-token advantage (detached)
    clip_ratio: float = 0.2,
    reduction: str = "mean",
) -> Tuple[torch.Tensor, dict]:
    """PPO-style importance-sampling surrogate; equals -(advantage*logp_new).mean() at ratio==1.

    `reduction="sum"` returns the unnormalised token sum, which is what a gradient
    accumulation loop needs so that every token in the batch carries equal weight
    regardless of how the batch was split into micro-batches.

    Returns (loss, diagnostics). `clip_frac` is the share of tokens where the clip bound was
    the binding branch -- if it climbs above ~0.2 the rollouts are too stale for the replay
    setting and `--buffer-max-age` should come down.
    """
    log_ratio = logp_new - logp_old.detach()
    ratio = log_ratio.exp()
    adv = advantage.detach()
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv
    # We want to *maximize* advantage-weighted log-prob, so minimize the negative min.
    objective = torch.min(unclipped, clipped)
    loss = -(objective.sum() if reduction == "sum" else objective.mean())
    with torch.no_grad():
        diag = {
            "clip_frac": float((unclipped > clipped).float().mean()),
            "ratio_mean": float(ratio.mean()),
            "ratio_max": float(ratio.max()),
            # Sample KL between the sampling policy and the live policy: a health check on
            # how far the student has drifted from the rollouts it is training on.
            "policy_drift": float((-log_ratio).mean()),
        }
    return loss, diag


def score_rollout_for_training(
    student_model,
    seq_ids: torch.Tensor,      # [T] prompt + completion, on the student's device
    prompt_len: int,
    teacher_logp_full: List[float] | torch.Tensor,  # [T], aligned to seq_ids
    student_logp_old: torch.Tensor,                  # [T-1], from sampling time (detached)
    kl_coef: float,
    rkl_clip: float = 0.0,
    clip_ratio: float = 0.2,
):
    """One rollout -> (loss, mean_reverse_kl). The single-sequence path, kept for the
    notebook's live demo; `train_distill.py` uses the batched two-phase path so it can
    whiten advantages across the batch first."""
    device = seq_ids.device
    T = seq_ids.shape[0]
    comp = completion_slice(prompt_len, T)

    rkl = rollout_reverse_kl(prompt_len, T, teacher_logp_full, student_logp_old,
                             rkl_clip=rkl_clip, device=device)
    advantage = -kl_coef * rkl

    logp_new = sequence_logprobs(student_model, seq_ids, with_grad=True)[comp]  # [C], grad
    loss, _ = distill_loss(logp_new, student_logp_old[comp], advantage, clip_ratio)
    return loss, rkl.mean().item()
