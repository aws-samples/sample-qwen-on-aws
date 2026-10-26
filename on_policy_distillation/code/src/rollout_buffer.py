"""Rollout buffer for on-policy distillation.

The buffer is the seam between two rates: the *generator* (student sampling + teacher
scoring) fills it, and the *trainer* drains it.

  * capacity      -- bounds memory; oldest rollouts are evicted (maxlen).
  * staleness     -- a rollout sampled under student weights theta_k but trained at
                     theta_{k+n} is *off-policy by n steps*. Each rollout is stamped with
                     the policy step it came from; `max_age` is enforced on every read
                     (`evict_stale`), and the importance ratio in `distill_step` corrects
                     what remains.

`max_age=0` gives a strictly on-policy loop -- generate, train on exactly those rollouts,
discard. With the vLLM sampler the trainer uses `max_age = SAMPLER_SYNC_EVERY`, the lag
between an adapter push and the rollouts generated from it. Sampling uniformly from a large,
unbounded buffer would make the loop off-policy by many steps and let the PPO clip zero most
of the gradient.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import List
import random


@dataclass
class Rollout:
    """One student trajectory, fully scored and ready to train on."""

    token_ids: List[int]           # prompt + student completion
    prompt_len: int                # number of prompt tokens (completion starts here)
    student_logp_old: List[float]  # student log-probs at sampling time (behaviour policy)
    teacher_logp: List[float]      # teacher log-probs for the same tokens
    produced_at_step: int = 0      # trainer step when this rollout was generated
    # True when generation stopped because it hit max_new_tokens rather than emitting EOS.
    # Training on unfinished reasoning teaches the student to never reach its final answer,
    # so the trainer drops (or masks) these -- see train_distill.--truncated-policy.
    truncated: bool = False
    prompt_text: str = ""          # kept for debugging / dumping example rollouts

    def num_completion_tokens(self) -> int:
        return len(self.token_ids) - self.prompt_len


class RolloutBuffer:
    """Bounded replay buffer with an enforced staleness ceiling.

    `max_age` is the maximum number of trainer steps a rollout may survive:
      * 0  -> strictly on-policy (the default): only rollouts produced this step are used.
      * n  -> allow n steps of replay, which the importance-sampling ratio corrects for.
    """

    def __init__(self, capacity: int = 1024, seed: int = 0, max_age: int = 0):
        self.buf: deque[Rollout] = deque(maxlen=capacity)
        self.rng = random.Random(seed)
        self.max_age = max_age
        self.total_added = 0
        self.total_evicted_stale = 0

    def add(self, rollout: Rollout) -> None:
        self.buf.append(rollout)
        self.total_added += 1

    def extend(self, rollouts: List[Rollout]) -> None:
        for r in rollouts:
            self.add(r)

    def __len__(self) -> int:
        return len(self.buf)

    def evict_stale(self, current_step: int) -> int:
        """Drop rollouts older than `max_age` steps. Returns how many were dropped."""
        keep = [r for r in self.buf if current_step - r.produced_at_step <= self.max_age]
        dropped = len(self.buf) - len(keep)
        if dropped:
            self.buf.clear()
            self.buf.extend(keep)
            self.total_evicted_stale += dropped
        return dropped

    def sample_batch(self, batch_size: int, current_step: int | None = None) -> List[Rollout]:
        """Uniformly sample without replacement, after enforcing the staleness ceiling."""
        if current_step is not None:
            self.evict_stale(current_step)
        n = min(batch_size, len(self.buf))
        return self.rng.sample(list(self.buf), n)

    def drain(self) -> List[Rollout]:
        """Take everything and empty the buffer -- the strictly on-policy read."""
        out = list(self.buf)
        self.buf.clear()
        return out

    def mean_staleness(self, current_step: int) -> float:
        """Average number of trainer steps since each buffered rollout was produced."""
        if not self.buf:
            return 0.0
        return sum(current_step - r.produced_at_step for r in self.buf) / len(self.buf)

    def clear(self) -> None:
        self.buf.clear()
