"""Prompt sourcing for the distillation trainer.

PROMPT DIVERSITY
----------------
`PromptStream` draws from the full train split (GSM8K: 7,473 problems), shuffles once per
epoch and never repeats a prompt within an epoch. A small pool that cycles lets the student
memorise its problems instead of learning anything that transfers to held-out questions.

CONTAMINATION GUARD
-------------------
`load_prompt_pool(..., exclude=...)` drops any training prompt whose normalised text also
appears in the eval set, so the held-out claim is checked rather than assumed.
"""

from __future__ import annotations

import random
import re
from typing import Any, Iterator, Sequence

from prompts import TaskSpec

# Field names that hold the question, in preference order -- datasets in this space are
# inconsistent ("question" for GSM8K, "problem" for MATH).
QUESTION_FIELDS = ("question", "problem", "prompt", "query", "input")


def _normalise_for_dedupe(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())[:400]


def _pick_field(row: dict[str, Any], candidates: Sequence[str], explicit: str | None) -> str:
    if explicit:
        if explicit not in row:
            raise KeyError(f"field {explicit!r} not in dataset row (have {sorted(row)})")
        return explicit
    for name in candidates:
        if name in row:
            return name
    raise KeyError(f"no question-like field in dataset row (have {sorted(row)})")


def _load_split(dataset: str, config: str | None, split: str) -> list[dict[str, Any]]:
    """Load one dataset split. `config` may be a comma-separated list of configs.

    The MATH train set is published per subject (`EleutherAI/hendrycks_math` has one config
    per topic), so concatenating configs has to be a first-class option rather than
    something the caller assembles by hand.
    """
    from datasets import load_dataset

    configs = [c.strip() or None for c in (config or "").split(",")] if config else [None]
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for cfg in configs:
        try:
            ds = load_dataset(dataset, cfg, split=split)
            rows.extend(dict(r) for r in ds)
        except Exception as exc:  # noqa: BLE001 - surfaced together below
            errors.append(f"{dataset}/{cfg}:{split} -> {type(exc).__name__}: {exc}")
    if not rows:
        raise RuntimeError(
            "could not load any training prompts. Tried:\n  " + "\n  ".join(errors)
            + "\nSet TRAIN_DATASET / TRAIN_DATASET_CONFIG / TRAIN_SPLIT in manifests/env_vars."
        )
    return rows


def load_eval_questions(spec: TaskSpec, limit: int | None = None,
                        select: str = "tail", seed: int = 42) -> set[str]:
    """Normalised question texts of the eval set, for the contamination guard."""
    try:
        rows = _load_split(spec.dataset, spec.dataset_name, spec.test_split)
    except RuntimeError:
        return set()
    if select == "shuffle":
        random.Random(seed).shuffle(rows)
    if limit:
        rows = rows[-limit:] if select == "tail" else rows[:limit]
    field = _pick_field(rows[0], QUESTION_FIELDS, spec.question_field)
    return {_normalise_for_dedupe(str(r[field])) for r in rows}


def load_prompt_pool(
    dataset: str,
    config: str | None,
    split: str,
    question_field: str | None = None,
    limit: int | None = None,
    seed: int = 0,
    exclude: set[str] | None = None,
) -> list[str]:
    """The full pool of training questions: deduped, contamination-filtered, shuffled."""
    rows = _load_split(dataset, config, split)
    field = _pick_field(rows[0], QUESTION_FIELDS, question_field)
    exclude = exclude or set()
    seen: set[str] = set()
    pool: list[str] = []
    dropped_contaminated = 0
    for row in rows:
        question = str(row[field]).strip()
        if not question:
            continue
        key = _normalise_for_dedupe(question)
        if key in exclude:
            dropped_contaminated += 1
            continue
        if key in seen:
            continue
        seen.add(key)
        pool.append(question)
    random.Random(seed).shuffle(pool)
    if limit:
        pool = pool[:limit]
    if dropped_contaminated:
        print(f"[data] dropped {dropped_contaminated} train prompts that also appear in "
              f"the eval set", flush=True)
    return pool




class PromptStream:
    """Epoch-aware, non-repeating prompt iterator.

    Hands out each prompt once per epoch, reshuffling between epochs. `epochs_done` is
    logged so a run that has started recycling prompts is visible rather than invisible.
    """

    def __init__(self, pool: Sequence[str], seed: int = 0):
        if not pool:
            raise ValueError("empty prompt pool")
        self.pool = list(pool)
        self.rng = random.Random(seed)
        self.epochs_done = 0
        self._order: list[int] = []
        self._reshuffle()

    def _reshuffle(self) -> None:
        self._order = list(range(len(self.pool)))
        self.rng.shuffle(self._order)

    def take(self, n: int) -> list[str]:
        out: list[str] = []
        while len(out) < n:
            if not self._order:
                self.epochs_done += 1
                self._reshuffle()
            out.append(self.pool[self._order.pop()])
        return out

    def __len__(self) -> int:
        return len(self.pool)

    def __iter__(self) -> Iterator[str]:
        while True:
            yield self.take(1)[0]
