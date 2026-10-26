"""Teacher scoring client for on-policy distillation.

The teacher's only job is to return, for a sequence the *student* generated, the
teacher's log-probability of every token. That single scalar per token is all we need to
compute the per-token reverse KL that supervises the student, and it is cheap enough to
send over the wire -- which is exactly why the teacher can live on its own GPU pool (a
vLLM inference Service: here one bf16 9B sharded over TP=2, reachable as a Service that can be
scaled to more replicas without the trainer noticing) instead of in the trainer process.

Two backends, same interface (`score(token_ids) -> list[float]`):

  * RemoteTeacher  -- talks to a vLLM OpenAI-compatible server. Uses `echo=True,
                      logprobs=1` to read back the log-prob of each *prompt* token, and
                      scores many sequences concurrently (`score_batch`).
  * LocalTeacher   -- loads the teacher in-process with transformers. Zero infra, used as
                      the de-risking fallback and for the unit check that the remote path
                      returns the same numbers.

Alignment convention (shared by both backends): the returned list has the same length as
`token_ids`; element t is  log p_teacher(token_ids[t] | token_ids[:t]). Element 0 is
undefined (no context) and is returned as 0.0 -- it is always masked out downstream
because we only score completion tokens (t >= prompt_len).

TWO SILENT-FAILURE MODES THIS FILE GUARDS
-----------------------------------------
1. **Null logprobs.** vLLM does not populate echoed prompt logprobs under every
   version/flag combination. Mapping `None -> 0.0` would turn the advantage into
   ``-kl_coef * log pi_student`` -- a *positive* advantage on the student's own samples,
   i.e. pure self-reinforcement -- while `teacher_kl` still printed a falling curve.
   `_parse` raises instead.
2. **Vocabulary mismatch.** `score()` sends raw token ids. If the teacher and student do
   not share a tokenizer, the teacher scores *different tokens* than the student
   generated and the KL is meaningless, with no error anywhere. Call
   `assert_shared_vocabulary()` once at startup.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List, Sequence


class TeacherScoringError(RuntimeError):
    """The teacher returned something we must not silently treat as a log-prob."""


class RemoteTeacher:
    """Score token sequences against a vLLM OpenAI-compatible server."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "EMPTY",
        timeout: float = 600.0,
        max_concurrency: int = 16,
        max_retries: int = 3,
    ):
        # Imported lazily so the module is importable without the openai package.
        from openai import OpenAI

        self.model = model
        self.max_concurrency = max_concurrency
        self.client = OpenAI(
            base_url=base_url.rstrip("/"),
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
        )

    def score(self, token_ids: Sequence[int]) -> List[float]:
        """Return per-token teacher log-probs aligned to `token_ids`."""
        resp = self.client.completions.create(
            model=self.model,
            prompt=list(token_ids),  # vLLM accepts a list of token ids as the prompt
            max_tokens=1,            # must be >= 1; we ignore the generated token
            echo=True,               # echo the prompt so we get its per-token log-probs
            logprobs=1,              # >=1: some vLLM builds return nothing at logprobs=0
            temperature=0.0,
        )
        return self._parse(resp, len(token_ids))

    @staticmethod
    def _parse(resp: Any, n_tokens: int) -> List[float]:
        choice = resp.choices[0]
        logprobs = getattr(choice, "logprobs", None)
        if logprobs is None or getattr(logprobs, "token_logprobs", None) is None:
            raise TeacherScoringError(
                "teacher returned no logprobs at all. The vLLM server must be started so "
                "that echoed prompt logprobs are available: pass `--max-logprobs 1` (or "
                "higher) and confirm the build supports `echo=True` on /v1/completions."
            )
        # token_logprobs is aligned to the echoed prompt tokens (index 0 is None because
        # the first token has no context), possibly followed by the one generated token --
        # keep only the prompt span.
        raw = list(logprobs.token_logprobs)[:n_tokens]
        if len(raw) < n_tokens:
            raise TeacherScoringError(
                f"teacher returned {len(raw)} logprobs for {n_tokens} tokens; the echoed "
                "prompt was truncated. Raise the server's --max-model-len."
            )
        # Index 0 is legitimately None. Anything else being None means the server is not
        # actually scoring the prompt -- that must NOT become 0.0 (see module docstring).
        nulls = [i for i, v in enumerate(raw) if v is None and i != 0]
        if nulls:
            raise TeacherScoringError(
                f"teacher returned null logprobs at {len(nulls)} of {n_tokens} positions "
                f"(first at index {nulls[0]}). Treating these as 0.0 would invert the "
                "advantage into self-reinforcement. Check --max-logprobs on the vLLM "
                "server and that `echo=True` is honoured by this vLLM version."
            )
        return [0.0 if v is None else float(v) for v in raw]

    def score_batch(self, batch: Sequence[Sequence[int]]) -> List[List[float]]:
        """Score many sequences concurrently.

        One HTTP round-trip per rollout, issued sequentially, would dominate the step. vLLM
        batches internally, so it is enough to keep several requests in flight.
        """
        if not batch:
            return []
        workers = min(self.max_concurrency, len(batch))
        if workers <= 1:
            return [self.score(ids) for ids in batch]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(self.score, batch))

    # -- introspection used by preflight ------------------------------------------------
    def served_models(self) -> list[str]:
        return [m.id for m in self.client.models.list().data]


class LocalTeacher:
    """Score token sequences with an in-process transformers model (fallback / unit check)."""

    def __init__(self, model_name: str, device: str = "cuda", dtype: str = "bfloat16"):
        import torch
        from transformers import AutoModelForCausalLM

        self.torch = torch
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=getattr(torch, dtype)
        ).to(device).eval()

    def score(self, token_ids: Sequence[int]) -> List[float]:
        torch = self.torch
        ids = torch.tensor(token_ids, device=self.device).unsqueeze(0)  # [1, T]
        with torch.no_grad():
            logits = self.model(ids).logits[0]                          # [T, V]
            logp = logits[:-1].log_softmax(-1)                          # predict tokens 1..T-1
            tok_logp = logp.gather(-1, ids[0, 1:, None]).squeeze(-1)    # [T-1]
        # Prepend 0.0 for position 0 so the output aligns 1:1 with token_ids.
        return [0.0] + tok_logp.float().cpu().tolist()

    def score_batch(self, batch: Sequence[Sequence[int]]) -> List[List[float]]:
        return [self.score(ids) for ids in batch]


def make_teacher(mode: str, **kwargs):
    """Factory: mode='remote' -> RemoteTeacher, mode='local' -> LocalTeacher."""
    if mode == "remote":
        return RemoteTeacher(
            base_url=kwargs["base_url"],
            model=kwargs["model"],
            max_concurrency=int(kwargs.get("max_concurrency", os.environ.get("TEACHER_CONCURRENCY", 16))),
            timeout=float(kwargs.get("timeout", 600.0)),
        )
    if mode == "local":
        return LocalTeacher(
            model_name=kwargs["model_name"],
            device=kwargs.get("device", "cuda"),
            dtype=kwargs.get("dtype", "bfloat16"),
        )
    raise ValueError(f"unknown teacher mode: {mode!r}")


def assert_shared_vocabulary(teacher_model: str, student_model: str) -> int:
    """Raise unless the teacher and student tokenise identically. Returns the vocab size.

    `RemoteTeacher.score()` sends raw token ids, so a differing tokenizer means the
    teacher scores a *different* string than the student generated -- a catastrophic,
    completely silent failure. Cheap to check, so always check.
    """
    from transformers import AutoTokenizer

    t_tok = AutoTokenizer.from_pretrained(teacher_model)
    s_tok = AutoTokenizer.from_pretrained(student_model)
    if t_tok.vocab_size != s_tok.vocab_size:
        raise RuntimeError(
            f"teacher vocab_size={t_tok.vocab_size} != student vocab_size={s_tok.vocab_size}. "
            "Raw token-id teacher scoring is invalid across vocabularies; pick models from "
            "the same family or score by text instead."
        )
    probe = (
        "Solve: if 3x + 7 = 22, what is x? \\boxed{5} #### 5\n"
        "Unicode & punctuation: café — 1,234.56 ≤ 2000"
    )
    t_ids, s_ids = t_tok(probe).input_ids, s_tok(probe).input_ids
    if t_ids != s_ids:
        raise RuntimeError(
            "teacher and student tokenizers disagree on a probe string even though "
            f"vocab sizes match ({len(t_ids)} vs {len(s_ids)} tokens). Raw token-id "
            "scoring is invalid."
        )
    return int(s_tok.vocab_size)
