"""Batched student rollout generation.

Two backends, same interface:

  * HFSampler   -- batched `generate()` with left padding, taking `logp_old` straight out
                   of `output_scores` (no second forward pass). No extra infra; slow.
  * VLLMSampler -- the student served by its own vLLM process with `--enable-lora`, on its
                   own GPU. The trainer pushes the LoRA adapter every N steps
                   (`sync_adapter`) instead of synchronising full weights; ~5-10x faster.

WHY logp_old MUST BE THE *RAW* POLICY LOG-PROB
----------------------------------------------
The importance ratio is ``exp(logp_new - logp_old)``, and `logp_new` is computed from raw
logits. So `logp_old` has to be the raw model distribution too. HuggingFace's
`output_scores` are the logits *after* the sampling warpers (temperature, top-p, top-k,
min-p, repetition penalty), and Qwen ships a `generation_config.json` that sets several of
them. Taken at face value under, say, `temperature=0.7, top_k=20`, the ratio would compare
two different distributions and the clip/gradient would be quietly wrong.

`_neutral_generation_kwargs` therefore pins every warper off and `HFSampler` asserts it.
When sampling with a temperature other than 1.0 (`--temperature`), that free path is not
available and the sampler falls back to one extra forward pass to recover raw log-probs.
vLLM returns raw log-probs at any temperature.

Only the *completion* span of `logp_old` is ever read (see `distill_step.completion_slice`),
so prompt positions are filled with 0.0 and no prompt-token log-probs are needed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, List, Sequence

import torch


@dataclass
class Sample:
    """One generated rollout, with everything the teacher and the loss need."""

    token_ids: List[int]      # unpadded: prompt + completion (including EOS if emitted)
    prompt_len: int
    logp_old: List[float]     # length len(token_ids) - 1; prompt positions are 0.0
    finished: bool            # False => hit max_new_tokens without emitting EOS
    prompt_text: str = ""

    @property
    def truncated(self) -> bool:
        return not self.finished


def _neutral_generation_kwargs(temperature: float) -> dict[str, Any]:
    """Generation kwargs with every distribution-distorting warper explicitly disabled.

    Pinned rather than inherited, because a model's own `generation_config.json` (Qwen sets
    top_k=20, top_p≈0.8, temperature≈0.7) would otherwise silently change the policy that
    `logp_old` is supposed to describe.
    """
    return {
        "do_sample": temperature > 0,
        "temperature": temperature if temperature > 0 else None,
        "top_p": 1.0,
        "top_k": 0,
        "min_p": 0.0,
        "typical_p": 1.0,
        "repetition_penalty": 1.0,
        "no_repeat_ngram_size": 0,
        "renormalize_logits": False,
    }


# PEFT names the adapter after the text-only module tree the trainer loads
# (AutoModelForCausalLM -> `model.layers.N`), but vLLM serves a Qwen3.5 checkpoint as the
# multimodal Qwen3_5ForConditionalGeneration, whose language layers the HF->vLLM mapper only
# finds under `model.language_model.layers.N`. vLLM matches nothing and IGNORES THE ADAPTER
# WITHOUT A WARNING: every rollout then comes from the base student, teacher_kl (computed on
# those rollouts) cannot move, and ~15% of tokens get PPO-clipped by the trainer/sampler mismatch.
_TEXT_PREFIX = "base_model.model.model.layers."
_VL_PREFIX = "base_model.model.model.language_model.layers."


def export_adapter_for_vllm(adapter_dir: str, base_model: str) -> str:
    """Return a dir vLLM will actually apply: a renamed copy for multimodal checkpoints.

    Text-only architectures (and adapters already in the VL layout) are returned as-is.
    """
    import shutil

    from safetensors.torch import load_file, save_file
    from transformers import AutoConfig

    archs = getattr(AutoConfig.from_pretrained(base_model), "architectures", None) or []
    if not any(a.endswith("ForConditionalGeneration") for a in archs):
        return adapter_dir
    weights = os.path.join(adapter_dir, "adapter_model.safetensors")
    state = load_file(weights)
    if not any(k.startswith(_TEXT_PREFIX) for k in state):
        return adapter_dir
    out = adapter_dir.rstrip("/") + "-vllm"
    shutil.copytree(adapter_dir, out, dirs_exist_ok=True)
    save_file({k.replace(_TEXT_PREFIX, _VL_PREFIX, 1): v for k, v in state.items()},
              os.path.join(out, "adapter_model.safetensors"))
    return out


def _scores_are_raw_logits(temperature: float) -> bool:
    """Can we read logp_old out of `output_scores` for free?

    Only when no warper is active, i.e. temperature exactly 1.0 on top of the neutral
    kwargs above. Any other temperature scales the logits before they are recorded.
    """
    return abs(temperature - 1.0) < 1e-9


class HFSampler:
    """Batched HuggingFace generation on the training GPU."""

    def __init__(self, model, tokenizer, device: str, max_new_tokens: int = 512,
                 temperature: float = 1.0, micro_batch: int = 8,
                 stop_strings: Sequence[str] | None = None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.micro_batch = micro_batch
        self.stop_strings = list(stop_strings or [])
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

    # -- public ------------------------------------------------------------------------
    def generate(self, prompts: Sequence[str], step: int = 0) -> List[Sample]:
        out: List[Sample] = []
        for i in range(0, len(prompts), self.micro_batch):
            out.extend(self._generate_chunk(list(prompts[i : i + self.micro_batch])))
        return out

    def sync_adapter(self, path: str) -> None:  # no-op: we generate from the live weights
        return None

    # -- internals ---------------------------------------------------------------------
    def _generate_chunk(self, prompts: List[str]) -> List[Sample]:
        tok = self.tokenizer
        prev_side, tok.padding_side = tok.padding_side, "left"
        try:
            enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
        finally:
            tok.padding_side = prev_side
        enc = {k: v.to(self.device) for k, v in enc.items()}
        pad_len = enc["input_ids"].shape[1]

        gen_kwargs = _neutral_generation_kwargs(self.temperature)
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}
        was_training = self.model.training
        self.model.eval()  # logp_old must describe the policy that actually sampled
        try:
            with torch.no_grad():
                out = self.model.generate(
                    **enc,
                    max_new_tokens=self.max_new_tokens,
                    pad_token_id=tok.pad_token_id,
                    eos_token_id=tok.eos_token_id,
                    output_scores=True,
                    return_dict_in_generate=True,
                    use_cache=True,
                    **({"stop_strings": self.stop_strings, "tokenizer": tok}
                       if self.stop_strings else {}),
                    **gen_kwargs,
                )
        finally:
            if was_training:
                self.model.train()

        gen_ids = out.sequences[:, pad_len:]                       # [B, G]
        if _scores_are_raw_logits(self.temperature):
            # Free path: scores are untouched logits, so normalising them gives the raw
            # policy log-probs we need for the importance ratio.
            trans = self.model.compute_transition_scores(
                out.sequences, out.scores, normalize_logits=True
            )                                                      # [B, G]
        else:
            trans = None                                           # recovered per sequence below

        samples: List[Sample] = []
        for i, prompt in enumerate(prompts):
            mask = enc["attention_mask"][i].bool()
            prompt_ids = enc["input_ids"][i][mask].tolist()
            g = gen_ids[i]
            eos_hits = (g == tok.eos_token_id).nonzero(as_tuple=True)[0]
            if eos_hits.numel() > 0:
                end = int(eos_hits[0]) + 1                         # keep EOS: stopping is learned
                finished = True
            else:
                end = int(g.shape[0])
                finished = False
            comp_ids = g[:end].tolist()
            token_ids = prompt_ids + comp_ids
            if trans is not None:
                logp_comp = trans[i, :end].float().cpu().tolist()
            else:
                logp_comp = self._raw_completion_logprobs(token_ids, len(prompt_ids))
            # Only the completion span is ever read; prompt positions are placeholders.
            logp_old = [0.0] * (len(prompt_ids) - 1) + logp_comp
            samples.append(Sample(token_ids=token_ids, prompt_len=len(prompt_ids),
                                  logp_old=logp_old, finished=finished, prompt_text=prompt))
        return samples

    def _raw_completion_logprobs(self, token_ids: List[int], prompt_len: int) -> List[float]:
        """One extra forward pass to get raw log-probs when temperature != 1.0."""
        ids = torch.tensor(token_ids, device=self.device).unsqueeze(0)
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                logits = self.model(ids).logits[0]
                logp = logits[:-1].log_softmax(-1)
                tok_logp = logp.gather(-1, ids[0, 1:, None]).squeeze(-1)
        finally:
            if was_training:
                self.model.train()
        return tok_logp[prompt_len - 1 :].float().cpu().tolist()


class VLLMSampler:
    """Student served by its own vLLM process, with hot-swappable LoRA adapters.

    Needs the server started with ``--enable-lora`` and
    ``VLLM_ALLOW_RUNTIME_LORA_UPDATING=1`` so `sync_adapter` can push a freshly saved
    adapter without a restart. Token *ids* come back exactly (not re-tokenised text) via
    ``return_tokens_as_token_ids``; re-tokenising the generated text is not a safe
    substitute because the round-trip is not always the identity, and any drift would
    silently misalign the teacher's per-token scores.
    """

    ADAPTER_NAME = "student-live"

    def __init__(self, base_url: str, model: str, tokenizer, api_key: str = "EMPTY",
                 max_new_tokens: int = 512, temperature: float = 1.0,
                 max_concurrency: int = 32, timeout: float = 600.0,
                 stop_strings: Sequence[str] | None = None):
        from openai import OpenAI

        self.base_url = base_url.rstrip("/")
        self.base_model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.max_concurrency = max_concurrency
        self.stop_strings = list(stop_strings or [])
        self.client = OpenAI(base_url=self.base_url, api_key=api_key, timeout=timeout)
        self._adapter_loaded = False
        self._adapter_version = 0
        # Per-process tag in every adapter name: vLLM keeps names across trainer restarts, so
        # a restarted trainer counting from 1 again collides with 'student-live-1' from the
        # previous attempt and the push is refused, crash-looping the restarted pod.
        import time as _time
        self._run_tag = format(int(_time.time()) % 36**5, "x")

    # -- public ------------------------------------------------------------------------
    @property
    def model_name(self) -> str:
        """Serve from the adapter once one has been pushed, else the pristine base."""
        return (f"{self.ADAPTER_NAME}-{self._run_tag}-{self._adapter_version}"
                if self._adapter_loaded else self.base_model)

    def generate(self, prompts: Sequence[str], step: int = 0) -> List[Sample]:
        from concurrent.futures import ThreadPoolExecutor

        if not prompts:
            return []
        workers = min(self.max_concurrency, len(prompts))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(self._generate_one, prompts))

    def served_models(self) -> List[str]:
        """Model ids the sampler serves: its base model plus any loaded adapters."""
        return [m.id for m in self.client.models.list().data]

    def assert_serves_base(self) -> None:
        """The sampler must serve the trainer's student. Otherwise the first adapter push fails
        inside vLLM with a bare tensor-size mismatch (e.g. a 2B adapter on a 0.8B sampler) --
        the usual cause is a sampler left on another model by the baseline loop."""
        served = self.served_models()
        if self.base_model not in served:
            raise RuntimeError(
                f"the student sampler at {self.base_url} serves {served}, but the trainer's "
                f"student is {self.base_model!r}. Re-apply manifests/student-sampler.yaml-template "
                f"from the same env_vars as the trainer (STUDENT_MODEL, LORA_R), wait for it, "
                f"then restart the trainer.")

    def sync_adapter(self, path: str) -> None:
        """Load the freshly-saved LoRA adapter into the running server.

        Registered under a new name each time: vLLM caches adapters by name, so reusing one
        name can serve stale weights.
        """
        import urllib.error
        import urllib.request

        if self._adapter_version == 0:
            self.assert_serves_base()
        version = self._adapter_version + 1
        name = f"{self.ADAPTER_NAME}-{self._run_tag}-{version}"
        path = export_adapter_for_vllm(path, self.base_model)
        body = f'{{"lora_name": "{name}", "lora_path": "{path}"}}'.encode()
        req = urllib.request.Request(
            f"{self.base_url}/load_lora_adapter", data=body,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            urllib.request.urlopen(req, timeout=300).read()
        except urllib.error.HTTPError as exc:
            raise RuntimeError(
                f"vLLM refused the adapter push ({exc.code}): {exc.read()[:500]!r}. The sampler "
                f"must run --enable-lora with VLLM_ALLOW_RUNTIME_LORA_UPDATING=1, serve "
                f"{self.base_model!r}, and allow --max-lora-rank >= the adapter's rank."
            ) from exc
        self._adapter_version = version
        self._adapter_loaded = True
        if version == self.APPLIED_CHECK_AT:
            self._assert_adapter_applied(name)

    # By the Nth push the adapter has trained long enough to move log-probs measurably.
    APPLIED_CHECK_AT = 5

    def _assert_adapter_applied(self, name: str) -> None:
        """Fail loudly if vLLM serves `name` identically to the base model.

        A silently-ignored adapter makes every rollout base-model and the whole run a no-op
        (see export_adapter_for_vllm), so one extra scoring request pair is cheap insurance.
        """
        ids = self.tokenizer("Compute 17 * 23 and explain each step of the method.",
                             add_special_tokens=False).input_ids

        def logps(model: str) -> list[float]:
            resp = self.client.completions.create(model=model, prompt=ids, max_tokens=1,
                                                  echo=True, logprobs=0, temperature=0.0)
            return [float(v) for v in resp.choices[0].logprobs.token_logprobs[1:len(ids)]]

        diff = max(abs(a - b) for a, b in zip(logps(name), logps(self.base_model)))
        if diff == 0.0:
            raise RuntimeError(
                f"vLLM serves adapter '{name}' identically to {self.base_model}: the LoRA "
                f"is not being applied (module names do not match the served architecture). "
                f"Every rollout would come from the base student."
            )
        print(f"[sampler] adapter '{name}' is applied (max prompt log-prob shift {diff:.4f})",
              flush=True)

    # -- internals ---------------------------------------------------------------------
    def _generate_one(self, prompt: str) -> Sample:
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids
        resp = self.client.completions.create(
            model=self.model_name,
            prompt=prompt_ids,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=1.0,
            logprobs=1,
            stop=self.stop_strings or None,
            extra_body={
                # Return "token_id:NNNN" instead of decoded text so we recover exact ids.
                "return_tokens_as_token_ids": True,
                "top_k": -1,
                "min_p": 0.0,
                "repetition_penalty": 1.0,
                "skip_special_tokens": False,
            },
        )
        choice = resp.choices[0]
        lp = getattr(choice, "logprobs", None)
        if lp is None or not getattr(lp, "tokens", None):
            raise RuntimeError(
                "student sampler returned no logprobs; start the vLLM student with "
                "--max-logprobs 1 or higher."
            )
        comp_ids = [self._token_id(t) for t in lp.tokens]
        logp_comp = [float(v) for v in lp.token_logprobs]
        if len(comp_ids) != len(logp_comp):
            raise RuntimeError(
                f"sampler returned {len(comp_ids)} tokens but {len(logp_comp)} logprobs"
            )
        # "length" means we hit max_tokens mid-thought: the rollout is truncated, and
        # train_distill's --truncated-policy decides what to do with it.
        finished = choice.finish_reason == "stop"
        token_ids = list(prompt_ids) + comp_ids
        logp_old = [0.0] * (len(prompt_ids) - 1) + logp_comp
        return Sample(token_ids=token_ids, prompt_len=len(prompt_ids),
                      logp_old=logp_old, finished=finished, prompt_text=prompt)

    @staticmethod
    def _token_id(token: str) -> int:
        if not token.startswith("token_id:"):
            raise RuntimeError(
                f"expected 'token_id:N' but got {token!r}. This vLLM build does not honour "
                "return_tokens_as_token_ids; re-tokenising the text is not a safe fallback "
                "because it can misalign the teacher's per-token scores. Upgrade vLLM or "
                "use --sampler hf."
            )
        return int(token.split(":", 1)[1])


def make_sampler(
    kind: str, *, model=None, tokenizer=None, device: str = "cuda:0",
    base_url: str | None = None, base_model: str | None = None,
    max_new_tokens: int = 512, temperature: float = 1.0,
    micro_batch: int = 8, max_concurrency: int = 32,
    stop_strings: Sequence[str] | None = None,
):
    """Factory. `kind='auto'` uses vLLM when a sampler URL is configured, else HF."""
    if kind == "auto":
        kind = "vllm" if (base_url or os.environ.get("STUDENT_SAMPLER_URL")) else "hf"
    if kind == "vllm":
        url = base_url or os.environ["STUDENT_SAMPLER_URL"]
        return VLLMSampler(base_url=url, model=base_model or "", tokenizer=tokenizer,
                           max_new_tokens=max_new_tokens, temperature=temperature,
                           max_concurrency=max_concurrency, stop_strings=stop_strings)
    if kind == "hf":
        return HFSampler(model=model, tokenizer=tokenizer, device=device,
                         max_new_tokens=max_new_tokens, temperature=temperature,
                         micro_batch=micro_batch, stop_strings=stop_strings)
    raise ValueError(f"unknown sampler kind: {kind!r}")
