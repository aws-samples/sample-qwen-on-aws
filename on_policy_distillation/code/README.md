# On-Policy Distillation on SageMaker HyperPod EKS — Code Talk

Distil a **teacher** (`Qwen/Qwen3.5-9B`, bf16, one whole copy sharded over TP=2) into a lean
**student** (`Qwen/Qwen3.5-2B`, instruct, LoRA r=128) by training on the *student's own* GSM8K
rollouts with **dense, per-token feedback** (reverse KL to the teacher). No verifier, no reward
model and no supervised warm-up. Everything runs on **one `ml.g5.24xlarge`** (4× A10G 24 GB).

**Evaluation** (all 1,319 GSM8K test problems): the base student and the teacher greedy, the
distilled student greedy and with an **8-vote majority** (`EVAL_VOTES=8`, `EVAL_VOTE_PHASES=after`:
8 answers at temperature 0.7, the most common one scored) — a decoding choice, so the dashboards
label it. The student is chosen by a **baseline eval** first: the strongest candidate that still
leaves a real gap to the teacher. Per-run configs and measured results are in
[`experiments/`](experiments/).

**Target cluster:** `hp-cluster-onpolicy-distillation` (instantstart) · EKS, `us-west-2`, ns
`default`. Storage: **FSx `/fsx` (`fsx-claim`) = high-throughput working store** (HF cache,
checkpoints, metrics); **S3 `/s3` (`s3-claim`) = model registry**.

## The whole run fits on four GPUs — here is how they are divided

K8s allocates GPUs to a pod **exclusively** (no MPS or time-slicing on this cluster), so the
four devices are *partitioned*, never shared:

| GPU | Occupant | Footprint |
|---|---|---|
| 0 | `teacher-vllm` rank 0 — half of one bf16 `Qwen/Qwen3.5-9B`, TP=2, `replicas: 1` | ~9.0 GB weights, ~7.8 GB cache |
| 1 | `teacher-vllm` rank 1 — the other half | ~9.0 GB weights, ~7.8 GB cache |
| 2 | `opd-student` — the trainer (bf16 2B + fp32 LoRA r=128) | peaks on 1024-token rollouts (`MICRO_BATCH_SIZE=2`) |
| 3 | `student-sampler` (vLLM rollouts, LoRA hot-swap; also serves the baseline and eval) | ~4 GB + KV |

`preflight.py` check **[8]** does this subtraction before anything launches, because an
oversubscribed partition surfaces as a pod stuck `Pending` forever with no error in any log. It
multiplies `TEACHER_TP × TEACHER_REPLICAS`, which is 2 here — so **all 4 devices are claimed**
and the node is exactly full.

**No spare GPU as configured** — exact (unquantised) teacher log-probs at the cost of a full node.
The eval does not need one: it generates through the sampler and the teacher over HTTP
(`EVAL_STUDENT_URL`, `EVAL_GPUS=0`), so it runs alongside training. Raising `TEACHER_MAX_NUM_SEQS`
does nothing to the wall clock: bf16 9B at TP=2 is prefill **compute**-bound at this batch size,
not queue-bound.

> ### Why bf16 at TP=2, and why not a quantised TP=1 alternative
> Scoring is a **single prefill forward pass** per request (echo-logprobs, zero generated tokens),
> and bf16 `Qwen/Qwen3.5-9B` is **18.0 GiB** against a ~20.2 GiB per-GPU budget
> (`0.90 × 22.5`) — it does not fit one A10G. Sharded over TP=2 that becomes ~9.0 GB/rank, which
> leaves **~7.8 GB/GPU** of KV/recurrent-state cache. preflight check **[2]** hard-fails the bf16-
> at-TP=1 case by name. TP=2 revives an all-reduce, and A10G has no NVLink: every all-reduce
> crosses PCIe, and on `g5.24xlarge` GPUs 0-1 and 2-3 sit under *separate* PCIe switches — a
> split pair is measurably slower than a same-switch pair. preflight check **[4]** warns when the
> kubelet hands out a `SYS` pair rather than a `PHB`/`PXB` one.
>
> **What we get for accepting TP=2.** The teacher's log-probs are **exact** — no quantisation
> error lands in the KL target. Quantising a 9B perturbs the supervision more than the 27B's
> MLP-only W4A16 did, because there are fewer parameters to absorb the error, and every 4-bit 9B
> checkpoint on the Hub is a third-party build. bf16 is the safe route when the signal, not the
> ceiling, is the concern.
>
> **What is closed on this hardware:**
> 1. **In-flight 4-bit is gone.** Serving a bf16 repo with
>    `--quantization bitsandbytes --load-format bitsandbytes` no longer works: vLLM deleted
>    bitsandbytes from its quantization registry, so the pod dies with `Unknown quantization method: bitsandbytes`.
>    Pinning an older image is not an escape hatch — `Qwen3_5ForConditionalGeneration` exists only
>    in builds *newer* than the removal.
> 2. **FP8/W8A8 is refused, not downgraded.** A10G is **sm_86** and vLLM's
>    `CompressedTensorsW8A8Fp8.get_min_capability()` is **89**, so an FP8 9B (e.g.
>    `RedHatAI/Qwen3.5-9B-FP8-dynamic`, 12.6 GiB — which *would* fit) dies with `Quantization
>    scheme is not supported for the current GPU. Min capability: 89. Current capability: 86.`
>    preflight check **[2]** screens this from a per-method capability table before you pull.
> 3. **W4A16 TP=1 is available as a fallback** (see `manifests/env_vars` — `sanskar003/Qwen3.5-9B-AWQ`,
>    `TEACHER_QUANTIZATION=compressed-tensors`). Choose it if you need the spare GPU back; accept
>    the log-prob perturbation and the third-party checkpoint.
>
> Two honest costs of the shipped (bf16, TP=2) layout:
> - **The ceiling.** The teacher is the accuracy *ceiling* — the student moves toward it and cannot
>   pass it. `headroom_recovered = (after − before) / (teacher − before)`. Run
>   the baseline first to measure the gap.
> - **The all-reduce.** TP=2 pays a PCIe all-reduce per prefill layer. On a split-switch pair it
>   is measurably slower than a same-switch pair, which is why preflight [4] warns.
>
> **If the ceiling is the problem**, `manifests/env_vars` documents the switch to
> `Qwen/Qwen3.5-27B-GPTQ-Int4` (28.2 GiB, MLP-only W4A16), which also needs TP=2 — same GPU
> partition, larger teacher. Note there is **no Qwen3.5-14B** — the dense line is 0.8B / 2B /
> 4B / 9B / 27B.

> **Confirm the node exists and really has 4 GPUs before applying anything:**
> ```bash
> kubectl --context "$CTX" get nodes -L node.kubernetes.io/instance-type
> ```
> `manifests/env_vars` defaults `INSTANCE_TYPE=ml.g5.24xlarge`. On `ml.g6.12xlarge` (4× L4
> 24 GB) every number above is unchanged and L4 is the faster GPU. Note g5 is **A10G**, g6 is
> **L4**.

## The pipeline is three stages, in this order

| # | Stage | Script / Job | Why it exists |
|---|---|---|---|
| 0 | **Baseline & model selection** | `src/eval_student.py` · `opd-student-eval` | The teacher and each candidate student on the same 1,319 problems: the gap decides which student to distil. |
| 1 | **On-policy distillation** | `src/train_distill.py` · `opd-student` | Per-token reverse KL against the teacher on the student's *own* rollouts, starting from the base student. |
| 2 | **Eval** | `src/eval_student.py` · `opd-student-eval` | The distilled checkpoints, against the baseline's base student **and teacher** (carried forward). |

**There is no supervised warm-up** — the trainer starts from the pristine instruct student, and
the 4-shot prompt (`TRAIN_FEWSHOT=EVAL_FEWSHOT=4`, enforced by preflight [7]) carries the answer
format. Five settings turned out to be essential, each found by a failed run or the baseline:

- **`LR=2e-5`, `STEPS=50`.** It reaches the plateau by step 25; 1e-5 takes about four times as long
  to the same level, and more steps do not raise it. At 1e-4 the student collapsed within ~4 steps
  (capped rollouts 9% → 50–70%).
- **The 2B student with LoRA r=128** (`LORA_ALPHA=256`). The ceiling is the student's capacity: the
  0.8B plateaus far lower whatever the schedule, and a larger LoRA rank adds little.
- **`ANSWER_SPAN_POLICY=mask`** — zero credit on the final-answer line and `<|im_end|>`. Those tokens
  carry the largest reverse KL (+0.54 vs. +0.33 for reasoning) because the teacher can tell when
  the number is wrong; without masking the student learns to stop writing answers.
- **`WHITEN_ADVANTAGES=1`.** The rollouts' reverse KL averages ~+0.3, so without a baseline nearly
  every sampled token is pushed down and the student's distribution flattens.
- **64 prompts × 4 samples per step** (`SAMPLES_PER_PROMPT=4`), the Thinking Machines default.

**Read `teacher_kl` with length.** Long, rambling text is easy for the teacher to predict, so a
student drifting toward longer answers *lowers* `teacher_kl` without learning. Watch it next to
`truncated_frac`, `completion_tokens_mean` and `answer_masked_rollouts`, and judge by the eval.

Run the baseline first (notebook Part 1, ~45 min for the teacher and two candidates) to measure the
gap before a ~4.5 h run.

## Contents
```
on_policy_distillation.ipynb        # the control panel: setup -> baseline + model selection -> train -> monitor -> eval
architecture.md                     # the GPU partition + data flow, as a mermaid diagram
src/
  prompts.py                        # THE prompt format, answer extraction and equivalence —
                                    #   one module shared by training and eval so the two
                                    #   formats cannot drift apart
  data.py                           # prompt pool: full train split, epoch-aware, no repeats,
                                    #   contamination-filtered against the eval set
  model_setup.py                    # LoRA student (bf16 base + fp32 adapter), optimizer, LR schedule
  sampler.py                        # batched rollout generation: HF `generate` or a vLLM server
  teacher_client.py                 # score student tokens via vLLM echo-logprobs; RAISES on nulls
  rollout_buffer.py                 # deque buffer with staleness EVICTION, not just reporting
  distill_step.py                   # reverse-KL advantage + clipped IS loss (core algorithm)
  train_distill.py                  # stage 1: the headless trainer the student Job runs
  eval_student.py                   # stage 2: Inspect AI eval, before/after/teacher (GSM8K or MATH-500)
  preflight.py                      # fails fast on the silent killers (see below)
  compare_ckpts.py, merge_debug.py  # debug tools for checkpoints and LoRA merges
manifests/
  env_vars                          # SINGLE source of truth: cluster, storage, models, both stages
  env_vars.local                    # git-ignored overrides for account values (CTX, S3_BUCKET)
  opd-config.yaml-template          # ConfigMap built from env_vars; every Job reads it via envFrom
  teacher-vllm.yaml-template        # Deployment + Service, bf16 teacher, TP=2 x 1 replica (GPUs 0-1)
  student-sampler.yaml-template     # vLLM rollout sampler with LoRA hot-swap (GPU 3)
  student-distill-job.yaml-template  # stage 1 Job
  student-eval-job.yaml-template     # stage 2 Job (no GPU: HTTP to the sampler + teacher)
  LAUNCH.md                         # how to create resources (admin context / instantstart) + monitor
```

The Jobs pass **no arguments** — every flag defaults from an environment variable, and
`manifests/opd-config.yaml-template` renders `env_vars` into the one ConfigMap they all read.
That is deliberate: the previous layout repeated a subset of the variables in each manifest, so
a pod could silently run with a value nobody set.

## Feasibility: `Qwen3.5-9B` as teacher on the node's four 24 GB GPUs

| | bf16, TP=1 (impossible) | FP8, TP=1 (refused) | **bf16, TP=2 × 1 (shipped)** | W4A16, TP=1 (fallback) |
|---|---|---|---|---|
| repo | `Qwen/Qwen3.5-9B` | `RedHatAI/Qwen3.5-9B-FP8-dynamic` | **`Qwen/Qwen3.5-9B`** | `sanskar003/Qwen3.5-9B-AWQ` |
| method | — | `compressed-tensors` (`float-quantized`) | **none (bf16)** | `compressed-tensors` (`pack-quantized`, W4A16) |
| weights | 18.0 GiB on one 22.5 GiB GPU | 13.0 GiB → fits on paper | **18.0 GiB → ~9.0 GB/GPU sharded** | 7.96 GiB on one GPU |
| GPUs consumed | 1 | 1 | **2** (`TEACHER_TP=2 × 1`) | 1 |
| status | **negative cache pool** — preflight [2] fails it | **`Min capability: 89. Current capability: 86.`** | shipped; no spare GPU | works; frees a GPU, third-party checkpoint |
| what stays bf16 | everything | activations only | **everything (unquantised)** | vision tower, 24 linear-attn layers, `lm_head` |
| left for KV cache | −1.2 GB ❌ | — (never loads) | **~7.8 GB/GPU** | ~9.8 GB/GPU |
| KV capacity | — | — | ~507k tokens ≈ **~141 concurrent 2048-token seqs** | ~320k tokens ≈ ~89 seqs |
| prefill speed | — | — | 2 sharded GPUs + PCIe all-reduce | 1 server, Marlin kernels, no all-reduce |
| teacher log-probs | exact | exact-ish (W8A8) | **exact** | ~0.02–0.08 nats/token of noise |

Read the KV rows carefully — the arithmetic is **not** the usual transformer arithmetic, because
Qwen3.5 is **hybrid**: `layer_types` alternates 3 `linear_attention` layers to 1 `full_attention`
(`full_attention_interval: 4`), so of the 9B's 32 layers only **8** keep a growing KV cache while
**24** hold a fixed recurrent (gated delta-net) state **per sequence**. Both terms are divided by TP.

- **Per token**: `2 × 8 layers × 4 KV heads × 256 head_dim × 2 bytes` = **32 KiB**, and at TP=2
  each rank holds half of it. Counting *all* layers instead would overstate it 4x.
- **Per sequence**: `24 layers × 32 value heads × 128 × 128` in **fp32** (`mamba_ssm_dtype`) =
  **48 MiB**. At 24+ concurrent sequences it *dominates*
  the per-token term, which is why the concurrency column is much lower than `cache ÷ 32 KiB ÷ 2048`
  would suggest.

`TEACHER_MAX_NUM_SEQS=48` is **per replica** — 48 in flight as configured — and sits well inside
~141-at-full-length, because a real scoring request is a 4-shot prompt plus a ≤512-token completion
(~700 tokens), not 2048. At this footprint the binding constraint is prefill **compute**, not memory,
which is why raising the queue depth would not buy throughput.

Three hardware facts matter:

- **No NVLink on A10G/L4**, so every tensor-parallel all-reduce crosses PCIe — which is the price
  of running the teacher unquantised on this hardware. Teacher scoring is a single prefill pass per
  request (echo-logprobs, zero generated tokens), so its per-step overhead is a small fixed cost
  each request pays; it does not stack with generation.
- **At TP=2, *which* two GPUs matters.** On g5.24xlarge, GPUs 0–1 and 2–3 sit under separate PCIe
  switches, and a split pair costs a slower all-reduce; the kubelet chooses. preflight check [4]
  warns on `SYS` pairings so you can `kubectl delete pod` and let the scheduler retry rather than
  discover the slower path from wall-clock alone.
- **A10G is sm_86, and low-precision schemes are *refused*, not downgraded.** FP8/W8A8 declares
  `get_min_capability() == 89`, so an FP8 checkpoint dies at load with `Quantization scheme is not
  supported for the current GPU. Min capability: 89. Current capability: 86.` **W4A16
  (`CompressedTensorsWNA16`) is 75** — "Turing and up" — and is the only quantised route available
  here; `gptq_marlin`/`awq_marlin` want 80, which A10G does meet. In-flight quantisation is also
  gone: bitsandbytes NF4 was the one scheme that could quantise a bf16 checkpoint as it loaded, and
  going back to a build that has it means going back before Qwen3.5 support existed. vLLM's newer
  "online quantization" shorthands do not substitute: `int8_per_channel_weight_only` applies to MoE
  experts only, `nvfp4_per_token` is Blackwell + FlashInfer, and every `fp8_*`/`mxfp8` variant needs
  FP8 silicon. preflight check [2] screens both the registry and the capability floor.

Two non-obvious manifest requirements: `/dev/shm` (**16Gi** at TP=2, because peer ranks exchange
tensors through shared memory and an undersized `/dev/shm` is the classic cause of a silent hang
partway through startup) and `VLLM_WORKER_MULTIPROC_METHOD=spawn` (`fork` deadlocks once CUDA is
initialised in the parent; required at TP=2 for vLLM's multi-GPU worker path). Note `medium:
Memory` bills against node RAM and `/dev/shm` is charged **per pod**, so it multiplies with
`TEACHER_REPLICAS` — comfortable in the node's 384 GiB.

**Wall clock.** A step (64 prompts × 4 rollouts) takes ~5.5 min on this node — ~50 s generation,
~80 s teacher scoring, ~200 s training — so the default 50 steps is ~4.5 h. Every `SAVE_EVERY=25` steps writes an
independently evaluable adapter.

**LoRA, with an fp32 adapter.** `PRECISION=lora` keeps the frozen base in bf16 and the trainable
adapter in **fp32**, which fits one 24 GB GPU (a 4B in fp32 with Adam would need ~68 GB).
The fp32 part is not optional: bf16 has 8 mantissa bits (~0.4% relative resolution), so an Adam
step of magnitude ≈ `lr` rounds away to nothing and the optimizer silently does nothing at all.

## `preflight.py` — catch silent failures before training
Run automatically by the distill Job before training starts; exits non-zero on failure.

1. **TP divisibility** — `num_key_value_heads % TP`, `num_attention_heads % TP`,
   `intermediate_size % TP`, and (Qwen3.5 being hybrid) `linear_num_key_heads % TP` and
   `linear_num_value_heads % TP`. vLLM only reports this *after* loading the weights. It reads
   dimensions out of `text_config`, because `Qwen3_5ForConditionalGeneration` is a **composite**
   config that does not forward attribute lookups — a plain `getattr(cfg, "num_hidden_layers")`
   returns `None` for this whole family, which made this check and check [2] silent no-ops on the
   one model actually served. At `TEACHER_TP=2`, note the 9B has 4 KV heads — divisible.
2. **Teacher quantisation and memory fit.** Two gates before any arithmetic. First, whether the
   serving vLLM build actually *has* `TEACHER_QUANTIZATION` — it imports the real registry from the
   image the teacher runs, and fails by name on methods vLLM retired (`bitsandbytes` among them),
   which otherwise kills the Deployment at startup with a pydantic `Unknown quantization method`
   error that looks nothing like a problem in these files. Second, whether the **GPU meets the
   scheme's minimum compute capability** — an FP8 checkpoint needs sm_89 and A10G is sm_86, so it is
   *refused* by the kernels. Then the fit at TP on the target GPU with the resulting cache headroom,
   hybrid-aware: per-token KV over the full-attention layers **plus** the fixed per-sequence
   recurrent state of the linear-attention layers, both sharded by TP. Set `TEACHER_WEIGHTS_GIB` to
   the checkpoint's measured size — for the shipped bf16 9B that is 18.0 GiB (9.653 B × 2 bytes),
   and real 4-bit repos quantise only some layers so bytes/param is off by >2× for
   `Qwen3.5-27B-GPTQ-Int4`.
3. **Shared tokenizer** — the teacher scores raw token ids from the student, so a differing
   vocabulary means it scores different text. Silent and fatal.
4. **GPU count and interconnect** (`nvidia-smi topo -m`), and compute capability. Checks what
   *this* pod needs, not `TEACHER_TP` — a 1-GPU trainer pod alongside a 2-GPU teacher is correct.
   At `TEACHER_TP=2` it warns on `SYS` pairings, since GPUs 0-1 and 2-3 sit under separate PCIe
   switches on g5.24xlarge and a split pair pays a slower all-reduce.
5. **Teacher echo-logprobs are actually populated.** The single most important check: vLLM
   returns `null` log-probs when the server was not started with `--max-logprobs`, the old
   client turned those into `0.0`, and the advantage became `-log π_student` — pure
   self-reinforcement, while `teacher_kl` still printed a convincing falling curve.
6. **Student training memory fit** for the chosen precision.
7. **Train/eval prompt-format agreement** — `TRAIN_FEWSHOT` vs `EVAL_FEWSHOT`, `CHAT_TEMPLATE`
   vs `EVAL_CHAT_TEMPLATE`, `MAX_NEW_TOKENS` vs `EVAL_MAX_TOKENS`.
8. **The single-node GPU partition** —
   `TEACHER_TP × TEACHER_REPLICAS + GPU_PER_NODE + SAMPLER_GPUS ≤ NODE_GPUS`. The **product** is the
   point: the teacher's claim is whole copies × GPUs-per-copy, so counting `TEACHER_TP` alone would
   silently undercount the moment `TEACHER_REPLICAS` goes above 1. As configured this reports **4 of
   4** claimed (2 + 1 + 1). Oversubscribing four GPUs does not raise an error anywhere; the losing
   pod simply sits `Pending` until someone notices. It also warns when the partition is *exactly*
   full, since the eval Job can then only schedule after the trainer pod releases its device.

The trainer adds a **step-0 gate** on top of this: it prints the measured teacher and student
log-probs and exits `3` if the reverse KL is not positive, rather than spend hours optimising the
student against noise.

## Did it work? — before / after / **teacher**
`teacher_kl` trending down says the student is matching the teacher's *distribution*; the eval
says whether that turned into **task accuracy**. `src/eval_student.py` runs one Inspect AI task
against three models, same samples and same prompt:

| phase | model | source |
|---|---|---|
| `before`  | `Qwen/Qwen3.5-2B` | the pristine student, generated by the sampler (the Part 1 baseline) |
| `after`   | `student-step<N>` | a LoRA checkpoint loaded into the sampler (`$EVAL_FT_MODEL_DIR`) |
| `teacher` | `Qwen/Qwen3.5-9B` (bf16) | the teacher Service (`$TEACHER_URL`) |

**The teacher phase is what makes the number mean anything.** A student cannot be distilled past
its teacher, so `after − before` in isolation is uninterpretable — +3 points is excellent if the
teacher is 5 ahead and dismal if it is 40 ahead. The reported metric is

```
headroom_recovered = (after − before) / (teacher − before)
```

and if the teacher turns out *not* to be ahead of the base student, the summary says so
explicitly: there is no headroom to distil and nothing downstream can create any.

**The benchmark is GSM8K, all 1,319 test rows** (`TASK=gsm8k`, `EVAL_LIMIT=1319`), 4-shot,
greedy, semantic match on the `#### N` answer; the standard error is ~±1.4 pp. `TASK=math500`
also works and suits a larger (≥4B) student — for the 0.8B it is too hard (most rollouts fail at the
training temperature).

**Phases carry forward.** `eval_student.py` reuses any phase already in `$EVAL_DIR/summary.json` that
was recorded with the same eval config, so the notebook measures `before` and `teacher` once in the
Part 1 baseline (one directory per model, `/fsx/opd/baseline-<task>-<model>`) and runs only `after`
for each checkpoint.

**Majority vote.** `EVAL_VOTES=8` samples each problem 8 times at `EVAL_VOTE_TEMPERATURE` (0.7) and
scores the most common final answer, for the phases in `EVAL_VOTE_PHASES` (default `after`; the
others stay greedy). Each result records its `decoding`; voted results add `maj_at` (accuracy over
the first 1..8 votes), `pass_at_k` and `mean_vote_share`, and a phase is only carried forward when
its decoding matches. The shipped headline votes on the distilled student only — greedy base, voted
distilled student, greedy teacher — and every chart labels it; set `EVAL_VOTES=1` for greedy.

```bash
set -a && source manifests/env_vars && set +a
aws s3 sync src s3://$S3_BUCKET/opd/src
envsubst < manifests/opd-config.yaml-template       | kubectl --context "$CTX" apply -f -
envsubst < manifests/student-eval-job.yaml-template | kubectl --context "$CTX" apply -f -
kubectl --context "$CTX" -n default logs -f job/opd-student-eval   # accuracy, delta, headroom
```

Outputs under `$EVAL_DIR`: `summary.json` (accuracy ± stderr per phase, `delta_accuracy`,
`teacher_gap`, `headroom_recovered`, paired `transitions` = fixed / broken / both_correct /
both_wrong, format-compliance and length metrics), `samples_<phase>.jsonl` (one row per eval
sample) and Inspect logs in `logs/<phase>` (`inspect view --log-dir …`). The same payload is
printed between `===EVAL_DASHBOARD_JSON_BEGIN/END===` sentinels. The notebook's dashboards read
`summary.json` and the per-sample files from `/fsx` through the teacher pod.

## The method in one screen
```
reverse_kl_t = log π_student(y_t|y_<t) − log π_teacher(y_t|y_<t)
advantage_t  = −kl_coef * clamp(reverse_kl_t, ±rkl_clip)        # then whitened across the batch
ratio_t      = exp( log π_new(y_t) − stop_grad(log π_old(y_t)) ) # ≈1 on fresh rollouts
loss         = −mean_t min( ratio_t·A_t, clip(ratio_t, 1±ε)·A_t )
```
Two conditioning departures from the raw recipe, both load-bearing at this model-size ratio:

- **`rkl_clip`** — per-token reverse KL between a small student and a 9B teacher is heavy-tailed. A
  handful of tokens carrying tens of nats consumed the entire `clip_grad_norm_(1.0)` budget, so
  the rest of the sequence contributed essentially nothing.
- **Batch whitening** — advantages are centred and scaled across the whole batch. The batch mean
  acts as a baseline (standard in policy gradient) and turns a uniformly negative push into a
  well-conditioned update. `kl_coef` then becomes a pure temperature on an already-normalised
  signal. `--no-whiten` recovers the unnormalised objective.

Health metrics to watch in `${RUN_DIR}/metrics_rank0.jsonl`: `teacher_kl` (should fall at stable length),
`clip_frac` (above ~0.2 means rollouts are too stale — lower `BUFFER_MAX_AGE` or
`SAMPLER_SYNC_EVERY`), `policy_drift`, `truncated_frac`, and `epochs_done` (above 1 means the
prompt pool has started recycling).

## Running it
- **Live algorithm demo (Parts 1 & 3):** runs in the notebook kernel — needs a **GPU kernel**.
  `TEACHER_MODE="local"` loads a teacher in-kernel; a bf16 9B does not fit one 24 GB GPU next to a
  student, so use
  the remote path (`port-forward svc/teacher-vllm 8000:8000`, `TEACHER_URL=http://localhost:8000/v1`)
  or a small stand-in teacher for the demo.
- **Cluster orchestration (Parts 2 & 4):** the notebook role has **EKS cluster-admin**, so
  `RUN_CLUSTER=True` creates the teacher `Service` and student `Job`s directly from the notebook
  (works from a CPU kernel too; the cluster Jobs do the GPU work). See `manifests/LAUNCH.md`.

```bash
set -a && source manifests/env_vars && set +a           # sets $CTX, storage, models
kubectl create secret generic hf-token --from-literal=token=hf_xxx -n default   # if pulling from HF
envsubst < manifests/opd-config.yaml-template          | kubectl --context "$CTX" apply -f -
envsubst < manifests/teacher-vllm.yaml-template        | kubectl --context "$CTX" apply -f -
kubectl --context "$CTX" -n default rollout status deploy/teacher-vllm --timeout=2400s
aws s3 sync src s3://$S3_BUCKET/opd/src
envsubst < manifests/student-distill-job.yaml-template | kubectl --context "$CTX" apply -f -
kubectl --context "$CTX" -n default get pods -o wide
```

## Who uses on-policy distillation (verified)
- **Qwen3** — student generates on-policy sequences, aligned to teacher (Qwen3-32B/235B) logits via
  KL divergence. *Qwen3 Technical Report, arXiv:2505.09388.*
- **Xiaomi MiMo-V2-Flash** — Multi-Teacher On-Policy Distillation (MOPD): teachers give dense,
  token-level reward on the student's rollouts. *arXiv:2601.02780.*
- **Method / recipe:** Thinking Machines Lab, *On-Policy Distillation* (2025); **GKD**: Agarwal et al.,
  *On-Policy Distillation of LMs*, ICLR 2024 (arXiv:2306.13649). Reference impl: Tinker Cookbook
  `distillation/` (`incorporate_kl_penalty`). Reverse-KL background: MiniLLM (arXiv:2306.08543).
- **One honest divergence from those reports:** they pair on-policy distillation with an
  off-policy SFT warm-up rather than running it from a cold student. This pipeline runs
  distillation only, from the pristine instruct student.

**Not on-policy (useful contrast):** DeepSeek-R1/V3 and Kimi k1.5/k2 *distill off-policy* — SFT on
teacher-generated traces. Gemma 2/3, Llama 3.2/4, NVIDIA Minitron use token-level KD on a **fixed
corpus** (also off-policy). GLM-4.5/4.6 use off-policy expert distillation; GLM-5's report emphasizes
RL (no confirmed on-policy-distillation claim — omitted).
