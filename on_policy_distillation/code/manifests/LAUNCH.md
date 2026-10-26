# Launching on the `hp-cluster-onpolicy-distillation` cluster
The SageMaker notebook role has been granted **EKS cluster-admin** (`AmazonEKSClusterAdminPolicy`
on its access entry), so it can create the teacher and student **directly from the notebook**
(`RUN_CLUSTER=True`) or from any admin `kubectl` context.

The pipeline is **two stages**: on-policy distillation → eval. There is no supervised warm-up;
the trainer starts from the base student. Run them in order.

## 0. One-time prerequisites

**Check the node first.** Everything runs on **one** node with **4 GPUs**:
```bash
kubectl --context "$CTX" get nodes -L node.kubernetes.io/instance-type
kubectl --context "$CTX" describe node <name> | grep -A3 Allocatable   # confirm nvidia.com/gpu: 4
```
`env_vars` defaults `INSTANCE_TYPE=ml.g5.24xlarge` (4× A10G 24 GB). On `ml.g6.12xlarge` (4× L4
24 GB) the arithmetic is identical and L4 is the faster GPU. (g5 = A10G, g6 = L4; the two are
often confused.)

**The four GPUs are partitioned, not shared** — K8s allocates them to pods exclusively:

| GPU | Occupant | `env_vars` |
|---|---|---|
| 0-1 | `teacher-vllm` — one bf16 `Qwen/Qwen3.5-9B`, sharded TP=2 | `TEACHER_TP=2`, `TEACHER_REPLICAS=1` |
| 2 | the student: stage 1 trainer → stage 2 eval | `GPU_PER_NODE=1` |
| 3 | `student-sampler` | `SAMPLER_GPUS=1` |

`preflight.py` check [8] verifies
`TEACHER_TP × TEACHER_REPLICAS + GPU_PER_NODE + SAMPLER_GPUS ≤ NODE_GPUS` — the **product**, since
each replica claims `TEACHER_TP` GPUs of its own. As configured that is 2 + 1 + 1 = **4 of 4**.
Oversubscribing raises no error — the losing pod just sits `Pending` — so run it before applying.

**One teacher sharded across TP=2 rather than one whole model at TP=1**, because bf16
`Qwen/Qwen3.5-9B` is 18.0 GiB against a ~20.2 GiB per-GPU budget — it does not fit one A10G.
Sharded at TP=2 it becomes ~9.0 GB/rank with ~7.8 GB/GPU left for cache. The trade-off is a PCIe
all-reduce per prefill layer (A10G has no NVLink) and the fact that *which* two GPUs the kubelet
picks matters — GPUs 0-1 and 2-3 sit under separate PCIe switches on g5.24xlarge and a split pair
is measurably slower; preflight [4] warns on `SYS` pairings. The full argument, and the W4A16 TP=1
fallback that frees a GPU, is at the top of `env_vars`.

**The node is exactly full, but the eval needs no GPU.** It generates through the sampler and the
teacher over HTTP (`EVAL_STUDENT_URL`, `EVAL_GPUS=0`), so it runs alongside training. Raising
`TEACHER_MAX_NUM_SEQS` is not a substitute for a faster teacher: the teacher is prefill
compute-bound, not queue-bound.

- **Download the teacher** `Qwen/Qwen3.5-9B` (bf16 safetensors, ~18 GiB) into the bucket
  (GUI → HF Download → S3, or the agent). Do the same for the `Qwen/Qwen3.5-2B` student (and `Qwen/Qwen3.5-0.8B` if you baseline it).
- **Download the bf16 repo**, unquantised. Two quantised routes are closed on this hardware:
  vLLM can no longer quantise a bf16 checkpoint as it loads (`--quantization bitsandbytes`;
  vLLM removed bitsandbytes from its registry, so that configuration
  fails at pod startup with `Unknown quantization method: bitsandbytes`, and pinning an older image
  does not help because Qwen3.5 support arrived only after the removal), and FP8/W8A8 is *refused*
  on this hardware — A10G is sm_86, the scheme needs sm_89, and the pod dies with `Min capability:
  89. Current capability: 86.` W4A16 (`sanskar003/Qwen3.5-9B-AWQ` with
  `TEACHER_QUANTIZATION=compressed-tensors`) is the documented fallback; see `env_vars`.
- Leave `TEACHER_MODEL_ID=Qwen/Qwen3.5-9B` — it is the name clients request and the repo preflight
  reads the config and tokenizer from, so it must stay a real HF id. `TEACHER_MODEL`, the thing
  vLLM loads, can be either the HF id or a `/s3` / `/fsx` path.
- **Sync the code** so the pods can read it:
  ```bash
  aws s3 sync src s3://<s3_bucket>/opd/src
  ```
- **Apply the ConfigMap.** Both Jobs read their entire configuration from it and pass no
  arguments, so this must exist and be current before any Job is created:
  ```bash
  set -a && source manifests/env_vars && set +a
  envsubst < manifests/opd-config.yaml-template | kubectl --context "$CTX" apply -f -
  ```
  Re-apply it after every `env_vars` edit. A running pod does **not** pick up ConfigMap changes
  to `env`, and Jobs are immutable — delete and recreate the Job.

**Storage layout:** `/fsx` (FSx for Lustre, PVC `fsx-claim`) is the high-throughput working store —
HF cache (`/fsx/hf_cache`), checkpoints, adapters and metrics (`${RUN_DIR}`, e.g. `/fsx/opd/gsm8k-opd`). `/s3` (PVC
`s3-claim`) is the model registry + code. Both are mounted in every
pod. *Optional, for fastest cold starts:* stage the weights onto FSx and point
`TEACHER_MODEL`/`STUDENT_MODEL` at `/fsx/...`. This matters at TP=2 in particular: two ranks pull
18.0 GiB of weights concurrently through the same node.

## 1. Teacher (vLLM Service, `Qwen/Qwen3.5-9B` at TP=2, one replica, GPUs 0-1)
```bash
set -a && source manifests/env_vars && set +a
envsubst < manifests/teacher-vllm.yaml-template | kubectl --context "$CTX" apply -f -
kubectl --context "$CTX" -n default rollout status deploy/teacher-vllm --timeout=1800s
```
**Expect several minutes.** 18.0 GiB of bf16 weights are read across two GPUs and CUDA-graph
capture runs per rank, so a cold first start is I/O-bound and does an NCCL init on top; the probes
allow ~20 minutes. Confirm the pod actually landed — if it sits `Pending`, the partition is
oversubscribed:
```bash
kubectl --context "$CTX" -n default get pods -l app=teacher-vllm -o wide
```
At TP=2, *which* pair of GPUs the kubelet chose matters — GPUs 0–1 and 2–3 sit under separate PCIe
switches on g5.24xlarge, so a split pair costs a slower all-reduce. `preflight` check [4] warns on
`SYS` pairings; the fix is to delete the pod and let the scheduler retry.

Three flags are not optional:

- **Distributed executor.** At TP=2 vLLM needs `--distributed-executor-backend mp` (added
  automatically via `TEACHER_DIST_FLAG` when `TEACHER_TP>1`). No `--quantization` flag is set for
  the shipped bf16 route (`TEACHER_QUANTIZATION=""` → empty `TEACHER_QUANT_FLAG`). If you switch to
  the W4A16 fallback, set `TEACHER_QUANTIZATION=compressed-tensors`; the flag is derived so the two
  cannot drift. preflight [2] verifies both the method (against the vLLM build's registry) and its
  minimum compute capability.

- `--max-logprobs 1` — without it, vLLM returns `null` for echo log-probs. Turned into `0.0`,
  nulls make the advantage `−log π_student` (self-reinforcement) while `teacher_kl` still prints
  a falling curve.
  `teacher_client.py` now **raises** instead of zeroing, and `preflight.py` check [5] catches it
  before any training starts.
- `--served-model-name ${TEACHER_MODEL_ID}` — clients must request this exact string.

Sanity-check it directly once it is up:
```bash
kubectl --context "$CTX" -n default port-forward svc/teacher-vllm 8000:8000 &
python3 src/preflight.py --teacher-url http://localhost:8000/v1
```

**or instantstart agent (kiro-cli):**
> "Deploy `vllm serve /s3/Qwen-Qwen3.5-9B --served-model-name Qwen/Qwen3.5-9B
> --tensor-parallel-size 2 --distributed-executor-backend mp
> --dtype bfloat16 --max-model-len 2048 --max-num-seqs 48 --max-logprobs 20
> --return-tokens-as-token-ids` with **1 replica × 2 GPUs** on ml.g5.24xlarge, /dev/shm 16Gi,
> `VLLM_WORKER_MULTIPROC_METHOD=spawn`, service type clusterip, name teacher-vllm — and wait
> until the pod is Running and /v1/models responds."
> Two GPUs total for the teacher: the other two belong to the student and the sampler.

## 2. Stage 1 — on-policy distillation

**First, the baseline (~45 min; notebook Part 1).** The trainer starts from the base student, so
before committing ~4.5 h, measure the teacher and every candidate student on the same problems —
that is what tells you whether there is anything to distil and which student to pick. For one
candidate by hand (the sampler must serve it):

```bash
export STUDENT_MODEL=Qwen/Qwen3.5-2B EVAL_BASE_MODEL=Qwen/Qwen3.5-2B
export EVAL_PHASE=before,teacher EVAL_DIR=/fsx/opd/baseline-gsm8k-qwen3.5-2b
envsubst < manifests/opd-config.yaml-template       | kubectl --context "$CTX" apply -f -
envsubst < manifests/student-sampler.yaml-template  | kubectl --context "$CTX" apply -f -
envsubst < manifests/student-eval-job.yaml-template | kubectl --context "$CTX" apply -f -
```
A wide `teacher − before` gap with **high format compliance** means there is headroom to work
with. Low format compliance means the reverse KL will spend the run fighting format instead of
reasoning — raise `TRAIN_FEWSHOT`/`EVAL_FEWSHOT` together (they must match) or fix the chat
template before starting. Re-source `manifests/env_vars` (it resets `STUDENT_MODEL` and the `EVAL_*`
overrides), re-apply the ConfigMap and delete the Job afterwards.

**Then the run itself:**
```bash
envsubst < manifests/student-distill-job.yaml-template | kubectl --context "$CTX" apply -f -
kubectl --context "$CTX" -n default logs -f job/opd-student
```
**or instantstart** — launch as a **HyperPod Job (generic .sh)** pointing at
`/s3/opd/src/entry-distill.sh` (no arguments — everything comes from the ConfigMap), instance
`ml.g5.24xlarge` — the same node as the teacher — with `nprocPerNode=1` so it claims one GPU,
mounting `/s3` and `/fsx`. The verl RayJob path (KubeRay) is the alternative for fault-tolerant
multi-node.

The Job runs `preflight.py` first and refuses to start on a tokenizer mismatch, null teacher
log-probs, or a train/eval format disagreement. It then gates on a **step-0 sanity check** and
exits `3` if the measured reverse KL is not positive, rather than spending hours optimising the
student against noise. An exit code of `3` in the pod status means exactly that — read the
printed teacher/student log-probs above it.

**If the run dies part-way through**, do not restart from zero: set `INIT_ADAPTER` to the newest
`${RUN_DIR}/student-step<N>` checkpoint (written every `SAVE_EVERY` steps), lower `STEPS` to the
remaining count, re-apply the ConfigMap, then delete and recreate the Job.

**The vLLM rollout sampler (GPU 3).** The loop spends most of its wall clock generating rollouts,
not computing gradients, and HF `generate` is roughly 5–10× slower than vLLM for this. GPU 3 is
the device left after the teacher (0-1) and the trainer (2), and at 25–40 s/step the sampler is
the difference between a run that finishes overnight and one that does not:
```bash
envsubst < manifests/student-sampler.yaml-template | kubectl --context "$CTX" apply -f -
# then in env_vars, and re-apply the ConfigMap:
export STUDENT_SAMPLER_URL=http://student-sampler.default.svc.cluster.local:8000/v1
export SAMPLER_GPUS=1
```
The trainer hot-swaps its LoRA adapter into that server every `SAMPLER_SYNC_EVERY` steps, so the
sampler lags the live policy by at most that many steps; `BUFFER_MAX_AGE` defaults to match, the
PPO ratio corrects the remainder, and `clip_frac` makes excess lag visible.

With the sampler enabled the node is 4 of 4 claimed (2 + 1 + 1). The eval still runs alongside
training, because it is an HTTP client of the sampler and the teacher. To run without the sampler,
set `STUDENT_SAMPLER_URL=` and `SAMPLER_GPUS=0` (and `EVAL_GPUS=1`, `EVAL_STUDENT_URL=` for a local
HF eval) so preflight's arithmetic matches reality. In-process sampling is correct, just slower.

## 3. Monitor (works with the read-only notebook role)
```bash
kubectl --context "$CTX" -n default get pods -o wide
kubectl --context "$CTX" -n default logs -f job/opd-student
kubectl --context "$CTX" -n default port-forward svc/teacher-vllm 8000:8000   # reach the teacher
```
Per-step metrics land in `${RUN_DIR}/metrics_rank0.jsonl` on `/fsx`. What to look at:

| metric | healthy | what it means when it isn't |
|---|---|---|
| `teacher_kl` | falling | flat ⇒ the student is not moving; check `grad_norm` and the sanity gate |
| `clip_frac` | < ~0.2 | high ⇒ rollouts too stale; lower `BUFFER_MAX_AGE` / `SAMPLER_SYNC_EVERY` |
| `policy_drift` | small | how far the live policy has moved from the rollouts it is training on |
| `truncated_frac` | low | high ⇒ raise `MAX_NEW_TOKENS`; the student is being cut off mid-reasoning |
| `epochs_done` | 0 for most of the run | ≥1 ⇒ the prompt pool has started recycling |
| `sec_gen` / `sec_score` / `sec_train` | — | where the wall clock actually goes; if `sec_gen` dominates, confirm the vLLM sampler is up |

## 4. Stage 2 — evaluate: before vs. after vs. **teacher**
`src/eval_student.py` scores final-answer accuracy on **all 500 rows of MATH-500**
(`TASK=math500`, `EVAL_LIMIT=500`), 4-shot (`EVAL_FEWSHOT`, matching `TRAIN_FEWSHOT`), greedy,
semantic `\boxed{}` match, for three phases:
`before` (the base student), `after` (`$EVAL_FT_MODEL_DIR`) and `teacher` (through
`$TEACHER_URL`). All knobs live in `manifests/env_vars`.

```bash
set -a && source manifests/env_vars && set +a
aws s3 sync src s3://$S3_BUCKET/opd/src
envsubst < manifests/opd-config.yaml-template        | kubectl --context "$CTX" apply -f -
envsubst < manifests/student-eval-job.yaml-template  | kubectl --context "$CTX" apply -f -
kubectl --context "$CTX" -n default logs -f job/opd-student-eval
```

Two changes from the earlier eval that matter:

- **The teacher is a phase**, so the teacher Service must still be running (set
  `EVAL_PHASE=before,after` if it has been torn down). Without it, `after − before` is
  uninterpretable — the student cannot pass its teacher, so the reported metric is
  `headroom_recovered = (after − before) / (teacher − before)`.
- **The whole test split.** GSM8K's 1,319 rows give a standard error of ~±1.4 pp; at n=100 it
  would be ~±5 pp, wider than most distillation gains.

The `after` phase blocks until `$EVAL_FT_MODEL_DIR` holds an adapter (or merged weights), so the
Job can be submitted while the trainer is still running. Set `EVAL_FT_MODEL_DIR` **before**
`envsubst`: the Job's wait loop is rendered from it. The `teacher` phase needs `teacher-vllm` still
up, so do **not** tear the teacher down after stage 1.

On the served route (`EVAL_STUDENT_URL` set, the default) `$EVAL_FT_MODEL_DIR` is a LoRA adapter
dir — any `student-step<N>` — which the eval registers with the sampler (renamed for vLLM by
`sampler.export_adapter_for_vllm`). On the local HF route it must hold **merged** weights:
`${RUN_DIR}/student-final`, which `train_distill.py`'s final save writes.

Results, all under `${EVAL_DIR}`: `summary.json` (per-phase accuracy/stderr, format + length
metrics, `delta_accuracy`, `teacher_gap`, `headroom_recovered`, and `transitions` = fixed /
broken / both_correct / both_wrong), `samples_<phase>.jsonl` (one row per eval sample — the
dashboard's input), and full Inspect logs in `logs/<phase>`
(`inspect view --log-dir ${EVAL_DIR}/logs/after`).

The same payload is printed to stdout between `===EVAL_DASHBOARD_JSON_BEGIN===` /
`===EVAL_DASHBOARD_JSON_END===`, so every chart can be rebuilt from the Job log alone, with no `/fsx`
access after the pod Completes (the notebook reads `/fsx` through the teacher pod instead):
```bash
kubectl --context "$CTX" -n default logs job/opd-student-eval --tail=-1   # contains the JSON payload
# or read the files directly through any pod that mounts /fsx:
kubectl --context "$CTX" -n default exec deploy/teacher-vllm -c vllm -- cat /fsx/opd/eval/summary.json
```
Jobs are immutable — `kubectl delete job opd-student-eval` before re-submitting with new `EVAL_*`.
The instantstart alternative: a **HyperPod Job (generic .sh)** pointing at
`/s3/opd/src/entry-eval.sh`, `nprocPerNode=1`, mounting `/s3` and `/fsx`.

## 5. Cleanup
```bash
kubectl --context "$CTX" -n default delete job opd-student --ignore-not-found
kubectl --context "$CTX" -n default delete job opd-student-eval --ignore-not-found
kubectl --context "$CTX" -n default delete deploy,svc student-sampler --ignore-not-found
kubectl --context "$CTX" -n default delete deploy,svc teacher-vllm --ignore-not-found
kubectl --context "$CTX" -n default delete configmap opd-config --ignore-not-found
```
The teacher Deployment holds `TEACHER_TP × TEACHER_REPLICAS` (2) GPUs for as long as it exists —
delete it once the eval has finished, not before, since the eval's `teacher` phase needs it.
Deleting the `student-sampler` Deployment is the way to free a GPU without touching the teacher.
