# Experiments

Every on-policy distillation run after the recipe first worked: what changed and what it showed.
Each run is `manifests/env_vars` plus the overrides in `configs/<run>.env`. On this branch the shipped
`env_vars` defaults are **run22**: run17's recipe (2B student, LoRA r=128) on the AWS Deep Learning
Containers. run13–run20 ran on the `vllm/vllm-openai` image and are the history that led to it.

All evals use the GSM8K test split (1,319 problems), 4-shot, `#### N` answer match, greedy unless
noted. Runs are compared with a paired sign test against the base student on the same problems, and
by the share of the base-to-teacher gap they close, `(after − before) / (teacher − before)`.
Differences within about one standard error of a single checkpoint are treated as ties.

## Baselines

Measured first, before any training (notebook Part 1): they pick the student and serve as the
`before` and `teacher` phases of every later eval.

| model | role | finding |
|---|---|---|
| Qwen3.5-9B | teacher | the ceiling |
| **Qwen3.5-2B** | **student, chosen** | a wide gap to the teacher, and enough capacity to close a large share of it |
| Qwen3.5-0.8B | candidate | the widest gap, but it plateaus far short of the teacher whatever the schedule |
| Qwen3.5-4B | candidate | already close to the teacher: too little left to distil |
| 0.8B / 9B, zero-shot `\boxed{}` (`TASK=gsm8k_native`) | prompt variant | lifts the base student a little; the teacher is unchanged |
| 0.8B / 9B, thinking mode | prompt variant | worse: the student thinks until it hits the token cap |

## Runs

| run | config | change vs. previous | outcome |
|---|---|---|---|
| run13 | [`configs/run13.env`](configs/run13.env) | first working recipe: LR 1e-5, 100 steps | clear gain over the 0.8B base, still rising at step 100 |
| run14 | [`configs/run14.env`](configs/run14.env) | 200 steps | same plateau as run13 from step 50 on |
| run15 | [`configs/run15.env`](configs/run15.env) | LR 2e-5 | same plateau, reached by step 25 |
| run16 | [`configs/run16.env`](configs/run16.env) | **2B student**, LR 2e-5, 50 steps | a much higher plateau than the 0.8B, reached by step 25 |
| run17 | [`configs/run17.env`](configs/run17.env) | 2B with **LoRA rank 128** (alpha 256) — the recipe run22 ships | slightly above run16, within about one standard error; flat from step 25 to 50 |
| run18 | [`configs/run18.env`](configs/run18.env) | run17 sampled at **T=1.0**, **8 samples × 32 prompts** | lags at step 25, ties run17 at step 50 |
| run19 | [`configs/run19.env`](configs/run19.env) | run17 + **KL to the teacher's top-20 distribution** | a tie with run17; the loss is no longer in the code |
| run20 | [`configs/run20.env`](configs/run20.env) | run17 with Qwen's **native zero-shot `\boxed{}` prompt** | higher base, same plateau as run17 |
| run22 | [`configs/run22.env`](configs/run22.env) | run17 on the **AWS DLC images** (`vllm:server-cuda-v2.6`, `pytorch:2.14.0-cu133-amzn2023`) — **shipped** | the same baselines and the same result as run17, greedy and with the 8-vote majority |

**Decoding with a majority vote** (`EVAL_VOTES=8`: 8 samples at T=0.7, most common answer scored):

| phases voting | finding |
|---|---|
| none (greedy) | the like-for-like distillation result |
| **`after` only — shipped headline** | a clear further gain for the distilled student, closing a much larger share of the gap |
| all | every model gains; distillation's share of the (larger) gap shrinks |

Accuracy keeps rising with the number of votes and flattens after about six. Two independent
8-vote runs agree on almost every problem. Per-checkpoint tables are printed by
`scripts/results_table.py` from the eval directories on `/fsx`.

## What the runs established

- **The limit is capacity, not training.** The 0.8B plateaus far below the teacher: more steps scored
  the same, and doubling the LR reached the plateau about four times sooner without raising it.
  `teacher_kl` falls briefly and then flattens, before one pass over the 7.5k training prompts.
- **Shipped defaults** follow from this: `LR=2e-5`, `STEPS=50` — the same result in half the time.
- **A larger student lifts the ceiling.** The 2B closes a much larger share of its gap than the 0.8B,
  plateaus by step 25 at LR 2e-5, and trains cleanly: answers keep their marker, few hit the cap,
  and length does not drift.
- **The adapter is not the main limit.** Rank 128 adds little over rank 32.
- **The 4B is not worth distilling here**: it starts within a few points of the teacher.
- **Exactly on-policy sampling does not raise the ceiling** (run18), and training stays stable.
- **A denser teacher signal does not either** (run19, top-20 KL).
- **The prompt moves the base, not the plateau** (run20).
- **The AWS DLC images reproduce the recipe** (run22): vLLM 0.30.0+amzn2023 and PyTorch 2.14.0 match
  run17 within noise.
- **Every 2B variant ends at the same greedy plateau**, within about one standard error: the ceiling
  is the student's capacity with this teacher. What still moves the reported number is decoding —
  the 8-vote majority — and a larger teacher (27B W4A16, documented in `env_vars`) is the remaining
  lever for training.

## Reproducing

```bash
cd code
set -a && source manifests/env_vars && source experiments/configs/run18.env && set +a
envsubst < manifests/opd-config.yaml-template        | kubectl --context "$CTX" apply -f -
envsubst < manifests/student-distill-job.yaml-template | kubectl --context "$CTX" apply -f -

# evaluate checkpoints as they appear (each eval needs no GPU and runs alongside training)
CRED_REFRESH_CMD='<your credential refresh command>' \
  experiments/scripts/eval_steps.sh "$RUN_DIR" run18 25 50
```

Changing `STUDENT_MODEL` or `LORA_R` also means re-applying `manifests/student-sampler.yaml-template`
(`--max-lora-rank` follows `LORA_R`), and a new student needs its baseline first (notebook Part 1,
with `BASE_DIR` pointing the scripts at it).

| script | purpose |
|---|---|
| [`scripts/eval_steps.sh`](scripts/eval_steps.sh) | waits for each `student-step<N>`, launches its eval Job, prints the comparison |
| [`scripts/summarize.py`](scripts/summarize.py) | accuracy, paired z, gap closed, marker/capped/length for any set of eval dirs |
| [`scripts/results_table.py`](scripts/results_table.py) | prints every run's per-checkpoint table (CSV) from the eval dirs on `/fsx` |
