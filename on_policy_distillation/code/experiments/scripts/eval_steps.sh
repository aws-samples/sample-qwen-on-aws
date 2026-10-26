#!/bin/bash
# eval_steps.sh <run_dir> <eval_prefix> <steps...>  -- evaluate each student-step<N> as it appears.
# Run from anywhere (it cds to code/ and sources manifests/env_vars + env_vars.local). Every cluster call has a
# hard timeout, and the script waits rather than acting on a failed read. Optional: CRED_REFRESH_CMD
# (e.g. an `ada credentials update ...` line) is run every 10 minutes to keep credentials fresh.
# BASE_DIR picks the baseline the eval dirs are seeded from and compared against. VOTES (default 1:
# greedy, the like-for-like sweep) overrides EVAL_VOTES for these evals; 8 = majority-vote the after phase.
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR/../.." || exit 1
RD=$1; PFX=$2; shift 2
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
T() { perl -e 'alarm shift; exec @ARGV' "$@"; }
K() { T 120 kubectl --context $C --request-timeout=90s -n default "$@"; }
say() { echo "[$(date -u +%m-%dT%H:%MZ)] $*"; }
last=0; refresh() { local now; now=$(date +%s); [ $((now-last)) -lt 600 ] && return; [ -z "${CRED_REFRESH_CMD:-}" ] && { last=$now; return; }; T 90 bash -c "$CRED_REFRESH_CMD" >/dev/null 2>&1 && last=$now || say "WARN credential refresh failed"; }
SP() { K get pods -o name 2>/dev/null | grep student-sampler | head -1 | sed 's|pod/||'; }
set -a; . manifests/env_vars >/dev/null 2>&1; set +a
C=$CTX
SUMMARIZE=$SCRIPT_DIR/summarize.py
ARGS=""
for s in "$@"; do
  AD=$RD/student-step$s; DIR=/fsx/opd/eval-$PFX/step$s; J=opd-eval-$PFX-s$s
  until sp=$(SP) && [ -n "$sp" ] && K exec "$sp" -c vllm -- test -f "$AD/adapter_config.json" 2>/dev/null; do refresh; sleep 120; done
  sleep 30; say "$PFX step $s checkpoint found"; refresh
  if ! K exec "$(SP)" -c vllm -- test -f "$DIR/samples_after.jsonl" 2>/dev/null; then
    until K exec "$(SP)" -c vllm -- sh -c "mkdir -p $DIR && cp ${BASE_DIR:-/fsx/opd/baseline-gsm8k-2b}/summary.json $DIR/" 2>/dev/null; do refresh; sleep 30; done
    export EVAL_FT_MODEL_DIR=$AD EVAL_PHASE=after EVAL_DIR=$DIR
    K delete job "$J" --ignore-not-found >/dev/null 2>&1
    until envsubst < manifests/student-eval-job.yaml-template | sed -e "s|name: opd-student-eval|name: $J|" -e "s|app: opd-student-eval|app: $J|" -e "s|^          env:|          env:\\
            - { name: EVAL_FT_MODEL_DIR, value: \"$AD\" }\\
            - { name: EVAL_PHASE,        value: \"after\" }\\
            - { name: EVAL_DIR,          value: \"$DIR\" }\\
            - { name: EVAL_VOTES,        value: \"${VOTES:-1}\" }|" | T 120 kubectl --context $C apply -f - >/dev/null 2>&1; do refresh; sleep 30; done
    until K get job "$J" -o jsonpath='{.status.conditions[?(@.status=="True")].type}' 2>/dev/null | grep -qE 'Complete|Failed'; do refresh; sleep 60; done
  fi
  ARGS="$ARGS $PFX-s$s=$DIR/samples_after.jsonl"
  sp=$(SP); K cp $SUMMARIZE "$sp":/tmp/summarize.py -c vllm >/dev/null 2>&1
  say "results so far:"; K exec "$sp" -c vllm -- env BASE_DIR=${BASE_DIR:-/fsx/opd/baseline-gsm8k-2b} python3 /tmp/summarize.py ${COMPARE:-} $ARGS 2>/dev/null | grep -E "^(base|teacher|run1)"
done
say "=== $PFX evals complete ==="
