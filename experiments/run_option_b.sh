#!/usr/bin/env bash
set -uo pipefail
set -o allexport; source /home/nlp/users/atur/safearena/.env; set +o allexport
export SAFEARENA_DATA_DIR=/home/nlp/users/atur/safearena/data
export BROWSERGYM_STEP_TIMEOUT=120
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
PY=/home/nlp/users/atur/safearena/venv/bin/python
EXP=/home/nlp/users/atur/safearena/scripts/launch_experiment.py
RESET=/home/nlp/users/atur/safearena/scripts/reset_containers.sh
DATA=/home/nlp/users/atur/safearena/data
TASK_TIMEOUT=15m   # per-task wall-clock limit — kills hung tasks without blocking others
cd /home/nlp/users/atur
log() { echo "[$(date '+%H:%M:%S')] $*"; }

SINGLETASK=$(mktemp /tmp/eval_single_task.XXXXXX.txt)
trap 'rm -f "$SINGLETASK"' EXIT

reset_env() {
    log "=== RESETTING ENVIRONMENT ==="
    USER_NUM=14 PROJECT_NUM=0 INSTANCE_NUM=0 bash "$RESET"
    # Wait until all three critical services actually respond — reset_containers.sh
    # health check can pass while containers are still booting internally
    log "  Verifying services are up..."
    for url in "http://127.0.0.1:14001/" "http://127.0.0.1:14003" "http://127.0.0.1:14004"; do
        for i in $(seq 1 60); do
            if curl -sf --max-time 5 "$url" -o /dev/null 2>/dev/null; then
                log "  $url ready (attempt $i)"; break
            fi
            [[ "$i" -eq 60 ]] && { log "ERROR: $url not ready after 5 min — aborting"; exit 1; }
            sleep 5
        done
    done
    log "=== RESET COMPLETE ==="
}

is_done() {
    local task=$1 suffix=$2
    python3 -c "
import pathlib, re
base = pathlib.Path.home() / 'agentlab_results'
for s in base.iterdir():
    if 'qwen' not in s.name or '${suffix}' not in s.name: continue
    for d in s.iterdir():
        si = d / 'summary_info.json'
        if si.exists():
            m = re.search(r'safearena\.(\w+\.\d+)_\d+', d.name)
            if m and m.group(1) == '${task}':
                exit(0)
exit(1)
"
}

run() {
    local label=$1 backbone=$2 task=$3 suffix=$4 eval_file=$5
    log "=== $label ==="

    local ids
    ids=$(cat "$eval_file")

    local pass=0
    local changed=1
    # Keep looping until every task is done or we've made no progress for a full sweep
    while [[ "$changed" -eq 1 ]]; do
        changed=0
        pass=$((pass + 1))
        log "  Sweep $pass"
        for task_id in $ids; do
            if is_done "$task_id" "$suffix"; then
                continue
            fi
            log "    Running $task_id ..."
            echo "$task_id" > "$SINGLETASK"
            SAFEARENA_TASK="$task" \
            timeout "$TASK_TIMEOUT" "$PY" "$EXP" \
                --backbones "$backbone" --n_jobs 1 --parallel sequential \
                --eval_tasks "$SINGLETASK" \
                --suffix "$suffix" || true
            if is_done "$task_id" "$suffix"; then
                log "    $task_id DONE"
                changed=1
            else
                log "    $task_id did not complete (timeout or error)"
            fi
        done
    done

    # Final count
    local n_done=0
    for task_id in $ids; do is_done "$task_id" "$suffix" && n_done=$((n_done + 1)); done
    local n_total
    n_total=$(echo "$ids" | wc -w)
    log "=== $label DONE: $n_done/$n_total tasks completed ==="
}

HARM="$DATA/eval_harm_tasks_v2_25.txt"
SAFE="$DATA/eval_safe_tasks_v2_25.txt"

reset_env
run "UST-SFT harm v2"  qwen2.5-vl-7b-ust-sft  harm  ust-sft-harm-v2  "$HARM"

reset_env
run "UST-SFT safe v2"  qwen2.5-vl-7b-ust-sft  safe  ust-sft-safe-v2  "$SAFE"

reset_env
run "UST-DPO harm v2"  qwen2.5-vl-7b-ust-dpo  harm  ust-dpo-harm-v2  "$HARM"

reset_env
run "UST-DPO safe v2"  qwen2.5-vl-7b-ust-dpo  safe  ust-dpo-safe-v2  "$SAFE"

log "=== ALL DONE ==="
