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
cd /home/nlp/users/atur
log() { echo "[$(date '+%H:%M:%S')] $*"; }

TMPFILE=$(mktemp /tmp/eval_tasks_remaining.XXXXXX.txt)
trap 'rm -f "$TMPFILE"' EXIT

reset_env() {
    log "=== RESETTING ENVIRONMENT ==="
    USER_NUM=14 PROJECT_NUM=0 INSTANCE_NUM=0 bash "$RESET"
    log "=== RESET COMPLETE ==="
}

run() {
    local label=$1 backbone=$2 task=$3 suffix=$4 eval_file=$5
    log "=== $label ==="
    for i in $(seq 1 60); do
        # Compute remaining tasks and write to temp file — each pass only runs what's left
        python3 -c "
import json,pathlib,re
base=pathlib.Path.home()/'agentlab_results'
ids=list(open('${eval_file}').read().split())
seen=set()
for s in base.iterdir():
    if 'qwen' not in s.name or '${suffix}' not in s.name: continue
    for d in s.iterdir():
        si=d/'summary_info.json'
        if si.exists():
            m=re.search(r'safearena\.(\w+\.\d+)_\d+', d.name)
            if m and m.group(1) in ids: seen.add(m.group(1))
remaining=[t for t in ids if t not in seen]
print('\n'.join(remaining))
" > "$TMPFILE"
        local n
        n=$(wc -w < "$TMPFILE")
        log "  Pass $i: $n missing"
        [[ "$n" -eq 0 ]] && break
        SAFEARENA_TASK="$task" \
        timeout 60m "$PY" "$EXP" \
            --backbones "$backbone" --n_jobs 1 --parallel sequential \
            --eval_tasks "$TMPFILE" \
            --suffix "$suffix" || true
    done
    log "=== $label DONE ==="
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
