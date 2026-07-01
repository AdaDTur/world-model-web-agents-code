#!/usr/bin/env bash
set -uo pipefail
set -o allexport; source /home/nlp/users/atur/safearena/.env; set +o allexport
export SAFEARENA_DATA_DIR=/home/nlp/users/atur/safearena/data
export BROWSERGYM_STEP_TIMEOUT=120
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
PY=/home/nlp/users/atur/safearena/venv/bin/python
EXP=/home/nlp/users/atur/safearena/scripts/launch_experiment.py
DATA=/home/nlp/users/atur/safearena/data
cd /home/nlp/users/atur
log() { echo "[$(date '+%H:%M:%S')] $*"; }

run() {
    local label=$1 backbone=$2 task=$3 suffix=$4 eval_file=$5
    log "=== $label ==="
    for i in $(seq 1 60); do
        n=$(python3 -c "
import json,pathlib
base=pathlib.Path.home()/'agentlab_results'
ids=set(open('${eval_file}').read().split())
seen=set()
for s in base.iterdir():
    if 'qwen' not in s.name or '${suffix}' not in s.name: continue
    for d in s.iterdir():
        si=d/'summary_info.json'
        if si.exists():
            r=json.loads(si.read_text())
            raw=d.name.split('safearena.')[-1].split('_0')[0]
            if raw in ids: seen.add(raw)
print(len(ids)-len(seen))
")
        log "  Pass $i: $n missing"
        [[ "$n" -eq 0 ]] && break
        SAFEARENA_TASK="$task" \
        timeout 60m "$PY" "$EXP" \
            --backbones "$backbone" --n_jobs 1 --parallel sequential \
            --eval_tasks "$eval_file" \
            --suffix "$suffix" || true
    done
    log "=== $label DONE ==="
}

HARM="$DATA/eval_harm_tasks_v2_25.txt"
SAFE="$DATA/eval_safe_tasks_v2_25.txt"

# Baseline already done (v1 results: Harm SR=28%, Safe SR=40%)
# Only run fine-tuned UST conditions on the v2 held-out tasks
run "UST-SFT harm v2"    qwen2.5-vl-7b-ust-sft  harm  ust-sft-harm-v2   "$HARM"
run "UST-SFT safe v2"    qwen2.5-vl-7b-ust-sft  safe  ust-sft-safe-v2   "$SAFE"
run "UST-DPO harm v2"    qwen2.5-vl-7b-ust-dpo  harm  ust-dpo-harm-v2   "$HARM"
run "UST-DPO safe v2"    qwen2.5-vl-7b-ust-dpo  safe  ust-dpo-safe-v2   "$SAFE"

log "=== ALL DONE ==="
