# SafeArena Safety World Model — Code

Code for fine-tuning and evaluating safety world models (SWM) on the SafeArena benchmark using Qwen2.5-VL-7B-Instruct.

## Overview

We train a per-step safety classifier (Unified Safety Tool, UST) via LoRA fine-tuning on v1 SafeArena trajectories, then evaluate it online against held-out v2 tasks.

**Training paradigms:**
- **SFT** — supervised fine-tuning with trajectory-level labels (harm task steps → UNSAFE, safe task steps → SAFE)
- **DPO** — preference pairs (chosen = correct verdict, rejected = wrong verdict)
- **GRPO** — group relative policy optimization with verifiable reward (WIP)

**Evaluation:** Online Option B — run the fine-tuned UST agent on 25 held-out harm tasks + 25 held-out safe tasks, measure Harm SR (↓) and Safe SR (↑).

## Structure

```
training/
  train_safety_world_model.py   # SFT training (LoRA on Qwen2.5-VL-7B)
  train_safety_wm_dpo.py        # DPO training
  train_safety_wm_grpo.py       # GRPO training
  grpo_config.yaml              # GRPO hyperparameters

agents/
  finetuned_ust_agent.py        # FinetunedUSTAgent — calls fine-tuned UST via HTTP API
  unified_safety_agent.py       # Base UnifiedSafetyAgent (upstream SafeArena, included for reference)

servers/
  ust_server.py                 # OpenAI-compatible FastAPI server for fine-tuned UST (SFT/DPO)
  qwen_server.py                # OpenAI-compatible server for base Qwen2.5-VL-7B

experiments/
  launch_experiment.py          # Main SafeArena experiment launcher (supports all backbones)
  run_option_b.sh               # Run script for Option B evaluation (SFT/DPO on v2 held-out tasks)

analysis/
  analyze_transitions.py        # Trajectory transition analysis
  validate_results.py           # Result validation utilities
```

## Setup

Requires the [SafeArena](https://github.com/McGill-NLP/safearena) benchmark installed in a separate venv. Training requires the `arena-env` conda environment with PyTorch + CUDA 13.0.

### Start inference servers

```bash
# Base Qwen agent (port 9000, GPU 0)
conda run -n arena-env python servers/qwen_server.py \
  --model Qwen/Qwen2.5-VL-7B-Instruct --port 9000 --gpu 0

# Fine-tuned UST — SFT adapter (port 8010, GPU 4)
conda run -n arena-env python servers/ust_server.py \
  --adapter ~/safearena-swm/final --port 8010 --gpu 4

# Fine-tuned UST — DPO adapter (port 8011, GPU 5)
conda run -n arena-env python servers/ust_server.py \
  --adapter ~/safearena-swm-dpo/final --port 8011 --gpu 5
```

### Run evaluation

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 bash experiments/run_option_b.sh
```

### Train SFT

```bash
conda run -n arena-env python training/train_safety_world_model.py \
  --output_dir ~/safearena-swm
```

### Train DPO

```bash
conda run -n arena-env python training/train_safety_wm_dpo.py \
  --output_dir ~/safearena-swm-dpo \
  --sft_adapter ~/safearena-swm/final
```

## Results (preliminary)

| Model | Safe SR ↑ | Harm SR ↓ |
|---|---|---|
| Qwen2.5-VL-7B Baseline | 40% | 28% |
| + OWM (zero-shot) | 40% | 34% |
| + UST (zero-shot) | 48% | 26% |
| + UST-SFT (fine-tuned) | TBD | ~10% |
| + UST-DPO (fine-tuned) | TBD | TBD |

## Notes

- Model checkpoints not included (too large). SFT adapter: `~/safearena-swm/final`, DPO adapter: `~/safearena-swm-dpo/final`
- Evaluation uses held-out v2 task sets (`data/eval_harm_tasks_v2_25.txt`, `data/eval_safe_tasks_v2_25.txt`) that do not overlap with training data
- Server memory: each UST server uses ~17GB GPU memory; base Qwen server uses ~17GB
- Environment: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1` required on shared servers to avoid mmap exhaustion
# world-model-web-agents-code
# world-model-web-agents-code
