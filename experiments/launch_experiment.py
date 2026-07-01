import argparse
import logging
import os
import pathlib
from pathlib import Path
from typing import List

from agentlab.agents.generic_agent.agent_configs import GenericAgentArgs
from agentlab.experiments.study import Study
from safearena import create_default_benchmark
from safearena.config import SAFE_TASK_IDS, HARM_TASK_IDS
from safearena.modeling import (
    prepare_gpt,
    prepare_vllm_model,
    prepare_claude,
    prepare_together,
    prepare_gemini,
    prepare_gemini_direct,
    prepare_exploration_gemini_direct,
    prepare_world_model_gemini_direct,
    prepare_rag_world_model_gemini_direct,
    prepare_optional_world_model_gemini_direct,
    prepare_unified_safety_gemini_direct,
    prepare_optional_world_model_vllm,
    prepare_unified_safety_vllm,
)
from safearena.finetuned_ust_agent import FinetunedUSTAgentArgs

def _make_finetuned_ust(adapter_path: str, server_url: str, harmful: bool = False):
    from safearena.modeling import prepare_vllm_model
    base_args = prepare_vllm_model(
        "Qwen/Qwen2.5-VL-7B-Instruct", harmful=harmful,
        max_new_tokens=4096, max_prompt_tokens=28672, max_total_tokens=32768,
    )
    return FinetunedUSTAgentArgs(
        adapter_path=adapter_path,
        server_url=server_url,
        chat_model_args=base_args.chat_model_args,
        flags=base_args.flags,
        max_retry=base_args.max_retry,
    )

logging.getLogger().setLevel(logging.INFO)


backbone_to_args = {
    "claude-3.5-sonnet": lambda harmful: prepare_claude("anthropic/claude-3.5-sonnet-20240620", harmful=harmful),
    "gpt-4o": lambda harmful: prepare_gpt("gpt-4o-2024-11-20",harmful=harmful),
    "gpt-4o-mini": lambda harmful: prepare_gpt("gpt-4o-mini-2024-07-18", harmful=harmful),
    "gemini-2.5-flash": lambda harmful: prepare_gemini(
        "google/gemini-2.5-flash-preview",
        max_new_tokens=4096,
        max_prompt_tokens=28672,
        max_total_tokens=32768,
        harmful=harmful,
    ),
    "gemini-2.5-flash-direct": lambda harmful: prepare_gemini_direct(
        "gemini-2.5-flash",
        max_new_tokens=4096,
        max_prompt_tokens=28672,
        max_total_tokens=32768,
        harmful=harmful,
    ),
    # RAG world model — retrieved examples ground the safety simulation
    "gemini-2.5-flash-rag-worldmodel": lambda harmful: prepare_rag_world_model_gemini_direct(
        "gemini-2.5-flash",
        max_new_tokens=4096,
        max_prompt_tokens=28672,
        max_total_tokens=32768,
    ),
    # Per-step world model: N candidates → VLM predictions → safety verdicts → selection
    "gemini-2.5-flash-worldmodel": lambda harmful: prepare_world_model_gemini_direct(
        "gemini-2.5-flash",
        max_new_tokens=4096,
        max_prompt_tokens=28672,
        max_total_tokens=32768,
        n_candidates=3,
    ),
    # Optional world model: agent may query WM as a tool before acting (1-4 calls/step)
    "gemini-2.5-flash-optional-worldmodel": lambda harmful: prepare_optional_world_model_gemini_direct(
        "gemini-2.5-flash",
        max_new_tokens=4096,
        max_prompt_tokens=28672,
        max_total_tokens=32768,
    ),
    # Unified safety tool: single-call WM simulation + safety verdict (2-3 calls/step)
    "gemini-2.5-flash-unified": lambda harmful: prepare_unified_safety_gemini_direct(
        "gemini-2.5-flash",
        max_new_tokens=4096,
        max_prompt_tokens=28672,
        max_total_tokens=32768,
    ),
    # UST backed by Qwen2.5-VL-7B via local server (VLLM_BASE_URL + VLLM_API_KEY)
    "qwen2.5-vl-7b-unified": lambda harmful: prepare_unified_safety_vllm(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        max_new_tokens=4096,
        max_prompt_tokens=28672,
        max_total_tokens=32768,
    ),
    # 5-minute exploration phase before task; always uses ExplorationHarmfulAgent
    "gemini-2.5-flash-explore": lambda harmful: prepare_exploration_gemini_direct(
        "gemini-2.5-flash",
        max_new_tokens=4096,
        max_prompt_tokens=28672,
        max_total_tokens=32768,
    ),
    # Fine-tuned UST models — served via ust_server.py on GPUs 4-5
    "qwen2.5-vl-7b-ust-sft": lambda harmful: _make_finetuned_ust(
        str(pathlib.Path.home() / "safearena-swm/final"),
        server_url="http://localhost:8010", harmful=harmful),
    "qwen2.5-vl-7b-ust-dpo": lambda harmful: _make_finetuned_ust(
        str(pathlib.Path.home() / "safearena-swm-dpo/final"),
        server_url="http://localhost:8011", harmful=harmful),
    # Qwen2.5-VL-7B via local server (VLLM_BASE_URL + VLLM_API_KEY)
    "qwen2.5-vl-7b": lambda harmful: prepare_vllm_model(
        "Qwen/Qwen2.5-VL-7B-Instruct", harmful=harmful,
        max_new_tokens=4096, max_prompt_tokens=28672, max_total_tokens=32768,
    ),
    "qwen2.5-vl-7b-optional-worldmodel": lambda harmful: prepare_optional_world_model_vllm(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        max_new_tokens=4096, max_prompt_tokens=28672, max_total_tokens=32768,
    ),
    "llama-3.2-90b": lambda harmful: prepare_vllm_model("meta-llama/Llama-3.2-90B-Vision-Instruct", harmful=harmful),
    "llama-3.2-90b-together": lambda harmful: prepare_together("meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo", harmful=harmful),
    "llama-3.3-70b": lambda harmful: prepare_vllm_model("meta-llama/Llama-3.3-70B-Instruct", use_vision=False, harmful=harmful),
    "qwen-2-vl-72b": lambda harmful: prepare_vllm_model("Qwen/Qwen2-VL-72B-Instruct", harmful=harmful),
    "qwen-2.5-vl-72b": lambda harmful: prepare_vllm_model("Qwen/Qwen2.5-VL-72B-Instruct", harmful=harmful),
}

def _preflight_check():
    """Abort early if any required SafeArena service is unreachable."""
    import urllib.request, urllib.error
    vllm_url = os.environ.get("VLLM_BASE_URL", "")
    required = {
        "WA_HOMEPAGE":       os.environ.get("WA_HOMEPAGE", ""),
        "WA_SHOPPING":       os.environ.get("WA_SHOPPING", ""),
        "WA_REDDIT":         os.environ.get("WA_REDDIT", ""),
        "WA_GITLAB":         os.environ.get("WA_GITLAB", ""),
        **({"VLLM_BASE_URL (models)": vllm_url.rstrip("/") + "/models"} if vllm_url else {}),
    }
    failed = []
    for name, url in required.items():
        if not url:
            failed.append(f"{name} not set")
            continue
        try:
            urllib.request.urlopen(url, timeout=5)
        except Exception as e:
            failed.append(f"{name} ({url}): {e}")
    if failed:
        raise RuntimeError(
            "Pre-flight check failed — aborting before wasting task budget:\n"
            + "\n".join(f"  ✗ {f}" for f in failed)
        )
    logging.info(f"Pre-flight check passed: {list(required.keys())} all reachable.")


def run_experiment(backbones, n_jobs, suffix, relaunch, reproduce, benchmark, parallel="sequential", harmful=False):
    _preflight_check()
    agent_args: List[GenericAgentArgs] = []

    for backbone in backbones:
        if backbone not in backbone_to_args:
            raise ValueError(f"Backbone {backbone} not found in available backbones: {list(backbone_to_args.keys())}")
        agent_args.append(backbone_to_args[backbone](harmful))

    if relaunch is not None:
        print("Relaunching study from directory containing:", relaunch)
        root_dir = Path(os.getenv("AGENTLAB_EXP_ROOT", "~/agentlab_results/")).expanduser()
        study = Study.load_most_recent(contains=relaunch, root_dir=root_dir)
        study.find_incomplete(include_errors=True)
    else:
        study = Study(agent_args, benchmark, logging_level_stdout=logging.INFO, suffix=suffix)  # type: ignore

    if reproduce:
        [a.set_reproducibility_mode() for a in agent_args]
    
    study.run(
        n_jobs=n_jobs,
        parallel_backend=parallel,
        strict_reproducibility=False,
        n_relaunch=1,
    )


if __name__ == "__main__":  # necessary for dask backend
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-b",
        "--backbones",
        type=str,
        default=["gpt-4o-mini"],
        nargs="+",
        choices=list(backbone_to_args.keys()),
    )
    parser.add_argument(
        "-n",
        "--n_jobs",
        type=int,
        default=1,
        help="""Number of jobs to run experiments. Defaults to : 1.""",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help="""Suffix for experiment name. Defaults to : None.""",
    )
    parser.add_argument(
        "--reproduce",
        type=bool,
        default=False,
        help="""Bool for reproducibility mode. Defaults to : False""",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--relaunch",
        type=str,
        default=None,
        help="""Keyword to find exp dir if relaunch is set to true. Defaults to empty""",
    )
    parser.add_argument(
        "--parallel",
        type=str,
        default="sequential",
        help="""Parallel backend to use. Defaults to : sequential.""",
        choices=["sequential", "ray", "joblib"],
    )
    parser.add_argument(
        "--eval_tasks",
        type=str,
        default=None,
        help=(
            "Explicit path to a task-ID list file (one ID per line). "
            "Overrides the default eval_{task_type}_tasks.txt lookup so you "
            "can run on any subset (e.g. --eval_tasks data/eval_harm_tasks_v2.txt). "
            "Use this whenever running on a non-default eval set to avoid silently "
            "using the wrong task list."
        ),
    )
    parser.add_argument(
        "--jailbreak",
        type=bool,
        default=False,
        help="""Bool for reproducibility mode. Defaults to : False""",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Randomly sample N tasks (0 = all). Pairs harm.i / safe.i by index.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for --sample (default: 42).",
    )

    args, unknown = parser.parse_known_args()

    task_type = os.getenv("SAFEARENA_TASK", "harm")
    if task_type == "harm":
        task_ids = HARM_TASK_IDS
    elif task_type == "safe":
        task_ids = SAFE_TASK_IDS
    else:
        raise ValueError(f"Task type {task_type} not found in available task types: ['harm', 'safe']")

    # Fixed eval set: explicit --eval_tasks path takes priority, then the
    # default data/eval_{task_type}_tasks.txt, then random sampling.
    safearena_data_dir = os.getenv("SAFEARENA_DATA_DIR", "data")

    if args.eval_tasks:
        eval_tasks_file = Path(args.eval_tasks)
        if not eval_tasks_file.exists():
            raise FileNotFoundError(
                f"--eval_tasks file not found: {eval_tasks_file}\n"
                f"Double-check the path — running on the wrong task list wastes compute."
            )
    else:
        eval_tasks_file = Path(safearena_data_dir) / f"eval_{task_type}_tasks.txt"

    if eval_tasks_file.exists():
        task_ids = [t for t in eval_tasks_file.read_text().split() if t]
        print(f"[eval] Fixed eval set loaded from {eval_tasks_file} ({len(task_ids)} tasks): "
              f"{task_ids[0]} … {task_ids[-1]}")
    else:
        # Legacy: random sampling via --sample / --seed (or env vars)
        sample_n    = args.sample or int(os.getenv("SAFEARENA_SAMPLE_N", "0"))
        sample_seed = args.seed   or int(os.getenv("SAFEARENA_SAMPLE_SEED", "42"))
        if sample_n and sample_n < len(task_ids):
            import random
            rng = random.Random(sample_seed)
            indices = list(range(len(task_ids)))
            sampled = sorted(rng.sample(indices, sample_n))
            task_ids = [task_ids[i] for i in sampled]
            print(f"[sample] {sample_n} tasks sampled (seed={sample_seed}): "
                  f"{task_ids[0]} … {task_ids[-1]}")

    benchmark = create_default_benchmark(task_ids=task_ids, name=f"safearena-{task_type}")

    run_experiment(
        backbones=args.backbones,
        n_jobs=args.n_jobs,
        suffix=args.suffix,
        relaunch=args.relaunch,
        reproduce=args.reproduce,
        benchmark=benchmark,
        parallel=args.parallel,
        harmful=args.jailbreak,
    )
