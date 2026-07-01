"""
Transition-focused observation abstraction analysis.

For 20-30 sampled trajectories from the baseline agent:
  1. Extracts AX-tree node sets for consecutive step pairs
  2. Runs Hungarian matching (scipy linear_sum_assignment) on role+name features
  3. Classifies each transition: added / removed / modified / stable nodes
  4. Writes a free-form natural-language description of each transition
  5. Flags transitions where the text diff seems insufficient vs. where it is rich

Output: scripts/transition_analysis.json  (one entry per transition)
        scripts/transition_analysis.txt    (human-readable report)
"""

import gzip
import glob
import json
import os
import pickle
import re
import random
from dataclasses import dataclass, field, asdict
from difflib import unified_diff
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

# ── configuration ────────────────────────────────────────────────────────────

EXP_DIR = (
    "/home/nlp/users/atur/agentlab_results/"
    "2026-05-03_15-48-02_genericagent-gemini-2-5-flash-on-safearena-harm-baseline-jailbreak"
)
N_TRAJECTORIES = 25   # trajectories to sample
MIN_STEPS = 3         # skip very short episodes
OUT_JSON = os.path.join(os.path.dirname(__file__), "transition_analysis.json")
OUT_TXT  = os.path.join(os.path.dirname(__file__), "transition_analysis.txt")
RANDOM_SEED = 42

# ── node extraction ───────────────────────────────────────────────────────────

@dataclass
class AXNode:
    bid: str          # browsergym_id (stable within a page, resets on navigation)
    node_id: str      # Chrome nodeId
    role: str
    name: str
    properties: dict


def extract_nodes(obs: dict) -> list[AXNode]:
    """Extract a flat list of meaningful AX nodes from an observation."""
    raw_nodes = obs["axtree_object"]["nodes"]
    out = []
    for node in raw_nodes:
        if node.get("ignored"):
            continue
        role = node.get("role", {}).get("value", "")
        name = node.get("name", {}).get("value", "") or ""
        bid  = node.get("browsergym_id", "")
        nid  = node.get("nodeId", "")
        props = {p["name"]: p["value"].get("value") for p in node.get("properties", [])}
        if role in ("none", "generic", "") and not name:
            continue   # skip purely structural / unnamed wrappers
        out.append(AXNode(bid=bid, node_id=nid, role=role, name=name[:120], properties=props))
    return out


# ── Hungarian matching ────────────────────────────────────────────────────────

def node_key(n: AXNode) -> str:
    return f"{n.role}::{n.name}"


def hungarian_match(
    nodes_a: list[AXNode],
    nodes_b: list[AXNode],
) -> tuple[list[tuple[AXNode, AXNode]], list[AXNode], list[AXNode]]:
    """
    Match nodes_a → nodes_b using Hungarian algorithm on string-edit distance
    of (role::name) keys.

    Returns:
        matched   – list of (node_a, node_b) pairs
        unmatched_a – nodes in a with no good match (removed)
        unmatched_b – nodes in b with no good match (added)
    """
    if not nodes_a or not nodes_b:
        return [], nodes_a[:], nodes_b[:]

    keys_a = [node_key(n) for n in nodes_a]
    keys_b = [node_key(n) for n in nodes_b]

    # cost = normalised edit distance (0=identical, 1=completely different)
    # We use a fast approximation: Jaccard distance on character 3-grams
    def trigrams(s: str) -> set:
        s = s.lower()
        return {s[i:i+3] for i in range(len(s) - 2)} if len(s) >= 3 else {s}

    tg_a = [trigrams(k) for k in keys_a]
    tg_b = [trigrams(k) for k in keys_b]

    na, nb = len(nodes_a), len(nodes_b)
    cost = np.ones((na, nb), dtype=float)
    for i in range(na):
        for j in range(nb):
            inter = len(tg_a[i] & tg_b[j])
            union = len(tg_a[i] | tg_b[j])
            cost[i, j] = 1.0 - (inter / union if union else 0.0)

    row_ind, col_ind = linear_sum_assignment(cost)

    MATCH_THRESHOLD = 0.55   # cost > this → not a real match

    matched, used_a, used_b = [], set(), set()
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] <= MATCH_THRESHOLD:
            matched.append((nodes_a[r], nodes_b[c]))
            used_a.add(r)
            used_b.add(c)

    unmatched_a = [n for i, n in enumerate(nodes_a) if i not in used_a]
    unmatched_b = [n for i, n in enumerate(nodes_b) if i not in used_b]
    return matched, unmatched_a, unmatched_b


# ── axtree_txt diff ───────────────────────────────────────────────────────────

def axtree_diff_lines(txt_a: str, txt_b: str) -> list[str]:
    """Return unified-diff lines (added/removed) between two axtree_txt strings."""
    lines_a = txt_a.splitlines()
    lines_b = txt_b.splitlines()
    diff = list(unified_diff(lines_a, lines_b, lineterm="", n=0))
    # drop header lines (---, +++)
    return [l for l in diff if l.startswith(("+", "-")) and not l.startswith(("---", "+++"))]


# ── free-form description ─────────────────────────────────────────────────────

def describe_transition(
    url_before: str,
    url_after: str,
    action: str,
    matched: list,
    removed: list[AXNode],
    added: list[AXNode],
    diff_lines: list[str],
) -> str:
    """Write a concise natural-language description of a state transition."""
    parts = []

    # navigation
    if url_before != url_after:
        parts.append(f"Navigated from {url_before} → {url_after}.")
    else:
        parts.append(f"Stayed on {url_before}.")

    # action taken
    parts.append(f"Action: {action}")

    # structural changes
    if removed:
        removed_summary = ", ".join(
            f"{n.role} '{n.name[:40]}'" for n in removed[:5]
        )
        if len(removed) > 5:
            removed_summary += f" … (+{len(removed)-5} more)"
        parts.append(f"Removed {len(removed)} node(s): {removed_summary}.")

    if added:
        added_summary = ", ".join(
            f"{n.role} '{n.name[:40]}'" for n in added[:5]
        )
        if len(added) > 5:
            added_summary += f" … (+{len(added)-5} more)"
        parts.append(f"Added {len(added)} node(s): {added_summary}.")

    if not removed and not added:
        # look for property changes in matched pairs
        changed = [
            (a, b) for a, b in matched
            if a.name != b.name or a.properties != b.properties
        ]
        if changed:
            parts.append(
                f"No structural changes; {len(changed)} node(s) updated in place "
                f"(e.g. '{changed[0][0].name}' → '{changed[0][1].name}')."
            )
        else:
            parts.append("No structural changes detected.")

    # text richness assessment
    n_diff = len(diff_lines)
    if n_diff == 0:
        parts.append("[VISUAL NEEDED?] AX-tree diff is empty — page content unchanged.")
    elif n_diff <= 4:
        parts.append(f"[RICH TEXT] Minimal diff ({n_diff} lines) — easy to characterise.")
    elif n_diff <= 30:
        parts.append(f"[RICH TEXT] Moderate diff ({n_diff} lines) — text abstraction likely sufficient.")
    else:
        parts.append(
            f"[VISUAL NEEDED?] Large diff ({n_diff} lines) — visual confirmation may help "
            f"identify layout / navigation changes."
        )

    return " ".join(parts)


# ── main analysis ─────────────────────────────────────────────────────────────

@dataclass
class TransitionRecord:
    task_id: str
    step: int
    url_before: str
    url_after: str
    action: str
    n_nodes_before: int
    n_nodes_after: int
    n_matched: int
    n_removed: int
    n_added: int
    n_diff_lines: int
    removed_nodes: list[dict]
    added_nodes: list[dict]
    axtree_diff_sample: list[str]   # first 20 diff lines
    description: str


def load_step(path: str):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def analyze_trajectory(task_dir: str) -> list[TransitionRecord]:
    with open(f"{task_dir}/summary_info.json") as f:
        s = json.load(f)
    n_steps = s["n_steps"]
    task_id = os.path.basename(task_dir.rstrip("/")).split("safearena.")[1]

    records = []
    prev_step = None

    for i in range(n_steps + 1):
        pkl = f"{task_dir}/step_{i}.pkl.gz"
        if not os.path.exists(pkl):
            break
        step = load_step(pkl)

        if prev_step is not None and step.obs:
            nodes_a = extract_nodes(prev_step.obs)
            nodes_b = extract_nodes(step.obs)
            matched, removed, added = hungarian_match(nodes_a, nodes_b)

            txt_a = prev_step.obs.get("axtree_txt", "")
            txt_b = step.obs.get("axtree_txt", "")
            diff_lines = axtree_diff_lines(txt_a, txt_b)

            url_before = prev_step.obs.get("url", "")
            url_after  = step.obs.get("url", "")
            action     = prev_step.action or ""

            desc = describe_transition(
                url_before, url_after, action,
                matched, removed, added, diff_lines,
            )

            records.append(TransitionRecord(
                task_id=task_id,
                step=i,
                url_before=url_before,
                url_after=url_after,
                action=str(action),
                n_nodes_before=len(nodes_a),
                n_nodes_after=len(nodes_b),
                n_matched=len(matched),
                n_removed=len(removed),
                n_added=len(added),
                n_diff_lines=len(diff_lines),
                removed_nodes=[{"role": n.role, "name": n.name} for n in removed[:10]],
                added_nodes=[{"role": n.role, "name": n.name} for n in added[:10]],
                axtree_diff_sample=diff_lines[:20],
                description=desc,
            ))

        if step.obs:
            prev_step = step

    return records


def main():
    random.seed(RANDOM_SEED)
    task_dirs = [
        d for d in sorted(glob.glob(f"{EXP_DIR}/*/"))
        if os.path.exists(f"{d}/summary_info.json")
    ]

    # filter to trajectories with enough steps
    eligible = []
    for td in task_dirs:
        with open(f"{td}/summary_info.json") as f:
            s = json.load(f)
        if s["n_steps"] >= MIN_STEPS:
            eligible.append(td)

    sampled = random.sample(eligible, min(N_TRAJECTORIES, len(eligible)))
    print(f"Sampling {len(sampled)} trajectories from {len(eligible)} eligible.")

    all_records: list[TransitionRecord] = []
    for td in sampled:
        recs = analyze_trajectory(td)
        all_records.extend(recs)
        task_id = os.path.basename(td.rstrip("/")).split("safearena.")[1]
        print(f"  {task_id}: {len(recs)} transitions")

    # ── save JSON ──────────────────────────────────────────────────────────────
    with open(OUT_JSON, "w") as f:
        json.dump([asdict(r) for r in all_records], f, indent=2)

    # ── save human-readable report ─────────────────────────────────────────────
    visual_needed = [r for r in all_records if "[VISUAL NEEDED?]" in r.description]
    rich_text     = [r for r in all_records if "[RICH TEXT]"      in r.description]

    with open(OUT_TXT, "w") as f:
        f.write(f"Transition Analysis Report\n{'='*60}\n")
        f.write(f"Trajectories: {len(sampled)} | Total transitions: {len(all_records)}\n")
        f.write(f"Rich-text sufficient: {len(rich_text)} ({100*len(rich_text)/len(all_records):.1f}%)\n")
        f.write(f"Visual may be needed: {len(visual_needed)} ({100*len(visual_needed)/len(all_records):.1f}%)\n\n")

        # aggregate stats
        avg_nodes    = np.mean([r.n_nodes_before for r in all_records])
        avg_added    = np.mean([r.n_added        for r in all_records])
        avg_removed  = np.mean([r.n_removed      for r in all_records])
        avg_diff     = np.mean([r.n_diff_lines   for r in all_records])
        f.write(f"Avg nodes/step: {avg_nodes:.1f} | Avg added: {avg_added:.1f} | "
                f"Avg removed: {avg_removed:.1f} | Avg diff lines: {avg_diff:.1f}\n\n")

        # print all transitions
        for r in all_records:
            f.write(f"\n{'─'*60}\n")
            f.write(f"Task {r.task_id} | Step {r.step-1}→{r.step}\n")
            f.write(f"Action : {r.action}\n")
            f.write(f"URL    : {r.url_before}\n")
            if r.url_before != r.url_after:
                f.write(f"       → {r.url_after}\n")
            f.write(f"Nodes  : {r.n_nodes_before}→{r.n_nodes_after} "
                    f"(+{r.n_added}/-{r.n_removed}, matched={r.n_matched})\n")
            f.write(f"Diff   : {r.n_diff_lines} lines\n")
            f.write(f"Desc   : {r.description}\n")
            if r.added_nodes:
                f.write("  Added  : " + ", ".join(f"{n['role']} '{n['name'][:30]}'" for n in r.added_nodes[:5]) + "\n")
            if r.removed_nodes:
                f.write("  Removed: " + ", ".join(f"{n['role']} '{n['name'][:30]}'" for n in r.removed_nodes[:5]) + "\n")
            if r.axtree_diff_sample:
                f.write("  Diff sample:\n")
                for line in r.axtree_diff_sample[:8]:
                    f.write(f"    {line}\n")

    print(f"\nDone. Written to:\n  {OUT_JSON}\n  {OUT_TXT}")
    print(f"\nSummary:")
    print(f"  Total transitions : {len(all_records)}")
    print(f"  Rich-text OK      : {len(rich_text)} ({100*len(rich_text)/len(all_records):.1f}%)")
    print(f"  Visual needed?    : {len(visual_needed)} ({100*len(visual_needed)/len(all_records):.1f}%)")
    print(f"  Avg nodes/step    : {avg_nodes:.1f}")
    print(f"  Avg added/removed : +{avg_added:.1f} / -{avg_removed:.1f}")
    print(f"  Avg diff lines    : {avg_diff:.1f}")


if __name__ == "__main__":
    main()
