"""Rollout logging for the CA study (RESEARCH_PLAN Phase 0.1).

Serializes a hierarchical-agent episode to a JSON record: per-turn ledger entries
(the CA substrate) plus vitals/inventory snapshots before and after each subgoal —
the state features the DAG necessity labeler (`credit_eval.py`) and the CA methods
consume — and an episode summary. One JSON object per line (JSONL) = one episode.

Torch-free; pure harness (imports only the env/executor/agent modules + numpy).
"""
import json
from typing import Any, Dict, List, Optional

import numpy as np

from agent import HierarchicalAgent
from craftax_env import CraftaxTextEnv
from executor import Executor

# vitals that drive the survival skills + duration/credit analysis
_VITAL_FIELDS = ["player_health", "player_food", "player_drink", "player_energy"]
# inventory fields (scalars are levels/counts; arrays like potions/armour are summed)
_INV_FIELDS = ["wood", "stone", "coal", "iron", "diamond", "sapphire", "ruby",
               "sapling", "torches", "arrows", "pickaxe", "sword", "bow",
               "armour", "potions", "books"]


def snapshot(state) -> Dict[str, Any]:
    """Vitals + inventory at a point in time (JSON-serializable ints/bools)."""
    vitals = {f.replace("player_", ""): int(getattr(state, f)) for f in _VITAL_FIELDS}
    vitals["is_sleeping"] = bool(state.is_sleeping)
    vitals["floor"] = int(state.player_level)
    inv = state.inventory
    inventory = {}
    for f in _INV_FIELDS:
        v = getattr(inv, f, None)
        if v is None:
            continue
        arr = np.asarray(v)
        inventory[f] = int(arr.sum()) if arr.ndim else int(arr)
    return {"vitals": vitals, "inventory": inventory}


def record_rollout(policy, seed: int = 0, max_turns: int = 64,
                   max_env_steps: Optional[int] = None) -> Dict[str, Any]:
    """Run one hierarchical-agent episode with `policy` and capture everything the
    downstream CA analysis needs. Returns {"summary": {...}, "turns": [{...}]}."""
    env = CraftaxTextEnv(seed=seed)
    ex = Executor(env)
    agent = HierarchicalAgent(env, ex, policy, max_turns=max_turns,
                              max_env_steps=max_env_steps)
    turns: List[Dict[str, Any]] = []
    while not agent.done and len(agent.ledger) < agent.max_turns:
        pre = snapshot(env.state)
        entry = agent.step_turn()
        post = snapshot(env.state)
        turns.append({
            "turn": entry.turn,
            "think": entry.think,
            "subgoal": entry.subgoal,
            "status": entry.status,
            "reason": entry.reason,
            "steps": int(entry.steps),           # option duration k (primitives)
            "reward": float(entry.reward),
            "achievements": list(entry.achievements),
            "floor": int(entry.floor),
            "env_step": int(entry.env_step),
            "vitals_pre": pre["vitals"],
            "inventory_pre": pre["inventory"],
            "vitals_post": post["vitals"],
            "inventory_post": post["inventory"],
        })

    achievements = sorted({a for t in turns for a in t["achievements"]})
    summary = {
        "seed": int(seed),
        "policy": type(policy).__name__,
        "n_turns": len(turns),
        "env_steps": int(env.t),
        "total_reward": float(agent.total_reward),
        "n_achievements": len(achievements),
        "achievements": achievements,
        "max_floor": max((t["floor"] for t in turns), default=0),
        "done": bool(agent.done),
        "term_reason": turns[-1]["reason"] if turns else "",
    }
    return {"summary": summary, "turns": turns}


def write_rollouts(records: List[Dict[str, Any]], path: str) -> None:
    """Append-free write: one JSON record per line (JSONL)."""
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def load_rollouts(path: str) -> List[Dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
