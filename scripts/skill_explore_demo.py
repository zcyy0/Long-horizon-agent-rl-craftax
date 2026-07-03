"""Test/demo for the explore skill (the navigation fallback).

explore expands SeenMemory by walking to the frontier of seen space — the place
where the 9x11 window pulls fresh tiles out of the fog — so the planner can then
navigate/mine/descend toward something it couldn't previously see.

Checks:
 (1) expands the map — known-tile count grows and the player actually moves;
 (2) directional bias — when a frontier exists in the requested half-plane,
     _nearest_frontier returns one that advances that way;
 (3) discovery integrity — anything explore reports as "found" is really in the
     known map afterward (no phantom reveals);
 (4) monotonic memory — repeated explores never forget a tile (obs discipline);
 (5) max_steps is respected;
 (6) a fully-known reachable region returns interrupted (nothing left to reveal).

Run:
    /workspace/envs/craftax/bin/python scripts/skill_explore_demo.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "harness"))

import numpy as np  # noqa: E402
from craftax.craftax.constants import BlockType, ItemType  # noqa: E402

from craftax_env import CraftaxTextEnv  # noqa: E402
from executor import UNKNOWN, _ACTION_TO_DELTA, Executor  # noqa: E402

_NB = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def known_count(ex):
    return ex._n_known(ex._level())


def known_mask(ex):
    return ex.mem.known[ex._level()] != UNKNOWN


def all_frontiers(ex):
    """Recompute frontier tiles independently (to cross-check the bias logic)."""
    lvl = ex._level()
    known = ex.mem.known[lvl]
    h, w = known.shape
    out = []
    for (x, y) in ex.reachable():
        for dr, dc in _NB:
            nx, ny = x + dr, y + dc
            if 0 <= nx < h and 0 <= ny < w and int(known[nx, ny]) == UNKNOWN:
                out.append((x, y))
                break
    return out


def main():
    checks = []

    # (1) expands the map
    env = CraftaxTextEnv(seed=1)
    ex = Executor(env)
    start, before = ex._pos(), known_count(ex)
    res = ex.explore(max_steps=80)
    after, moved = known_count(ex), ex._pos() != start
    print(f"explore: status={res.status} reason='{res.reason}' steps={res.steps} "
          f"reward={res.reward:.2f}")
    print(f"  known tiles {before} -> {after} (+{after - before}) | moved={moved}")
    checks.append(("expands map", res.ok and after > before and moved))

    # (2) directional bias: when a directional frontier exists, the pick advances that way
    env = CraftaxTextEnv(seed=3)
    ex = Executor(env)
    ex.explore(max_steps=30, stop_on_discovery=False)  # reveal enough to have choices
    px, py = ex._pos()
    lvl = ex._level()
    fronts = all_frontiers(ex)
    bias_ok = True
    for d, (ddr, ddc) in _ACTION_TO_DELTA.items():
        has_dir = any((x - px) * ddr + (y - py) * ddc > 0 for (x, y) in fronts)
        tgt = ex._nearest_frontier(lvl, d, set())
        prog = None if tgt is None else (tgt[0] - px) * ddr + (tgt[1] - py) * ddc
        print(f"  dir={d:<5} directional_frontier_exists={has_dir} pick={tgt} progress={prog}")
        if has_dir and (tgt is None or prog <= 0):
            bias_ok = False
    checks.append(("directional bias", bias_ok))

    # (3) discovery integrity: every "found" thing is actually in the known map now
    env = CraftaxTextEnv(seed=2)
    ex = Executor(env)
    res3 = ex.explore(max_steps=120)
    lvl = ex._level()
    block_names = {BlockType(b).name.lower() for b in ex._seen_block_ids(lvl)}
    item_names = {ItemType(i).name.lower() for i in ex._seen_item_ids(lvl)}
    found = []
    if "found" in res3.reason:
        found = [t.strip() for t in res3.reason.split("found", 1)[1].split(",") if t.strip()]
    disc_ok = all(name in block_names or name in item_names for name in found)
    print(f"explore (seed2): status={res3.status} reason='{res3.reason}'")
    print(f"  reported found={found} | all present in known map: {disc_ok}")
    checks.append(("discovery integrity", disc_ok))

    # (4) monotonic memory across repeated explores (never forgets)
    env = CraftaxTextEnv(seed=4)
    ex = Executor(env)
    ex.explore(max_steps=40)
    m1 = known_mask(ex).copy()
    ex.explore(max_steps=40)
    m2 = known_mask(ex)
    forgot = int(np.count_nonzero(m1 & ~m2))
    print(f"explore x2 (seed4): known after#1={int(m1.sum())} after#2={int(m2.sum())} "
          f"| tiles forgotten={forgot}")
    checks.append(("monotonic memory", forgot == 0))

    # (5) max_steps respected
    env = CraftaxTextEnv(seed=5)
    ex = Executor(env)
    res5 = ex.explore(max_steps=10)
    print(f"explore (max_steps=10): steps={res5.steps}")
    checks.append(("max_steps respected", res5.steps <= 10))

    # (6) fully-known reachable region -> interrupted (nothing left to reveal)
    env = CraftaxTextEnv(seed=6)
    ex = Executor(env)
    lvl = ex._level()
    ex.mem.known[lvl] = np.asarray(ex.env.state.map[lvl]).astype(np.int32)  # pretend all seen
    res6 = ex.explore(max_steps=20)
    print(f"explore (all-known): status={res6.status} reason='{res6.reason}'")
    checks.append(("fully-known -> interrupted", res6.status == "interrupted"))

    print("\n=== results ===")
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print("EXPLORE_OK" if all(ok for _, ok in checks) else "EXPLORE_FAIL")


if __name__ == "__main__":
    main()
