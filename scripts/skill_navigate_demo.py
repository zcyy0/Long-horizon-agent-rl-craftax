"""Test/demo for the navigate_to skill.

Checks: (1) navigates to a reachable known walkable tile and arrives exactly;
(2) adjacent mode stops next to a solid target (for mining/fighting);
(3) observation discipline — a target outside seen space returns "interrupted",
never routing through fog.

Run:
    /workspace/envs/craftax/bin/python scripts/skill_navigate_demo.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "harness"))

import numpy as np  # noqa: E402
from craftax.craftax.constants import BlockType, SOLID_BLOCKS  # noqa: E402

from craftax_env import CraftaxTextEnv  # noqa: E402
from executor import Executor  # noqa: E402

SOLID = {int(b) for b in SOLID_BLOCKS}


def main():
    checks = []

    # (1) navigate to a far reachable known walkable tile
    env = CraftaxTextEnv(seed=0)
    ex = Executor(env)
    dist = ex.reachable()
    start = ex._pos()
    target, d = max(dist.items(), key=lambda kv: kv[1])
    print(f"start={start} | reachable known tiles={len(dist)} | farthest={target} (dist {d})")
    res = ex.navigate_to(target)
    arrived = ex._pos() == target
    print(f"navigate_to{target}: status={res.status} reason='{res.reason}' steps={res.steps} "
          f"reward={res.reward:.2f} ach={res.achievements}")
    print(f"  arrived exactly: {arrived} | steps==dist: {res.steps == d}")
    checks.append(("navigate arrives", res.ok and arrived))

    # (2) adjacent mode to a known solid tile (e.g. a tree/stone) if one is reachable-adjacent
    env = CraftaxTextEnv(seed=0)
    ex = Executor(env)
    lvl = ex._level()
    known = ex.mem.known[lvl]
    solids = np.argwhere(np.isin(known, list(SOLID)))
    solid_target = None
    for (sx, sy) in solids:
        # reachable if some 4-neighbor is reachable-known-walkable
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            if (sx + dr, sy + dc) in ex.reachable():
                solid_target = (int(sx), int(sy))
                break
        if solid_target:
            break
    if solid_target is not None:
        res2 = ex.navigate_to(solid_target, adjacent=True)
        px, py = ex._pos()
        is_adj = abs(px - solid_target[0]) + abs(py - solid_target[1]) == 1
        bname = BlockType(int(known[solid_target])).name.lower()
        print(f"navigate_to {solid_target} ({bname}, adjacent): status={res2.status} "
              f"steps={res2.steps} | adjacent now: {is_adj}")
        checks.append(("adjacent mode", res2.ok and is_adj))
    else:
        print("no reachable-adjacent solid tile in initial view; skipping adjacent check")

    # (3) observation discipline: target far outside seen space -> interrupted
    env = CraftaxTextEnv(seed=0)
    ex = Executor(env)
    px, py = ex._pos()
    far = (px + 20, py)  # well beyond the 9x11 seen window
    res3 = ex.navigate_to(far, max_steps=50)
    print(f"navigate_to far-unseen {far}: status={res3.status} reason='{res3.reason}' steps={res3.steps}")
    checks.append(("obs discipline (no fog routing)", res3.status == "interrupted"))

    print("\n=== results ===")
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print("NAV_OK" if all(ok for _, ok in checks) else "NAV_FAIL")


if __name__ == "__main__":
    main()
