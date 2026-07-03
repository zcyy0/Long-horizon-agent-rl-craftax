"""Test/demo for the ascend skill (Tier-2).

Checks: (1) standing on floor 1's up-ladder, ascend returns to floor 0 (no clear
requirement, unlike descend); (2) ascend from floor 0 fails (already on top). The
player is teleported onto the up-ladder (up_ladders[floor]) so it is in view and
reachable, mirroring the descend demo's setup.

Run:
    /workspace/envs/craftax/bin/python scripts/skill_ascend_demo.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "harness"))

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from craftax_env import CraftaxTextEnv  # noqa: E402
from executor import Executor  # noqa: E402


def place(env, floor, pos):
    s = env.state
    env.state = s.replace(
        player_level=jnp.array(floor, dtype=s.player_level.dtype),
        player_position=jnp.array(pos, dtype=s.player_position.dtype),
    )


def main():
    checks = []

    # (1) ascend 1 -> 0 from the up-ladder
    env = CraftaxTextEnv(seed=0)
    up1 = np.asarray(env.state.up_ladders)[1]
    place(env, 1, (int(up1[0]), int(up1[1])))
    ex = Executor(env)
    res = ex.ascend()
    print(f"ascend (floor 1 -> 0): status={res.status} reason='{res.reason}' "
          f"steps={res.steps} floor_now={int(env.state.player_level)}")
    for e in res.events:
        print("   -", e)
    checks.append(("ascended 1->0", res.ok and int(env.state.player_level) == 0))

    # (2) gate: can't ascend from the top floor
    env2 = CraftaxTextEnv(seed=0)
    ex2 = Executor(env2)
    res2 = ex2.ascend()
    print(f"\nascend (floor 0): status={res2.status} reason='{res2.reason}'")
    checks.append(("top floor gated", res2.status == "failure" and "top floor" in res2.reason))

    print("\n=== results ===")
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print("ASCEND_OK" if all(ok for _, ok in checks) else "ASCEND_FAIL")


if __name__ == "__main__":
    main()
