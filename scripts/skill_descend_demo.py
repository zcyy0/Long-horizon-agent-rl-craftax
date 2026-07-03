"""Test/demo for the descend skill.

Checks: (1) from floor 0 (starts cleared) the agent navigates onto the down-ladder
and descends to floor 1; (2) the clear precondition gates descent on an uncleared
floor (floor 1, 0/8 kills) with a clear failure. The agent is placed near each
ladder (we have no explore skill yet) so the ladder is in view.

Run:
    /workspace/envs/craftax/bin/python scripts/skill_descend_demo.py
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

    # (1) descend from floor 0 (cleared). Stand on the down-ladder (a path tile,
    # so it is walkable); the ladder pocket is surrounded by stone.
    env = CraftaxTextEnv(seed=0)
    ladder0 = np.asarray(env.state.down_ladders)[0]
    place(env, 0, (int(ladder0[0]), int(ladder0[1])))
    ex = Executor(env)
    res = ex.descend()
    print(f"descend (floor 0, cleared): status={res.status} reason='{res.reason}' steps={res.steps} ach={res.achievements}")
    for e in res.events:
        print("   -", e)
    print(f"  player_level now: {int(env.state.player_level)}")
    checks.append(("descended 0->1", res.ok and int(env.state.player_level) == 1))

    # (2) gate: floor 1 is not cleared (0/8). down-ladder at down_ladders[1].
    env2 = CraftaxTextEnv(seed=0)
    ladder1 = np.asarray(env2.state.down_ladders)[1]
    place(env2, 1, (int(ladder1[0]), int(ladder1[1])))
    ex2 = Executor(env2)
    res2 = ex2.descend()
    print(f"\ndescend (floor 1, uncleared): status={res2.status} reason='{res2.reason}'")
    checks.append(("uncleared floor gated", res2.status == "failure" and "not cleared" in res2.reason))

    print("\n=== results ===")
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print("DESCEND_OK" if all(ok for _, ok in checks) else "DESCEND_FAIL")


if __name__ == "__main__":
    main()
