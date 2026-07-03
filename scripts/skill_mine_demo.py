"""Test/demo for the mine skill (composes navigate_to).

Checks: (1) mining wood collects the requested count, unlocks collect_wood, and
inventory rises; (2) precondition gating — mining stone with no pickaxe fails
with a clear reason; (3) after getting a wood pickaxe (placed directly for the
test), stone becomes minable.

Run:
    /workspace/envs/craftax/bin/python scripts/skill_mine_demo.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "harness"))

from craftax_env import CraftaxTextEnv  # noqa: E402
from executor import Executor  # noqa: E402


def main():
    checks = []

    # (1) mine wood
    env = CraftaxTextEnv(seed=0)
    ex = Executor(env)
    w0 = int(env.state.inventory.wood)
    res = ex.mine("wood", count=2)
    w1 = int(env.state.inventory.wood)
    print(f"mine wood x2: status={res.status} reason='{res.reason}' steps={res.steps} "
          f"reward={res.reward:.2f} ach={res.achievements}")
    for e in res.events:
        print("   -", e)
    print(f"  wood {w0} -> {w1}")
    checks.append(("mine wood collects 2", res.ok and w1 - w0 == 2))
    checks.append(("collect_wood unlocked", "collect_wood" in res.achievements))

    # (2) precondition: stone needs a wood pickaxe (we have none at start)
    env2 = CraftaxTextEnv(seed=0)
    ex2 = Executor(env2)
    res2 = ex2.mine("stone")
    print(f"\nmine stone (no pickaxe): status={res2.status} reason='{res2.reason}'")
    checks.append(("stone gated without pickaxe", res2.status == "failure" and "pickaxe" in res2.reason))

    # (3) give a wood pickaxe directly, confirm gate opens
    import jax.numpy as jnp
    env3 = CraftaxTextEnv(seed=0)
    env3.state = env3.state.replace(
        inventory=env3.state.inventory.replace(pickaxe=jnp.array(1, dtype=env3.state.inventory.pickaxe.dtype))
    )
    ex3 = Executor(env3)
    res3 = ex3.mine("stone", count=1, max_steps=120)
    print(f"\nmine stone (wood pickaxe): status={res3.status} reason='{res3.reason}' "
          f"steps={res3.steps} ach={res3.achievements}")
    # success if a stone deposit was reachable; otherwise interrupted (explore) — both mean the gate opened
    gate_open = res3.status != "failure"
    checks.append(("stone gate opens with pickaxe", gate_open))

    print("\n=== results ===")
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print("MINE_OK" if all(ok for _, ok in checks) else "MINE_FAIL")


if __name__ == "__main__":
    main()
