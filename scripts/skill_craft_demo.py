"""Test/demo for the craft skill (places stations, then make_*).

Checks: (1) wood pickaxe (places a table, crafts) — pickaxe level rises,
place_table + make_wood_pickaxe unlocked; (2) precondition — crafting with no
materials fails with a shortfall; (3) iron pickaxe needs BOTH a table and a
furnace (the place->step->place maneuver). Materials are injected to isolate
crafting from gathering.

Run:
    /workspace/envs/craftax/bin/python scripts/skill_craft_demo.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "harness"))

import jax.numpy as jnp  # noqa: E402

from craftax_env import CraftaxTextEnv  # noqa: E402
from executor import Executor  # noqa: E402


def give(env, **fields):
    inv = env.state.inventory
    repl = {k: jnp.array(v, dtype=getattr(inv, k).dtype) for k, v in fields.items()}
    env.state = env.state.replace(inventory=inv.replace(**repl))


def main():
    checks = []

    # (1) wood pickaxe: needs a table (2 wood) + 1 wood
    env = CraftaxTextEnv(seed=0)
    give(env, wood=5)
    ex = Executor(env)
    res = ex.craft("wood_pickaxe")
    pick = int(env.state.inventory.pickaxe)
    print(f"craft wood_pickaxe: status={res.status} reason='{res.reason}' steps={res.steps} ach={res.achievements}")
    for e in res.events:
        print("   -", e)
    print(f"  pickaxe level -> {pick}")
    checks.append(("wood pickaxe crafted", res.ok and pick >= 1))
    checks.append(("place_table unlocked", "place_table" in res.achievements))

    # (2) precondition: no materials
    env2 = CraftaxTextEnv(seed=0)
    give(env2, wood=0)
    ex2 = Executor(env2)
    res2 = ex2.craft("wood_pickaxe")
    print(f"\ncraft wood_pickaxe (no wood): status={res2.status} reason='{res2.reason}'")
    checks.append(("gated without materials", res2.status == "failure" and "missing" in res2.reason))

    # (3) iron pickaxe: needs table AND furnace
    env3 = CraftaxTextEnv(seed=0)
    give(env3, wood=10, stone=10, iron=3, coal=3)
    ex3 = Executor(env3)
    res3 = ex3.craft("iron_pickaxe")
    pick3 = int(env3.state.inventory.pickaxe)
    print(f"\ncraft iron_pickaxe: status={res3.status} reason='{res3.reason}' steps={res3.steps} ach={res3.achievements}")
    for e in res3.events:
        print("   -", e)
    print(f"  pickaxe level -> {pick3}")
    checks.append(("iron pickaxe (table+furnace)", res3.ok and pick3 >= 3))

    print("\n=== results ===")
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print("CRAFT_OK" if all(ok for _, ok in checks) else "CRAFT_FAIL")


if __name__ == "__main__":
    main()
