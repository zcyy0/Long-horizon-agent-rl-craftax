"""Test/demo for the drink_water skill (Tier-2 survival).

Checks: (1) with a water tile adjacent and drink low, drinking refills drink to
max and unlocks collect_drink; (2) drinking when already full is a no-op success;
(3) no water in view returns interrupted ("explore first"). Water is injected via
state.replace since a given seed may not spawn water in the start window.

Run:
    /workspace/envs/craftax/bin/python scripts/skill_drink_demo.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "harness"))

import jax.numpy as jnp  # noqa: E402

from craftax.craftax.constants import BlockType  # noqa: E402
from craftax.craftax.util.game_logic_utils import get_max_drink  # noqa: E402
from craftax_env import CraftaxTextEnv  # noqa: E402
from executor import Executor  # noqa: E402


def set_faced_block_right(env, block):
    """Put `block` on the tile to the player's right and face them right."""
    s = env.state
    lvl = int(s.player_level)
    px, py = int(s.player_position[0]), int(s.player_position[1])
    env.state = s.replace(
        map=s.map.at[lvl, px, py + 1].set(jnp.array(block, dtype=s.map.dtype)),
        player_direction=jnp.array(2, dtype=s.player_direction.dtype),  # 2 = right
    )
    return px, py


def set_drink(env, value):
    s = env.state
    env.state = s.replace(player_drink=jnp.array(value, dtype=s.player_drink.dtype))


def main():
    checks = []

    # (1) drink from adjacent water, thirsty
    env = CraftaxTextEnv(seed=0)
    set_faced_block_right(env, BlockType.WATER.value)
    set_drink(env, 3)
    max_drink = int(get_max_drink(env.state))
    ex = Executor(env)
    d0 = int(env.state.player_drink)
    res = ex.drink_water()
    d1 = int(env.state.player_drink)
    print(f"drink_water (thirsty): status={res.status} reason='{res.reason}' "
          f"drink {d0}->{d1}/{max_drink} ach={res.achievements}")
    for e in res.events:
        print("   -", e)
    checks.append(("drink refills to max", res.ok and d1 == max_drink))
    checks.append(("collect_drink unlocked", "collect_drink" in res.achievements))

    # (2) already full -> no-op success
    env2 = CraftaxTextEnv(seed=0)
    set_faced_block_right(env2, BlockType.WATER.value)
    ex2 = Executor(env2)
    res2 = ex2.drink_water()
    print(f"\ndrink_water (already full): status={res2.status} reason='{res2.reason}' steps={res2.steps}")
    checks.append(("full is no-op success", res2.status == "success" and res2.steps == 0))

    # (3) no water in view -> interrupted
    env3 = CraftaxTextEnv(seed=0)
    set_drink(env3, 3)
    ex3 = Executor(env3)
    res3 = ex3.drink_water()
    print(f"\ndrink_water (no water): status={res3.status} reason='{res3.reason}'")
    checks.append(("no water interrupted", res3.status == "interrupted" and "explore first" in res3.reason))

    print("\n=== results ===")
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print("DRINK_OK" if all(ok for _, ok in checks) else "DRINK_FAIL")


if __name__ == "__main__":
    main()
