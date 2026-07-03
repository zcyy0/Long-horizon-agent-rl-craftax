"""Test/demo for the eat skill (Tier-2 survival).

Checks: (1) eating a ripe plant adjacent restores food (+4) and unlocks eat_plant;
(2) eating a cow (an injected adjacent passive mob) restores food and unlocks
eat_cow — note passive kills don't bump monsters_killed, so eat reads success from
FOOD GAINED, not fight's kill counter; (3) auto with no food source in view is
interrupted. Food is lowered first so the gain is observable (it caps at max).

Run:
    /workspace/envs/craftax/bin/python scripts/skill_eat_demo.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "harness"))

import jax.numpy as jnp  # noqa: E402

from craftax.craftax.constants import BlockType  # noqa: E402
from craftax_env import CraftaxTextEnv  # noqa: E402
from executor import Executor  # noqa: E402


def set_food(env, value):
    s = env.state
    env.state = s.replace(player_food=jnp.array(value, dtype=s.player_food.dtype))


def spawn_cow(env, pos, health=1.0):
    s = env.state
    lvl = int(s.player_level)
    pm = s.passive_mobs
    env.state = s.replace(passive_mobs=pm.replace(
        mask=pm.mask.at[lvl, 0].set(True),
        position=pm.position.at[lvl, 0].set(jnp.array(pos, dtype=pm.position.dtype)),
        health=pm.health.at[lvl, 0].set(jnp.array(health, dtype=pm.health.dtype)),
        type_id=pm.type_id.at[lvl, 0].set(jnp.array(0, dtype=pm.type_id.dtype)),  # cow
    ))


def main():
    checks = []

    # (1) eat a ripe plant to the right
    env = CraftaxTextEnv(seed=0)
    s = env.state
    lvl = int(s.player_level)
    px, py = int(s.player_position[0]), int(s.player_position[1])
    env.state = s.replace(
        map=s.map.at[lvl, px, py + 1].set(jnp.array(BlockType.RIPE_PLANT.value, dtype=s.map.dtype)),
        player_direction=jnp.array(2, dtype=s.player_direction.dtype),
    )
    set_food(env, 3)
    ex = Executor(env)
    f0 = int(env.state.player_food)
    res = ex.eat("plant")
    f1 = int(env.state.player_food)
    print(f"eat plant: status={res.status} reason='{res.reason}' food {f0}->{f1} ach={res.achievements}")
    for e in res.events:
        print("   -", e)
    checks.append(("plant restores food", res.ok and f1 > f0))
    checks.append(("eat_plant unlocked", "eat_plant" in res.achievements))

    # (2) eat a cow (injected adjacent passive mob)
    env2 = CraftaxTextEnv(seed=0)
    px, py = int(env2.state.player_position[0]), int(env2.state.player_position[1])
    spawn_cow(env2, (px, py + 1), health=1.0)
    set_food(env2, 3)
    ex2 = Executor(env2)
    f0 = int(env2.state.player_food)
    res2 = ex2.eat("cow", health_floor=0)
    f1 = int(env2.state.player_food)
    print(f"\neat cow: status={res2.status} reason='{res2.reason}' food {f0}->{f1} ach={res2.achievements}")
    for e in res2.events:
        print("   -", e)
    checks.append(("cow restores food", res2.ok and f1 > f0))
    checks.append(("eat_cow unlocked", "eat_cow" in res2.achievements))

    # (3) auto with nothing edible in view -> interrupted
    env3 = CraftaxTextEnv(seed=0)
    set_food(env3, 3)
    ex3 = Executor(env3)
    res3 = ex3.eat("auto")
    print(f"\neat auto (nothing edible): status={res3.status} reason='{res3.reason}'")
    checks.append(("auto no-food interrupted", res3.status == "interrupted"))

    print("\n=== results ===")
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print("EAT_OK" if all(ok for _, ok in checks) else "EAT_FAIL")


if __name__ == "__main__":
    main()
