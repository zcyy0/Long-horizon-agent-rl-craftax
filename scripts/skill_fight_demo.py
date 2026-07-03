"""Test/demo for the fight skill.

No mobs spawn at reset, so we inject a low-health zombie adjacent to the player
and verify fight kills it (monsters_killed rises, defeat achievement fires). Also
checks the no-target case returns interrupted.

Run:
    /workspace/envs/craftax/bin/python scripts/skill_fight_demo.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "harness"))

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from craftax_env import CraftaxTextEnv  # noqa: E402
from executor import Executor  # noqa: E402


def spawn_zombie(env, pos, health=2.0):
    s = env.state
    lvl = int(s.player_level)
    mm = s.melee_mobs
    env.state = s.replace(melee_mobs=mm.replace(
        mask=mm.mask.at[lvl, 0].set(True),
        position=mm.position.at[lvl, 0].set(jnp.array(pos, dtype=mm.position.dtype)),
        health=mm.health.at[lvl, 0].set(jnp.array(health, dtype=mm.health.dtype)),
        type_id=mm.type_id.at[lvl, 0].set(jnp.array(0, dtype=mm.type_id.dtype)),  # zombie
    ))


def main():
    checks = []

    # (1) kill an injected adjacent zombie
    env = CraftaxTextEnv(seed=0)
    px, py = int(env.state.player_position[0]), int(env.state.player_position[1])
    spawn_zombie(env, (px, py + 1), health=2.0)
    lvl = int(env.state.player_level)
    mk0 = int(np.asarray(env.state.monsters_killed)[lvl])
    ex = Executor(env)
    res = ex.fight(count=1, health_floor=0, max_steps=60)
    mk1 = int(np.asarray(env.state.monsters_killed)[lvl])
    print(f"fight zombie: status={res.status} reason='{res.reason}' steps={res.steps} ach={res.achievements}")
    for e in res.events:
        print("   -", e)
    print(f"  monsters_killed[{lvl}] {mk0} -> {mk1}")
    checks.append(("killed the zombie", res.ok and mk1 - mk0 == 1))
    checks.append(("defeat achievement fired", any("defeat" in a for a in res.achievements)))

    # (2) no hostiles in view -> interrupted
    env2 = CraftaxTextEnv(seed=0)
    ex2 = Executor(env2)
    res2 = ex2.fight(count=1, max_steps=10)
    print(f"\nfight (no mobs): status={res2.status} reason='{res2.reason}'")
    checks.append(("no-target interrupted", res2.status == "interrupted" and "no hostile" in res2.reason))

    print("\n=== results ===")
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print("FIGHT_OK" if all(ok for _, ok in checks) else "FIGHT_FAIL")


if __name__ == "__main__":
    main()
