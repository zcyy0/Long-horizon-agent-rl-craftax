"""Test/demo for the sleep skill (Tier-2 survival).

Checks: (1) with energy low, sleep restores energy to full and unlocks wake_up
(the engine forces auto-noop sleeping until energy maxes, then wakes); (2) sleeping
with full energy is a no-op success. We prime fatigue very negative so energy
regenerates within a couple of steps (energy +1 fires when fatigue < -10).

Run:
    /workspace/envs/craftax/bin/python scripts/skill_sleep_demo.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "harness"))

import jax.numpy as jnp  # noqa: E402

from craftax.craftax.util.game_logic_utils import get_max_energy  # noqa: E402
from craftax_env import CraftaxTextEnv  # noqa: E402
from executor import Executor  # noqa: E402


def set_fields(env, **kw):
    s = env.state
    env.state = s.replace(**{k: jnp.array(v, dtype=getattr(s, k).dtype) for k, v in kw.items()})


def main():
    checks = []

    # (1) tired -> sleep to full, wake_up
    env = CraftaxTextEnv(seed=0)
    max_e = int(get_max_energy(env.state))
    set_fields(env, player_energy=max_e - 2, player_fatigue=-10.5)
    ex = Executor(env)
    e0 = int(env.state.player_energy)
    res = ex.sleep(max_steps=100)
    e1 = int(env.state.player_energy)
    print(f"sleep (tired): status={res.status} reason='{res.reason}' energy {e0}->{e1}/{max_e} "
          f"steps={res.steps} ach={res.achievements} still_sleeping={bool(env.state.is_sleeping)}")
    for e in res.events:
        print("   -", e)
    checks.append(("energy restored to max", res.ok and e1 == max_e))
    checks.append(("wake_up unlocked", "wake_up" in res.achievements))
    checks.append(("not left asleep", not bool(env.state.is_sleeping)))

    # (2) already full -> no-op success
    env2 = CraftaxTextEnv(seed=0)
    ex2 = Executor(env2)
    res2 = ex2.sleep()
    print(f"\nsleep (full energy): status={res2.status} reason='{res2.reason}' steps={res2.steps}")
    checks.append(("full is no-op success", res2.status == "success" and res2.steps == 0))

    print("\n=== results ===")
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print("SLEEP_OK" if all(ok for _, ok in checks) else "SLEEP_FAIL")


if __name__ == "__main__":
    main()
