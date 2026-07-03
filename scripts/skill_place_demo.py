"""Test/demo for the place skill (Tier-2).

Checks: (1) placing stone on an open faced tile consumes a stone, puts a STONE
block there, and unlocks place_stone; (2) placing a torch consumes a torch, puts a
TORCH item on the faced tile, and unlocks place_torch (this is what lets the agent
light dark floors 2/5/7/8 so explore can proceed); (3) placing with no material
fails with a clear shortfall. Materials are injected via state.replace (which also
trips the has-item achievement, e.g. collect_stone — a harmless test artifact).

Run:
    /workspace/envs/craftax/bin/python scripts/skill_place_demo.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "harness"))

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from craftax.craftax.constants import BlockType, ItemType  # noqa: E402
from craftax_env import CraftaxTextEnv  # noqa: E402
from executor import Executor  # noqa: E402


def open_tile_right(env):
    """Clear the tile to the player's right (PATH, no item) and face them right."""
    s = env.state
    lvl = int(s.player_level)
    px, py = int(s.player_position[0]), int(s.player_position[1])
    env.state = s.replace(
        map=s.map.at[lvl, px, py + 1].set(jnp.array(BlockType.PATH.value, dtype=s.map.dtype)),
        item_map=s.item_map.at[lvl, px, py + 1].set(jnp.array(ItemType.NONE.value, dtype=s.item_map.dtype)),
        player_direction=jnp.array(2, dtype=s.player_direction.dtype),
    )
    return px, py


def give(env, **kw):
    inv = env.state.inventory
    env.state = env.state.replace(inventory=inv.replace(
        **{k: jnp.array(v, dtype=getattr(inv, k).dtype) for k, v in kw.items()}))


def main():
    checks = []

    # (1) place stone
    env = CraftaxTextEnv(seed=0)
    px, py = open_tile_right(env)
    give(env, stone=2)
    ex = Executor(env)
    st0 = int(env.state.inventory.stone)
    res = ex.place("stone")
    st1 = int(env.state.inventory.stone)
    blk = int(np.asarray(env.state.map[int(env.state.player_level)])[px, py + 1])
    print(f"place stone: status={res.status} reason='{res.reason}' stone {st0}->{st1} "
          f"faced_block={BlockType(blk).name} ach={res.achievements}")
    checks.append(("stone consumed", res.ok and st1 == st0 - 1))
    checks.append(("STONE block placed", blk == BlockType.STONE.value))
    checks.append(("place_stone unlocked", "place_stone" in res.achievements))

    # (2) place torch
    env2 = CraftaxTextEnv(seed=0)
    px, py = open_tile_right(env2)
    give(env2, torches=1)
    ex2 = Executor(env2)
    t0 = int(env2.state.inventory.torches)
    res2 = ex2.place("torch")
    t1 = int(env2.state.inventory.torches)
    itm = int(np.asarray(env2.state.item_map[int(env2.state.player_level)])[px, py + 1])
    print(f"\nplace torch: status={res2.status} reason='{res2.reason}' torches {t0}->{t1} "
          f"faced_item={ItemType(itm).name} ach={res2.achievements}")
    checks.append(("torch consumed", res2.ok and t1 == t0 - 1))
    checks.append(("TORCH item placed", itm == ItemType.TORCH.value))
    checks.append(("place_torch unlocked", "place_torch" in res2.achievements))

    # (3) gated: no material
    env3 = CraftaxTextEnv(seed=0)
    ex3 = Executor(env3)
    res3 = ex3.place("torch")
    print(f"\nplace torch (no torches): status={res3.status} reason='{res3.reason}'")
    checks.append(("no-material failure", res3.status == "failure" and "insufficient" in res3.reason))

    print("\n=== results ===")
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print("PLACE_OK" if all(ok for _, ok in checks) else "PLACE_FAIL")


if __name__ == "__main__":
    main()
