"""Verify the renderer's light-map fog masking matches native, in actual darkness.

Floors 2, 5, 7, 8 are dark dungeon levels (light_map has large 0.0 regions). We
teleport into each, let the light update with the player present, then compare our
visible_block_window against a reconstruction from the native obs. The test FAILS
if it never encounters dark tiles, so the fog branch cannot silently go untested.
"""
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "harness"))

from craftax.craftax.constants import BlockType, ItemType, OBS_DIM  # noqa: E402
from craftax.craftax.renderer import render_craftax_symbolic  # noqa: E402
from craftax.craftax_env import make_craftax_env_from_name  # noqa: E402

from craftax_text import visible_block_window  # noqa: E402

N_BLOCK = len(BlockType)
PER_CELL = N_BLOCK + len(ItemType) + 40 + 1
CELLS = OBS_DIM[0] * OBS_DIM[1]
DARK_FLOORS = [2, 5, 7, 8]


def native_vbw(state):
    obs = np.asarray(render_craftax_symbolic(state))
    block = obs[: CELLS * PER_CELL].reshape(OBS_DIM[0], OBS_DIM[1], PER_CELL)[:, :, :N_BLOCK]
    visible = block.sum(axis=-1) > 0.5
    return np.where(visible, block.argmax(axis=-1), -1)


def main():
    env = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=False)
    params = env.default_params
    rng = jax.random.PRNGKey(0)
    rng, rk = jax.random.split(rng)
    _, base = env.reset(rk, params)
    item_map = np.asarray(base.item_map)

    total_dark = 0
    mismatches = 0
    checked = 0
    for floor in DARK_FLOORS:
        ups = np.argwhere(item_map[floor] == ItemType.LADDER_UP.value)
        if not len(ups):
            continue
        pos = jnp.array(ups[0])
        state = base.replace(
            player_level=jnp.array(floor, dtype=base.player_level.dtype),
            player_position=pos,
        )
        # wander a little so lighting updates with the player present
        for _ in range(6):
            ours = visible_block_window(state)
            native = native_vbw(state)
            if not np.array_equal(ours, native):
                mismatches += 1
                if mismatches <= 2:
                    print(f"\nMISMATCH on floor {floor}:\n ours=\n{ours}\n native=\n{native}")
            total_dark += int((native == -1).sum())
            checked += 1
            rng, ak, sk = jax.random.split(rng, 3)
            action = env.action_space(params).sample(ak)
            _, state, _, _, _ = env.step(sk, state, action, params)

    print(f"checked {checked} dark-floor states across floors {DARK_FLOORS}")
    print(f"total dark(masked) tiles exercised: {total_dark}")
    print(f"mismatches vs native: {mismatches}")
    ok = mismatches == 0 and total_dark > 0
    print("FOG_VERIFY_OK" if ok else "FOG_VERIFY_FAIL"
          + ("" if total_dark else " (no dark tiles seen!)"))


if __name__ == "__main__":
    main()
