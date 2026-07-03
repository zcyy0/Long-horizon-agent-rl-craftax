"""Prove the full-Craftax text renderer matches the native symbolic observation.

Native obs per cell = 37 block one-hot + 5 item one-hot + 40 mob + 1 light = 83,
and the block one-hot is zeroed on dark tiles (light <= 0.05). We reconstruct the
visible block window from the obs vector and diff it against
craftax_text.visible_block_window over many random steps (including descents to
deeper, darker floors, which exercise the fog masking).
"""
import os
import sys

import jax
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "harness"))

from craftax.craftax.constants import BlockType, ItemType, OBS_DIM  # noqa: E402
from craftax.craftax_env import make_craftax_env_from_name  # noqa: E402

from craftax_text import visible_block_window  # noqa: E402

N_BLOCK = len(BlockType)
N_ITEM = len(ItemType)
N_MOB = 40
PER_CELL = N_BLOCK + N_ITEM + N_MOB + 1  # + light channel
MAP_CELLS = OBS_DIM[0] * OBS_DIM[1]


def native_visible_block_window(obs):
    all_map = np.asarray(obs[: MAP_CELLS * PER_CELL]).reshape(OBS_DIM[0], OBS_DIM[1], PER_CELL)
    block = all_map[:, :, :N_BLOCK]
    visible = block.sum(axis=-1) > 0.5  # dark tiles were zeroed
    return np.where(visible, block.argmax(axis=-1), -1)


def main():
    env = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=False)
    params = env.default_params
    rng = jax.random.PRNGKey(0)
    rng, rk = jax.random.split(rng)
    obs, state = env.reset(rk, params)

    print(f"OBS_DIM {OBS_DIM} | per-cell {PER_CELL} | obs len {obs.shape[0]} "
          f"(map section {MAP_CELLS * PER_CELL})")

    mismatches = 0
    dark_tiles = 0
    floors_seen = set()
    for i in range(400):
        ours = visible_block_window(state)
        native = native_visible_block_window(obs)
        if not np.array_equal(ours, native):
            mismatches += 1
            if mismatches <= 2:
                print(f"\nMISMATCH at step {i}:\n ours=\n{ours}\n native=\n{native}")
        dark_tiles += int((native == -1).sum())
        floors_seen.add(int(state.player_level))
        rng, ak, sk = jax.random.split(rng, 3)
        action = env.action_space(params).sample(ak)
        obs, state, reward, done, info = env.step(sk, state, action, params)
        if bool(done):
            rng, rk = jax.random.split(rng)
            obs, state = env.reset(rk, params)

    print(f"\nchecked 400 states | floors visited: {sorted(floors_seen)} | "
          f"total dark(masked) tiles seen: {dark_tiles}")
    print(f"block-window mismatches vs native: {mismatches}")
    print("VERIFY_OK" if mismatches == 0 else "VERIFY_FAIL")


if __name__ == "__main__":
    main()
