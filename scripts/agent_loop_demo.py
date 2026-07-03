"""Demo of the from-scratch CraftaxTextEnv loop on full Craftax.

Shows: native text obs, a random rollout with the achievement ledger, working
go_back (reset-and-replay), and the RNG properties (replay-deterministic per
seed, diverse across seeds).

Run:
    /workspace/envs/craftax/bin/python scripts/agent_loop_demo.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "harness"))

import jax  # noqa: E402

from craftax_env import ACTION_NAMES, CraftaxTextEnv  # noqa: E402


def random_actions(n, seed):
    rng = jax.random.PRNGKey(seed)
    out = []
    for _ in range(n):
        rng, k = jax.random.split(rng)
        out.append(int(jax.random.randint(k, (), 0, len(ACTION_NAMES))))
    return out


def main():
    print(f"action vocab ({len(ACTION_NAMES)}): {ACTION_NAMES}\n")

    env = CraftaxTextEnv(seed=0)
    print("=== INITIAL OBSERVATION (native full 9x11) ===")
    print(env.obs())

    actions = random_actions(300, seed=42)

    print("\n=== RANDOM ROLLOUT (seed=0, 300 steps) ===")
    results = env.multistep(actions)
    total = sum(r.reward for r in results)
    unlocked = [a for r in results for a in r.achievements_unlocked]
    print(f"steps: {len(results)} | total reward: {total:.3f} | final floor: {results[-1].floor}")
    print(f"achievements unlocked: {unlocked}")

    print("\n=== go_back CHECK ===")
    env = CraftaxTextEnv(seed=0)
    env.go_forward(actions[:50])
    obs_at_50 = env.obs()
    hist_at_50 = list(env.action_history)
    env.go_forward(actions[50:100])
    env.go_back(50)
    replay_ok = env.obs() == obs_at_50 and env.action_history == hist_at_50
    print(f"state after go_back matches state at step 50: {replay_ok}")

    print("\n=== RNG CHECK ===")
    def episode_obs(seed):
        e = CraftaxTextEnv(seed=seed)
        return [r.obs_text for r in e.multistep(actions[:50])]

    same = episode_obs(0) == episode_obs(0)
    diff = episode_obs(0) != episode_obs(1)
    print(f"seed0 == seed0 (replay-deterministic): {same}")
    print(f"seed0 != seed1 (diverse rollouts):     {diff}")

    print("\nDEMO_OK" if (replay_ok and same and diff) else "\nDEMO_FAIL")


if __name__ == "__main__":
    main()
