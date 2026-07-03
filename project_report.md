# Progress & Results

A living summary of what has been built and what the experiments show — kept up to date
in place, so it always reads as one coherent report rather than a running changelog. The
project overview is in [`README.md`](README.md).

## What's built

- **Environment harness** — full Craftax as a text game, with a renderer verified to match
  the native observation exactly and real fog-of-war.
- **Scripted skill library** — 11 skills spanning navigation, the crafting tech tree, and
  survival (`eat` / `drink_water` / `sleep`), each with a self-checking test.
- **Hierarchical agent loop** — the LLM planner picks one subgoal per turn, the scripted
  controller carries it out, and every decision is recorded in a ledger.
- **Credit-assignment evaluation** — rollout logging, a tech-tree prerequisite graph built
  from the game's recipes, a labeler for which earlier decisions were causally necessary
  for each achievement, and a rank-based credit-quality metric (AUC).
- **GPU serving + rollout pipeline** — the planner (Qwen3-4B) runs on the local GPU; the
  agent plays and logs full games against it.

## Results so far

### Baseline: the untrained planner

Before any training, the raw Qwen3-4B planner already plays coherently. Measured over a
batch of games (untrained, sampling temperature 0.7):

- **Achievements per game: ~7 on average, up to 10** (of 67 total; only ~25 are reachable
  on the surface). It reliably works up the tech tree to a stone or iron pickaxe.
- **It never descends — floor 0 (the surface) in every game.** It exhausts itself crafting
  on the surface and **starves to death ~230 game-steps in**; the single game that used the
  survival skills survived to 501 steps. So the ~40 dungeon achievements are unreached
  simply because the model does not yet *survive-and-descend*, not because they are gated.
- **The hierarchy compresses the game as intended:** ~38 LLM decisions expand to ~276 game
  actions per game (~7× fewer decisions than raw actions).
- **Reward is sparse:** only ~18% of decisions carry any reward, and the gap between
  rewarded decisions runs a mean of ~4 and up to ~28 decisions. This measured delay is what
  sets the credit-assignment hyperparameters, rather than being guessed.
- **Output format is perfect:** every subgoal the model emitted was valid, with no
  malformed outputs — so training can focus on *competence*, not syntax.
- **The credit-quality metric covers 100%** of the achievement types the model unlocks.

### What this tells us

- The untrained model is a **competent surface crafter with large headroom** — the ideal
  starting point: enough tech-tree structure for the credit-assignment methods to work
  over, and one clear thing to improve.
- **Getting past floor 0** — better survival, and a reason to go down — is exactly what
  training with good credit assignment is meant to unlock.

## Next

- The training loop: policy optimization over subgoal decisions, comparing the
  credit-assignment methods on both learning performance and the credit-quality metric.
- Faster rollout collection (playing games in parallel).
