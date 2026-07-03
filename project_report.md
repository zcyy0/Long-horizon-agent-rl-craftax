# Progress Log

A dated record of milestones and findings — **oldest first, newest at the bottom.**
The stable project overview lives in `README.md`.

---

## 2026-07-02 — Evaluation scaffolding and first rollouts

**Built**
- **Survival skills** — `drink_water`, `eat`, `sleep`, `place`, `ascend`. The scripted
  skill library is now 11 skills, each with a self-checking test.
- **Credit-assignment evaluation harness** — rollout logging, a tech-tree
  **prerequisite graph** built from the game's recipes, a labeler for which earlier
  decisions were causally necessary for each achievement, and a rank-based
  credit-quality metric (AUC).
- **GPU serving + rollout pipeline** — the planner (Qwen3-4B) runs on the local GPU;
  the agent plays, and every game is logged for analysis.

**First rollout batch — 6 games, _untrained_ Qwen3-4B (seeds 0–5)**

| seed | LLM decisions | game steps | max floor | reward | achievements |
|---|---|---|---|---|---|
| 0 | 34 | 226 | 0 | 4.10 | 5 |
| 1 | 29 | 243 | 0 | 5.10 | 6 |
| 2 | 36 | 232 | 0 | 6.10 | 7 |
| 3 | 64 | 243 | 0 | 8.30 | 9 (reached the iron pickaxe) |
| 4 | 43 | 501 | 0 | 9.10 | 10 (used survival skills) |
| 5 | 25 | 210 | 0 | 6.10 | 7 |

Aggregate:
- Achievements per game: **mean 7.3, max 10** (of 67 total; ~25 are reachable on the surface).
- **~38 LLM decisions → ~276 game actions** per game (~7× compression from the hierarchy).
- Furthest floor: **0 (the surface) in every game** — the model never descends.
- Reward is sparse: **18%** of decisions carry reward; gaps between rewarded decisions
  run **mean 4.3, median 2, max 28** decisions.
- Format was perfect: **231/231** subgoals parsed, zero malformed outputs.
- The credit-quality metric covered **13/13** of the achievement types unlocked.
- Throughput: ~2 decisions/s, **~4,800 games/day** collected sequentially.

**Findings**
- **Untrained Qwen3-4B is already a competent surface crafter** — it reliably works up
  the tech tree to a stone/iron pickaxe. That gives the credit-assignment methods real
  structure to work over, with plenty of headroom.
- **It never descends.** It exhausts itself crafting on the surface and **starves ~230
  steps in**; only the one game that used `eat`/`drink`/`sleep` survived to 501 steps.
  So the ~40 dungeon achievements are unreached simply because the model doesn't yet
  survive-and-descend — that's the headroom training is meant to unlock.
- **The measured reward delays** (mean 4.3, max 28 decisions) directly set the
  credit-assignment hyperparameters, rather than being guessed.
- **Perfect output format** means training can focus on *competence*, not syntax.

**Next**
- Training loop: policy optimization over subgoal decisions, with the credit-assignment
  methods compared on both learning performance and the credit-quality metric.
- Faster rollout collection (games in parallel).
