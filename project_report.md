# Progress & Results

A living summary of what this project has built and what the experiments show. It is meant
to be read on its own — no other docs required.

---

## The problem

[**Craftax**](https://github.com/MichaelTMatthews/Craftax) is an open-source RL benchmark: a
procedurally-generated survival/crafting game (think Minecraft-in-2D) with **9 increasingly
hard floors**, ~100k steps per episode, and **67 sparse "achievement" rewards** (collect
wood, craft a pickaxe, open a chest, defeat a monster, descend a floor, …). An episode is a
long chain of actions with a few dozen reward moments — so the research question is *which
actions deserve credit for a later reward*. That's the credit-assignment problem, and it's
what makes long-horizon RL hard.

**The idea that makes it tractable.** Instead of learning over 100k raw joystick actions, use
a **hierarchy**: an LLM picks one **subgoal** at a time (a skill like "mine wood", "open the
chest", "go down a floor"), and a hand-written controller turns that subgoal into the
underlying moves. This collapses ~100k primitive steps into a few hundred high-level
decisions — a much shorter horizon to assign credit over. Keeping the skills single-purpose
means the *interesting* decisions live at the planner level, which is exactly where I study
credit assignment.

**The specific question.** Can a small **learned value model** ("outcome model") rerank the
LLM's proposed subgoals to act better — and does that skill **transfer from floors it has
seen to a floor it hasn't** — more sample-efficiently than training the LLM directly with
reinforcement learning? I compare three agents:

- **FROZEN** — the LLM takes its own top-choice subgoal (no learning).
- **MODEL** — frozen LLM proposes a few candidates; the learned value model reranks and
  picks the best (my proposed method).
- **PPO** — the LLM itself is trained with reinforcement learning (the standard baseline; in
  progress).

The study focuses on one clean transfer: **train on the surface (floor 0), test on the first
dungeon floor (floor 1)** the model has never seen.

---

## What I built

- **Game harness** — the full Craftax game wrapped as a text environment for the LLM, with a
  renderer verified byte-for-byte against the native observation (so the agent sees exactly
  what a standard RL agent would), real fog-of-war, and a hierarchical LLM decision loop.
- **17-skill library** — single-purpose skills (mine, craft, fight, eat, descend, open a
  chest, shoot a bow, cast a spell, …). I audited all 67 achievements and added skills until
  every one is reachable.
- **Exact save / restore / branch** — freeze the complete game state (including the RNG) at
  any decision, restore it independently, and try a *different* action without disturbing the
  original. This is the foundation for the evaluation oracle below.
- **Value model ("outcome model")** — a small (~30k-parameter) neural-net ensemble that, given
  a game state (63 hand-built features) and a candidate subgoal, predicts the **discounted
  future reward** plus auxiliary outcomes (will the agent die? unlock an achievement? descend?).
  It also reports **uncertainty** via ensemble disagreement. Trains on CPU in seconds.
- **Reranker + exact evaluation oracle** — the LLM proposes K candidate subgoals in one call;
  the value model scores them (preferring high value, penalizing high uncertainty). To measure
  whether a choice was *actually* good, I **branch-and-replay**: from a saved state, force each
  candidate and roll the game forward to measure its true finite-horizon reward. This gives
  ground truth to score the reranker against.
- **Reproducible data + training pipeline** — a versioned per-decision record, leak-free
  train/test splits by world seed, and a deterministic dataset build (same logs → identical
  data).
- **GPU serving** — Qwen3-4B served locally (on an NVIDIA Blackwell GPU) for fast LLM calls.

---

## Results

### 1. The starting point — what the raw LLM does

Out of the box, Qwen3-4B plays coherently: **~7 achievements per game**, reliably crafting a
stone/iron pickaxe — **but it never goes down a floor.** It exhausts itself on the surface and
**starves ~230 steps in.** So the gap to real depth is *survive-and-descend*, not a lack of
capability — which is what the value model and skills target.

### 2. The value model learns to predict outcomes

Trained on the LLM's own surface (floor-0) play, evaluated on held-out data:

| What it predicts | Model | Simple baseline | |
|---|---|---|---|
| Future reward (regression error, lower is better) | **0.469** | 0.715 (predict the average) | ✅ |
| Will it unlock a new achievement? (AUROC) | **0.828** | 0.5 (chance) | ✅ |
| What the skill's outcome will be (accuracy) | **0.839** | 0.453 (majority class) | ✅ |
| Will the agent die? (AUROC) | **0.682** | 0.5 | ✅ |
| How long the skill takes (error) | **4.57** | 6.48 | ✅ |
| Does its uncertainty track its error? (corr) | **+0.183** | 0 | ✅ |

The value model beats every baseline, and — usefully — **its uncertainty rises where its
predictions are worse.** (One thing it *can't* yet learn: predicting *descent*, because the
raw LLM almost never descends, so there are essentially no examples to learn from.)

### 3. The reranker beats the LLM's own choice (on seen states)

On 25 decision states where the choice actually matters, using branch-and-replay for ground
truth:

| Metric | LLM's own top pick | **Reranker** | |
|---|---|---|---|
| Regret (how far from the best action; lower is better) | 0.702 | **0.289** | ~59% lower |
| Picks the truly-best action | 20% | **52%** | 2.6× |

The reranker specifically rescues the states where the LLM's instinct is bad. *(Pilot-scale:
25 states.)*

### 4. It transfers to an unseen floor — but hits a ceiling

Testing on 38 held-out **floor-1** states (the value model trained only on floor 0):

| Metric | FROZEN (LLM's pick) | **MODEL** (reranked) |
|---|---|---|
| Regret (lower is better) | 0.888 | **0.242** |
| Reward earned | −0.28 | **+0.26** |
| Proposes the best action ("candidate recall") | — | **0.37** |

**The reranker transfers zero-shot** (regret 0.89 → 0.24 on the new floor). But there's a
ceiling: **the LLM only proposes the best action 37% of the time**, and the reranker can only
pick from what the LLM offers. The dominant missed action is **opening a chest** — worth a
large reward and present in ~half of these states, but the LLM rarely suggests it. *The
bottleneck moved from "can the model rank?" to "does the LLM propose the right options?"*

> **A debugging story worth telling.** The first version of this experiment reported the agent
> earning *zero* reward on floor 1 — a totally dead result. The cause was a subtle bug: reward
> was being read by absolute timestep, but the saved game states (created before a later code
> change) didn't carry the reward history, so every lookup silently returned empty. I traced
> it to the exact slicing line, fixed it to index by position, and the real signal appeared.
> Rigor in the evaluation harness mattered more than any single result.

### 5. Headline result — distilling the value model *into* the planner

Since the ceiling was the LLM's proposals, I trained the **LLM itself** to propose what the
value model prefers. Concretely: score the available actions with the value model, turn those
values into a target preference distribution, and **fine-tune the LLM (with LoRA) to match
it** — a supervised distillation, no reinforcement-learning loop and no extra game rollouts
during training.

*One honest failure along the way:* the first attempt made the LLM propose "open chest"
**everywhere**, even where no chest exists — because the training signal only *raised* the
probability of good actions and never *lowered* it elsewhere (and the value model itself
over-values chest-opening in states it never saw). **Fix:** explicitly include impossible
actions in the training targets with a near-zero value (they'd just fail), which teaches the
*conditional* — propose "open chest" only when a chest is actually reachable. That worked:

| "open chest" proposed | base LLM | **distilled LLM** |
|---|---|---|
| when a chest is reachable | 0.30 | **0.78** |
| when no chest exists | 0.00 | **0.00** (over-proposing fixed) |

**End-to-end result** — the distilled agent vs. the base reranking agent, on 30 held-out
worlds:

| Metric | base (rerank only) | **distilled** |
|---|---|---|
| Proposes the best action (recall) | 0.37 | **0.77** |
| **Reward earned** | 2.40 | **4.63** |
| Reward from the LLM's *own* top pick | 1.45 | **4.58** |
| Proposes "open chest" where it's the best action | 5 / 15 | **15 / 15** |

**Fixing the proposals nearly doubled the agent's reward** (2.4 → 4.6): it now proposes
opening a chest in *every* state where that's best, so it collects the rewards it was blind
to. The most interesting part: the distilled LLM's **own top choice** (4.58) is now nearly as
good as the reranked choice (4.63) — the LLM **internalized the value model**, so reranking
adds almost nothing on top. (Compare the base LLM, whose top pick was *never* the best.)

**Caveats.** Pilot-scale (30 worlds, one training run) — directional, not yet
significance-tested. The distillation trades a little recall on chest states (1.00 → 0.78) for
correctly *not* proposing chests elsewhere. *Reward* is the trustworthy cross-agent number; the
regret metric isn't cleanly comparable between agents with different candidate sets.

### A note on measuring cost fairly

For the eventual comparison against reinforcement learning, "did it learn more efficiently?"
only means something if you count the resources honestly. I added a **per-stage budget
ledger** that records environment steps, LLM tokens, and GPU time — keeping *ordinary* game
steps and *privileged* (save/restore) steps on separate lines so no cost is hidden. So far
this approach has spent ~11,300 environment steps and **zero** privileged steps to build its
data.

---

## What's next

- **PPO reinforcement-learning baseline** — the comparison the whole project is built around:
  is distilling a learned value model into the planner **more sample-efficient** than training
  the LLM directly with model-free RL? Measured on the shared budget ledger above.
- **Scale for significance** — more evaluation worlds and a second training run (current
  results are a 30-world pilot), with bootstrap confidence intervals.
- **Harden the distillation** — one more iteration that re-collects data from the *improved*
  planner and re-trains (a DAgger-style loop), to close the remaining recall gap.
- **Final three-way write-up** — FROZEN vs PPO vs MODEL, on both performance and cost.
