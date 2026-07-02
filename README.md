# Long-Horizon Credit Assignment for LLM Agents on Craftax

A research project on **credit assignment** for a hierarchical LLM agent that plays
[Craftax](https://github.com/MichaelTMatthews/Craftax), an open-ended survival and
crafting game with a deep tech tree spread across nine dungeon floors.

The core idea: instead of having the LLM press buttons, we let it choose *subgoals*, and a
scripted controller turns each subgoal into button presses. A game that takes ~100,000
button presses then becomes a sequence of only a few hundred decisions. That is short
enough to study a question that is hopeless at the button level — **when a reward finally
arrives, which earlier decisions actually earned it?**

---

## 1. The problem

Reinforcement learning is hard over long horizons because rewards are late and vague about
their cause. You do something useful now, but you only find out much later, and by then
it is unclear what helped. This is the **credit-assignment problem**, and it is the main
thing standing between an agent and long-term planning.

Craftax is a good place to study it because progress is concrete and checkable. The game
tracks 67 **achievements** — collect wood, make a stone pickaxe, enter the dungeon, defeat
the necromancer, and so on. Each one is a sparse, one-time reward, and each has clear
prerequisites. When you unlock "make a stone pickaxe," the reward lands on a single step,
but you earned it over many earlier steps: finding trees, mining wood, placing a table,
making a wood pickaxe first. The distance between *when the reward shows up* and *which
decisions produced it* is the whole problem.

At the raw button level, none of this is workable for a language model. Episodes run to
about 100,000 steps — far more than an LLM can act over, and far more than any
reward-redistribution method can untangle. Standard baselines on full Craftax stall well
short of finishing the tech tree.

So we change the level we work at.

---

## 2. The approach

### An LLM planner on top of scripted skills

The LLM never touches the joystick. Each turn it reads the situation and picks one
subgoal, like "mine 3 wood" or "craft a stone pickaxe." A fixed, scripted controller then
carries that subgoal out with the actual game controls and reports back what happened.

```
 observation ─▶  LLM planner  ─▶  {"think": ..., "subgoal": {"name", "args"}}
                                      │
                                      ▼
                            scripted controller  ─▶  1–100+ game actions
                                      │
                                      ▼
                    result (status, reward, achievements)  ─▶  ledger
```

Every turn becomes one entry in a running **ledger**. That ledger — the list of decisions
and what each one led to — is the data we do credit assignment on. Because the achievement
rewards show up inside these entries, the rewards and the decisions live on the same short
timeline.

This is the key move. A 100,000-step episode becomes a few hundred decisions. Methods that
have no hope of decomposing 100,000 steps become practical over a few hundred.

### The skills

The planner chooses from ten skills. Each one handles its own navigation, checks its
preconditions, and acts:

| Skill | What it does |
|---|---|
| `explore` | uncover new map by walking to the edge of what's been seen |
| `mine` | gather a resource (wood up through diamond); needs the right pickaxe |
| `craft` | build an item, placing a table or furnace if the recipe needs one |
| `place` | put down stone, a table, a furnace, a **torch** (to light dark floors), or a plant |
| `fight` | approach and defeat nearby enemies |
| `descend` / `ascend` | move between floors by ladder |
| `drink_water` / `eat` / `sleep` | stay alive: restore drink, food, and energy |

The skills are kept deliberately narrow — `craft`, for example, will not gather materials
for you. That matters: if the controller were allowed to be clever, it would quietly
absorb the very credit we are trying to measure.

### The environment

A text-based harness built from scratch on top of Craftax (the full game: a 9×11 view, 67
achievements, 43 actions, 9 floors). The renderer matches the game's native observation
exactly, keeps real fog-of-war, and only lets the agent plan over tiles it has actually
seen.

---

## 3. What we're actually asking

The hierarchy is the setup, not the experiment. We are not claiming that hierarchy beats a
flat agent — a flat 100,000-step LLM agent isn't feasible in the first place, which is the
whole reason we build the hierarchy. Published flat-Craftax results are just the outside
reference point. The experiment is only about the credit signal over the ledger:

> **Which way of assigning credit to subgoal decisions produces the best learning — and
> does that credit actually land on the decisions that mattered?**

We frame it as three predictions we can be wrong about:

- **Timing matters.** Assigning credit close in time to where it was earned beats spreading
  it evenly across the whole episode.
- **Learning the credit matters more.** A model that *learns* which decisions to reward
  beats hand-tuned temporal methods, and its rewards concentrate on the decisions that were
  genuinely necessary.
- **The credit metric predicts learning.** How well a method assigns credit *early* in
  training predicts how efficiently it learns overall — which would make our evaluation a
  useful benchmark on its own.

If a prediction fails, that's still a result. For example, if learned decomposition
*doesn't* beat a well-tuned temporal method, that tells us something worth knowing: at a
few-hundred-decision horizon, simple temporal credit is already good enough.

---

## 4. The methods we compare

The agent is an open LLM, fine-tuned with LoRA and trained with PPO plus a value critic
over the sequence of subgoal decisions. (The short horizon is what makes a value function
learnable here — that's the payoff of working at the subgoal level.)

Every method runs through the same training pipeline on an equal budget. They differ only
in *how each decision gets its credit*. Each one is chosen to isolate a single factor:

| Method | How it assigns credit | What it isolates |
|---|---|---|
| Sequence advantage (GRPO) | one value for the whole episode | the floor: no timing at all |
| Monte-Carlo reward-to-go | reward that follows each decision | timing alone |
| GAE(λ) | timing, smoothed by a value estimate | what the value critic adds |
| Potential-based shaping | hand-built from the tech-tree graph | expert knowledge vs. learning |
| Randomized Return Decomposition | a *learned* per-decision reward | learned credit vs. simple propagation |
| Uniform redistribution (IRCR) | spread the total reward evenly | a sanity check on the setup |
| Optimal-transport redistribution | learned + structural | *(stretch goal)* |

The one we expect to matter most is **Randomized Return Decomposition**. It learns a small
reward for each decision such that, added up, they reproduce the episode's total reward.
That kind of decomposition is usually too unstable to train — but over a few hundred ledger
entries instead of 100,000 raw steps, it becomes both stable and cheap.

The **potential-based shaping** baseline is the "expert knowledge" ceiling: we hand-build
it from the game's known recipes. If a learned method can't beat it, that's an honest and
useful thing to report.

---

## 5. Evaluation — and a benchmark

We measure two things.

**Does it learn?** How many achievements it reaches, and how deep into the tech tree, per
unit of compute. Compute is measured in generated tokens, since for an LLM agent the
generation is what actually costs — the game steps are nearly free. We report across
multiple seeds with confidence bands and test on held-out worlds.

**Is the credit correct?** This is the part that makes the project more than a bake-off.
Craftax's tech tree hands us ground-truth credit labels for free. We know a stone pickaxe
*requires* wood, stone, a table, and a wood pickaxe first — so we can build a prerequisite
graph straight from the game's recipes. For any achievement the agent unlocks, we know
which earlier decisions were genuinely necessary.

That lets us ask, directly: **do the decisions that mattered get more credit than the ones
that didn't?** We score this with a simple rank statistic (AUC), so it compares fairly
across methods even when their credit values live on different scales.

This turns "is this method assigning credit well?" into a number we can read off each
trajectory, instead of guessing from noisy reward curves. It's a richer, more realistic
version of the classic single-key "Key-to-Door" credit-assignment tests — a full tech tree
with many interacting prerequisites. We intend the graph-labeled evaluation to be reusable
as a benchmark in its own right.

---

## 6. Where things stand

**Done and tested**
- Full-Craftax text harness; renderer matches the native observation exactly.
- Scripted controller: 11 skills (navigation, tech tree, survival), each with its own test.
- The hierarchical loop: skill menu, dispatch, prompts, and the per-turn ledger.
- LLM-policy plumbing, plus a client that talks to a served model.
- Evaluation tools: rollout logging, the prerequisite graph and necessity labeler, and the
  credit-quality metric.

**In progress**
- Serve the planner model and collect a batch of games to measure how it behaves — reward
  sparsity, how far apart cause and reward sit, and throughput.

**Planned**
- The training loop, with a short supervised warm-up so the model reliably emits valid
  moves.
- The main study: run the methods from Section 4, score them on both axes above, and
  ablate.

**First real milestone:** the harness, the credit-quality metric, and a clean comparison
of the four core methods, with proper seeds and held-out evaluation.

---

## 7. Related work

- **Environment:** Craftax — Matthews et al., 2024 ([paper](https://arxiv.org/abs/2402.16801)).
- **Reward redistribution:** RUDDER (Arjona-Medina et al., 2019); Randomized Return
  Decomposition (Ren et al.).
- **Advantage estimation:** GAE (Schulman et al., 2016); RLOO (Ahmadian et al., 2024); GRPO.
- **Reward shaping:** potential-based shaping (Ng, Harada & Russell, 1999).
- **Credit-assignment testbeds:** Key-to-Door (Hung et al., 2019; Mesnard et al., 2021).
- **Hierarchy:** the options framework and semi-MDPs (Sutton, Precup & Singh, 1999).
- **LLM agents with skill libraries:** Voyager, ELLM, SmartPlay.

---

## 8. Repository

The harness is written from scratch against Craftax-core, with no heavy dependencies in the
environment layer. See `README.md` for setup and the per-skill demos — each demo script
checks itself. Contributions and discussion are welcome.
