# Long-Horizon Credit Assignment for LLM Agents on Craftax

A research project on **credit assignment (CA)** for a hierarchical LLM agent playing
[Craftax](https://github.com/MichaelTMatthews/Craftax) — the open-ended survival-and-
crafting benchmark with a deep achievement tech tree over nine dungeon floors.

**One-line thesis:** by having an LLM choose *subgoals* (not joystick actions) and a
scripted executor carry them out, a ~100,000-step episode collapses into a few hundred
decisions — short enough that reward-redistribution methods which are hopeless at the
primitive level become both tractable *and* measurable. We ask **which credit signal
over these subgoal decisions learns best, and whether the credit it assigns lands on the
decisions that causally mattered.**

---

## 1. Why this problem

**Credit assignment** — figuring out which earlier decisions deserve credit for a later
reward — is the core difficulty in long-horizon reinforcement learning. Craftax makes it
concrete and *verifiable*: progress is measured by 67 **achievements** (collect wood,
make a stone pickaxe, enter the dungeon, defeat the necromancer…), each a sparse,
first-completion milestone with a **known tech-tree prerequisite structure**. A milestone
reward lands on a single step, but it was *earned* by a chain of enabling steps before it.
The gap between "when the reward arrives" and "which decisions produced it" is exactly the
credit-assignment problem.

At the raw action level this is intractable for an LLM: episodes run to ~100k primitive
steps, far beyond what any language-model policy can act over or any redistribution method
can decompose. Published flat baselines (PPO-RNN in the Craftax paper) plateau well short
of full achievement coverage.

**The idea:** put a hierarchy in the middle. An LLM **planner** picks one *subgoal* per
turn from a fixed skill library; a **scripted executor** expands that subgoal into
primitive moves. This turns the problem into a **semi-MDP over ~a few hundred subgoal
decisions** — and that is the regime where credit-assignment methods become practical.
Crucially, the scripted executor is single-purpose and fixed, so the *high-level* credit
(over subgoal choices) is cleanly isolated as the object of study.

---

## 2. The setup

### The hierarchical agent

```
 observation ──▶  LLM planner  ──▶  {"think": ..., "subgoal": {"name", "args"}}
                                        │
                                        ▼
                              scripted executor  ──▶  1–100+ primitive actions
                                        │
                                        ▼
                       SkillResult(status, reward, achievements, ...)  ──▶  ledger
```

Each turn is one **decision** appended to a **ledger**. The ledger — the sequence of
`(subgoal, status, reward, achievements, duration, …)` records — *is* the dataset credit
assignment operates on. Because achievements fire inside these records, the reward signal
lives on the same timeline as the decisions.

### The skill library (the planner's action space)

Ten macro-actions, each a self-contained option that navigates, checks preconditions, and
acts:

| Skill | What it does |
|---|---|
| `explore` | reveal new map by walking to the frontier of seen space |
| `mine` | gather a resource (wood → … → diamond), gated by pickaxe tier |
| `craft` | build an item, placing a table/furnace when a recipe needs one |
| `place` | place stone / table / furnace / **torch** (light dark floors) / plant |
| `fight` | approach and defeat visible hostile mobs |
| `descend` / `ascend` | move between floors via ladders |
| `drink_water` / `eat` / `sleep` | survival: restore drink / food / energy |

The skills are deliberately **single-purpose** (e.g. `craft` never gathers materials) so
the executor can't quietly absorb the credit we're trying to study.

### The environment

A from-scratch text harness over Craftax-Symbolic (full game: 9×11 view, 67 achievements,
43 primitive actions, 9 floors) with a renderer verified byte-exact against the native
observation, real fog-of-war, and observation discipline (the agent only ever plans over
tiles it has actually seen).

---

## 3. Research question & hypotheses

The hierarchy is **fixed infrastructure**, not the variable — we do not claim or run a
flat-vs-hierarchical comparison (a flat 100k-step LLM baseline is infeasible, which is the
whole motivation). Published flat-Craftax numbers are the external reference point. The
experiment is entirely about the credit signal over the subgoal ledger:

> **Which per-decision credit signal best converts sparse milestone rewards into
> learning — and does the credit it assigns land on the decisions that causally
> mattered?**

Three falsifiable, ranked hypotheses:

- **H1 — temporal locality pays.** Credit signals local in time (Monte-Carlo
  reward-to-go, GAE(λ)) beat trajectory-uniform credit on both sample efficiency and
  causal-credit accuracy.
- **H2 — learned decomposition pays more.** A learned return-decomposition (RRD) beats the
  best value-based method, concentrating credit on causally-necessary decisions rather
  than filler.
- **H3 — the metric is a leading indicator.** Credit-assignment quality measured *early*
  in training predicts final sample efficiency — which would make the evaluation itself a
  reusable benchmark.

Negative results are informative and reported: e.g. if H2 fails while H1 holds, that says
"at a few-hundred-step horizon a tuned critic already saturates what decomposition
offers" — a real, publishable bound on when these methods matter.

---

## 4. Credit-assignment methods

The training policy is a LoRA-tuned open LLM optimized with **PPO + a value critic over
the subgoal semi-MDP** (the short horizon makes a value function genuinely learnable — the
payoff of the hierarchy). Every method below shares one rollout+update machinery under an
identical, **token-matched** budget; they differ only in the per-decision reward/credit
signal. Each pairwise contrast is designed to isolate a single factor:

| Method | Credit signal | Isolates |
|---|---|---|
| GRPO sequence-advantage | trajectory-uniform, critic-free | floor: no temporal credit |
| Monte-Carlo reward-to-go | temporal, critic-free | temporal locality alone |
| GAE(λ) sweep | temporal, value-bootstrapped | the value critic's contribution |
| Potential-based shaping | hand-crafted dense (from the tech-tree graph) | prior knowledge vs learned |
| Randomized Return Decomposition (RRD) | **learned** per-decision reward | decomposition vs propagation |
| Uniform redistribution (IRCR) | uniform dense | diagnostic: does "delayed reward" framing apply |
| Optimal-transport redistribution | learned + structured | *(stretch)* transport vs regression |

**Randomized Return Decomposition** is the method we expect to matter most: it learns a
per-decision reward whose sum over a trajectory reconstructs the episodic return. This is
normally intractable — but over a few-hundred-entry ledger rather than 100k primitive
steps, it is both stable and cheap. The **potential-based shaping** baseline, built from
the tech-tree prerequisite graph, is the "prior-knowledge ceiling": if hand-crafted
shaping matches learned redistribution, that itself is an honest and useful finding.

---

## 5. Evaluation — and a benchmark contribution

Two axes:

**1. Does it learn?** Achievements per episode and tech-tree depth reached, as a function
of **generated tokens** (the honest compute axis for an LLM policy — env steps are nearly
free), across ≥3 seeds with confidence bands and held-out world seeds.

**2. Is the credit *correct*?** This is the part that makes the project more than a method
bake-off. Craftax's tech tree gives us **ground-truth credit labels**: for every
achievement, we know its prerequisites (a stone pickaxe *needs* wood, stone, a crafting
table, and a wood pickaxe first). We build this **prerequisite DAG** directly from the
game's recipe tables, and for each unlocked achievement we label which earlier decisions
were causally necessary. We then score a method's assigned credit with a threshold-free
rank statistic (**AUC**): *do the causally-necessary decisions rank above the filler ones
by the credit the method assigns?* Because it is rank-based, it compares fairly across
methods whose "credit" lives on different scales.

This gives a direct, per-trajectory measurement of credit-assignment quality — not just
squinting at final-return curves. It generalizes the classic single-key **Key-to-Door**
credit-assignment testbeds to a rich, naturalistic tech tree with many interacting
prerequisites. The DAG-labeled evaluation is intended to stand alone as a reusable
benchmark.

---

## 6. Status & roadmap

**Built and tested**
- ✅ Full-Craftax text harness; renderer verified exact vs. native observation.
- ✅ Scripted executor: 11 skills (navigation, tech-tree, survival), each unit-tested.
- ✅ Hierarchical agent loop: skill menu, dispatch, prompts, per-turn ledger.
- ✅ Local-LLM policy plumbing (small-model smoke test) and a served-LLM client policy.
- ✅ Evaluation scaffolding: rollout logging, the prerequisite DAG + necessity labeler,
  and the AUC credit-quality metric.

**In progress**
- ⏳ Serve the planner LLM; collect and analyze a rollout batch (delay distribution,
  reward sparsity, throughput) to fix the CA hyperparameters from data.

**Planned**
- Training loop (PPO + critic over subgoals; a supervised format/behavior warm-start).
- The credit-assignment study: baselines → value-based → learned decomposition, scored on
  both axes above, with ablations.

**Minimum showable milestone:** the harness + the DAG credit-quality metric + a clean
comparison of {GRPO, MC reward-to-go, GAE(λ), RRD} with seeds and held-out evaluation.

---

## 7. Related work

- **Environment:** Craftax — Matthews et al., 2024 ([paper](https://arxiv.org/abs/2402.16801)).
- **Reward redistribution / return decomposition:** RUDDER (Arjona-Medina et al., 2019);
  Randomized Return Decomposition (Ren et al.).
- **Advantage estimation:** GAE (Schulman et al., 2016); RLOO / REINFORCE-style LLM RL
  (Ahmadian et al., 2024); GRPO.
- **Reward shaping:** potential-based shaping and its policy-invariance guarantee
  (Ng, Harada & Russell, 1999).
- **Ground-truth credit testbeds:** Key-to-Door (Hung et al., 2019; Mesnard et al., 2021).
- **Hierarchy / options:** the options framework and semi-MDPs (Sutton, Precup & Singh,
  1999).
- **LLM agents with skill libraries:** Voyager, ELLM, SmartPlay.

---

## 8. Repository

The harness is written from scratch against Craftax-core (no heavy dependencies for the
environment layer). See `README.md` for setup and the per-skill demos; each demo script is
self-verifying. Contributions and discussion welcome.
