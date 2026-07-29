# Project report: LLM priors vs. learned world models on Craftax

*A controlled study of four agent systems under matched interaction budgets — what
each one learns, what each one cannot learn, and what it takes to measure the
difference honestly.*

---

## 1. Motivation

Two schools of thought dominate current agent research:

1. **The prior school**: take a pretrained LLM, give it tools and an action interface,
   and its internet-scale knowledge substitutes for environment experience. Finetune
   with RL if you want more.
2. **The learning school**: model the environment from data, plan against the model,
   and let interaction — not pretraining — be the source of competence.

These are usually compared across papers, with incompatible budgets, action spaces,
and evaluation protocols, which makes the comparisons nearly meaningless. This project
puts both schools in one harness on one benchmark under one budget and asks a narrow,
decidable question:

> **Given exactly 3,200 environment decisions of training interaction, does a
> world-model planner trained from zeros beat (a) a frozen pretrained LLM planner and
> (b) the same LLM finetuned with PPO on those 3,200 decisions?**

The testbed is **Craftax** (full version): a Crafter-descendant with a deep tech tree
(wood → stone → iron → diamond), survival meters (food, water, energy, health), day/night
cycles, hostile mobs, and a multi-floor dungeon. It is hard enough that neither school
trivially wins, and it has a property that turned out to matter more than expected:
part of its mechanics (the tech tree) is Minecraft folklore an LLM already knows, and
part (the thirst meter, dungeon mob density) is not.

This comparison is stage one of a larger program (see the README's roadmap): the same
harness — grounded menus, semi-MDP credit assignment, and exact branch-and-replay
snapshotting — is built to carry the study to full nine-floor training and evaluation
and to counterfactual long-horizon credit assignment beyond GAE, once the agents
produce trajectories deep enough to need it. The floor-1 transfer and few-shot results
below (§3.4, §3.6) are the first steps down that ladder.

The four systems:

| arm | policy | training signal | compute |
|---|---|---|---|
| FROZEN | Qwen3-4B ranks a menu of grounded actions | none | inference only |
| PPO | same + LoRA, duration-aware PPO+GAE | 3,200 decisions | ~376M tokens, 3.2 GPU-h |
| TEACHER | ensemble world model + value net, no LLM anywhere | 3,200 decisions, from zeros | ~0 GPU (CPU minutes) |
| DISTILL | TEACHER's decisions distilled into the LLM | inherits TEACHER's 3,200 | designed, not yet run |

---

## 2. Design choices, and why

### 2.1 One grounded action menu for every system

Every system chooses from the **same menu of macro-actions** (up to 20 per state),
built by a single constructor from the game state: `go_to(tree)`, `craft(stone_pickaxe)`,
`fight(zombie)`, `drink_water`, `descend`, `open_chest`, … A scripted executor carries
out the chosen subgoal over multiple game steps.

The invariant is **"offered ⟹ doable"**: the constructor offers an action only when
its preconditions hold — checked by the *same predicate functions the executor uses*
(one rule, one function). Sleep is not offered at full energy; craft is not offered
without materials in reach; placing a table is not offered where no tile accepts it.

Why this matters more than it sounds: an earlier version offered `drink`/`sleep`
unconditionally, and the world-model planner — which, unlike the LLM, has no prior
telling it these are pointless at full meters — spent **61.4% of its entire training
budget** on zero-step no-ops. The headline comparison was garbage until the action
space was honest. Every action's gating now has a regression test (a gate script that
fails if any offered action can be a no-op), because the menu *is* the experiment: a
systematically different action space per system would make every downstream number
uninterpretable.

### 2.2 Semi-MDP credit assignment

Macro-actions take wildly different numbers of game steps (a `fight` may take 3, a
`go_to` 40). Discounting by decision count would systematically favor slow actions.
Returns, advantages (GAE), and the planner's Q-values all discount by **elapsed game
time**: Q = r̂ + γ^max(τ,1) · (1 − p̂_death) · V(ŝ′), with the τ floor shared by every
consumer through one function. The RL core is unit-tested against hand-computed
references.

### 2.3 The world model: structure where the game has laws

The teacher's model is deliberately not a monolith:

- **Typed prediction heads.** Each of 132 state features is typed (continuous / count /
  binary / categorical) and gets a matching head — the model cannot look accurate on
  average while being useless per field.
- **Compositional reward head.** Craftax's reward is almost entirely *one-time
  achievement unlocks*. The head predicts per-achievement unlock probabilities and
  assembles reward as Σ p̂(unlock_k)·coef_k, with the game's coefficients as constants.
  Crucially, the achievement *set* is input (67 bits) and spent achievements are
  **hard-masked to zero** — "already unlocked ⟹ zero reward" is a law of the game, so
  it lives in the architecture, not in the training data's ability to teach it. Before
  this, the planner farmed already-spent achievements 471 times per 60 worlds, because
  a scalar head fed an achievement *count* was mathematically unable to represent
  exhaustion.
- **Empirical-Bayes shrinkage** (the decisive repair — §3.2): per-achievement
  probabilities are blended with a pooled any-unlock head by wₖ = nₖ/(nₖ+10), so
  achievements with ~5 training examples borrow strength from the pool, and the prior
  retires itself as evidence accumulates.
- **Ensemble + two decision rules.** Three bootstrap-resampled members. *Collection*
  uses Thompson sampling (one member per episode — coherent optimism, so a member that
  believes in a plan executes the whole plan); *deployment* uses the pessimistic LCB
  (μ − 0.5σ). Evaluating under the exploration rule would report the exploration bonus
  as skill.

### 2.4 The fairness contract

Held constant across arms, enforced by executable checks rather than intentions:
budget (3,200 decisions, audited to the unit from each run's own transition files —
the ledger deliberately isn't trusted for per-run identity), action space
(constructor version stamped into every data file; version mismatch is a **fatal**
preflight error), worlds (paired seeds), evaluator (one script computes every system's
metrics), exploration floor (ε=0.05 during collection for both learned arms, never
during eval).

Deliberately *not* matched, and reported instead: mechanism and compute. The LLM arms
inherit an internet-scale prior; the teacher starts from zeros. That asymmetry is the
subject of the study, not a confound in it.

### 2.5 Measurement discipline

- **Power before verdicts.** Early evals used n=10 worlds; four separate "trends"
  from that era (including a published-grade "PPO +1.36, p=0.002") died at n=60. The
  dev instrument is 60 paired worlds, MDE ≈ 0.9–1.5 reward, paired bootstrap with
  10,000 resamples; every verdict is significance-gated, and "n.s." is written "n.s.",
  not narrated as a trend.
- **Pre-registration.** Readouts are written down before each run (in the driver
  script's header, where they cannot be quietly edited afterward). The negative
  results in §3.2 were reported because the pre-registered readouts forced them into
  view.
- **Gates before compute.** Twelve-plus gate scripts (menu grounding, reward-head
  exactness, PPO gradient-path correctness, evaluator consistency, model-exploitation
  detection) must pass before a run spends budget. The PPO gradient gate is built
  around a *deliberately broken* implementation that produces perfect scores with
  silently wrong gradients, and asserts the real path is measurably different.
- **A held-out test set that has never been touched.** All results below are dev-set;
  80 test worlds (seeds 100–179) stay pristine until the roster is final and are spent
  once.

---

## 3. Results

### 3.1 The headline: the from-scratch planner wins its training distribution

60 paired dev worlds, matched budgets, one evaluator:

| system | reward | achievements gained | survival | vs FROZEN (paired) |
|---|---:|---:|---:|---|
| **TEACHER (v5c)** | **10.61** | **10.6** | **90%** | **+3.32, p<0.001** |
| PPO (ckpt4) | 7.66 | 8.5 | 52% | +0.36, p=0.22 n.s. |
| FROZEN | 7.29 | 8.1 | 48% | — |

Two results in one table. First, a small structured world model trained from zeros on
3,200 decisions — no language prior, no GPU — beats a 4-billion-parameter pretrained
planner on reward, achievements, and survival simultaneously. Second, **PPO on the
same budget is statistically indistinguishable from not training at all**: the
earlier-design "+1.36, p=0.002" PPO effect does not reproduce under the corrected
action space and the powered instrument (new CI [−0.23, +0.92]).

### 3.2 How the teacher got there: a ladder of controlled interventions

No single trick produced 10.61. The ladder, each rung a one-variable change measured
on the same 60 worlds:

| arm | change | reward |
|---|---|---:|
| ε=0 | honest menu, blind scalar reward head | 5.75 (significantly *worse* than FROZEN) |
| **+ ε=0.05 floor** | mirror the exploration floor PPO already had | **9.41** |
| + honest reward head | per-achievement compositional head, hard masking | 7.35 (**−2.06**) |
| + value-iteration targets | fitted-VI backups for V | 7.50 (Δ n.s. — score-neutral) |
| **+ shrinkage** | empirical-Bayes blend, pooled prior, self-retiring | **10.61** |

Three mechanisms, each verified causally:

**Exploration was the binding constraint (+3.66).** Not model quality, not credit
assignment — a 5% action floor. Chain events (mine coal, smelt, fight) rose 2–7× in
the training stream, and the model then valued them correctly on its own. The lesson
generalizes: before diagnosing a learner, check what its data ever contained.

**An honest model made things worse (−2.06), and understanding why mattered.** The
old achievement-blind head averaged reward credit over everything — wrong, but a
smooth accidental *breadth prior*. The honest per-achievement head is sharp where it
has data and silent where it does not: coal had ~5 training examples, so predicted
coal value was ~0, so the planner never mined coal, so coal stayed at ~5 examples — a
self-sealing error loop, made worse because a planner *chooses* its own training
distribution (endogenous data). At 3,200 decisions, **wrong-but-smooth beat
right-but-starved**. This is a bias-variance statement about small-data model-based
RL, and it reproduced across two independent arms.

**Making the accidental prior explicit beat both (+3.27 over the honest head).**
Shrinkage keeps the honest head's exactness (spent achievements predict exactly zero,
by construction) while restoring breadth through a pooled prior (furnace: 30 worlds
vs 2; zombie: 29 vs 1) — and unlike the accidental version, it *retires itself* as
per-achievement evidence accumulates. Bonus: the compositional structure crafts stone
pickaxes with **zero training examples** of doing so — cross-tier template
generalization the monolithic head cannot express.

### 3.3 The systems know different things

Behavioral fingerprints on the same worlds:

| behavior | FROZEN / PPO | TEACHER |
|---|---|---|
| deep chain (stone pickaxe → iron) | **25–37/60 stone pick, 14/60 iron** | 6/60 best, iron 0/60 |
| drinking (Craftax-specific mechanic) | ~0 drinks; 29+26 dehydration deaths | learned; 90% survival |
| easy-tier breadth | partial | **best** (10.6 achievements) |

The LLM planners walk the wood→stone→iron chain because *Minecraft folklore contains
it* — and die of thirst because Minecraft has no thirst meter. 3,200 decisions of PPO
fixed neither the missing mechanic nor anything else measurable. The from-scratch
teacher learned the meters this world actually has, and never discovered iron because
at this budget its exploration never got there — the depth is proven feasible within
40-turn episodes (the LLMs do it), so this is a pure discovery gap, and it triggered
the pre-registered next intervention (randomized-prior exploration).

### 3.4 One floor down: the transfer study, and a symmetry

All three systems were evaluated from 40 banked floor-1 entry states (a mob-dense
dungeon; near-zero training exposure, labelled per arm: TEACHER 3 lifetime decisions,
PPO 13, FROZEN 0):

| system | reward | ach gained | survived |
|---|---:|---:|---:|
| TEACHER | 4.81 | **3.20** | 18/40 |
| PPO | 4.93 | 2.43 | 28/40 |
| FROZEN | 3.98 | 1.93 | **29/40** |

No pairwise reward difference is significant at n=40 (MDE ≈ 2) — reported as
descriptive. The finding that transfers is the **death-mechanism symmetry**:

> On the surface, the mechanic missing from the LLM's prior (thirst) kills the LLMs;
> the teacher's learned upkeep saves it. On floor 1 the roles invert *exactly*: the
> habit the teacher learned on the surface — sleep at low energy, safe above ground,
> 219 naps at 90% survival — is lethal in a dungeon where sleeping multiplies incoming
> damage. 13 of its 22 floor-1 deaths have `sleep` among the final actions. FROZEN,
> whose prior lacks the sleep mechanic entirely, cannot make that mistake and
> out-survives it 29/40 vs 18/40 (fight rates are equal — it is not aggression).
>
> **Each system dies precisely where its knowledge's origin diverges from the world
> it is standing in** — the prior's game for the LLMs, the training floor for the
> learner.

This reframes "which system is better?" as "which *knowledge source* fails where?",
and it set up two follow-ups with pre-registered readouts: few-shot floor-1
adaptation — does 400 decisions of dungeon experience reprice the teacher's sleep
habit faster than it patches the LLM's gaps? — answered in §3.6, and distillation —
can the LLM's depth prior and the teacher's learned upkeep be *unioned*? — designed
in [docs/DISTILL_DESIGN.md](docs/DISTILL_DESIGN.md), whose central risk (importing
the teacher's lethal sleep habit along with its upkeep) is exactly the symmetry
finding applied to training data.

### 3.5 The null results, reported as such

- PPO ≈ FROZEN at n=60 (above). Under this action interface — where the menu already
  does the affordance work and the policy only ranks ~12 grounded options — 3,200
  decisions of duration-aware PPO produced no detectable improvement. The gradient
  path itself is gate-verified correct; the null is about signal, not machinery.
- Fitted value-iteration targets for the teacher's V: score-neutral at n=60
  (Δ −0.16 n.s.), and behaviorally it *suppressed* upkeep — a reminder that a
  max-over-model-Q backup imports the model's optimism into exactly the states the
  data least supports (the deadly triad, observed in vivo, caught by a pre-registered
  exploitation auditor rather than by intuition).

### 3.6 Few-shot adaptation: the same 400 decisions, very different returns

Both learned systems then received B=400 extra floor-1 decisions (from the same
snapshot bank the eval uses) followed by their own standard update — a full model
refit for the teacher, one more KL-bounded LoRA step for PPO; FROZEN is the B=0
control. Paired on the same 40 snapshots (these arms have spent 3,600 total decisions
and are labelled B=400 — never mixed into the matched-3,200 tables):

| arm | reward | ach gained | survived | vs its own B=0 |
|---|---:|---:|---:|---|
| **Teacher + 400** | **13.46** | **8.45** | 32/40 | **+8.65, p<0.001** (all metrics) |
| PPO + 400 | 5.60 | 2.85 | **33/40** | +0.68 n.s. (survival +12.5pts, p=0.007) |
| FROZEN (B=0) | 3.98 | 1.93 | 29/40 | — |

The pre-registered mechanism readout confirmed: the teacher's deaths fell 22→8 and
its sleep-death fraction from 59% to 38% — roughly forty sleep-outcome examples
repriced the lethal habit. More striking, with ore actually exposed on floor 1 the
from-scratch learner acquired the whole dungeon economy in 400 decisions — coal,
furnaces, potions (an action it had never once taken), bows, **iron in 7/40 and
diamond in 3/40 worlds** — with zero forgetting on the surface (paired surface
subset: 9.94 vs 9.24). So the surface iron gap of §3.3 was about the long chain to
*reach* ore, not about valuing it — an exploration/credit-assignment problem, not a
valuation one, which is what roadmap stages 4–5 attack.

The LLM arm, updated through one trust-region-bounded PPO step, barely changed
(action mix moved <10% in every category; deaths 12→7). **Per decision of
experience, the world model converts new evidence into behavior change roughly an
order of magnitude faster** — model refit vs. KL-bounded policy step is itself a
structural difference between the schools, not just a hyperparameter choice. The
step-size caution that protects PPO on-distribution is exactly what caps its
few-shot adaptation off-distribution.

---

## 4. What it took to measure this

The results above consumed roughly a third of the project's effort. The other two
thirds went into making them *measurable*, and the failures found along the way are
part of the contribution ([docs/LESSONS.md](docs/LESSONS.md) holds the full set):

- **The no-op audit.** The teacher's first "failure" was 61.4% of its budget spent on
  actions that did nothing (sleep at full energy). The fix was in the action
  interface, not the learner — and every "system X is worse" diagnosis since starts
  with "what did X's budget actually buy?"
- **The terminal-state bug.** Craftax raises `done` one step *after* health hits zero.
  Five separate decision loops each needed the same guard; the one that lacked it
  charged 81 post-death decisions to one arm's budget and wrote (corpse, action) → 0
  rows into its training data. The rule now lives in one function (`env.is_terminal()`)
  asserted by a consistency gate across all four loops — "one rule, one function"
  stopped being a style preference and became a correctness requirement.
- **Instrument before verdict.** Four early trends died when the eval grew from 10 to
  60 worlds. A comparison without a minimum-detectable-effect number attached is a
  mood, not a measurement.
- **Suspect the instrument first.** The single most valuable reflex this project
  trained: when a number is surprising, the first hypothesis is that the *measurement*
  is broken. It was, repeatedly — a budget ledger without run identity (summing
  all-time totals across runs), a resumed log read as a fresh runtime, an eval window
  dominated by server startup.

---

## 5. Limitations

- **Dev-set results.** Seeds 40–99, used throughout development. The 80-world test
  set is reserved, unspent, and will be spent once.
- **One training seed per arm.** Training-seed variance is unmeasured; a second
  teacher seed is queued before any publication-grade claim. The teacher-vs-FROZEN
  headline is p<0.001, but one internal rung of the ablation ladder sits at p=0.050.
- **One environment, one model size, 40-turn episodes.** The bias-variance and
  symmetry findings are hypotheses elsewhere, results here.
- **The action interface does real work.** All systems benefit from the grounded
  menu and scripted skills; these results say nothing about end-to-end token-level
  control.
- **Training so far is surface-floor.** Floor-1 results are transfer and few-shot
  studies; full-depth training and evaluation (roadmap stage 4) has not begun.

## 6. Status and next steps

Against the program roadmap (README):

- **Stage 1 — matched-budget comparison: complete** (§3.1–3.3, §3.5), pending a
  second training seed and the one-shot test-set protocol for publication-grade
  claims.
- **Stage 2 — floor transfer and few-shot adaptation: complete for floor 1**
  (§3.4, §3.6), with both pre-registered mechanism readouts resolved.
- **Stage 3 — DISTILL: designed and gated, not yet run**
  ([docs/DISTILL_DESIGN.md](docs/DISTILL_DESIGN.md)): knowledge-union distillation
  with floor-conditional targets, confidence gating from ensemble variance, an
  upkeep exemption from the trust region, and a KL anchor against interference.
  The few-shot result added an option worth measuring: the floor-adapted teacher is
  a stronger distillation source than the surface-only one, since its dungeon
  habits are no longer lethal.
- **Stage 4 — full-depth training** via the snapshot-bank curriculum: next up, with
  a queued prerequisite on the surface side — randomized-prior exploration for the
  discovery gap §3.6 diagnosed (the pre-registered trigger has fired: depth is
  proven feasible by the LLM arms, and proven learnable-when-reached by the
  few-shot teacher; what's missing is the exploration to connect the chain).
- **Stage 5 — counterfactual long-horizon credit assignment**: infrastructure ready
  (exact branch-and-replay), waiting on stage 4's deep trajectories.

---

*All numbers in this report are reproducible from the repo: every table has a driver
script, every claim a gate or verdict script, and every run's budget is audited from
its own transition files. See [PROGRESS.md](PROGRESS.md) for the living lab notebook.*
