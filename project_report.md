# Project report: LLM priors vs. learned world models on Craftax
---

## 1. Motivation

There are two common answers to "how should an agent get good at an environment?"

1. **Start from a pretrained LLM.** Its general knowledge stands in for experience.
   Add RL fine-tuning if you want more.
2. **Learn a model of the environment.** Predict what actions do, plan against those
   predictions, and let interaction be the source of competence.

These two approaches are usually compared across different papers, with different
budgets, different action spaces, and different evaluation setups. That makes the
comparisons hard to trust. This project puts both approaches in the same harness, on
the same benchmark, with the same budget, and asks one narrow question:

> **Given exactly 3,200 environment decisions to learn from, does a world model
> trained from scratch beat a frozen pretrained LLM planner? And does it beat the
> same LLM fine-tuned with PPO on those 3,200 decisions?**

The testbed is **Craftax** (the full version). It is a 2D survival game in the
Minecraft family: a tech tree that runs from wood to stone to iron to diamond,
hunger/thirst/energy/health meters, day and night, hostile monsters, and a dungeon
with nine floors. Two properties make it a good testbed. It is hard enough that
neither approach wins trivially. And, in a way that turned out to matter a lot, *part*
of the game matches Minecraft folklore an LLM already knows (the tech tree), and part
of it does not (the thirst meter, the dungeon).

This comparison is stage one of a larger program (the README has the roadmap). The
same infrastructure is built to eventually support training and evaluation across all
nine floors, and counterfactual credit assignment over long horizons — "which decision
two hundred steps ago made this diamond possible?" The floor-1 studies in sections 3.4
and 3.6 are the first steps in that direction.

The four systems:

| arm | what it is | trained on | compute |
|---|---|---|---|
| FROZEN | Qwen3-4B picks actions from a menu | nothing | inference only |
| PPO | same LLM, fine-tuned with PPO+GAE | 3,200 decisions | ~376M tokens, 3.2 GPU-h |
| TEACHER | small world model + value net, no LLM at all | 3,200 decisions, from scratch | minutes of CPU |
| DISTILL | the teacher's decisions taught back to the LLM | inherits the teacher's data | running now |

---

## 2. Design choices, and why

### 2.1 Every system picks from the same menu

At each decision point, the harness builds a menu of up to 20 macro-actions from the
game state — things like `go_to(tree)`, `craft(stone_pickaxe)`, `fight(zombie)`,
`drink_water`, `descend`, `open_chest`. The system under test picks one. A scripted
controller then carries it out over however many game steps it takes.

One rule governs the menu: **if an action is offered, it can actually be done.** Sleep
is not offered when energy is full. Craft is not offered without materials. The menu
constructor checks these conditions using the same code the controller uses, so the
two can never disagree.

This sounds like a detail. It nearly sank the project. An early version offered
`drink` and `sleep` unconditionally. The LLM ignored them (its prior "knows" they're
usually pointless), but the world-model planner had no such instinct and spent **61%
of its entire training budget** on actions that did nothing. Every comparison made
with that menu was garbage. The menu is now covered by regression tests, because the
menu *is* the experiment: if different systems saw different action spaces, none of
the downstream numbers would mean anything.

### 2.2 Credit assignment that respects time

Macro-actions take very different amounts of game time — a `fight` might take 3 steps,
a long `go_to` forty. If you discount rewards per *decision*, slow actions get an
unfair advantage. So everything here — returns, advantages, the planner's Q-values —
discounts by **elapsed game time** instead. This "semi-Markov" accounting is
implemented once, shared by every consumer, and unit-tested against hand-computed
examples.

### 2.3 A world model with structure where the game has rules

The teacher's world model is deliberately not one big black box:

- **Typed prediction heads.** Each of the 132 state features is typed (a count, a
  binary flag, a category, a continuous value) and gets a matching prediction head.
  The model can't look accurate on average while being useless on the fields that
  matter.
- **A reward head that mirrors how the game actually pays.** Almost all reward in
  Craftax comes from one-time achievement unlocks. So the reward head predicts, per
  achievement, the probability of unlocking it, and multiplies by the game's own
  reward coefficients. Already-unlocked achievements are **forced to predict zero** —
  that's a rule of the game, so it is built into the architecture rather than left
  for training data to teach. Before this existed, the planner "farmed" spent
  achievements 471 times per 60 worlds, chasing reward the game would never pay
  again.
- **Shrinkage for rare events.** With only 3,200 decisions, many achievements have a
  handful of training examples. Their predictions are blended with a pooled
  "any-achievement" estimate, weighted by how much data each one has. Rare events
  borrow strength from the pool; as real evidence accumulates, the pool's influence
  fades away on its own. Section 3.2 explains why this one change was worth +3.3
  reward.
- **An ensemble, used two different ways.** Three copies of the model are trained on
  bootstrap resamples of the data. During *data collection*, one copy is sampled per
  episode and trusted fully — so a copy that believes in a plan executes the whole
  plan (this is Thompson sampling, and it produces coherent exploration rather than
  random jitter). During *evaluation*, the planner scores actions by the mean minus
  half the spread — a pessimistic rule. Evaluating with the exploration rule would
  count the exploration bonus as skill.

### 2.4 What is held equal, and what is not

Held equal across all systems, enforced by scripts rather than good intentions:

- the interaction budget (3,200 decisions, counted from each run's own log files),
- the action menu (a version number is stamped into every data file, and comparing
  across versions is a hard error),
- the evaluation worlds (the same seeds, paired),
- the evaluation code (one script computes every system's metrics),
- the exploration floor (both learning systems get ε=0.05 random-ish actions during
  collection, never during evaluation).

Deliberately *not* equal: the starting knowledge and the compute. The LLM arms begin
with an internet-scale prior and spend GPU-hours; the teacher begins with random
weights and trains in CPU minutes. That asymmetry is not a flaw in the study — it is
the subject of the study.

### 2.5 Measurement discipline

Four habits did a lot of work here:

- **Check the statistical power first.** Early evaluations used 10 worlds. Four
  separate "findings" from that era — including one at p=0.002 — disappeared when the
  evaluation grew to 60 worlds. Every comparison now reports the smallest effect it
  could have detected, and anything not significant is called not significant.
- **Write down predictions before running.** Each experiment's expected readouts are
  recorded in the driver script's header before it runs. Two of the most useful
  results in this report are negative ones that pre-registration forced into view.
- **Gate before spending.** More than a dozen check scripts have to pass before any
  run spends real budget: menu correctness, reward-head exactness, gradient-path
  correctness, evaluator consistency. The gradient check is built around a
  deliberately broken implementation that produces perfect-looking scores with wrong
  gradients — the gate proves the real path differs from it.
- **Keep a clean test set.** Everything in this report uses development worlds
  (seeds 40–99). Eighty test worlds are reserved and have never been touched. They
  will be spent once, at the end.

---

## 3. Results

### 3.1 The headline: the from-scratch planner wins on its home turf

Sixty paired development worlds, matched budgets, one evaluator:

| system | reward | achievements | survival | vs FROZEN (paired) |
|---|---:|---:|---:|---|
| **TEACHER** | **10.61** | **10.6** | **90%** | **+3.32, p<0.001** |
| PPO | 7.66 | 8.5 | 52% | +0.36, p=0.22 (not significant) |
| FROZEN | 7.29 | 8.1 | 48% | — |

Two results live in this table. First: a small world model trained from random
weights on 3,200 decisions — no language prior, essentially no GPU — beats a
4-billion-parameter pretrained planner on reward, achievements, and survival at the
same time. Second: **PPO on this budget is statistically indistinguishable from not
training at all.** An earlier version of this project reported a significant PPO gain;
under the corrected action menu and the properly powered evaluation, that effect does
not reproduce.

### 3.2 How the teacher got there: one change at a time

The 10.61 did not come from one trick. It came from a ladder of single-variable
changes, each measured on the same 60 worlds:

| arm | change | reward |
|---|---|---:|
| baseline | honest menu, naive reward head | 5.75 (significantly *worse* than FROZEN) |
| **+ exploration floor** | give the teacher the same ε=0.05 PPO had | **9.41** |
| + honest reward head | per-achievement predictions, spent ones masked to zero | 7.35 (a 2-point *drop*) |
| + value-iteration targets | fancier value-network training | 7.50 (no significant change) |
| **+ shrinkage** | blend rare-event predictions with a pooled estimate | **10.61** |

Three lessons, each verified rather than assumed:

**Exploration was the bottleneck, not the model.** A 5% floor of forced variety was
worth +3.7 reward on its own. With it, the training data finally contained coal
mining, smelting, and fighting (2–7× more of each), and the model priced them
correctly once it had seen them. Before diagnosing a learner, check what its data
ever contained.

**Making the reward model *more correct* made the agent *worse* — and the reason
matters.** The old, naive head spread reward credit over everything. Wrong, but
smooth: it acted as an accidental "try things broadly" prior. The honest
per-achievement head is sharp where it has data and silent where it does not. Coal
had about five training examples, so predicted coal value was near zero, so the
planner never mined coal, so coal stayed at five examples. The error seals itself in,
because a planner chooses its own future training data. With this little data,
wrong-but-smooth beat right-but-starved by two full points. That is a bias-variance
tradeoff showing up at the system level.

**The fix was to make the accidental prior explicit.** Shrinkage keeps the honest
head's exactness (spent achievements still predict exactly zero) while restoring the
breadth the naive head had provided by accident — furnaces built in 30 worlds instead
of 2, zombies fought in 29 instead of 1. And unlike the accident, it fades out
gracefully as real evidence arrives. A bonus from the compositional structure: the
model crafts stone pickaxes correctly with *zero* training examples of doing so,
generalizing the pattern from other craft actions.

### 3.3 The systems know different things

| behavior | FROZEN / PPO | TEACHER |
|---|---|---|
| the deep chain (stone pickaxe → iron) | yes — iron in 14 of 60 worlds | never (0 of 60) |
| drinking (a mechanic Minecraft lacks) | almost never; 55 dehydration deaths | learned it; 90% survival |
| breadth on the easy tier | partial | best |

The LLM planners walk the wood → stone → iron chain out of the box, because Minecraft
folklore contains that chain. The same planners die of thirst, because Minecraft has
no thirst meter — and 3,200 decisions of PPO did not patch that hole in the prior.
The teacher learned the meters this world actually has and survives 90% of episodes,
but never discovered iron: at this budget, its exploration never completed the long
chain to reach ore. The LLMs prove the chain fits inside a 40-turn episode, so the
teacher's gap is purely about discovery, not capability. That fired the pre-registered
trigger for the next exploration upgrade (randomized priors).

### 3.4 One floor down: where knowledge comes from decides where you die

All three systems were then dropped into the dungeon: 40 saved floor-1 entry states,
essentially untrained territory for everyone (the teacher had seen 3 floor-1
decisions in its life, PPO 13, FROZEN none).

| system | reward | achievements | survived |
|---|---:|---:|---:|
| TEACHER | 4.81 | **3.20** | 18/40 |
| PPO | 4.93 | 2.43 | 28/40 |
| FROZEN | 3.98 | 1.93 | **29/40** |

No reward difference here is statistically significant at 40 worlds, so treat the
ranking as descriptive. The finding that matters is a symmetry in *how they die*:

> On the surface, the LLMs die of thirst — the one mechanic their prior lacks. The
> teacher's learned upkeep saves it. One floor down, the roles flip exactly. The
> habit the teacher learned on the surface — sleep when energy is low, which is safe
> up there — is lethal in a dungeon full of monsters, where sleeping multiplies the
> damage you take. Thirteen of its 22 floor-1 deaths have `sleep` among the last few
> actions. FROZEN's prior doesn't have the sleep habit at all, so it cannot make that
> mistake, and it survives the most.
>
> **Each system dies exactly where its knowledge came from somewhere else** — the
> LLMs where their prior's game differs from this one, the teacher where its training
> floor differs from this floor.

That reframes the question from "which system is better?" to "which knowledge source
fails where?" — and it set up the two experiments that follow. Few-shot adaptation:
give each system a little dungeon experience and see who fixes their blind spot
faster (section 3.6). Distillation: try to combine the LLM's deep knowledge with the
teacher's learned survival into one policy (running now).

### 3.5 The negative results, stated plainly

- **PPO ≈ FROZEN.** With the menu already handling "what is possible" and the policy
  just ranking about 12 grounded options, 3,200 decisions of PPO produced no
  detectable improvement. The training machinery itself is verified correct by the
  gates; the problem is signal, not bugs.
- **Value-iteration targets didn't help.** A more sophisticated way of training the
  value network (backing up the best model-predicted action instead of what actually
  happened) changed the score by nothing, and behaviorally it *suppressed* the
  survival habits. Taking a max over model predictions imports the model's optimism
  exactly where data is thinnest — a textbook risk, observed live, caught by a
  pre-registered check.

### 3.6 Few-shot adaptation: same 400 decisions, wildly different returns

Both learning systems then got 400 extra decisions of dungeon experience, drawn from
the same saved entry states the evaluation uses, followed by their own standard
update — a full model refit for the teacher, one more PPO step for the LLM. FROZEN,
which has no update mechanism, is the control. (These arms have now used 3,600 total
decisions, so they are labeled separately and never mixed into the matched-budget
tables above.)

| arm | reward | achievements | survived | change vs its own baseline |
|---|---:|---:|---:|---|
| **Teacher + 400** | **13.46** | **8.45** | 32/40 | **+8.65, p<0.001, every metric** |
| PPO + 400 | 5.60 | 2.85 | **33/40** | +0.68, not significant |
| FROZEN (control) | 3.98 | 1.93 | 29/40 | — |

The teacher nearly tripled its dungeon reward. The pre-registered prediction — that
its death-model would re-price the sleep habit — came true: deaths fell from 22 to 8.
And it went much further than fixing one habit. With ore actually exposed on floor 1,
the teacher picked up the whole dungeon economy in 400 decisions: coal, furnaces,
torches, bows, potions (an action it had never taken even once), **iron in 7 worlds
and diamond in 3** — with zero forgetting of its surface skills. Remember that iron
was the thing it could *never* discover on the surface. So the surface gap was about
the long chain needed to *reach* ore, not about valuing it.

Why did PPO barely move? We took the comparison apart layer by layer, and the answer
is **not the data**. PPO's 400 collected decisions were about as rich as the
teacher's (99 achievement unlocks, including iron and diamond), and its own advantage
estimates pointed at the right lessons — "open chests" carried the strongest positive
signal in the batch. The bottleneck is *absorption*. PPO's trust region — the safety
rule that stops the policy from moving too far per update — tripped after **one
16-state step**, because off-distribution states make the policy hypersensitive (the
first gradient step was ~30× larger than normal). The same rule allows 40–110 steps
per update on familiar territory. So PPO absorbed 16 states of its 400; the teacher's
supervised refit absorbed all 3,600 rows it had ever seen, and then re-planning
spread the update everywhere: re-scoring the same floor-1 situations under the old
and new models, the preferred action changed in **78% of states**, in exactly the
directions the evidence pointed (chests and potions way up, mining unchanged).

Even the sleep fix is smarter than "stop sleeping": the adapted teacher actually
sleeps *more* (58 takes vs 38), but its predicted danger of sleeping rose 14× in the
risky states, so it now picks its moments — deaths per sleep fell from 34% to 5%. It
learned *when* sleep kills.

The takeaway: **per decision of experience, the world model converts evidence into
behavior change roughly ten times faster.** That is not a tuning issue. Learning
facts and re-planning against them has no speed limit; an on-policy policy gradient
must move slowly to stay valid. The caution that protects PPO at home is exactly
what prevents it from adapting quickly somewhere new.

---

## 4. What it took to measure this

The results took about a third of the effort. The other two thirds went into making
them measurable, and the failures along the way are part of what this project has to
offer ([docs/LESSONS.md](docs/LESSONS.md) has the full list):

- **Audit what the budget actually bought.** The teacher's first "failure" turned out
  to be 61% of its budget spent on do-nothing actions. The fix was in the action
  menu, not the learner. Every diagnosis since starts with "what did this system's
  budget actually buy?"
- **One rule, one function.** Craftax reports death one step *late*, so every
  decision loop needs the same guard. It existed in five copies; the one copy that
  was missing it charged 81 post-death decisions to one arm and wrote corpse-state
  rows into its training data. The rule now lives in exactly one function, and a
  consistency check asserts every loop calls it.
- **Instrument before verdict.** Four early "trends" died when the evaluation grew
  from 10 worlds to 60. A comparison that doesn't know its own detection threshold
  is a mood, not a measurement.
- **When a number surprises you, suspect the measurement first.** It was the
  measurement, repeatedly: a budget ledger that couldn't tell runs apart, a resumed
  log read as a fresh one, a timing window dominated by server startup.

---

## 5. Limitations

- **These are development-set results.** The 80 held-out test worlds are still
  untouched and will be spent once, at the end.
- **One training seed per arm.** Run-to-run training variance is unmeasured. The
  headline result is p<0.001, but one internal rung of the ablation ladder sits
  right at p=0.05; a second seed is queued before any publication-grade claim.
- **One game, one model size, 40-turn episodes.** The bias-variance and symmetry
  findings are results here and hypotheses anywhere else.
- **The action menu does real work.** Every system benefits from the grounded menu
  and scripted skills. This study says nothing about end-to-end control from raw
  tokens.
- **Training so far happens on the surface floor.** The dungeon results are transfer
  and few-shot studies. Full multi-floor training hasn't started yet.

## 6. Status and next steps

Against the roadmap in the README:

- **Stage 1 — the matched-budget comparison: done** (sections 3.1–3.3, 3.5). Still
  owed: a second training seed and the one-shot test-set run.
- **Stage 2 — floor transfer and few-shot adaptation: done for floor 1**
  (sections 3.4, 3.6), with both pre-registered predictions resolved.
- **Stage 3 — distillation: running now.** The adapted teacher is the source — its
  dungeon habits are no longer lethal, which removes the main risk of teaching them
  to the LLM. Design in [docs/DISTILL_DESIGN.md](docs/DISTILL_DESIGN.md).
- **Stage 4 — full-depth training** using the saved-state curriculum. One
  prerequisite queued first: the exploration upgrade for the surface discovery gap,
  now well-motivated — the few-shot result proved the teacher learns deep resources
  easily once it can reach them.
- **Stage 5 — counterfactual credit assignment over long horizons.** The
  infrastructure (exact save/restore and branching) is built; it waits on stage 4
  producing trajectories deep enough to need it.

---

*Every number in this report is reproducible from the repo. Each table has a driver
script, each claim has a check or verdict script, and each run's budget is counted
from its own log files. [PROGRESS.md](PROGRESS.md) is the running lab notebook.*
