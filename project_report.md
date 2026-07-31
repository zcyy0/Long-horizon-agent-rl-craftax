# Project report: LLM priors vs. learned world models on Craftax

*Four agent systems, one game, one interaction budget. What does each one learn?
What can each one not learn? And what does it take to measure the difference
honestly?*

This report is self-contained: sections 1–2 give the question and the design
decisions, section 3 gives every result, and the appendix gives the exact contracts
and formulas for the four things whose details decide whether the comparison is fair
— the action menu, the world model, the semi-Markov PPO objective, and the
distillation objective. The [README](README.md) is the short version.

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
| DISTILL | the teacher's decisions taught back to the LLM | inherits the teacher's data | ~2.8 GPU-h |

The LLM arms run Qwen3-4B behind a vLLM inference server, and training updates it
through LoRA adapters that are merged back into the served weights between
iterations. Training gradients are computed by a separate HuggingFace/PyTorch stack —
same weights, different kernels — which is why the report keeps returning to the
discipline of never trusting one stack's probabilities in the other's computation.

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

(The full constructor contract — its two invariants, the shared grounding predicates,
and the gate that enforces them empirically — is in appendix A.1.)

### 2.2 Credit assignment that respects time

Macro-actions take very different amounts of game time — a `fight` might take 3 steps,
a long `go_to` forty. If you discount rewards per *decision*, slow actions get an
unfair advantage. So everything here — returns, advantages, the planner's Q-values —
discounts by **elapsed game time** instead. This "semi-Markov" accounting is
implemented once, shared by every consumer, and unit-tested against hand-computed
examples. (The exact return, advantage, and policy-loss formulas are in appendix
A.3.)

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

(The architecture, the per-type head parameterizations, and the assembled reward
formula are in appendix A.2.)

### 2.4 Distillation as a test of knowledge union, not of imitation

The fourth arm exists because of a result, not a plan: the LLM and the teacher end up
knowing *complementary* things (section 3.3). So the interesting question is not "can
a student copy a teacher" — it is whether one policy can hold both knowledge sources
at once. That framing decides the design.

The teacher and the LLM already rank the same menu, so every decision the teacher
logged during training is a supervised example, and the arm needs **zero new
environment interactions**. The failure mode to fear is not underfitting but
**interference**: dragging the whole policy toward the teacher and destroying the
depth prior — the LLM forgetting iron in order to learn drinking. Two guards address
it directly: the teacher's targets are damped in states where the base LLM finds the
action absurd (a trust region *in the prior*), and a KL anchor holds the student at
the prior wherever the teacher has nothing confident to say. Sections 3.7 and 3.8 are
mostly the story of those two guards being too crude and then too coarse; appendix
A.4 has the targets, the loss, and both calibrations.

### 2.5 What is held equal, and what is not

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

### 2.6 Measurement discipline

Every headline number is a **paired** comparison: each system plays the same 60
development worlds (seeds 40–99), scores are differenced world by world, and the
interval and p-value come from a paired bootstrap over worlds (10,000 resamples,
two-sided). Pairing matters because world difficulty dominates score variance — a
brutal seed drags every system down together, and differencing removes it.

Alongside every verdict the same script reports the **minimum detectable effect**:
the smallest true difference this evaluation would have caught with 80% power at
α=0.05, given the measured paired spread. If the observed gap is smaller than that,
the verdict written down is *"this evaluation cannot resolve these systems"* — never
"they are equal."

Four habits did the rest of the work:

- **Check the power first.** Early evaluations used 10 worlds, where the detectable
  effect was ±4.6 reward — larger than any effect in this project. Four separate
  "findings" from that era, including one at p=0.002, disappeared when the evaluation
  grew to 60 worlds. All four are reported in section 3.5 and section 4 rather than
  quietly dropped.
- **Write down predictions before running.** Each experiment's expected readouts are
  recorded in the driver script's header before it runs, with a declared refutation
  branch where possible. Two of the most useful results in this report are negative
  ones that pre-registration forced into view.
- **Gate before spending.** More than a dozen check scripts have to pass before any
  run spends real budget: menu correctness, reward-head exactness, gradient-path
  correctness, evaluator consistency. The gradient check is built around a
  deliberately broken implementation that produces perfect-looking scores with wrong
  gradients — the gate proves the real path differs from it.
- **Keep a clean test set.** Everything in this report uses development worlds.
  Eighty test worlds (seeds 100–179) are reserved and have never been touched. They
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

The teacher's row has since been replicated. A second teacher, trained from scratch
with a different random seed on the same recipe, scored 11.75 reward, 11.9
achievements, and the same 90% survival — again beating FROZEN at p<0.001 on all
three metrics. The two seeds differ from each other by about one reward point, which
is a real (and now measured) training-seed variance, but the headline does not
depend on which seed you look at.

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
teacher's learned survival into one policy (sections 3.7–3.8).

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
must move slowly to stay valid.

A follow-up A/B then tested the tempting version of that explanation — "so if we
remove the caution, PPO will adapt" — and refuted it. The update was re-run with the
data-collection bug fixed (the full 400 decisions this time) and the KL early-stop
switched off, so nothing halted it partway: it absorbed **22 minibatch steps instead
of one**, with only the per-token clip still in the loss. The dungeon score did not
move — 4.5 and 4.9 across the two arms against 4.9 for not adapting at all (n=40,
detectable effect ≈ 2) — and nothing was forgotten on the surface either. The
constraint tripping was therefore a symptom, not the cause: the update was free to
run and had nothing to absorb.

What that leaves is a structural reading: a policy gradient can only reweight the
actions it happened to sample, one noisy example at a time, while the teacher turns
the same 400 experiences into an updated model of the world and replans every
decision against it. Stated at the strength the evidence supports: this A/B rules out
the throttle as the explanation, and the structural account is the reading that
survives it — not a separately tested claim. A fully unbounded policy gradient (no
clip, larger steps) was not run.

### 3.7 Distillation: the union is real, and so are its costs

The last arm teaches the teacher's decisions back to the LLM and asks whether one
policy can hold both knowledge sources: the prior's deep game knowledge *and* the
teacher's learned survival. The target is the teacher's own deployment score over the
same menu, damped by the base LLM's preferences (the trust region) except for
survival actions, with a KL anchor holding the student at the prior wherever the
teacher is unconfident — two guards against the pre-registered null hypothesis of
interference. Appendix A.4 gives the formulas. Two LoRA rounds, ~2.8 GPU-hours, zero
new environment interactions; the source stream came from the *dungeon-adapted*
teacher, so this arm's lineage is 3,600 decisions and it is kept out of the
matched-budget table.

Five predictions were registered before the run. Two passed, two failed, one was
report-only — and the pattern of which is the finding:

| prediction | result |
|---|---|
| P1: upkeep transfers (≥5 drinks/60 worlds, dehydration deaths drop) | **PASS** — 339 drinks, **zero** dehydration deaths (FROZEN: 0 and 29) |
| P2: survival above FROZEN's | **PASS** — 53/60 vs 29/60, teacher-level |
| P3: depth retained (iron/stone counts stay in the LLM band) | **FAIL** — stone retained (29), iron dropped 14 → 8 |
| P4: headline reward (report-only) | 9.75 — between the parents, closer to the teacher |
| P5: floor-1 sleep habit not imported | **FAIL** — 5 of 8 dungeon deaths involve sleep |

Against the frozen model the distilled one is better by +2.46 reward, +1.97
achievements and +40 points of survival, all at p<0.001; against its own teacher it
is statistically indistinguishable on all three metrics (−0.86 reward, p=0.066) —
while costing a single forward pass per decision, where the teacher runs an ensemble
and a value network over every menu option. The mechanic 3,200 decisions of RL could
not install (PPO after training: 7 drinks, 26 dehydration deaths) arrived intact
through supervised distillation of the same experience, and arrived *well-timed*: the
distilled model's drink spacing matches the teacher's almost exactly.

The two failures are the informative part.

**P3 is real but is not forgetting.** The iron capability survived: half of the
distilled model's iron worlds are ones the frozen model never cracked, and in the ten
worlds where it "lost" iron it still crafted stone pickaxes in five and out-scored
frozen in six. What happened is arithmetic — **27.7% of its decisions go to upkeep**
(frozen: 0.0%), and eleven decisions of drinking and sleeping do not fit inside a
40-turn episode next to a five-step crafting chain. Reported as failed; diagnosed
post-hoc as opportunity cost. The distinction decides what to fix next: forgetting
would indict the anchor, while opportunity cost is the teacher's *policy* being
faithfully copied, upkeep-heaviness included.

**P5 is the exact risk accepted going in.** Distilling from the dungeon-adapted
teacher (rather than surface states only) was a deliberate choice whose known risk
was importing the teacher's sleep habit into the dungeon. It happened: 50 floor-1
sleeps, five of eight deaths involving one — a mistake frozen structurally cannot
make. The habit arrived with partial selectivity (10% of dungeon sleeps end in death,
versus 34% for the unadapted teacher and 5% for the adapted one) and the floor-1
package is still strongly net-positive (+3.2 reward over frozen, p=0.002, survival
not worse). But the prediction failed, and the lesson is symmetrical with section
3.4: distillation transfers habits with less of the context that made them safe.

One more question the data answers: why does the student *match* its teacher rather
than beat it, given it also holds the prior? Because imitation's ceiling is the
teacher's own policy, and measured achievement-by-achievement the prior's gains and
the imitation's losses nearly cancel. The gains are the entire mining chain the
teacher never had. The losses concentrate in one mechanism: the teacher earns a point
in nearly every world by planting saplings, and the distilled model **never places
anything**, because the frozen LLM never places anything (0 times in 1,766
decisions) — so the trust region crushed every place target exactly the way it would
have crushed drinking, had drinking not been explicitly exempted. One category
missing from a hand-written exemption list, roughly the whole teacher gap. It is a
precise illustration of the design's real dial: **every category exempted from the
trust region transfers one more teacher lesson and removes one more guard on the
prior.**

The arm in one sentence: **distillation produced the second-best system measured here
— a single-forward-pass policy that matches its search-based teacher on the surface,
fixes the LLM's fatal blind spot, and pays for it with a measured share of the depth
prior and one imported habit.** The union is real, and it is not free.

### 3.8 Second try: replace the hand list with evidence — and learn why the guard was there

Any hand-maintained exemption list will always be one entry short of something. So
the second run replaced it with a rule: **each action family's veto is retired in
proportion to how much real experience the teacher has with it** — the same
shrinkage form the reward head uses, applied one level up, calibrated on the
teacher's own training stream with no evaluation data touched (formula and weights in
A.4). Placing, which the teacher did 566 times, keeps almost none of the veto;
dungeon sleeping, which it did 15 times in its life, keeps most of it.

One pre-registered prediction was deliberately two-sided: if the rule really measures
evidence, it should transfer *more* in one place and *less* in another at the same
time. Both ends fired.

| readout | run 1 (hand list) | run 2 (evidence-weighted) | pre-registered verdict |
|---|---|---|---|
| dev60 reward / ach / survival | 9.75 / 10.10 / 53-60 | 9.56 / 9.93 / 52-60 | V4 ✗ — Δ−0.19, n.s. (MDE 0.81) |
| place takes; sapling worlds | 0; 0 | **713; 48/60** | V1 ✓ |
| drinks; dehydration deaths | 339; 0 | 33; 7 | V2 ✗ |
| iron; stone-pickaxe worlds | 8; 29 | 2; 4 | V3 ✗ — the declared refutation branch |
| floor-1 reward / sleeps / sleep-deaths | 7.20 / 50 / 5 | **8.83 / 0 / 0** | V5 ✓ |
| decision mix (mine/place/drink) | 29% / 0% / 16% | 40% / **31%** / 1% | — |

The direction of the idea is confirmed on both ends of the dial: the deleted skill
came back in earnest, and the risky dungeon habit was filtered out entirely — the
best floor-1 score any LLM-based system has posted here, at zero sleeps.

The calibration failed, and the last row explains why. The teacher spends 16% of its
decisions placing; the student spends 31%, double its own teacher. The training label
is a normalized distribution, so it is a **fixed budget of preference**: every point
given to one family is taken from the rest. Un-vetoing the teacher's single strongest
lesson didn't restore it to its natural share — it let it outbid everything else.
Drinking, whose winning margins were always slim, lost the auction; and an agent
spending a third of a 40-turn episode placing runs out of turns for the five-step
chain to iron. One number per family, blind to margins, can say "transfer this
lesson" but not "transfer it *at the teacher's rate*."

Both written-down conclusions fired. The depth collapse was the pre-declared
refutation branch — *"if iron collapses, the original trust region was earning its
keep"* — so the crude veto keeps its job and **the first run remains the reported
distillation arm** (`v2` in the code and artifacts; the second is `v3`). And
the general lesson joins section 3.2's: distillation hands out a fixed budget of
preference, and whatever damps the teacher's lessons is really deciding who wins that
auction. A hand list is too rigid, a family-level count too coarse; a third attempt
would have to allocate mass, not merely remove vetoes. The keeper from this run is
the dungeon policy — proof that filtering *by evidence* beats filtering *by category*
exactly where the teacher's knowledge is newest.

---

## 4. What it took to measure this

The results took about a third of the effort. The other two thirds went into making
them measurable, and the failures along the way are part of what this project has to
offer:

- **Audit what the budget actually bought.** The teacher's first "failure" turned out
  to be 61% of its budget spent on do-nothing actions. The fix was in the action
  menu, not the learner. Every diagnosis since starts with "what did this system's
  budget actually buy?"
- **Don't infer a state you can read directly.** The same bug shape appeared four
  times: a component needed fact X and checked an easier proxy assumed to imply it —
  "always dispatchable" for *energy is not full*, "material held" for *a valid tile*,
  "table reachable" for *item + gem + mana*, an error string for *health ≤ 0*. A
  planner then actively seeks the states where the proxy fails, because a free action
  is an attractive one, so a minority-state defect became a majority of the budget.
- **One rule, one function.** Craftax reports death one step *late*, so every
  decision loop needs the same guard. It existed in five copies; the one copy missing
  it charged 81 post-death decisions to one arm and wrote corpse-state rows into its
  training data. The rule now lives in exactly one function, and a consistency check
  asserts every loop calls it.
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
- **Two training seeds for the teacher; one for everything else.** A second
  independently trained teacher reproduced the headline (11.75 reward vs. the first
  seed's 10.61; each beats FROZEN at p<0.001 on all three metrics). The two seeds
  differ by ~1.1 reward (p=0.022), which is the first real estimate of training-seed
  variance — and a reason to keep reading internal ablation rungs of similar size
  with caution: one of them sits right at p=0.05, inside that between-seed spread.
  The PPO arm still has a single training seed.
- **One game, one model size, 40-turn episodes.** The bias-variance and symmetry
  findings are results here and hypotheses anywhere else.
- **The action menu does real work.** Every system benefits from the grounded menu
  and scripted skills. This study says nothing about end-to-end control from raw
  tokens.
- **Training so far happens on the surface floor.** The dungeon results are transfer
  and few-shot studies. Full multi-floor training hasn't started yet.
- **Floor-1 exposure during training is small but not exactly zero** (teacher 3
  decisions, PPO 13, frozen none), so section 3.4 is reported as *near*-zero-shot
  with those counts attached.

## 6. Status and next steps

Against the roadmap in the README:

- **Stage 1 — the matched-budget comparison: done** (sections 3.1–3.3, 3.5). The
  second training seed is done and replicates (section 3.1). Still owed: the
  one-shot test-set run.
- **Stage 2 — floor transfer and few-shot adaptation: done for floor 1**
  (sections 3.4, 3.6), with both pre-registered predictions resolved.
- **Stage 3 — distillation: done, two runs** (sections 3.7–3.8). The first run is
  the reported arm: a partial union, with its two failed predictions reported as
  failures. The second replaced the hand-tuned guard with an evidence rule — right in
  direction on both pre-registered ends, wrong in calibration, and it proved by
  ablation that the original guard was protecting the deep knowledge.
- **Stage 4 — full-depth training** using the saved-state curriculum. One
  prerequisite queued first: the exploration upgrade for the surface discovery gap,
  now well-motivated — the few-shot result proved the teacher learns deep resources
  easily once it can reach them.
- **Stage 5 — counterfactual credit assignment over long horizons.** The
  infrastructure (exact save/restore and branching) is built; it waits on stage 4
  producing trajectories deep enough to need it.

---

## Appendix: the four core designs, precisely

The main text describes *what* was built and *why*. This appendix walks through the
core components in more detail, with the exact contracts and formulas — because the
fairness of the comparison lives in these details. Everything here is implemented in
the named files, and each design is guarded by the checks described alongside it.

### A.1 The grounded action constructor

*`harness/candidate_actions.py`, with every grounding predicate owned by
`harness/executor.py`.*

The constructor builds the menu that every system chooses from. That makes it the
carrier of the study's core fairness requirement: the LLM's own suggestions must
never define the action space. So the menu is assembled from exactly two sources —
the public skill schema, and what the agent can currently observe.

Two invariants govern it.

**First: if an action is offered, it can actually be done.** The two worst bugs of
the project were both violations of this rule — 30% of decisions spent on impossible
crafts, and 61% of a training budget spent on actions that did nothing. It is now
enforced by construction and checked empirically.

**Second: availability is not value.** The menu says what is *possible*, never what
is *good*. A chest appears on the menu the moment one is visible and reachable, long
before any system knows whether opening it pays off. Keeping those two things
separate is what makes "learning to use an affordance" measurable at all, as
something distinct from "being shown it".

#### What is on the menu, and what puts it there

Here is the complete set of macro-actions and the condition each one has to meet.
"Reachable" throughout means *the agent has seen the tile and a walking path to it
exists in its own map memory* — not merely that the thing is on screen.

| action | offered when… | the reason for the condition |
|---|---|---|
| `explore`, `explore(up/down/left/right)` | always | the guaranteed fallback, so the menu is never empty |
| `open_chest` | a chest has been seen and is reachable | availability only; nothing is implied about the loot |
| `descend` | a down-ladder is reachable **and** ≥ 8 monsters have been killed on this floor | the game itself refuses the descent below the quota, so offering it earlier would put an impossible action on the menu |
| `ascend` | an up-ladder has been seen and is reachable | |
| `drink_water` | water or a fountain is reachable **and** the drink meter is not full | drinking at a full meter consumes zero game steps and returns nothing |
| `eat(plant)` | a ripe plant is reachable **and** the food meter is not full | same zero-step problem |
| `sleep` | energy is not full | sleeping at full energy does nothing — the single defect that consumed 61% of a training budget |
| `fight(passive)` | a cow, bat or snail is reachable | hunting. Deliberately *not* gated on hunger: the kill is legal at any food level, and whether it is worth a decision is the policy's judgment |
| `fight`, `fight(melee)`, `fight(ranged)` | a hostile of that class is reachable | reachable, not just visible — a zombie behind a wall can be seen but never approached, so the skill would spend zero steps |
| `mine(resource)` | a deposit is reachable **and** the held pickaxe is good enough | Craftax gates ore by tool tier: wood needs no pickaxe, stone and coal need a wood one, iron needs stone, diamond needs iron, sapphire and ruby need diamond |
| `go_to(crafting_table)`, `go_to(furnace)` | the station is remembered and reachable but not adjacent | without this, an agent with one wood and a table ten tiles away had *no* path to any craft — it would have had to gather three wood and build a second table |
| `craft(item)` | every material is held — **including** the two wood for a crafting table this call would have to place — **and** the result would actually be an upgrade | charging only the recipe made 30% of all decisions impossible crafts; tools are tiered, so re-crafting a tier you already hold is a silent no-op |
| `place(item)` | the material is held **and** the tile the agent faces will accept it | a plant needs *grass* specifically, not merely an empty tile — that one rule was 7 of 11 observed failures |
| `collect_sapling` | grass is reachable | |
| `shoot` | a bow and at least one arrow are held | |
| `read_book` | a book is held | |
| `drink_potion` | a potion is held | |
| `enchant(sword/armour/bow)` | an enchantment table is reachable, the item is held, mana ≥ 9, **and** the held gem matches that table's element (fire wants ruby, ice wants sapphire) | four separate conditions; checking only the first three still yields a zero-step action |

Craft and place draw from fixed shortlists — the progression spine (wood/stone/iron
pickaxes and swords, torches, arrows) and the five placeable items — rather than
every recipe in the game, so one satisfiable recipe family cannot crowd out the rest
of the menu.

A few design decisions keep the two invariants true in practice.

The constructor only reads what the agent has seen. Every predicate works from the
agent's own memory of explored tiles, plus one shared reachability map computed from
that memory — never from the true game map. This matters because a menu built from
privileged information would leak that information through *which actions appear*,
even if no system ever read the map directly.

Each grounding rule lives in exactly one function, shared with the controller. When
the menu asks "would this craft succeed?", it calls the same `craft_requirements()`
the crafting skill itself uses — which counts the full cost, including the crafting
table this particular call might have to place. The same pattern covers tiered tools
(re-crafting a tier you already hold does nothing, so it is not offered), placement
rules (a plant needs grass, not just an empty tile), enchanting (which needs the
table, the item, enough mana, and the right gem all at once), the survival meters,
and reachability (a monster behind a wall is visible but unreachable). Because the
menu and the skill share one function, they cannot drift apart. They are the same
code.

There is one judgment call in the design, and it has a clean rule: **the constructor
rejects the impossible; the policy owns the unwise.** Descending is gated on the
floor's kill quota because the *game* refuses the descent before eight kills —
offering it earlier would make "descending was blocked" indistinguishable from
"descending was a bad idea", which corrupts credit assignment. Fighting at low
health, by contrast, stays on the menu: the game permits it, so it is the agent's
decision to get wrong and learn from. If it is a game rule, gate it in the
constructor. If it is strategy, leave it to the policy.

Menu assembly is deterministic. Candidates are added in priority tiers — from a
guaranteed core (the explore fallback, the chest, the descent) down to
directional-explore filler — then deduplicated, stably sorted, and capped at 20. The
cap has never bound; the largest menu ever observed is 17. Each candidate records the
observable fact that put it on the menu, and the menu's version number is stamped
into every data file. Comparing data across menu versions is a hard error, not a
warning.

Finally, the invariants are verified empirically rather than by code review. A gate
replays every candidate offered across ~3,000 real and synthesized states and asserts
that none of them consumes zero game steps; a companion check asserts that every
survival action that *would* help is actually offered. The empirical form earned its
cost immediately: it caught do-nothing cases in `eat` and `fight` that nobody had
predicted.

### A.2 The world model

*`models/macro_world_model.py`, including the ensemble; retrained each round on the
teacher's own experience.*

The model answers one question: *if I take this action from this state, what
happens?* Answering it well is what lets the teacher plan without ever consulting a
language model — it can score all twelve menu options by imagining each one, instead
of having to try one and find out.

It is deliberately a **one-step, direct** model: it predicts the situation after the
whole macro-action finishes, in the same feature space it took as input. There is no
learned latent space and no multi-step rollout. That is a real limitation, honestly
stated — the model cannot see six decisions ahead, and section 3 reports a
measurement where exactly that shows up — but it buys something valuable: the
predictions are in human-readable units, so when the planner does something strange
you can look directly at what it believed would happen.

$$M_\psi(x_t, a_t) \rightarrow (\hat{x}_{t+1},\ \hat{r}_t,\ \hat{\tau}_t,\ \hat{d}_t,\ \hat{y}_t)$$

**What goes in.** The state $x$ is 132 numbers describing everything the agent can
legitimately know:

| group | examples | count |
|---|---|---|
| body and status | health, food, drink, energy, whether asleep | ~6 |
| inventory and tools | pickaxe tier, sword tier, bow, armour pieces, potions, books | ~7 |
| the achievement set | one flag per achievement: *have I already unlocked this one?* | 67 |
| what is visible | water, hostiles, passive mobs, ladders, and the distance to each | ~10 |
| exploration state | tiles seen, fraction explored, reachable tiles, whether standing in the dark | ~6 |
| position in the game | floor, facing direction, monsters killed here, floor cleared | ~5 |
| what just happened | the previous skill, its outcome, its duration, its reward | ~8 |
| adjacency | standing next to a crafting table, next to a furnace | 2 |

The 67 achievement flags are the least obvious group and the most important. Craftax
pays for each achievement exactly *once*, so the true reward function depends on
which ones are already spent. An earlier version carried only the achievement
*count*, which made two states with the same count but different sets look
identical — and the reward head was then mathematically forced to give the same
answer where the truth differed by a full point. The consequences were exactly as
bad as that sounds: the model kept predicting about +0.9 for collecting a sapling
long after that reward was spent, and the teacher farmed it 471 times across 60
worlds. The value network, denied the variable that actually explained the reward,
latched onto whatever correlated with time instead and concluded that *holding wood
is bad* and *low energy is good*. Adding the set fixed the reward head and the value
network in one change. None of this is privileged information: the agent unlocked
those achievements itself.

The action $a$ is a 61-dimensional encoding — a one-hot for which skill, plus typed
slots for its arguments (which resource, which item, which direction, which target)
and a scalar count. The layout is strictly **append-only**. New action types are
added at the end, never inserted, because inserting one shifts every later index and
would silently re-label every row of previously collected data — the model would
decode old experience as the wrong skills entirely, and nothing would crash.

**What comes out.** Five predictions: the next state, the reward, how many game steps
the action will consume, whether it ends in death, and 72 named events. The events
are 67 "did achievement *k* unlock on this action?" flags plus five general ones —
a chest opened, the floor went down, the floor went up, the action succeeded, and
whether *any* achievement unlocked at all. That last one turns out to matter; it
becomes the pooled prior in the reward head below.

**How it is built.** A shared 2-layer, 256-unit MLP trunk feeds a separate prediction
head for each *type* of thing being predicted:

| feature type | what the head predicts | loss |
|---|---|---|
| continuous (health, distances) | the change, in normalized units | Huber |
| counts (inventory, tiles seen) | the change, on a `log1p` scale | Huber |
| binary flags | a logit | cross-entropy |
| categorical (direction, floor) | a softmax per field | cross-entropy |
| duration $\hat{\tau}$ | `log1p` of the step count | Huber |
| death $\hat{d}$ | a logit — true death only, never a truncated episode | cross-entropy |
| events $\hat{y}$ | one logit per event | cross-entropy |

Two choices here do most of the work. **Typing the heads** stops a model from looking
accurate on average while being useless where it counts: a facing-direction code and
a distance in tiles are not the same kind of number and must not share a loss
function. **Predicting changes rather than next values** exploits the fact that most
of the 132 features do not move on any given action — inventory does not change when
you walk. That makes "nothing happened" the zero-effort default and pushes the
model's capacity onto whatever actually moved. The `log1p` scale for counts handles
both ends of a wide range at once: spending three wood down to zero, and a
step counter in the thousands.

**The reward head is assembled, not learned end-to-end.** This is the part of the
design with the most thought behind it, so it is worth stating the reasoning before
the formula.

Craftax's reward is not a mystery function to be approximated. It genuinely *is* a
sum: each achievement, the first time you unlock it, pays a constant that we can
read straight out of the game's source, plus a small term for health changes. A
naive design asks a network to regress that sum from scratch, which throws away the
structure and forces the model to rediscover — from a few thousand examples — a
formula we already know exactly.

So instead the model predicts the *ingredients* (how likely is each achievement to
unlock if I take this action?) and the reward is assembled from them using the game's
own coefficients:

$$\hat{r} \;=\; \underbrace{\sum_k w_k\, \hat{p}_k\, c_k}_{\text{sharp, per-achievement}} \;+\; \underbrace{\hat{p}_{\text{any}} \cdot \frac{\sum_k (1-w_k)\, c_k\, \mathbb{1}[\text{locked}_k]}{\#\text{locked}}}_{\text{pooled prior on the untrusted mass}} \;+\; 0.1\,\widehat{\Delta\text{health}}$$

Reading it left to right: the first term is the model's own per-achievement belief,
the second is a fallback for achievements it has barely seen, and the third is the
health term. Three properties of this arrangement each carry real weight.

**The mask makes the model exact where the game is exact.** An achievement already
unlocked pays nothing, ever. That is a rule, not a pattern, so $\hat{p}_k$ is forced
to zero by the architecture whenever the corresponding input flag is set — the
network's learned opinion only gets consulted for achievements still available. The
alternative is hoping the network infers the rule from data, and we know what that
costs: before the mask, the planner chased already-spent achievements 471 times
across 60 worlds.

**The shrinkage weights handle rare events honestly.** Here is the problem they
solve. With only 3,200 decisions of experience, some achievements have hundreds of
examples and others have five. A per-achievement head with five examples predicts
near-zero simply for lack of evidence — and a planner that believes coal is
worthless never mines coal, so coal stays at five examples forever. The error seals
itself in, because a planner chooses its own future training data.

The fix is to trust each head in proportion to how much it has seen:
$w_k = n_k/(n_k + 10)$, computed from the training counts when the model is fit. An
achievement seen 5 times gets weight 0.33 and leans mostly on the pooled fallback;
one seen 200 times gets 0.95 and stands on its own evidence. The leftover weight
$(1-w_k)$ is carried by that "did *anything* unlock?" event — a much easier
prediction, since every achievement's data contributes to it — spread across the
achievements still locked. So a rarely-seen achievement inherits a sensible generic
optimism instead of a confident zero. And because $n_k$ grows every round, $w_k$
climbs toward 1 on its own: the fallback quietly retires itself wherever real
evidence arrives. This is the single change worth +3.3 reward in section 3.2.

**And there is a built-in lie detector.** A plain scalar reward head is trained
alongside the assembled one and never used for planning. Its only job is
disagreement: if the composed prediction and the scalar prediction drift apart
beyond the slack expected from within-action discounting, something is wrong with
the labels or the wiring, and a gate fails loudly. A model that is confidently wrong
in both halves at once is far less likely than one that is wrong in one.

**The value network.** The reward head only sees one action ahead. What makes
planning possible is a second small network, $V_\omega$, trained to predict the
discounted return from a state — so the planner can ask "what is the situation I end
up in worth?" rather than only "what does this action pay right now". It is refit
from scratch every round, because it estimates the value of states *under the
current policy*, and the policy changes every round. A stale value function is one of
the easier ways to make a model-based agent chase its own tail.

**The ensemble, and its two rules.** Three copies of the model are trained on
different bootstrap resamples of the same data. Where they agree, the data has
spoken clearly; where they disagree, the model is guessing. That spread is the
teacher's only sense of its own ignorance, and both of the rules below are built on
it.

To score a candidate action $a$, each copy $m$ computes

$$Q_m(h,a) \;=\; \hat{r}_m(h,a) \;+\; \gamma^{\max(\hat{\tau}_m,\,1)} \cdot \big(1-\hat{d}_m(h,a)\big) \cdot V_\omega\big(\hat{x}'_m(h,a)\big)$$

— the predicted reward, plus the discounted value of wherever the action leads,
zeroed out if the action is predicted to end in death (there is no future left to
discount). The $\max(\hat{\tau},1)$ floor in the exponent is the same shared
function the PPO objective uses; A.3 explains the bug it prevents. The planner then
works with the mean $\mu_Q$ and the spread $U$ across the three copies.

The two selection rules matter enough to restate. During data collection, the
planner samples one ensemble member per episode and trusts it fully — a member that
believes in a plan gets to execute the whole plan, which produces coherent
exploration rather than random jitter. (Resampling every decision sounds more
thorough and is actually worse: it averages the members out, which looks like
exploration while exploring strictly less.) An ε = 0.05 random floor, mirrored to
PPO's, sits on top. During evaluation, the planner switches to the pessimistic rule
$\arg\max_a\, (\mu_Q - \kappa U)$ with κ = 0.5 — preferring an action it is confident
about over one that merely *might* be brilliant. Reporting a collection-mode run as
"the planner's score" would count the exploration bonus as skill, and conflating
these two rules is, in our experience, the main way a model-based result gets
overstated.

**How it all trains.** The teacher starts genuinely from nothing: three randomly
initialized networks that disagree about everything, which makes its first decisions
effectively arbitrary. It then repeats a simple loop eight times, spending 400 of its
3,200 decisions per round:

1. **Play.** Collect 400 decisions on the training worlds under the Thompson rule.
2. **Refit the world model.** Retrain all three copies on everything collected so
   far, with fresh bootstrap resamples. The shrinkage weights $w_k$ are recomputed
   here, so they tighten automatically as evidence accumulates.
3. **Refit the value network** against the updated model and the new data.

Nothing is carried between rounds except the data — models are refit rather than
fine-tuned, which is affordable precisely because they are small. The whole eight
rounds take minutes of CPU and no GPU at all. That the resulting agent beats a
4-billion-parameter language model on this task is the headline of section 3.1.

### A.3 Semi-Markov PPO + GAE

*`rl/smdp_returns.py`, `rl/gae.py`, `rl/ppo.py` — plain numpy and torch, imported by
the GPU trainer but testable without it.*

**Accounting for time.** A decision here is a macro-action: decision $t$ runs for
$\tau_t$ game steps and collects a reward on each. Its reward, and the episode's
return, are

$$R_t^{\text{macro}} = \sum_{j=0}^{\tau_t-1} \gamma^j r_{t,j}, \qquad G = \sum_t \gamma^{T_t} R_t^{\text{macro}}, \quad T_t = \sum_{i<t}\tau_i$$

The regrouped return $G$ provably equals the ordinary discounted return over the
flat stream of game steps — and that identity is a unit test, so "discount at macro
boundaries" is checked rather than assumed.

One subtlety deserves its own paragraph. The objective charges $\gamma^{\max(\tau,1)}$
per decision — never $\gamma^0$. Without that floor, an action that takes zero steps
and changes nothing scores $Q = V(s)$ undiscounted, which makes it the only *free*
action on the menu, and a planner will happily loop on it forever. This exact
loophole once consumed 61% of a training budget. The floor lives in one shared
function (`tau_effective`) used by the GAE pass, the teacher's Q-values, and the
value-fitting targets alike, so the objective cannot quietly fork between the
systems being compared.

**Advantages.** The TD error and the advantage are the standard GAE pair, with
elapsed game time in the discount exponent:

$$\delta_t = R_t^{\text{macro}} + \gamma^{\tau_t}(1-d_t)\,V(h_{t+1}) - V(h_t), \qquad \hat{A}_t = \delta_t + \gamma^{\tau_t}\lambda\,(1-d_t)\,\hat{A}_{t+1}$$

with γ = 0.99 per game step and λ = 0.95, one λ-decay per decision. The flag $d_t$
means true death, which zeroes the bootstrap. An episode merely cut short by the
step limit instead bootstraps from the value of its final state. Death and
truncation look identical in a reward trace and mean opposite things for the value
target — mixing them up is one of the quiet bugs the unit tests construct on
purpose.

**The policy is a distribution over the menu.** For a candidate action written out
as tokens $y_1..y_L$, the policy scores it by its average token log-probability and
softmaxes across the menu:

$$s_\theta(h,a) = \frac{1}{L^\alpha}\sum_{j} \log p_\theta(y_j \mid h, y_{<j}), \qquad \pi_\theta(a\mid h) = \frac{e^{s_\theta(h,a)/T}}{\sum_{a' \in C(h)} e^{s_\theta(h,a')/T}}$$

Both parameters were measured, not defaulted. The length normalization (α = 1)
exists because raw summed log-probs vary by about 27 nats with the sheer length of
the action string, against only ~1.4 nats of actual preference — an unnormalized
policy is opinionated about verbosity, not value. The temperature $T$ is calibrated
on development worlds until the policy's entropy hits a set target, then frozen; it
is recalibrated whenever the menu changes shape, because it depends on the menu
size. Rollout collection, the PPO ratio, and the distillation loss all share the
same $(\alpha, T)$ — if they disagreed, the behavior policy would silently differ
from the trained one while every diagnostic stayed green.

**The update** is the clipped PPO surrogate, applied per token over the action's
tokens, with the decision's advantage broadcast to each:

$$\rho_{t,j} = \exp\big(\log\pi_{\text{new}}(y_j) - \log\pi_{\text{old}}(y_j)\big), \qquad \mathcal{L} = -\,\mathbb{E}_{t,j}\Big[\min\big(\rho_{t,j}\hat{A}_t,\ \text{clip}(\rho_{t,j}, 1\pm\epsilon)\,\hat{A}_t\big)\Big]$$

Why per token? A sequence-level ratio multiplies the drifts of every token together,
so a tiny per-token change compounds into an exploding ratio over a multi-token
action. We measured it: a single optimizer step of size ~10⁻⁶ produced an
approximate KL of 3.2 under the sequence form. Per-token ratios do not compound, so
the clip and the trust region behave as designed. Updates also stop early when the
approximate KL (Schulman's low-variance $k_3$ estimator) passes 0.02.

Three disciplines surround the formula, each motivated by a bug it prevents.

The "old" log-probs in the ratio are recomputed once, without gradients, on the
exact weights that collected the data — never trusted from the inference server. The
serving and training stacks disagree by up to 3.8 percentage points at the operating
temperature; taking the server's numbers would book that disagreement as policy
improvement.

Masking is done by selection, not multiplication. Positions outside the action span
contribute exactly zero via `torch.where`, because 0 × NaN is NaN — one degenerate
logit elsewhere in the sequence must not be able to poison the action's
log-probability.

And the gradient path is gated against its own most plausible bug. The fast scoring
path shares one cached computation across all ~12 candidates, and there is a
specific way to get that sharing wrong that produces *perfectly correct scores* with
silently wrong gradients. The gate builds that broken version on purpose and asserts
the real path measurably differs from it — a class of check that comparing outputs
can never provide.

### A.4 Distillation: the target, the loss, and the two guards

*`scripts/build_distill_data.py`, `scripts/train_distill_v2.py`; the student's policy
representation is exactly the $\pi_\theta$ of A.3, same $\alpha$ and same calibrated
$T$ — a gate asserts the match, so "the distilled policy" and "the PPO policy" are
the same object trained two ways.*

**What is distilled.** The teacher and the student rank the *same* grounded menu, so
every decision the teacher logged during its own training is a supervised example:
the state $h$, its menu $C(h)$, and the teacher's scores over that menu. Nothing new
is played. The arm inherits the teacher's interaction bill and adds only GPU time —
which is why it is reported with a lineage label (3,600 decisions, because the source
stream is the dungeon-adapted teacher of section 3.6) rather than as a new
matched-budget arm.

The design's default was to distill *surface* states only, since that is where the
teacher's competence is demonstrated and where section 3.4 showed its habits are
safe. The run deliberately relaxed that and used the dungeon-adapted teacher's whole
stream — more transferable knowledge, at a named risk that was then written into
prediction P5 and duly realized.

**The target.** The teacher's *deployment* score $S_{\text{lcb}}(h,a) = \mu_Q - \kappa U$
(κ = 0.5 — the pessimistic rule it actually acts under, never the optimistic
collection rule) becomes a distribution over the menu:

$$q^*(a\mid h)\;\propto\;\pi_{\text{base}}(a\mid h)\;\cdot\;\exp\!\big(S_{\text{lcb}}(h,a)/\beta\big),\qquad \beta = 1$$

The $\pi_{\text{base}}$ factor is a **trust region in the prior**: the student is
pulled toward what the teacher prefers, but damped wherever the base LLM assigns an
action essentially no mass. It exists for the null hypothesis of this arm —
interference, not underfitting — and it is the single most consequential line in the
design; sections 3.7 and 3.8 are largely the story of calibrating it.

**The loss**, over menu candidates only:

$$\mathcal{L}(h)\;=\;w(h)\sum_{a\in C(h)} q^*(a\mid h)\,\big(-\log \pi_\theta(a\mid h)\big)\;+\;\lambda\,\mathrm{KL}\big(\pi_\theta(\cdot\mid h)\,\big\|\,\pi_{\text{base}}(\cdot\mid h)\big),\qquad \lambda = 0.1$$

The KL anchor is the second guard. In states where the teacher's signal is switched
off ($w(h) = 0$), it holds $\pi_\theta$ at the prior instead of letting gradient
leakage erode it — that is what protects the deep knowledge in the states the
teacher has nothing to say about.

**The confidence gate** decides where the teacher gets to teach at all:

$$w(h) \;=\; \mathbb{1}\big[\text{margin}(h) \ge m_{\min}\big]\cdot\mathbb{1}\big[U_{\text{chosen}}(h) \le u_{\max}\big]$$

with the two thresholds set at the 20th and 80th percentiles of the teacher's own
final-round decision log — quantities already logged per decision, so the calibration
costs no new interactions. 2,305 of 3,600 states pass. Uncertain teacher states
contribute the anchor term only: the student is not taught the teacher's guesses.
(The cost is visible in section 3.7's post-hoc ledger — 36% of states gated out, and
breadth partly lives in low-margin states.)

**The one hand-written exception.** Survival actions (`drink`/`eat`/`sleep`) skip the
$\pi_{\text{base}}$ factor and use the raw $\exp(S_{\text{lcb}}/\beta)$ target. The
reason is arithmetic: a multiplicative trust region can never lift an action whose
prior mass is ≈ 0, and "the prior gives drinking ≈ 0 mass" is exactly the pathology
being treated (frozen: 0 drinks, 29 dehydration deaths in 60 worlds). One explicit,
auditable exception, not a knob. It is what made P1 pass — and, being a hand-written
list, it is also what made the sapling failure inevitable: `place` was not on it, so
every place target was crushed the same way drinking would have been.

**Training.** Two LoRA rounds, ~2.8 GPU-hours, 2 epochs each, lr 1e-5. Round 1 is
hard cross-entropy onto the teacher's argmax (final top-1 agreement 0.687 against an
abort bar of 0.5). Round 2 applies the soft target above with the gate and the
anchor; agreement settles at 0.459, which is by design rather than a regression — the
soft objective deliberately moves about a quarter of gated states off teacher-argmax
while the anchor equilibrates from 3.32 to 2.5 nats. Four gates run before any GPU
time: $q^*$ reproducibility (same menu + same checkpoint ⇒ identical targets),
non-degenerate $w$ (some states pass, some don't; the exception fires on > 0 states),
a 200-row overfit reaching ~1.0 agreement, and anchor-only training moving
$\mathrm{KL}(\pi_\theta\|\pi_{\text{base}}) \approx 0$.

**Deployment cost**, which is half the point of the arm: one forward pass per
decision. The teacher scores every menu option through three world-model copies and a
value network.

#### The second run: an evidence-weighted trust region

Section 3.8's run (`v3` in the code) replaced the hand-written exception with a
measured one. Each action family's veto is retired in proportion to how much real
experience the teacher has with it:

$$q^*(a\mid h)\;\propto\;\max\!\big(\pi_{\text{base}}(a\mid h),\;e^{-V}\big)^{\,1-w_a}\cdot\exp\!\big(S_{\text{lcb}}(h,a)/\beta\big),\qquad w_a=\frac{n_{f(a),b}}{n_{f(a),b}+m}$$

where $f(a)$ is the action's family, $b$ its floor bucket (surface or dungeon), and
$n$ counts how many times the teacher took that family on that bucket in its own
3,600-decision stream; $m = 25$, $V = 5$ nats. The form is the reward head's
shrinkage (A.2) applied one level up — the prior's veto retires where the teacher has
receipts. $w = 1$ reproduces the hand exemption; $w = 0$ reproduces the original
run's full damping. The hand list becomes a special case of a measured quantity, and
a pre-run gate required the rule to reproduce it (weights for drinking and placing on
the surface both ≥ 0.9) or the design would have been refuted before spending GPU.

Calibrated weights, the auditable artifact of the run: place|surface $n$=566
$w$=0.96 · drink|surface $n$=311 $w$=0.93 · sleep|surface $n$=155 $w$=0.86 ·
**sleep|dungeon $n$=15 $w$=0.38** · place|dungeon $n$=3 $w$=0.11 · unseen families
$w$=0. That single bolded row is why the rule could transfer *more* and *less* at the
same time, which is exactly what section 3.8 measured.

Two details are worth keeping for anyone who repeats this.

*The floor $e^{-V}$ was added before the run, by hand-checking the geometry.* With a
raw $\pi_{\text{base}}^{1-w}$ factor, $\ln \pi_{\text{base}}$ spans −1 to −14 nats, so
even at $w = 0.96$ a prior mass of $10^{-6}$ leaves a residual veto of −0.55 nats —
larger than the median place margin (0.577). The rule would have failed to transfer
the exact lesson it was built to transfer. Capping the veto in *score* units makes
the price bounded: evidence buys off at most $(1-w)\cdot V$, and $V = 5$ sits just
above the largest gated margin ever observed (4.17), so $w = 0$ still vetoes
everything, exactly as the first run's factor did.

*σ-based weights were measured and rejected before the run.* Weighting by ensemble
spread instead of counts sounds more principled, but the measured spread across
families spans only 0.31–0.63 — it is dominated by state-value disagreement, not by
action-outcome evidence — so the implied damping barely moves. Direction right,
magnitude useless. Recorded so the next person does not re-derive it.

**Why the calibration failed, in one property of the formula.** $q^*$ is normalized
over the menu, so it is a *fixed budget of preference*: mass returned to one family
is taken from all the others. $w_a$ is one number per family and blind to margins, so
it can express "transfer this lesson" but not "transfer it at the teacher's rate" —
and the measured result was a place rate of 31% against the teacher's 16%, with
drinking (slim margins) and the five-step craft chain paying the bill. A third
attempt has to allocate mass — margin-aware or budget-constrained retirement — rather
than merely remove vetoes.

---

*Every number in this report is reproducible from the repo. Each table has a driver
script under `scripts/`, each claim has a check or verdict script, and each run's
budget is counted from its own log files rather than from what the driver intended.*
