# Project report: LLM priors vs. learned world models on Craftax
---

## 1. Motivation

There are two common answers to "how should an agent get good at an environment?"

1. **Start from a pretrained LLM.** Its general knowledge stands in for experience.
   Add RL fine-tuning if you want more.
2. **Learn a model of the environment.** Predict what actions do, plan against those
   predictions, and let interaction be the source of competence.

This project puts both approaches in the same harness, on the same benchmark, with the same budget, and asks one narrow question:

> **Given exactly 3,200 environment decisions to learn from, does a world model
> trained from scratch beat a frozen pretrained LLM planner? And does it beat the
> same LLM fine-tuned with PPO on those 3,200 decisions?**

The testbed is **Craftax** (the full version). It is a 2D survival game in the
Minecraft family: a tech tree that runs from wood to stone to iron to diamond,
hunger/thirst/energy/health meters, day and night, hostile monsters, and a dungeon
with nine floors. 

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

## 4. Limitations

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

## 5. Status and next steps

Against the roadmap in the README:

- **Stage 1 — the matched-budget comparison: done** (sections 3.1–3.3, 3.5). Still
  owed: a second training seed and the one-shot test-set run.
- **Stage 2 — floor transfer and few-shot adaptation: done for floor 1**
  (sections 3.4, 3.6), with both pre-registered predictions resolved.
- **Stage 3 — distillation: running now.** The adapted teacher is the source — its
  dungeon habits are no longer lethal, which removes the main risk of teaching them
  to the LLM. 
- **Stage 4 — full-depth training** using the saved-state curriculum. One
  prerequisite queued first: the exploration upgrade for the surface discovery gap,
  now well-motivated — the few-shot result proved the teacher learns deep resources
  easily once it can reach them.
- **Stage 5 — counterfactual credit assignment over long horizons.** The
  infrastructure (exact save/restore and branching) is built; it waits on stage 4
  producing trajectories deep enough to need it.

---
## Appendix: the three core designs, precisely

The main text describes *what* was built and *why*. This appendix gives the exact
contracts and formulas, because the details are where the fairness of the comparison
actually lives. Everything here is implemented in the named files and covered by the
gates described alongside it.

### A.1 The grounded action constructor

*`harness/candidate_actions.py` (~380 lines), with every grounding predicate owned by
`harness/executor.py`.*

The constructor builds the menu every system chooses from, so it carries the study's
core fairness requirement: the LLM's own generations must never define the action
space. The menu is assembled from the **public skill schema** plus **observable
state** only. Two invariants govern it:

1. **Offered ⇒ dispatchable.** Every action on the menu can actually be executed
   from this state. 
2. **Availability is not value.** The menu says what is *possible*, never what is
   *good*. A chest is offered the moment one is visible and reachable — long before
   any system knows whether opening it pays. This is what makes "learning to use an
   affordance" measurable separately from "being shown it".

Design decisions that keep those invariants true:

- **Observation discipline.** Every predicate reads the agent's own seen-memory and
  one shared BFS reachability map computed from it — never the true game map. A menu
  built from god-mode state would leak information through *which actions appear*,
  even if no system ever read the map directly.
- **One rule, one function, shared with the controller.** Each grounding predicate is
  the executor's own: `craft_requirements()` (full cost, *including* the crafting
  table this call would have to place), `craft_would_upgrade()` (tools are tiered;
  re-crafting a held tier is a silent no-op), `place_would_succeed()` (Craftax's
  placement rules transcribed exactly once — a plant needs grass, not merely a free
  tile), `enchant_would_succeed()` (table ∧ item ∧ mana ≥ 9 ∧ the gem matching that
  table), the survival-meter checks, and reachability that excludes mobs behind
  walls. The menu and the skill executing its promise cannot drift apart, because
  they are the same code.
- **The constructor rejects the impossible; the policy owns the unwise.** `descend`
  is gated on the floor's kill quota because the *game* refuses it below 8 kills —
  offering it earlier makes "descending was blocked" indistinguishable from
  "descending was a bad idea", which corrupts credit assignment. Low-health `fight`,
  by contrast, stays on the menu: the game permits it, so it is the agent's decision
  to get wrong and learn from. Rule ⇒ gate in the constructor; strategy ⇒ leave to
  the policy.
- **Deterministic assembly.** Candidates are added in priority tiers (a guaranteed
  core — explore fallback, chest, descend — down to directional-explore ballast),
  deduplicated by canonical form, stably sorted, and capped at 20 (the cap never
  binds; the largest observed menu is 17). Each candidate carries its *provenance* —
  the observable fact that licensed it. The schema version is stamped into every
  data file, and the preflight gate makes cross-version comparisons a hard error.
- **Verified empirically, not by review.** The `NOOPGATE` check branch-and-replays
  every candidate offered across ~3,000 real and synthesized states and asserts none
  consumes zero environment steps; its complement asserts every survival action that
  *would* help is actually offered. The empirical form earned its cost: it found
  no-op cases in `eat` and `fight` that nobody had predicted.

### A.2 The world model

*`models/macro_world_model.py`; ensemble in the same file; trained per-round from the
teacher's own transition stream.*

One-step, direct, structured — no learned latent:

$$M_\psi(x_t, a_t) \rightarrow (\hat{x}_{t+1},\ \hat{r}_t,\ \hat{\tau}_t,\ \hat{d}_t,\ \hat{y}_t)$$

where $x$ is the 132-feature typed state, $a$ a 61-dim action encoding (skill one-hot
plus arguments; append-only, because inserting a dimension silently re-labels every
row of existing data), and the outputs are next state, macro-reward, duration in game
steps, death probability, and 72 named event probabilities. A shared 2×256 MLP trunk
feeds per-type heads:

| feature type | prediction | loss |
|---|---|---|
| continuous | residual in normalized units | Huber |
| count | residual on the `log1p` scale | Huber |
| binary | logit | BCE |
| categorical | per-field softmax | CE |
| duration $\hat{\tau}$ | `log1p` scale | Huber |
| death $\hat{d}$ | logit (true termination only — truncation is not death) | BCE |
| events $\hat{y}$ | one logit per event | BCE |

Two parameterization choices do real work. **Typing** prevents the failure where one
homogeneous regression head looks accurate on average while being useless per field
(a skill-id code is not a distance). **Residuals** exploit the fact that most
features are unchanged by most macro-actions: predicting deltas makes "nothing
happened" the zero-effort default, so capacity is spent on what actually moved.

**The reward head is assembled, not learned end-to-end.** Craftax's reward *is*
$\sum_k \text{unlock}_k \cdot c_k + 0.1\,\Delta\text{health}$ with known constants
$c_k$, so the model predicts per-achievement unlock probabilities and the reward is
composed from them:

$$\hat{r} \;=\; \underbrace{\sum_k w_k\, \hat{p}_k\, c_k}_{\text{sharp, per-achievement}} \;+\; \underbrace{\hat{p}_{\text{any}} \cdot \frac{\sum_k (1-w_k)\, c_k\, \mathbb{1}[\text{locked}_k]}{\#\text{locked}}}_{\text{pooled prior on the untrusted mass}} \;+\; 0.1\,\widehat{\Delta\text{health}}$$

with three properties, each load-bearing:

- **Exactness by construction:** $\hat{p}_k$ is hard-masked to zero when achievement
  $k$ is already unlocked. That is a rule of the game, so it is architecture, not
  training data. (Before the mask, the planner "farmed" spent achievements 471 times
  per 60 worlds.)
- **Empirical-Bayes shrinkage:** $w_k = n_k / (n_k + 10)$, computed from training
  counts at fit time. A head with five coal examples is mostly pooled prior; a head
  with hundreds is mostly its own evidence. The prior retires itself as counts grow —
  this is the single change worth +3.3 reward in section 3.2.
- **A built-in lie detector:** a plain scalar reward head is trained in parallel and
  never used for planning. If the composed and scalar predictions diverge beyond the
  within-macro discounting slack, the `REWARDHEAD` gate fails — mislabeled targets
  cannot pass silently.

**The ensemble and the two selection rules.** Three members are trained on bootstrap
resamples. For a candidate $a$, each member $m$ scores

$$Q_m(h,a) \;=\; \hat{r}_m(h,a) \;+\; \gamma^{\max(\hat{\tau}_m,\,1)} \cdot \big(1-\hat{d}_m(h,a)\big) \cdot V_\omega\big(\hat{x}'_m(h,a)\big)$$

with $\mu_Q$ and $U$ the mean and spread across members. Predicted death zeroes the
continuation (no future to discount), and the $\max(\hat{\tau},1)$ floor is the *same
shared function* the PPO objective uses — see A.3. Then, deliberately, two different
rules:

- **Collection: Thompson sampling.** One member is sampled *per episode* and trusted
  fully, so a member that believes in a plan executes the whole plan. Per-decision
  resampling would average the members out — it looks like exploration while
  exploring strictly less. An ε = 0.05 floor (mirrored to PPO's) sits on top.
- **Evaluation: lower confidence bound,** $\arg\max_a\, (\mu_Q - \kappa U)$ with
  κ = 0.5. Reporting a Thompson run as "the planner's score" would count the
  exploration bonus as skill; the conflation of these two rules is, in our
  experience, the main way a model-based result gets overstated.

### A.3 Semi-Markov PPO + GAE

*`rl/smdp_returns.py`, `rl/gae.py`, `rl/ppo.py` — pure-numpy/torch math, imported by
the GPU trainer but unit-testable without it.*

**Time accounting.** Decision $t$ executes $\tau_t$ primitive game steps with rewards
$r_{t,0..\tau_t-1}$. Its macro-reward and the episode return are

$$R_t^{\text{macro}} = \sum_{j=0}^{\tau_t-1} \gamma^j r_{t,j}, \qquad G = \sum_t \gamma^{T_t} R_t^{\text{macro}}, \quad T_t = \sum_{i<t}\tau_i$$

and $G$ provably equals the flat primitive-discounted return over the concatenated
step stream — that identity is a unit test, so "discounting at macro boundaries" is
checked rather than asserted. The *objective* charges $\gamma^{\max(\tau,1)}$ per
decision through one shared function (`tau_effective`) used by GAE, the teacher's
Q-values, and the value-fitting targets alike. The floor exists because a zero-step
action returning to the same state would otherwise score $Q = V(s)$ *undiscounted* —
the only free action on the menu, an absorbing self-loop the planner actively
prefers (this exact loophole consumed 61% of a training budget before it was closed).
Putting it in one function means the objective cannot fork across the systems being
compared.

**Advantage estimation.** Semi-Markov TD error and GAE, per decision:

$$\delta_t = R_t^{\text{macro}} + \gamma^{\tau_t}(1-d_t)\,V(h_{t+1}) - V(h_t), \qquad \hat{A}_t = \delta_t + \gamma^{\tau_t}\lambda\,(1-d_t)\,\hat{A}_{t+1}$$

with γ = 0.99 per game step, λ = 0.95, and one λ-decay per *decision*. $d_t$ marks
true death only, which zeroes the bootstrap; an episode cut short by the rollout
limit instead bootstraps from $V(h_T)$. Conflating those two endings is one of the
quiet bugs the unit suite constructs on purpose.

**The policy is a distribution over the menu.** For candidate $a$ with canonical form
$y_1..y_L$:

$$s_\theta(h,a) = \frac{1}{L^\alpha}\sum_{j} \log p_\theta(y_j \mid h, y_{<j}), \qquad \pi_\theta(a\mid h) = \frac{e^{s_\theta(h,a)/T}}{\sum_{a' \in C(h)} e^{s_\theta(h,a')/T}}$$

Both parameters were measured, not defaulted. α = 1 (a per-token mean) exists
because raw sums span ~27 nats *by string length* against ~1.4 nats of actual
preference — an unnormalized policy is sharp about verbosity, not value. $T$ is
calibrated on development worlds to a target entropy of $0.5\ln|C|$ and then frozen;
it is re-calibrated whenever the menu changes, because it depends on $|C|$. The same
$(\alpha, T)$ is shared by rollout collection, the PPO ratio, and the distillation
loss — a mismatch between collection and training silently changes the behavior
policy while every diagnostic stays green.

**The update** is a token-level clipped surrogate over the action-span tokens, with
the per-sequence advantage broadcast to each of its tokens:

$$\rho_{t,j} = \exp\big(\log\pi_{\text{new}}(y_j) - \log\pi_{\text{old}}(y_j)\big), \qquad \mathcal{L} = -\,\mathbb{E}_{t,j}\Big[\min\big(\rho_{t,j}\hat{A}_t,\ \text{clip}(\rho_{t,j}, 1\pm\epsilon)\,\hat{A}_t\big)\Big]$$

Token-level rather than sequence-level because a sequence ratio
$\exp(\sum_j \Delta\log p_j)$ *compounds* per-token drift multiplicatively over a
multi-token action — measured directly: a single 10⁻⁶-scale step produced
approx-KL ≈ 3.2 under the sequence form. Per-token ratios do not compound, so the
clip and the trust region behave as intended. Updates early-stop when approx-KL
(Schulman's low-variance $k_3$ estimator) exceeds 0.02.

Three disciplines around the formula, each motivated by a bug it prevents:

- **Old log-probs are recomputed** once, under `no_grad`, on the collection-time
  weights — never trusted from the inference server. The serving and training stacks
  disagree by up to 3.8 percentage points at the operating temperature; trusting the
  server's numbers would book that disagreement as policy improvement.
- **Masking is selection, not multiplication.** Off-span positions contribute
  exactly zero via `torch.where`, because `0 × NaN = NaN`: one degenerate logit
  outside the action span must not be able to poison the action's log-prob.
- **The gradient path is gated against its own plausible bug.** The fast scoring
  path shares one cached computation across all ~12 candidates; a specific mistake
  in that sharing yields *perfectly correct scores* with silently wrong gradients.
  The gate builds that broken version on purpose and asserts the real path is
  measurably different — the class of check that output-comparison tests
  structurally cannot provide.
