# Long-horizon Credit Assignment for a Hierarchical LLM Agent in Craftax

## 1. Project objective

This project studies one focused question:

> Can counterfactual action-replacement signals assign better credit than GAE to long-horizon subgoal decisions made by an LLM planner in Craftax?

The agent is hierarchical. An LLM selects a structured subgoal, and an 11-skill scripted controller executes it for a variable number of primitive game actions. Each planner decision, execution result, duration, reward, achievement, and state change is stored in a ledger.

The project deliberately compares only:

1. **Primary baseline:** semi-Markov PPO with duration-aware GAE.
2. **Main method:** counterfactual branch-and-replay.
3. **Optional extension:** a small rule-blind critic trained to predict branch-and-replay effects.

The recipe graph is used only for analysis. It is not given to the baseline or main learned method.

---

## 2. Current status

Already implemented:

- verified text interface for full Craftax;
- 11 narrow scripted skills;
- hierarchical LLM-planner loop;
- decision ledger and rollout logging;
- prerequisite-graph analysis;
- local GPU inference and rollout collection.

The current untrained planner can craft surface tools but does not reliably survive and descend. Therefore, the immediate bottleneck is **trajectory coverage**, not credit assignment.

---

## 3. Scope

### In scope

- teach one shared initialization to survive and descend sometimes;
- implement and validate duration-aware PPO+GAE;
- collect counterfactual action-replacement data from exact simulator snapshots;
- compare GAE credit with measured counterfactual effects;
- optionally distill those effects into a lightweight critic;
- report a small, honest, compute-matched study.

### Out of scope for the first version

- seven-way algorithm comparisons;
- exhaustive hyperparameter sweeps;
- large ablation matrices;
- pixel-level world models or MuZero-style search;
- full implementations of HCA, COCOA, RUDDER, and RRD;
- claims of state-of-the-art Craftax performance;
- strong statistical claims from many training seeds.

HCA, COCOA, RUDDER, RRD, and learned world models remain possible follow-up work after the core project is complete.

---

## 4. Stage 0: obtain useful trajectory coverage

No credit method can learn that descending is useful if the data never contains a state where descent is attempted or succeeds.

Create one shared **coverage checkpoint** before comparing credit methods. Possible low-cost tools are:

- a small supervised dataset of manually written or scripted subgoal trajectories;
- a curriculum that first rewards survival, ladder discovery, and one successful descent;
- occasional forced exploration or forced `descend` when the action is valid;
- reset snapshots near a ladder or immediately after descent;
- a mixture of LLM trajectories and scripted successful trajectories.

These mechanisms are allowed to use privileged information for data collection, provided that:

1. the resulting initialization is shared by all methods;
2. the main credit models do not receive the recipe graph;
3. performance is also evaluated from normal episode starts.

### Exit criterion

Proceed to the credit-assignment study once the shared checkpoint produces repeatable nonzero coverage, for example:

- at least 10 successful descents across a fixed evaluation suite; and
- trajectories containing both successful and failed survival/descent attempts.

The exact threshold may be adjusted to fit compute, but it should be fixed before comparing methods.

---

## 5. Primary baseline: semi-Markov PPO with GAE

Planner actions have different durations. Let:

- $h_t$ be the planner history or state representation;
- $a_t$ be the selected subgoal;
- $\tau_t$ be the number of primitive game steps consumed by that subgoal;
- $r_{t,j}$ be the primitive reward at internal step $j$.

The discounted reward collected within one planner action is

$$
R_t^{\mathrm{macro}}
=
\sum_{j=0}^{\tau_t-1}\gamma^j r_{t,j}.
$$

The duration-aware TD residual is

$$
\delta_t
=
R_t^{\mathrm{macro}}
+
\gamma^{\tau_t}V(h_{t+1})
-
V(h_t).
$$

Use the recursive GAE estimate

$$
\hat A_t^{\mathrm{GAE}}
=
\delta_t
+
\gamma^{\tau_t}\lambda\hat A_{t+1}^{\mathrm{GAE}}.
$$

The baseline uses these advantages in the ordinary PPO objective.

### Required work

- implement duration-aware rewards and bootstrapping;
- distinguish true termination from rollout truncation;
- train a value critic over the ledger state;
- apply policy gradients to the structured subgoal and argument tokens;
- add unit tests using hand-constructed trajectories;
- log return, achievements, survival, floor depth, value loss, KL, entropy, and clip fraction.

### What is not required

Do not run a full $\gamma$ or $\lambda$ sweep. Select one reasonable configuration through a small pilot, document it, and freeze it before the final comparison.

Do not run all proposed PPO ablations. Correctness tests are mandatory; large empirical ablations are optional.

---

## 6. Main method: counterfactual branch-and-replay

At a selected planner state $h_t$:

1. save the exact Craftax state and random-number-generator state;
2. replay the factual action $a_t$;
3. replace it with two or three valid alternatives;
4. continue every branch with the same frozen continuation policy;
5. use the same continuation horizon and paired randomness;
6. compare future achievements, survival, floor depth, and discounted return.

For candidate action $a$:

$$
\hat Q_{\mathrm{branch}}(h_t,a)
=
\frac{1}{K}\sum_{k=1}^{K}G^{(k)}(h_t,a).
$$

The counterfactual advantage is

$$
\hat A_t^{\mathrm{CF}}
=
\hat Q_{\mathrm{branch}}(h_t,a_t)
-
\frac{1}{|\mathcal A_t'|}
\sum_{a'\in\mathcal A_t'}
\hat Q_{\mathrm{branch}}(h_t,a').
$$

This measures the effect of replacing the selected action under a stated continuation policy. It is more direct than inferring credit only from temporal proximity.

### Keep branching affordable

- branch only a stratified subset of states;
- prioritize states near achievements, survival failures, ladders, and large GAE magnitudes;
- compare only two or three plausible alternatives;
- use a short or medium continuation horizon first;
- use deterministic or low-temperature continuation during measurement;
- reuse every branch result for evaluation and model training.

### First scientific result

Before using branch credit for policy training, measure:

- correlation between GAE and branch advantages;
- sign agreement;
- pairwise action-ranking accuracy;
- examples where GAE and intervention effects disagree;
- whether disagreements concentrate at long delays, distractor actions, or critic errors.

This analysis alone is a useful benchmark contribution.

---

## 7. Optional main extension: rule-blind counterfactual critic

If the branch-and-replay benchmark works, train a lightweight model

$$
Q_\psi(h_t,a)
$$

to predict branch returns or branch outcome vectors from the ledger state and candidate action.

The model must not receive the recipe graph. It may predict:

- future achievement probabilities;
- survival probability;
- floor depth;
- discounted return;
- uncertainty.

Compute its advantage as

$$
\hat A_\psi(h_t,a_t)
=
Q_\psi(h_t,a_t)
-
\frac{1}{|\mathcal A_t'|}
\sum_{a'\in\mathcal A_t'}Q_\psi(h_t,a').
$$

Use it in either:

- PPO as a replacement or supplement for GAE; or
- an offline pairwise ranking loss that makes the planner prefer actions with larger predicted branch return.

This is the strongest resume milestone because it turns expensive simulator interventions into a learned, deployable credit model.

---

## 8. Minimal experiment matrix

Only the following conditions are required:

| Condition | Purpose |
|---|---|
| Frozen coverage checkpoint | Free reference; no additional algorithm implementation |
| SMDP PPO + GAE | Primary RL baseline |
| Counterfactual method | Main contribution; direct branch credit or learned counterfactual critic |

One optional diagnostic condition may be added only if compute remains:

- GAE without duration correction; or
- counterfactual critic without branch supervision.

Do not implement Monte Carlo, potential shaping, HCA, COCOA, RUDDER, RRD, and a world model in the first version.

---

## 9. Compute-conscious evaluation

### Development

- use one training seed while debugging;
- run short jobs and stop early when diagnostics are broken;
- evaluate all checkpoints on the same fixed set of environment seeds;
- reuse stored rollouts for critic and credit analysis.

### Final comparison

Minimum affordable design:

- one final training run for GAE;
- one final training run for the counterfactual method;
- identical initialization and compute budget;
- paired evaluation on many fixed environment seeds;
- bootstrap confidence intervals across evaluation episodes.

Preferred design, only if affordable:

- repeat the headline comparison with a second independent training seed.

A single training seed is acceptable for a clearly labeled portfolio pilot, but it is not enough for a strong general scientific claim. Report this limitation directly.

### Primary metrics

- achievements per episode;
- probability of first descent;
- deepest floor reached;
- survival time;
- reward per generated token;
- learning curve under a matched token budget;
- agreement with measured branch effects.

Avoid a large metric suite in the public README. Put additional diagnostics in the experiment report.

---

## 10. Success criteria

The project is complete when it delivers:

1. a correct, stable SMDP PPO+GAE implementation;
2. a shared policy checkpoint with repeatable descent coverage;
3. a reproducible branch-and-replay dataset;
4. a comparison between GAE credit and measured action-replacement effects;
5. one compute-matched policy-learning comparison;
6. a concise report containing quantitative results and failure cases.

The result is still valuable if the counterfactual method does not beat GAE, provided the evaluation is correct and the failure is analyzed.

---

## 11. Priority order

| Priority | Work item | Status in first version |
|---:|---|---|
| **P0** | Coverage checkpoint for survival and descent | Required prerequisite |
| **P1** | SMDP PPO + GAE | Required baseline |
| **P2** | Counterfactual branch-and-replay benchmark | Required main contribution |
| **P3** | Rule-blind counterfactual critic | Recommended extension |
| **P4** | HCA/COCOA or RUDDER/RRD comparison | Future work |
| **P5** | Learned world model and imagined counterfactuals | Future work |

---

## 12. Public presentation

The GitHub landing page should be short enough to skim in roughly one or two minutes. It should contain:

- the problem in one paragraph;
- one architecture diagram;
- current system status;
- the GAE baseline;
- the branch-and-replay idea;
- one result table or figure;
- reproduction commands;
- a link to a longer technical report.

Keep detailed derivations, implementation notes, failed experiments, and auxiliary metrics in `docs/RESEARCH_NOTES.md` or a final report rather than the main README.

### Resume-ready completion point

Do not wait to implement every method in the credit-assignment literature. A strong project consists of:

> a reliable hierarchical agent system, a correct SMDP PPO+GAE baseline, a simulator-grounded counterfactual benchmark, and one learned critic that distills intervention effects.

That combination demonstrates agent infrastructure, RL implementation, experimental design, and an original long-horizon credit-assignment contribution.
