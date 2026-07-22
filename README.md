# Long-Horizon Reinforcement Learning for Hierarchical LLM Agents in Craftax

> **Current focus:** sample-efficient planning and adaptation with semi-Markov PPO+GAE and a structured action-outcome model.  
> **Planned extension:** long-horizon counterfactual credit assignment once the agent can reliably produce deeper, longer trajectories.

> **Status:** the Craftax environment, 11-skill controller, hierarchical LLM loop, trajectory ledger, replay tooling, and GPU rollout pipeline are implemented. The learning experiments below are planned; this document does not claim completed training results.

## Project overview

This project studies how a hierarchical LLM agent can learn to act over long horizons in Craftax while using interaction data efficiently.

The research is staged:

1. **Phase I — sample-efficient planning and adaptation.** Learn reusable action-outcome structure from earlier Craftax experience and use it to make better decisions at unseen floor frontiers.
2. **Phase II — long-horizon credit assignment.** Use exact simulator interventions to learn which earlier subgoal decisions caused delayed achievements, survival, depth, and return.

The LLM does not press individual game buttons. It selects a structured subgoal such as `mine wood`, `eat`, `craft stone pickaxe`, `fight`, or `descend`. A narrow scripted controller executes that subgoal for a variable number of primitive game steps and records the result in a decision ledger.

```text
observation + recent ledger
           |
           v
      LLM planner
   proposes K subgoals
           |
           v
structured outcome model
predicts reward, risk, progress,
and uncertainty for each subgoal
           |
           v
      candidate reranker
           |
           v
 scripted skill controller
           |
           v
 Craftax transition + ledger entry
```

This converts a very long button-level episode into a shorter sequence of semantically meaningful planner decisions while preserving Craftax's survival, exploration, combat, and crafting challenges.

## Motivation

Model-free reinforcement learning can require many complete trajectories before a useful action receives a reliable learning signal. This is especially costly when an agent encounters a new environment or when rewards arrive long after the decisions that caused them.

A more capable agent should be able to:

- learn reusable action-outcome relationships from prior experience;
- transfer those relationships to unfamiliar states;
- predict the consequences of candidate actions before acting;
- recognize uncertainty and adapt from a small amount of new experience; and
- assign delayed success or failure to the earlier decisions that actually influenced it.

Craftax is a useful controlled testbed. Earlier and deeper floors share broad concepts—survival, tools, resources, enemies, exploration, and ladders—but introduce new states and combinations. The simulator also supports exact state snapshots, allowing alternative subgoals to be executed from the same starting state.

## Current system

Already implemented:

- a text interface for full Craftax, verified against the native observation;
- real fog-of-war;
- an 11-skill scripted controller for navigation, crafting, survival, combat, and floor transitions;
- a hierarchical loop in which a Qwen3-4B planner chooses one subgoal per turn;
- a ledger containing decisions, durations, state changes, rewards, and achievements;
- deterministic replay and rollout logging;
- local GPU serving and rollout collection.

The untrained planner currently:

- earns about seven achievements per game on average;
- often reaches stone or iron tools on the surface;
- compresses roughly 276 primitive actions into about 38 LLM decisions;
- produces valid structured subgoals reliably; and
- does not yet survive and descend consistently.

The first coverage milestone is therefore a **shared survival-and-descent checkpoint** used by every learned system in the comparison.

## Research questions

### Phase I: generalization and sample-efficient planning

> Can an action-conditioned structured outcome model, trained on prior Craftax experience, improve zero-shot and few-shot decisions at unseen floor frontiers and reach deeper states more efficiently than a frozen LLM planner or model-free PPO+GAE?

The study measures:

- **zero-shot generalization:** decisions made before using target-floor training data;
- **few-shot adaptation:** improvement after a small target-floor interaction budget;
- **end-to-end competence:** native reward and maximum floor from the normal game start; and
- **sample efficiency:** performance versus interactions and generated LLM tokens.

### Phase II: long-horizon credit assignment

> After the agent can produce sufficiently long and diverse trajectories, can a branch-supervised counterfactual critic assign delayed credit more accurately and improve policy learning relative to semi-Markov GAE?

Phase II is an extension, not a prerequisite for completing Phase I.

## Phase I systems

The first complete study compares only three systems.

| System | Role |
|---|---|
| **Frozen LLM planner** | Measures pretrained zero-shot behavior without learning. |
| **Semi-Markov PPO + GAE** | Model-free RL baseline over variable-duration subgoals. |
| **LLM + structured outcome-model reranking** | Proposed model-based method. |

Exact simulator branch-and-replay is initially an **evaluation and calibration tool**, not a fourth full training algorithm.

### Model-free baseline: semi-Markov PPO + GAE

Each skill can consume a different number of primitive steps. The PPO baseline therefore uses duration-aware bootstrapping rather than treating every subgoal as one equal-duration step.

For skill duration $\tau_t$, internal primitive rewards $r_{t,j}$, and critic $V(h_t)$:

$$
R_t^{\mathrm{macro}}
=
\sum_{j=0}^{\tau_t-1}\gamma^j r_{t,j},
$$

$$
\delta_t
=
R_t^{\mathrm{macro}}
+
\gamma^{\tau_t}V(h_{t+1})
-
V(h_t).
$$

The baseline computes GAE over the planner-level trajectory and updates the LLM policy with PPO.

### Proposed method: structured outcome-model reranking

At each planner decision, the LLM proposes a small set of diverse valid subgoals in one generation call. A lightweight ensemble predicts planner-relevant outcomes for each candidate, including:

- short-horizon native return;
- skill success;
- new-achievement probability;
- death or survival risk;
- floor-transition probability;
- expected duration; and
- predictive uncertainty.

The first version performs one-step candidate reranking:

$$
a_t
=
\arg\max_{a\in\mathcal A_t}
\left(
\widehat Q_{\mathrm{native}}(h_t,a)
-
\kappa\widehat\sigma(h_t,a)
\right).
$$

The primary score optimizes predicted native return while penalizing uncertainty. Survival, depth, and achievement predictions are auxiliary targets and diagnostics.

Multi-step planning is optional. It is attempted only after one-step reranking works and requires a compact learned next-state dynamics model.

## Generalization protocol

The first target is floor 1.

1. Train on surface trajectories only.
2. Create held-out floor-1 entry snapshots with a scripted, curriculum, or archive policy.
3. Exclude target-floor transitions from zero-shot training.
4. Give control to each evaluated system immediately after floor entry.
5. Measure performance before any floor-1 update: **zero-shot**.
6. Give adaptive systems the same small floor-1 interaction budget and reevaluate on untouched snapshots: **few-shot**.
7. Separately evaluate each integrated agent from the normal surface start.

A later transfer study may train on floors 0–1 and hold out floor 2. Repeating the protocol through all nine floors is not required.

## Exact branch-and-replay evaluation

At a selected saved state, the evaluator executes several candidate subgoals from the exact same simulator and PRNG state, then continues each branch for the same limited horizon with a fixed continuation policy.

This provides a direct action-ranking target under a specified continuation procedure. It does not claim a universal causal effect independent of the policy or horizon.

Phase I reports:

- top-1 action accuracy against branch outcomes;
- pairwise ranking accuracy;
- regret relative to the best tested candidate;
- calibration of predicted return and death risk; and
- performance on seen versus unseen floors.

Branching is concentrated at important states—new floors, ladders, low-health decisions, unfamiliar enemies, and achievement frontiers—to control compute.

## Planned Phase II: long-horizon credit assignment

Phase II begins only after Phase I produces trajectories with enough successful and unsuccessful long-horizon outcomes. Credit assignment cannot learn the consequence of a floor or achievement that never appears in the data.

### Counterfactual credit target

At selected history state $h_t$, execute alternative actions from the same snapshot and continue with a frozen continuation policy $\pi_c$ for horizon $H$:

$$
Q_H^{\mathrm{branch}}(h_t,a)
=
\mathbb E
\left[
G_{t,H}
\mid
\operatorname{do}(A_t=a),\pi_c
\right].
$$

A learned critic $Q_\psi(h,a)$ is trained to predict these branch outcomes. Its centered action credit is

$$
A_t^{\mathrm{CF}}
=
Q_\psi(h_t,a_t)
-
\mathbb E_{a'\sim b(\cdot\mid h_t)}
\left[Q_\psi(h_t,a')\right],
$$

where $b$ is a documented alternative-action baseline over valid candidates.

The first online extension will compare:

| System | Credit signal |
|---|---|
| **PPO+GAE** | Standard duration-aware GAE. |
| **PPO+GAE+CF** | The same PPO pipeline plus a calibrated counterfactual-credit term from the branch-supervised critic. |

The study will avoid a large algorithm bake-off. HCA, COCOA, RUDDER, and RRD remain optional later comparisons.

### Credit-assignment stress tests

The extension can evaluate two reward regimes:

1. **Native Craftax rewards**, where achievements provide intermediate feedback.
2. **Controlled delayed rewards**, where the same total reward is shifted later or bundled at episode end.

This tests whether counterfactual credit becomes more useful as temporal delay increases.

### Phase II evaluation

- agreement with held-out branch action rankings;
- regret and calibration of counterfactual value predictions;
- native reward and maximum floor;
- learning speed versus interactions and LLM tokens;
- sensitivity to reward delay; and
- qualitative analysis of decisions where GAE and counterfactual credit disagree.

The recipe graph may be reported as a secondary structural-relevance diagnostic, but exact branch comparisons are the primary intervention-grounded metric.

## Evaluation and budget

### Frontier evaluation

- native reward during the first $H$ planner decisions;
- survival and catastrophic-action rate;
- new achievements;
- branch-based action-ranking regret;
- zero-shot and few-shot adaptation curves.

### End-to-end evaluation

- total native achievement reward;
- maximum floor reached;
- probability of first descent;
- survival duration;
- generated LLM tokens; and
- primitive environment actions.

### Solo-research reporting

The minimum final Phase I study uses one training run per learned system and paired evaluation on the same held-out world seeds. Bootstrap intervals over evaluation worlds will be reported with an explicit note that they do not measure retraining variance. A second training seed is optional after the complete pipeline works.

Phase II reuses the Phase I agent, snapshots, encoder, branch runner, and logging infrastructure. It should not begin with a new large model or a many-method comparison.

## Scope

### Required for Phase I

1. Shared survival-and-descent coverage checkpoint.
2. Structured transition dataset and held-out-floor protocol.
3. One-step structured outcome model and candidate reranker.
4. Correct semi-Markov PPO+GAE baseline.
5. Targeted branch-and-replay evaluator.
6. One compute-matched end-to-end comparison.

### Optional within Phase I

- online adaptation of a small outcome-model adapter;
- two- or three-subgoal model-predictive planning;
- branch-supervised outcome-model fine-tuning; and
- a second independent training seed.

### Phase II extension

- collect diverse long-horizon trajectories with delayed outcomes;
- create a fixed branch-comparison credit dataset;
- train a branch-supervised counterfactual critic;
- compare PPO+GAE with PPO+GAE+CF under native and delayed rewards; and
- publish an intervention-grounded credit-quality analysis.

### Out of scope unless the earlier stages succeed

- a pixel-level Dreamer or MuZero implementation;
- seven-way credit-assignment comparisons;
- exhaustive hyperparameter sweeps;
- full HCA, COCOA, RUDDER, or RRD implementations;
- claims that the agent can infer arbitrary unseen rules without evidence; and
- state-of-the-art Craftax performance.

## Roadmap

### Phase I — planning and generalization

- [x] Craftax text environment and verification
- [x] 11-skill controller
- [x] hierarchical Qwen planner and ledger
- [x] replayable rollout pipeline
- [ ] shared survival-and-descent checkpoint
- [ ] held-out frontier snapshot dataset
- [ ] structured outcome model
- [ ] multi-candidate LLM interface and reranker
- [ ] branch-based action-ranking evaluator
- [ ] semi-Markov PPO+GAE baseline
- [ ] zero-shot and few-shot frontier evaluation
- [ ] end-to-end compute-matched comparison
- [ ] Phase I report, plots, and reproducibility instructions

### Phase II — optional long-horizon credit assignment

- [ ] define the continuation policy, branch horizon, and alternative-action baseline
- [ ] collect deeper successful and unsuccessful trajectories
- [ ] build a held-out branch-comparison credit dataset
- [ ] train and calibrate the counterfactual critic
- [ ] integrate the credit term into the PPO pipeline
- [ ] run native-reward and controlled-delay comparisons
- [ ] publish credit-quality and policy-learning results

## What would count as success?

Phase I succeeds if it shows a clear, reproducible capability gain such as:

- lower branch regret on unseen frontier states;
- higher first-visit survival;
- faster adaptation after limited target-floor experience;
- more successful descents or greater maximum depth under the same budget; or
- a useful failure analysis showing when the learned model should not be trusted.

Phase II succeeds if the branch-supervised critic predicts held-out action-replacement effects and either improves long-horizon policy learning or clearly identifies conditions under which standard GAE is already sufficient.

The complete research-engineering story is:

```text
instrument a long-horizon environment
→ implement a correct model-free RL baseline
→ learn reusable action-outcome structure
→ plan and adapt under uncertainty
→ collect deeper and more diverse trajectories
→ learn intervention-grounded delayed credit
→ test whether better credit improves policy learning
```

Detailed engineering decisions, schemas, tests, budget gates, and the Phase II extension contract are documented in `docs/CRAFTAX_INTERNAL_IMPLEMENTATION_PLAN.md`.
