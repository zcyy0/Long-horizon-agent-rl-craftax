# Long-Horizon Causal Credit Assignment for Hierarchical LLM Agents in Craftax

## 0. Executive summary

This project studies **how to assign credit to earlier subgoal decisions in long-horizon LLM-agent trajectories**.

The agent is hierarchical:

1. An LLM planner reads the current observation and decision history.
2. The planner selects one structured subgoal from an 11-skill library.
3. A narrow scripted controller executes that subgoal for a variable number of primitive Craftax actions.
4. The result, rewards, achievements, duration, and state changes are written to a ledger.
5. Reinforcement learning updates the planner from the ledger-level trajectory.

The project will compare three increasingly strong notions of credit:

1. **Temporal credit:** propagate observed rewards backward using returns or TD errors.
2. **Mechanistic relevance:** determine whether a decision lies on a realized resource/achievement production path.
3. **Interventional causal credit:** estimate what would have happened if the planner had chosen a different subgoal at the same state.

The **primary baseline** is:

> **Semi-Markov PPO with duration-corrected GAE, using no recipe graph or privileged game rules.**

The **main proposed method** is:

> **Counterfactual Branch-and-Replay:** snapshot a Craftax state, replace one planner decision with valid alternatives, continue under a fixed policy, and measure the effect on achievements and return.

The main generalization method is:

> **A rule-blind counterfactual critic trained to predict intervention effects from ordinary trajectories plus a limited set of branch-and-replay labels.**

The game’s recipe graph is used as an **oracle evaluator and shaping control**, not as an input to the main learned method.

---

## 1. Motivation

Long-horizon agents receive rewards after many decisions, often after irrelevant or redundant actions have occurred. A policy-gradient learner therefore needs an estimate of which earlier decisions increased or decreased expected future return.

Craftax is useful because:

- episodes are long and contain survival, exploration, combat, and a deep crafting tree;
- achievements provide sparse but meaningful intermediate rewards;
- the environment is a simulator, so exact state snapshots and counterfactual replays are possible;
- the recipe graph provides structural knowledge that can be reserved for evaluation;
- rules can be hidden from the learner to study unknown-dynamics settings.

The hierarchy makes the problem tractable. Instead of asking an LLM to optimize approximately 100,000 primitive button presses, the planner makes a few hundred semantically meaningful decisions such as:

- `mine wood`
- `craft stone pickaxe`
- `eat`
- `explore`
- `descend`
- `fight`

This creates a natural long-horizon agent-RL benchmark at the subgoal level.

---

## 2. Research questions

### RQ1 — Temporal credit

How strong is a correctly implemented semi-Markov GAE baseline at the current planner-level reward density and horizon?

### RQ2 — Reward redistribution

Do RUDDER or Randomized Return Decomposition assign better credit than GAE when rewards are delayed, bundled, or separated by distractor actions?

### RQ3 — Causal credit

Does simulator-based counterfactual branching produce credit estimates that are more faithful to action interventions than temporal or correlational methods?

### RQ4 — Rule-blind generalization

Can a learned counterfactual critic infer action–outcome structure without receiving Craftax recipes or prerequisite rules?

### RQ5 — Learning relevance

Does early credit quality predict later sample efficiency, achievement count, survival, and dungeon depth?

### RQ6 — Transfer beyond Craftax

Which parts of the method require a resettable simulator, and which can be transferred to settings with only logged trajectories and limited real-world interventions?

---

## 3. Falsifiable hypotheses

### H1 — GAE is a competitive native-reward baseline

Because current achievement rewards occur at moderate planner-level delays, duration-corrected GAE may match or outperform more complex methods under native rewards.

### H2 — Causal methods matter more as the horizon becomes harder

Counterfactual and reward-redistribution methods will show larger gains under delayed, episode-bundled, distractor-heavy, and deeper-floor regimes.

### H3 — Static recipe labels are useful but incomplete

Recipe-graph labels will identify many relevant crafting decisions, but they will miss survival, exploration, information-gathering, redundancy, and alternative causal paths.

### H4 — Branch-and-replay gives a stronger target

Credit scores that better predict measured intervention effects will yield more efficient policy improvement than scores that only reconstruct returns or correlate with prerequisites.

### H5 — A rule-blind critic can distill interventions

A learned achievement-conditioned critic trained from factual trajectories and sparse branch labels will approximate branch-and-replay effects and generalize to unbranched states.

### H6 — Credit quality predicts learning

Methods with better early intervention agreement, calibration, and top-$k$ decision ranking will later achieve better reward-per-token and deeper progression.

Failure of any hypothesis is still a useful result.

---

## 4. Scope, terminology, and claims

### 4.1 Temporal credit

A score based on when rewards and TD errors occur. Examples:

- episode return;
- reward-to-go;
- GAE;
- RUDDER;
- RRD.

Temporal credit can be useful without identifying a causal mechanism.

### 4.2 Structural relevance

A decision is structurally relevant when its resource or achievement type is on a known prerequisite path.

The existing recipe graph should be described as providing:

> **Recipe-structural relevance labels**

rather than complete ground-truth causal labels.

### 4.3 Event provenance

A stronger mechanistic label records the exact realized chain:

$$
\text{planner decision}
\rightarrow
\text{resource unit acquired}
\rightarrow
\text{resource consumed}
\rightarrow
\text{item crafted}
\rightarrow
\text{achievement}.
$$

This distinguishes a necessary acquisition from redundant surplus collection.

### 4.4 Interventional causal credit

The primary causal estimand is the effect of replacing one planner action while fixing the state and continuation policy.

For history $h_t$, factual action $a_t$, alternative-action distribution $b$, and continuation policy $\pi$:

$$
A_{\text{CF}}^\pi(h_t,a_t) = \mathbb{E}[G \mid do(A_t=a_t), h_t, \pi \text{ thereafter}] - \mathbb{E}_{a' \sim b} \mathbb{E}[G \mid do(A_t=a'), h_t, \pi \text{ thereafter}].
$$

This is a **policy-continuation causal advantage**. It is not a universal statement that an action was necessary under every possible future policy.

### 4.5 Non-goals

This project does not initially attempt to:

- compare the hierarchy against a flat 100,000-step LLM policy;
- infer a unique complete causal graph from passive trajectories alone;
- claim that return decomposition is automatically causal;
- build a full pixel-level MuZero system;
- rely on the recipe graph in the main rule-blind learner;
- prove that one scalar is the uniquely correct notion of credit.

---
## 5. Formal problem formulation

At planner decision $t$:

- $h_t$: observation plus relevant ledger history or recurrent state;
- $a_t$: structured planner subgoal;
- $\tau_t$: number of primitive Craftax actions consumed by the skill;
- $r_{t,j}$: primitive reward at internal skill step $j$;
- $h_{t+1}$: next planner history after skill execution.

The discounted macro reward is

$$
R_t^{\text{macro}} = \sum_{j=0}^{\tau_t-1}\gamma^j r_{t,j}.
$$

The planner therefore operates in a semi-Markov decision process.

The policy objective is

$$
J(\theta) = \mathbb{E}_{\pi_\theta} \left[\sum_t \gamma^{\sum_{i<t}\tau_i} R_t^{\text{macro}} \right].
$$

The critic estimates

$$
V_\phi(h_t) \approx \mathbb{E}[G_t \mid h_t].
$$

The rule-blind learner receives observations, history, actions, state changes, durations, and rewards, but not the recipe graph.

---

## 6. Shared agent and training architecture

All credit methods use the same architecture and training budget unless a method intrinsically requires an auxiliary model.

### 6.1 Planner policy

- Base model: Qwen3-4B or another fixed open model.
- Fine-tuning: LoRA or equivalent parameter-efficient adaptation.
- Output: constrained structured subgoal name and arguments.
- Optional natural-language reasoning is logged but is not treated as an independent environment action.

### 6.2 Policy-gradient token mask

The primary policy loss is applied to:

- subgoal-name tokens;
- argument tokens;
- any explicit action-selection tokens.

Free-form explanation or `think` tokens are excluded from the primary RL loss unless a dedicated ablation tests otherwise.

This avoids assigning reward to response verbosity or wording instead of the actual environment decision.

### 6.3 Critic

The critic consumes:

- current observation;
- inventory and equipment;
- health, food, drink, and energy;
- achievement flags;
- floor and map summary;
- recent subgoals and outcomes;
- skill status and duration;
- recurrent or transformer memory over the ledger.

Possible implementations:

- a value head on the planner representation;
- a separate lightweight transformer;
- a recurrent sequence critic.

The initial study should keep the critic architecture fixed across credit methods.

### 6.4 Initialization

A short supervised warm-up may teach:

- valid skill names;
- argument schemas;
- survival basics;
- use of all skills.

The same warm-up checkpoint must initialize every RL condition.

Because output syntax is already valid, the warm-up should not inject unnecessary recipe-graph knowledge into the rule-blind study.

### 6.5 Rollout and optimizer controls

Hold constant:

- initial checkpoint;
- policy architecture;
- critic architecture;
- optimizer;
- PPO clipping;
- batch size;
- rollout count;
- prompt/scaffold;
- generation parameters;
- curriculum;
- token budget;
- environment-action budget;
- seed set.

---

## 7. Baselines

## 7.1 Primary baseline: SMDP PPO + GAE

This is the main baseline that every proposed method must beat.

The duration-corrected TD residual is

$$
\delta_t = R_t^{\text{macro}} + \gamma^{\tau_t} V_{\phi_{\text{old}}}(h_{t+1}) - V_{\phi_{\text{old}}}(h_t).
$$

Using $\lambda$ per planner transition:

$$
\hat A_t^{\text{SMDP-GAE}}
=
\delta_t
+
\gamma^{\tau_t}\lambda
\hat A_{t+1}^{\text{SMDP-GAE}}.
$$

An ablation may instead decay $\lambda$ per primitive step:

$$
\hat A_t
=
\delta_t
+
(\gamma\lambda)^{\tau_t}
\hat A_{t+1}.
$$

The PPO objective is

$$
L_{\text{PPO}}
=
\mathbb{E}_t
\left[
\min
\left(
\rho_t\hat A_t,\,
\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)\hat A_t
\right)
\right].
$$

Required implementation details:

- bootstrap across rollout truncations;
- set bootstrap value to zero only after a true terminal transition;
- compute GAE from old, detached value predictions;
- train the critic toward $V_{\text{old}}+\hat A$;
- normalize advantages within an appropriate batch;
- log value error, explained variance, policy KL, entropy, and clip fraction;
- account for skill duration in reward accumulation and bootstrapping.

> **Baseline statement:** The headline comparison is against rule-blind SMDP PPO+GAE with the same policy, critic, rollout data, exploration support, and generated-token budget.

## 8.2 Lower-bound and control baselines

### B0 — No-RL planner

The frozen untrained or supervised planner establishes the starting capability.

### B1 — Episode-level centered return

Every planner decision receives the same centered episode score.

This should be named **episode-level centered return**, not GRPO, unless the complete group-relative sampling and optimization procedure is implemented.

### B2 — Monte Carlo reward-to-go

Each decision receives the discounted rewards observed after it, without value bootstrapping.

### B3 — Duration-incorrect GAE ablation

Treat every skill as one equal-length step. This demonstrates the importance of the SMDP correction.

### B4 — Oracle potential-based shaping

Use the known recipe graph to construct a potential $\Phi(h)$.

For a variable-duration macro action:

$$
F(h_t,a_t,h_{t+1})
=
\gamma^{\tau_t}\Phi(h_{t+1})-\Phi(h_t).
$$

This is an expert-knowledge control, not the main method.

### B5 — Random or heuristic credit

Useful for validating that the credit-quality metrics do not report improvement for arbitrary scores.

---

## 9. Algorithms ranked by implementation priority

The ranking below balances scientific value, engineering risk, and portfolio value.

| Priority | Method | Role | Why it is prioritized |
|---:|---|---|---|
| **P0** | **SMDP PPO + GAE** | Required primary baseline | Establishes correct end-to-end RL, variable-duration credit, critic learning, and reproducible training |
| **P1** | **Counterfactual Branch-and-Replay** | Main causal oracle and direct credit method | Uses the simulator to measure intervention effects rather than infer them only from temporal correlation |
| **P2** | **Rule-Blind Counterfactual Critic** | Main generalization algorithm | Distills sparse intervention labels into a deployable model that can assign credit without recipes or branching every state |
| **P3** | **Achievement-Conditioned COCOA/HCA** | Strong causal-credit comparator | Directly models action contribution to identified rewarding outcomes |
| **P4** | **Achievement-Conditioned RUDDER/RRD** | Strong reward-redistribution comparator | Tests learned temporal decomposition without claiming full causal identification |
| **P5** | **Learned Latent World Model + Counterfactual Imagination** | Ambitious unknown-rules extension | Replaces the true simulator with learned dynamics for settings where counterfactual replay is unavailable |
| **P6** | **Goal-Conditioned HER / Frontier Curriculum** | Exploration support, not a credit method | Ensures all methods receive trajectories involving survival, descent, and deeper achievements |

---

## 10. P0 — SMDP PPO + GAE

### Objective

Build a trustworthy, reproducible primary baseline before introducing new credit models.

### Required components

- duration-aware rollout storage;
- macro reward accumulation;
- recurrent or history-conditioned critic;
- PPO update on structured action tokens;
- value-loss clipping or another documented stable value update;
- training and evaluation seed separation;
- parallel rollout collection;
- checkpoint evaluation on held-out worlds.

### Required ablations

- $\lambda \in \{0, 0.9, 0.95, 0.99, 1\}$, subject to compute;
- duration-aware vs duration-ignorant GAE;
- current observation vs ledger-memory critic;
- action-token-only vs all-generated-token policy loss;
- native reward vs delayed-reward regimes.

### Exit criteria

- unit tests verify return and GAE calculations on hand-constructed trajectories;
- terminal and truncation handling are correct;
- policy updates are numerically stable;
- learning curves are reported over multiple seeds;
- the run is reproducible from a configuration file and checkpoint;
- the baseline improves at least one meaningful behavior or provides a clear, diagnosed null result.

---

## 11. P1 — Counterfactual Branch-and-Replay

### 11.1 Working definition

At selected planner states:

1. Save the complete Craftax state.
2. Save every relevant random-number-generator state.
3. Record the factual subgoal.
4. Enumerate or sample valid alternative subgoals.
5. Execute each first action from the identical snapshot.
6. Continue with a frozen continuation policy.
7. Use common random numbers or paired seeds when possible.
8. Compare achievement outcomes, survival, floor depth, and discounted return.

For candidate action $a$:

$$
\hat Q_{\text{branch}}(h_t,a)
=
\frac{1}{K}
\sum_{k=1}^{K}
G^{(k)}(h_t,a).
$$

The branch-based advantage is

$$
\hat A_t^{\text{branch}}
=
\hat Q_{\text{branch}}(h_t,a_t)
-
\sum_{a'} b(a' \mid h_t)
\hat Q_{\text{branch}}(h_t,a').
$$

### 11.2 Outcome vector

Do not predict only total return. Record:

$$
\mathbf{Y}
=
[
Y_1,\ldots,Y_{67},
Y_{\text{survival}},
Y_{\text{depth}},
Y_{\text{health}},
Y_{\text{resource changes}}
].
$$

This supports per-achievement causal effects and makes the result interpretable.

### 11.3 Alternative-action distribution

Report results for at least two baselines:

- **policy baseline:** $a' \sim \pi(\cdot \mid h_t)$;
- **valid-action baseline:** uniform or stratified over valid alternative skills.

The chosen baseline changes the meaning of the causal advantage and must be stated.

### 11.4 State sampling

Branching every state is unnecessary. Stratify states by:

- distance to achievement;
- floor;
- survival risk;
- reward gap;
- critic uncertainty;
- disagreement between GAE, RRD, and structural labels;
- occurrence of apparently irrelevant distractor actions.

### 11.5 Uses

Branch-and-replay serves three purposes:

1. **Evaluation oracle:** compare credit scores with measured effects.
2. **Direct training signal:** update the policy from branch advantages.
3. **Supervision:** train the rule-blind counterfactual critic.

### 11.6 Important validity controls

- frozen continuation policy during each comparison;
- paired environment randomness;
- repeated continuations for stochastic LLM decisions;
- identical maximum continuation horizon;
- action-validity masking;
- sensitivity analysis over continuation policy and alternative-action baseline.

### Exit criteria

- exact snapshot replay is deterministic before branching;
- paired branch estimates have confidence intervals;
- measured effects pass sanity tests on known prerequisite and survival scenarios;
- graph relevance, event provenance, and intervention effects are reported separately;
- branch labels are stored as a reusable benchmark dataset.

---

## 12. P2 — Rule-Blind Counterfactual Critic

### 12.1 Goal

Approximate branch-and-replay effects without giving the model the recipe graph and without branching every state.

### 12.2 Model

Learn an achievement-conditioned action-value model:

$$
Q_\psi(h_t,a,k)
\approx
P(\text{achievement }k\text{ occurs within horizon }H
\mid h_t, do(A_t=a), \pi).
$$

Also predict:

- discounted scalar return;
- death or survival;
- floor depth;
- resource changes;
- skill duration;
- uncertainty.

The estimated causal advantage for achievement $k$ is

$$
\hat A_{\psi,k}(h_t,a_t)
=
Q_\psi(h_t,a_t,k)
-
\mathbb{E}_{a'\sim b}
Q_\psi(h_t,a',k).
$$

A scalar training advantage can be formed as

$$
\hat A_\psi(h_t,a_t)
=
\sum_k w_k\hat A_{\psi,k}
+
w_{\text{survival}}\hat A_{\psi,\text{survival}}
+
w_{\text{depth}}\hat A_{\psi,\text{depth}}.
$$

### 12.3 Training data

Use:

- ordinary factual rollouts;
- sparse branch-and-replay labels;
- unsuccessful and successful trajectories;
- alternative actions with known validity;
- held-out environment seeds;
- explicit action propensities from the behavior policy.

### 12.4 Uncertainty and active branching

Use an ensemble or calibrated uncertainty estimator.

Spend new branch budget on states with:

- high uncertainty;
- high predicted action impact;
- disagreement between ensemble members;
- disagreement with GAE or reward-decomposition methods;
- insufficient action overlap.

### 12.5 Policy update

Possible update rules:

- PPO with the learned counterfactual advantage;
- advantage-weighted regression;
- pairwise ranking of candidate subgoals;
- a hybrid of GAE and counterfactual advantage.

The first implementation should keep PPO fixed and replace only the advantage target.

### 12.6 Generalization claim

The critic is **rule-blind**, not necessarily assumption-free. It learns from action interventions and trajectory data. Its transfer to the real world depends on:

- adequate state/history representation;
- action coverage;
- limited hidden confounding;
- model calibration under intervention;
- environment stationarity.

### Exit criteria

- predicts branch effects on held-out states and worlds;
- beats GAE, structural relevance, and return-only predictors on intervention ranking;
- is calibrated by effect magnitude;
- improves policy learning or produces a clearly analyzed negative result;
- does not receive recipe features.

---

## 13. P3 — Achievement-Conditioned COCOA/HCA

### Adaptation

Treat each achievement identity as a rewarding outcome $k$.

Learn how much an earlier action changed the probability of that outcome rather than conditioning on the complete future state.

### Why it is useful

- it addresses a counterfactual question;
- it naturally uses semantic achievement identities;
- it can reduce interference between unrelated rewards;
- it provides a principled comparison to the branch-based method.

### Evaluation

Compare COCOA/HCA scores with:

- branch effects;
- event provenance;
- recipe relevance;
- actual policy improvement.

### Risk

The original assumptions and estimator may not transfer directly to a partially observed, variable-duration LLM planner. Document every adaptation and keep the SMDP timing explicit.

---

## 14. P4 — Achievement-Conditioned RUDDER and RRD

### 14.1 RUDDER

Train a sequence model to predict future return or achievement outcomes from the ledger. Attribute changes in predicted return to earlier decisions and redistribute reward accordingly.

### 14.2 Randomized Return Decomposition

Learn per-decision proxy rewards whose sampled sum reconstructs trajectory-level outcomes.

### 14.3 Achievement conditioning

Prefer separate or factorized decomposition:

$$
\tilde r_t
=
\sum_{k=1}^{67}\tilde r_{t,k}.
$$

This provides an interpretable answer to:

- which decision contributed to collecting stone;
- which contributed to crafting a pickaxe;
- which contributed to survival;
- which contributed to dungeon entry.

### 14.4 Causal limitation

A decomposition that reconstructs total return is not automatically an intervention effect. Evaluate it against branch labels rather than treating its output as causal ground truth.

### Exit criteria

- reconstructed returns are accurate on held-out trajectories;
- reward redistribution is stable across seeds;
- per-achievement contributions are interpretable;
- comparison includes both native and delayed-reward regimes;
- intervention agreement is reported separately from return reconstruction.

---

## 15. P5 — Learned latent world model and counterfactual imagination

### Objective

Replace exact simulator branching with a learned action-conditioned dynamics model.

### Model components

$$
z_t = e_\psi(h_t)
$$

$$
(z_{t+1}, \hat R_t, \hat \tau_t)
=
f_\psi(z_t,a_t)
$$

$$
(\hat V_t,\hat{\mathbf Y}_t)
=
g_\psi(z_t)
$$

The model predicts:

- next latent planner state;
- macro reward;
- duration;
- achievement outcomes;
- survival;
- value.

### Use

For each candidate action, roll out imagined continuations and estimate:

$$
\hat Q_{\text{model}}(h_t,a)
=
\mathbb{E}_{\hat p_\psi,\pi}[G \mid h_t,a].
$$

Then compute a counterfactual action contrast.

### Validation

The true simulator remains available during research, so model-generated effects can be checked against branch-and-replay effects.

### Scope control

Start with the structured ledger rather than raw pixels. Full tree search or MuZero-style planning is an extension, not a requirement for the first paper-quality result.

---

## 16. P6 — Goal-conditioned learning and frontier curriculum

This is an exploration mechanism, not a credit-assignment method.

### Motivation

The current planner never descends, so no credit method can learn the causal value of descent from the existing support.

### Goal representation

Condition behavior on targets such as:

- survive 500 primitive steps;
- use food, water, and sleep before critical thresholds;
- find a ladder;
- descend once;
- unlock the next unseen achievement;
- reach a specified floor;
- obtain a specified resource or tool.

### Data reuse

Relabel trajectories with goals they actually achieved, following the logic of hindsight experience replay.

### Fairness requirement

Use the same curriculum and goal distribution for all online credit-method comparisons.

---

## 17. Evaluation framework

The evaluation has three layers.

## 17.1 Layer A — Agent performance

Primary metrics:

- achievements per episode;
- unique achievements across a run;
- deepest floor reached;
- frontier achievement reached;
- survival time;
- reward per generated token;
- reward per primitive environment action;
- reward per planner decision;
- success rate for survival and descent;
- area under the learning curve.

Compute metrics:

- generated and trained tokens;
- environment steps;
- planner decisions;
- GPU hours;
- wall-clock time;
- auxiliary-model compute.

## 17.2 Layer B — Credit quality

### Structural relevance

- ROC-AUC;
- average precision;
- precision at $k$;
- mean rank of relevant decisions.

### Event provenance

- precision and recall for realized resource ancestry;
- identification of redundant resource collection;
- attribution to survival and exploration decisions;
- attribution completeness for each achievement.

### Interventional agreement

- Spearman rank correlation with branch effects;
- pairwise action-ranking accuracy;
- calibration error by effect magnitude;
- top-$k$ decision replacement test;
- sign accuracy;
- mean squared error where scales are comparable;
- effect confidence interval coverage.

### Faithfulness tests

- remove or replace top-credited decisions and measure outcome degradation;
- replace low-credited decisions and measure whether outcomes remain stable;
- compare predicted and actual effect of alternative subgoals.

## 17.3 Layer C — Generalization

- held-out Craftax worlds;
- held-out random seeds;
- held-out achievement types where feasible;
- deeper floors than those used for early training;
- changed reward timing;
- unknown-rule condition;
- optional hidden or altered recipe variants;
- different continuation policies.

---

## 18. Experimental protocols

## 18.1 Protocol A — Fixed-data credit benchmark

Purpose: isolate credit estimation from exploration and policy-induced data differences.

### Dataset

Build one shared dataset containing:

- untrained-planner rollouts;
- supervised-planner rollouts;
- scripted or curriculum-assisted descent examples;
- successful and failed survival trajectories;
- surface and dungeon achievements;
- distractor actions;
- redundant resource collection;
- alternative paths to the same achievement;
- branch-and-replay labels on a stratified subset.

### Comparison

Train every credit estimator on the same trajectories.

Evaluate only credit quality and held-out prediction.

No method may collect its own additional online data in this protocol.

## 18.2 Protocol B — Online matched-budget learning

Purpose: measure end-to-end policy improvement.

Controls:

- same initial checkpoint;
- same curriculum;
- same rollout workers;
- same environment seeds for evaluation;
- same token budget;
- same primitive-action budget;
- same optimizer and PPO settings;
- same number of update epochs;
- same evaluation cadence.

Each method may generate its own training data after initialization, but the exploration mechanism must remain matched.

## 18.3 Protocol C — Native and controlled reward regimes

Run at least the following:

1. **Native achievements:** original reward timing.
2. **Fixed-delay rewards:** reveal achievement rewards after $D$ planner decisions.
3. **Episode-bundled rewards:** provide the achievement sum only at the end.
4. **Distractor rewards:** add irrelevant short-term rewards.
5. **Long-gap curriculum:** emphasize deeper achievements and survival dependencies.
6. **Reward identity removed:** compare scalar reward with achievement-labeled reward.
7. **No-rule learner:** hide recipe and prerequisite features.

These regimes identify when GAE is sufficient and when stronger credit mechanisms matter.

---

## 19. Causal-label hierarchy

Maintain three separate label sets.

### Level 1 — Recipe-structural relevance

Derived from the static prerequisite graph.

Use for inexpensive broad evaluation.

### Level 2 — Realized event provenance

Derived from exact inventory acquisitions, consumption, item production, placement, and achievement events.

Use for mechanistic path evaluation.

### Level 3 — Branch-and-replay intervention effect

Derived from action replacement at a fixed snapshot and continuation policy.

Use as the strongest causal evaluation target.

Never merge these labels into one undifferentiated “ground truth.”

---

## 20. Key ablations

### Time and SMDP structure

- duration-corrected vs equal-duration discounting;
- $\lambda$ per macro transition vs per primitive step;
- short vs long rollout fragments;
- correct vs incorrect truncation bootstrapping.

### State representation

- current observation only;
- finite ledger window;
- recurrent full-history state;
- explicit inventory/achievement features;
- no recipe-graph features.

### Credit representation

- scalar return;
- achievement-conditioned vector;
- reward timing only;
- provenance supervision;
- branch supervision.

### Intervention design

- number of alternatives;
- number of continuation samples;
- policy vs uniform alternative baseline;
- paired vs unpaired random seeds;
- frozen vs updated continuation policy;
- branch horizon.

### Policy training

- action-token-only loss;
- all-token loss;
- PPO;
- advantage-weighted regression;
- direct candidate ranking.

### Auxiliary-model capacity

- small MLP or recurrent critic;
- transformer critic;
- shared vs separate policy representation;
- ensemble size;
- uncertainty calibration.

---

## 21. Engineering and reproducibility requirements

### Determinism

- snapshot full environment state;
- snapshot JAX/NumPy/Python/model sampling RNG states;
- test replay equality;
- version environment and skill-controller code.

### Logging

Every planner decision should log:

- observation identifier;
- history identifier;
- chosen subgoal;
- action log probability;
- valid-action mask;
- primitive duration;
- internal primitive rewards;
- macro reward;
- value prediction;
- TD residual;
- every credit score;
- inventory delta;
- health/resource delta;
- achievements;
- terminal/truncation status;
- policy and model checkpoint hashes.

### Testing

- hand-computed SMDP returns;
- hand-computed GAE;
- terminal/truncation cases;
- duration-one equivalence with standard GAE;
- snapshot determinism;
- branch pairing;
- event-provenance accounting;
- no recipe leakage into rule-blind models.

### Experiment management

- configuration files checked into version control;
- deterministic evaluation suite;
- automatic checkpoint and metric recovery;
- parallel rollout monitoring;
- failure-rate and retry logging;
- separate train and evaluation worlds;
- artifact manifests for every reported figure.

### Reporting

Use multiple seeds. For expensive LLM runs:

- never report a single seed as a final result;
- report confidence intervals;
- include individual-seed traces;
- state every stopped or failed run;
- distinguish exploratory runs from preregistered final runs.

---

## 22. Milestones and exit criteria

## M0 — Baseline hardening

Deliver:

- corrected terminology;
- 11-skill documentation;
- SMDP trajectory schema;
- action-token masking;
- deterministic snapshot tests;
- event-provenance logger.

Exit when all tests pass.

## M1 — SMDP PPO+GAE baseline

Deliver:

- stable policy and critic updates;
- duration-aware GAE;
- fixed evaluation suite;
- multiple-seed learning curves;
- ablation against duration-ignorant GAE.

This is the minimum complete agent-RL training milestone.

## M2 — Coverage-oriented dataset

Deliver:

- shared fixed dataset;
- successful descent trajectories;
- survival successes and failures;
- deeper achievement examples;
- distractor and redundant-action examples;
- train/validation/test split.

Exit when the data supports every claimed credit question.

## M3 — Counterfactual Branch-and-Replay benchmark

Deliver:

- exact snapshot branching;
- alternative-action sampler;
- paired continuation runner;
- per-achievement effect labels;
- confidence intervals;
- benchmark files and evaluation code.

This is the first distinctive causal-research milestone.

## M4 — Rule-Blind Counterfactual Critic

Deliver:

- achievement-conditioned critic;
- uncertainty estimates;
- active branch selection;
- held-out intervention prediction;
- PPO or ranking update from learned effects.

This is the strongest initial resume and paper milestone.

## M5 — Comparator suite

Deliver:

- episode-level return;
- Monte Carlo reward-to-go;
- SMDP GAE;
- oracle potential shaping;
- COCOA/HCA adaptation;
- RUDDER/RRD adaptation.

Exit when all methods use matched data and budgets.

## M6 — Online learning study

Deliver:

- native and delayed reward experiments;
- matched-budget comparisons;
- learning efficiency and credit-quality curves;
- analysis of whether early credit quality predicts final performance.

## M7 — Unknown-rules extension

Deliver:

- graph-hidden learner;
- learned counterfactual model or latent world model;
- transfer to held-out worlds or altered dependencies;
- comparison with the true simulator intervention oracle.

---

## 23. Risks and mitigations

### Risk: no deep-task support

The planner never descends, so deeper action effects are absent.

**Mitigation:** shared curriculum, scripted seed trajectories, goal conditioning, and fixed-data experiments.

### Risk: exploration is mistaken for credit quality

A method may look better only because it happens to visit more informative states.

**Mitigation:** fixed-data credit benchmark before online learning.

### Risk: the critic is wrong

GAE and learned causal critics may reinforce bad actions due to value error.

**Mitigation:** Monte Carlo comparisons, branch labels, uncertainty estimates, value diagnostics, and held-out calibration.

### Risk: recipe labels overclaim causality

Static prerequisites may label redundant actions as necessary and miss survival or information-gathering actions.

**Mitigation:** rename the metric, add event provenance, and use interventions.

### Risk: counterfactual continuation is unstable

Changing one action can move the trajectory into a different history where the LLM policy behaves unpredictably.

**Mitigation:** freeze the continuation policy, use multiple continuations, report policy sensitivity, and define the causal estimand explicitly.

### Risk: hidden state or partial observability

The same visible observation may conceal different histories.

**Mitigation:** recurrent history representation and complete simulator snapshot for intervention evaluation.

### Risk: branch cost is large

Evaluating many alternatives multiplies rollout cost.

**Mitigation:** branch only informative states, use common random numbers, train a critic to distill labels, and select states actively.

### Risk: learned world-model exploitation

The policy may exploit model errors.

**Mitigation:** uncertainty penalties, short imagination horizons, true-environment validation, and branch-based calibration.

### Risk: reward hacking or controller leakage

The planner may exploit the harness, and scripted skills may silently perform planning.

**Mitigation:** narrow skill definitions, QA tests, trace inspection, adversarial evaluation, and skill-level invariants.

---

## 24. Minimum viable and strong research outcomes

### Minimum viable research result

- correct SMDP PPO+GAE baseline;
- fixed-data comparison with episode return and Monte Carlo;
- recipe relevance renamed and event provenance added;
- delayed-reward stress tests;
- multiple seeds;
- open-source, reproducible training and evaluation.

This is enough for a strong engineering portfolio even if no novel algorithm wins.

### Strong research result

- branch-and-replay intervention benchmark;
- direct comparison between temporal, structural, and causal credit;
- rule-blind counterfactual critic;
- measurable agreement with held-out intervention effects;
- improved reward-per-token or deeper progression under hard reward regimes;
- public technical report or workshop submission.

### Very strong research result

- learned dynamics or counterfactual model replaces most simulator branches;
- transfers to changed or unknown rules;
- calibrated causal credit improves learning across multiple long-horizon environments;
- clear evidence that the metric predicts later policy learning.

---

## 25. Career and resume milestones

## 25.1 You can list the project now

The current system already demonstrates:

- construction of an RL environment and agent harness;
- a hierarchical LLM agent with a skill library;
- long-horizon trajectory logging;
- evaluation and replay infrastructure;
- a GPU rollout pipeline;
- attention to credit-assignment metrics.

Describe it as an active project and do not claim training results that have not been obtained.

### Current-project resume bullet

> Built a hierarchical Qwen-based Craftax agent in which an LLM selects structured subgoals and an 11-skill controller executes variable-length actions; implemented a verified text environment, long-horizon ledger, replay pipeline, prerequisite-based credit evaluation, and GPU rollout serving.

## 25.2 Minimum milestone before a strong agent-RL application

Finish **P0: SMDP PPO+GAE** end to end.

This proves that you can:

- implement actor–critic RL rather than only agent scaffolding;
- train an LLM policy and critic;
- handle variable-duration actions correctly;
- debug on-policy optimization;
- build reproducible rollouts and evaluations;
- analyze bias, variance, truncation, and value learning.

### Resume bullet after P0

> Implemented semi-Markov PPO with duration-corrected GAE for variable-length LLM subgoals, including action-token policy gradients, recurrent value estimation, truncation bootstrapping, parallel rollouts, and multi-seed evaluation; improved **[metric]** from **[baseline]** to **[result]** under a matched token budget.

Only fill in the bracketed result after it is measured.

## 25.3 Best distinctive milestone for research-engineer applications

Finish **P1 + P2: Counterfactual Branch-and-Replay plus the Rule-Blind Counterfactual Critic**.

This is the best combination because it shows:

- a novel or uncommon research idea;
- careful definition of a causal estimand;
- environment snapshotting and interventions;
- data generation and evaluation design;
- learned models and policy optimization;
- empirical comparison against strong baselines;
- a path from simulator supervision to unknown-rule settings.

### Resume bullet after P1 + P2

> Developed a counterfactual credit-assignment system for long-horizon LLM agents by branching exact simulator snapshots over alternative subgoals and distilling intervention effects into an achievement-conditioned, rule-blind critic; compared against SMDP-GAE, Monte Carlo, shaping, and return decomposition on held-out worlds and delayed-reward regimes.

Add quantitative outcomes only after the final experiment.

## 25.4 Do not wait for the full world model

P5 is ambitious and valuable, but it should not delay applications.

A completed, reproducible baseline plus one distinctive causal method is stronger than an unfinished broad world-model project.

## 25.5 Portfolio package

Before sending applications, publish:

1. a clean repository with one-command setup;
2. architecture and SMDP diagrams;
3. a two-minute rollout video;
4. baseline and proposed-method learning curves;
5. credit heat maps over real trajectories;
6. branch-and-replay counterfactual examples;
7. a concise technical report;
8. tests for returns, GAE, snapshots, and provenance;
9. a section on negative results and failure analysis;
10. exact hardware, token, environment-step, and seed budgets.

## 25.6 Role mapping

### Agent RL research engineer

Emphasize:

- SMDP PPO+GAE;
- causal credit method;
- model training;
- controlled experiments;
- paper implementation;
- research conclusions.

### RL environments and evaluations

Emphasize:

- Craftax harness;
- skill controller;
- state snapshotting;
- branch benchmark;
- long-horizon graders;
- reliability and variance analysis.

### RL infrastructure engineer

Emphasize:

- parallel rollout collection;
- GPU serving;
- checkpointing;
- deterministic replay;
- monitoring;
- failure recovery;
- throughput and cost profiling.

---

## 26. Recommended application point

Do not define readiness as “after every algorithm is complete.”

Use this sequence:

1. **Apply now** to relevant environment, evaluation, agent-infrastructure, and applied-agent roles with the existing system clearly marked as ongoing.
2. **Broaden applications to core agent-RL research-engineering roles after P0**, when SMDP PPO+GAE trains end to end and has reproducible results.
3. **Lead with the project as a distinctive research artifact after P1 + P2**, when counterfactual branch labels and the learned rule-blind critic are complete.
4. Continue P3–P5 while interviewing rather than treating them as prerequisites.

---

## 27. Expected public artifacts

- `README.md`: project motivation, setup, and current results;
- `PROJECT_PLAN.md`: this plan;
- `METHOD.md`: mathematical definition of SMDP GAE and causal estimands;
- `BENCHMARK.md`: structural, provenance, and intervention labels;
- `RESULTS.md`: frozen final experiment table;
- `configs/`: all baseline and method configurations;
- `tests/`: return, GAE, replay, branch, and provenance tests;
- `data/branch_benchmark/`: branch labels where licensing permits;
- model cards and checkpoints;
- blog post or preprint.

---

## 28. Decision rule for project scope

The core project is complete when it can answer:

> Under matched data and compute, when does temporal SMDP-GAE suffice, when do learned redistribution methods help, and does a rule-blind intervention-trained critic assign credit more faithfully and improve long-horizon LLM planning?

Anything not needed to answer that question is an extension.

---

## 29. References

- Matthews et al., **Craftax: A Lightning-Fast Benchmark for Open-Ended Reinforcement Learning**  
  <https://arxiv.org/abs/2402.16801>
- Schulman et al., **High-Dimensional Continuous Control Using Generalized Advantage Estimation**  
  <https://arxiv.org/abs/1506.02438>
- Schulman et al., **Proximal Policy Optimization Algorithms**  
  <https://arxiv.org/abs/1707.06347>
- Sutton, Precup, and Singh, **Between MDPs and Semi-MDPs: A Framework for Temporal Abstraction in Reinforcement Learning**  
  <https://www.sciencedirect.com/science/article/pii/S0004370299000521>
- Ng, Harada, and Russell, **Policy Invariance Under Reward Transformations**  
  <https://ai.stanford.edu/~ang/papers/icml99-shaping.pdf>
- Arjona-Medina et al., **RUDDER: Return Decomposition for Delayed Rewards**  
  <https://proceedings.neurips.cc/paper/2019/hash/16105fb9cc614fc29e1bda00dab60d41-Abstract.html>
- Ren et al., **Learning Long-Term Reward Redistribution via Randomized Return Decomposition**  
  <https://arxiv.org/abs/2111.13485>
- Harutyunyan et al., **Hindsight Credit Assignment**  
  <https://arxiv.org/abs/1912.02503>
- Meulemans et al., **Would I Have Gotten That Reward? Long-Term Credit Assignment by Counterfactual Contribution Analysis**  
  <https://arxiv.org/abs/2306.16803>
- Andrychowicz et al., **Hindsight Experience Replay**  
  <https://arxiv.org/abs/1707.01495>
- Schrittwieser et al., **Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model**  
  <https://www.nature.com/articles/s41586-020-03051-4>
