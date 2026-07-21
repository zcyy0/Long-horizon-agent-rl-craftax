# Long Horizon Credit Assignment for a Hierarchical LLM Agent in Craftax

This project studies how an LLM planner should assign credit to earlier subgoal decisions when rewards arrive much later.

## Why Craftax?

Craftax combines survival, exploration, combat, and a deep crafting tree. Its achievements provide meaningful intermediate rewards, but an achievement may depend on many earlier decisions. For example, crafting a stone pickaxe requires earlier decisions to find wood, place a table, craft a wooden pickaxe, and collect stone.

A raw Craftax episode is too long for an LLM to control one button at a time. I therefore use a hierarchical agent:

```text
observation + ledger
        |
        v
   LLM planner  ---> structured subgoal
        |                 |
        |                 v
        |          scripted skill controller
        |                 |
        <--- result, duration, rewards, state changes
```

The planner chooses from 11 narrow skills such as `explore`, `mine`, `craft`, `eat`, and `descend`. The controller executes the selected skill for a variable number of primitive game steps. Each planner decision and its outcome is stored in a ledger.

## Research question

> Does counterfactual action replacement assign more faithful and useful credit than GAE to long-horizon LLM-planner decisions?

The first version compares only one baseline and one proposed method.

## Baseline: semi-Markov PPO with GAE

Planner actions have different durations, so the baseline accounts for the number of primitive game steps used by each skill.

For macro-action duration $\tau_t$ and internal rewards $r_{t,j}$:

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
V(h_t),
$$

and the GAE recursion is

$$
\hat A_t
=
\delta_t
+
\gamma^{\tau_t}\lambda\hat A_{t+1}.
$$

This is the primary RL baseline.

## Main method: counterfactual branch-and-replay

At selected planner states, I save the exact simulator and random-number-generator state. I then:

1. execute the factual subgoal;
2. replace it with two or three valid alternatives;
3. continue each branch under the same frozen policy and horizon;
4. compare future achievements, survival, floor depth, and return.

This directly measures the consequence of changing one planner decision while holding the preceding state fixed.

The first analysis compares GAE advantages with measured branch effects using:

- sign agreement;
- action-ranking accuracy;
- rank correlation;
- qualitative disagreement cases.

## Optional extension: rule-blind counterfactual critic

Exact branching is expensive and unavailable in many real-world settings. A lightweight critic can be trained to predict branch outcomes from the ledger state and a candidate subgoal:

$$
Q_\psi(h_t,a).
$$

The learned counterfactual advantage is

$$
\hat A_\psi(h_t,a_t)
=
Q_\psi(h_t,a_t)
-
\frac{1}{|\mathcal A_t'|}
\sum_{a'\in\mathcal A_t'}Q_\psi(h_t,a').
$$

The model is rule-blind: it sees observations, actions, state changes, and rewards, but not the Craftax recipe graph.

## Compute-conscious scope

The first version does **not** implement every credit-assignment algorithm. The required conditions are:

| Condition | Role |
|---|---|
| Frozen coverage checkpoint | Starting reference |
| SMDP PPO + GAE | Primary baseline |
| Branch-and-replay or learned counterfactual critic | Main contribution |

HCA, COCOA, RUDDER, RRD, and learned world models are follow-up work.

Development uses one training seed and fixed evaluation worlds. The final comparison uses equal token and interaction budgets, paired evaluation worlds, and bootstrap intervals across evaluation episodes. A second training seed is added only if affordable.

## Evaluation

Primary outcomes:

- achievements per episode;
- probability of first descent;
- deepest floor reached;
- survival time;
- reward per generated token;
- agreement with measured counterfactual effects.

The project is successful if it produces a correct GAE baseline, a reproducible branch dataset, an honest comparison, and interpretable failure cases. The counterfactual method does not need to outperform GAE for the experiment to be informative.

## Roadmap

1. Build a shared survival-and-descent coverage checkpoint.
2. Implement and validate SMDP PPO+GAE.
3. Collect a stratified branch-and-replay dataset.
4. Compare GAE credit with intervention effects.
5. Train the optional rule-blind counterfactual critic.
6. Run one compute-matched policy-learning comparison.

A longer implementation and experiment plan is available in `docs/SOLO_RESEARCH_PLAN.md`.
