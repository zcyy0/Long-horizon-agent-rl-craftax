# Technical Appendix: Craftax Agent Comparison

This appendix contains the implementation contracts and equations behind the public report, [`PROJECT_REPORT.md`](PROJECT_REPORT.md). It is intentionally more technical than the main report.

## A. Grounded macro-action interface

All systems choose from the same deterministic menu $C(h)$, constructed from:

- the public macro-action schema;
- the agent's current observation;
- the agent's memory of previously observed tiles and entities;
- the same feasibility predicates used by the scripted executor.

The menu is capped at 20 actions; the largest observed menu in the current experiments is 17.

### A.1 Two invariants

**Executable actions only.** If an action is offered, executing it must consume at least one primitive game step and satisfy the game's hard preconditions.

**Availability is not value.** The constructor filters impossible actions but does not remove legal actions merely because they appear strategically poor. For example, fighting at low health remains available if the game permits it.

The rule is:

> **The constructor rejects the impossible; the policy owns the unwise.**

### A.2 Examples of grounding conditions

| Action | Offered when |
|---|---|
| `explore` / directional exploration | Always; provides a fallback |
| `open_chest` | A chest has been observed and is reachable |
| `descend` | A down ladder is reachable and the floor's required kill count is satisfied |
| `ascend` | An up ladder is reachable |
| `drink_water` | Water is reachable and the drink meter is not full |
| `eat(plant)` | A ripe plant is reachable and food is not full |
| `sleep` | Energy is not full |
| `fight(...)` | A legal target of the requested class is reachable |
| `mine(resource)` | The resource is reachable and the held tool meets the tier requirement |
| `craft(item)` | All materials, including any station-placement cost, are available and the result is an upgrade |
| `place(item)` | The item is held and the facing tile accepts it |
| `shoot` | A bow and arrow are held |
| `drink_potion` | A potion is held |
| `read_book` | A book is held |
| `enchant(...)` | The station, item, mana, and matching gem are available |

No hidden map information is used. Reachability is calculated from the agent's own map memory.

### A.3 Interface validation

Regression checks replay offered actions over real and synthesized states and verify that:

- no candidate is a zero-step no-op;
- helpful survival actions are not accidentally omitted;
- the menu and executor use the same rule implementation;
- data files carry an action-menu version, and cross-version comparisons fail rather than warn.

---

## B. Semi-Markov objective

Planner decisions are macro-actions. Decision $t$ lasts $\tau_t$ primitive steps and yields primitive rewards $r_{t,0},\ldots,r_{t,\tau_t-1}$.

The within-action discounted reward is

$$
R_t^{\mathrm{macro}}=\sum_{j=0}^{\tau_t-1}\gamma^j r_{t,j}.
$$

If

$$
T_t= \sum_{i\lt t}\tau_i,
$$

then the planner-level return is

$$
G=\sum_t\gamma^{T_t}R_t^{\mathrm{macro}}.
$$

This is algebraically equal to the ordinary discounted return over the flat primitive-step reward stream.

A shared helper uses

$$
\tau_t^{\mathrm{eff}}=\max(\tau_t,1)
$$

at bootstrap boundaries. The menu should already eliminate zero-step actions; the floor prevents any remaining zero-step transition from becoming an undiscounted self-loop.

---

## C. PPO and GAE

### C.1 Duration-aware GAE

For true termination flag $d_t$,

$$
\delta_t=R_t^{\mathrm{macro}}+\gamma^{\tau_t^{\mathrm{eff}}}(1-d_t)V(h_{t+1})-V(h_t),
$$

$$
\widehat A_t=\delta_t+\gamma^{\tau_t^{\mathrm{eff}}}\lambda(1-d_t)\widehat A_{t+1}.
$$

Current settings are $\gamma=0.99$ per primitive game step and $\lambda=0.95$ per macro-decision.

True death sets the bootstrap to zero. A rollout cut off by a time or decision limit bootstraps from the final-state value.

### C.2 LLM policy over the grounded menu

For candidate action $a$ serialized as tokens $y_{1:L_a}$,

$$
s_\theta(h,a)=\frac{1}{L_a^\alpha}\sum_{j=1}^{L_a}\log p_\theta(y_j\mid h,y_{\lt j}),
$$

with $\alpha=1$.

The categorical menu policy is

$$
\pi_\theta(a\mid h,C)=\frac{\exp(s_\theta(h,a)/T)}{\sum_{a'\in C}\exp(s_\theta(h,a')/T)}.
$$

The current calibrated temperature is approximately $T=0.129$. The same action serialization, length normalization, and temperature are used during rollout collection, PPO training, and distillation.

### C.3 Exploration mixture and PPO ratio

Collection uses an $\varepsilon=0.05$ category-balanced exploration floor $u(h,a)$:

$$
\mu_\theta(a\mid h)=(1-\varepsilon)\pi_\theta(a\mid h)+\varepsilon u(h,a).
$$

The sampled environment action is one menu index, so PPO uses one scalar ratio per decision:

$$
\rho_t=\frac{\mu_{\theta}(a_t\mid h_t)}{\mu_{\theta_{\mathrm{old}}}(a_t\mid h_t)}.
$$

The clipped objective is

$$
\mathcal L_{\mathrm{policy}}=-\mathbb E_t\left[\min\left(\rho_t\widehat A_t,\mathrm{clip}(\rho_t,1-\epsilon_{\mathrm{clip}},1+\epsilon_{\mathrm{clip}})\widehat A_t\right)\right].
$$

Although $\rho_t$ is scalar, gradients flow through the complete menu softmax, including the alternative candidates in the denominator.

### C.4 KL stopping and scoring-stack mismatch

The implementation stops taking additional optimizer steps after the menu-policy KL crosses 0.02. Because all candidate probabilities are available, exact categorical KL can be logged alongside the low-variance sampled estimator.

Rollouts are served through vLLM, while training uses Hugging Face/PyTorch. The two stacks produce measurably different scores even at the same nominal weights. The implementation therefore recomputes old probabilities in the training stack so that

$$
\rho_t(\theta_{\mathrm{old}})=1
$$

at initialization of an update. The server-side behavior probability is also stored and the discrepancy is monitored. This preserves a stable surrogate gradient but does not make the training-stack distribution identical to the true serving behavior distribution; the mismatch remains a documented limitation.

### C.5 PPO correctness checks

The test suite verifies:

- macro-return equivalence to primitive-step discounting;
- GAE against hand-calculated trajectories;
- terminal versus truncated bootstrapping;
- exploration-mixture probability accounting;
- old-policy detachment;
- menu-softmax gradients through all candidates;
- masking by selection rather than multiplication, avoiding $0\times\mathrm{NaN}$ contamination;
- a deliberately broken cached-scoring gradient path differs from the production path.

---

## D. Structured world-model planner

### D.1 Model inputs and outputs

The observable state has 132 features, including:

- health, food, drink, energy, and sleep status;
- inventory, tools, weapons, armour, potions, and books;
- 67 achievement-unlocked indicators;
- visible entities and distances;
- map exploration statistics;
- floor, direction, kill count, and floor-clear status;
- previous macro-action result, duration, and reward;
- adjacency to crafting stations.

The action is encoded using a 61-dimensional typed representation containing the skill identity and its arguments.

Each ensemble member predicts

$$
M_{\psi_m}(x_t,a_t)\rightarrow\left(\widehat x_{t+1},\widehat r_t,\widehat\tau_t,\widehat d_t,\widehat y_t\right)
$$

The event vector $\widehat y_t$ contains 67 per-achievement unlock events and general events such as chest opening, floor transition, action success, and any-achievement unlock.

### D.2 Typed prediction heads

A shared two-layer, 256-unit MLP trunk feeds feature-specific heads:

| Feature type | Prediction | Loss |
|---|---|---|
| Continuous | Normalized residual change | Huber |
| Count | Residual change on `log1p` scale | Huber |
| Binary | Logit | Binary cross-entropy |
| Categorical | Field-specific softmax | Cross-entropy |
| Duration | `log1p` duration | Huber |
| Death | True-termination logit | Binary cross-entropy |
| Events | Per-event logits | Binary cross-entropy |

Predicting residual changes makes "no state change" the default, which is appropriate for sparse macro-action effects.

### D.3 Structured reward prediction

Craftax reward is largely a known sum of one-time achievement rewards plus a health-change term. The model predicts the ingredients and assembles reward:

$$
\widehat r=\underbrace{\sum_k w_k\widehat p_k c_k}_{\text{per-achievement term}}+\underbrace{\widehat p_{\mathrm{any}}\frac{\sum_k(1-w_k)c_k\mathbf 1[\text{locked}_k]}{\text{#locked}}}_{\text{pooled rare-event term}}+0.1\widehat{\Delta\mathrm{health}}.
$$

Already-unlocked achievements are masked to zero because the game never pays them twice.

The shrinkage weight is

$$
w_k=\frac{n_k}{n_k+10},
$$

where $n_k$ is the count of observed training examples for achievement $k$. Rare achievement heads borrow more from the pooled any-achievement estimate; common heads rely on their own evidence.

A separate scalar reward head is trained as a diagnostic and is not used for planning.

### D.4 Continuation value and action score

A separate value network $V_\omega(x)$ estimates discounted return under the current teacher policy. It is refit each round because the teacher changes after every model update.

For ensemble member $m$,

$$
Q_m(h,a)=\widehat r_m(h,a)+\gamma^{\max(\widehat\tau_m,1)}\left(1-\widehat d_m(h,a)\right)V_\omega(\widehat x'_m(h,a)).
$$

Let

$$
\mu_Q(h,a)=\frac{1}{M}\sum_m Q_m(h,a),
$$

and let $U(h,a)$ be the ensemble standard deviation.

During evaluation,

$$
a^*(h)=\arg\max_{a\in C(h)}\left[\mu_Q(h,a)-0.5U(h,a)\right].
$$

During collection, one bootstrap member is sampled for the full episode, with an additional $\varepsilon=0.05$ exploration floor.

### D.5 Training loop

The teacher trains for eight rounds:

1. collect 400 decisions using the current Thompson policy;
2. append the data to replay;
3. retrain each bootstrap world model on a fresh resample of all collected data;
4. recompute rare-event shrinkage weights;
5. refit the value network.

Only the data persists between rounds. The small models are retrained rather than incrementally fine-tuned.

---

## E. Policy distillation

### E.1 Teacher targets

The teacher's deployment score is

$$
S_{\mathrm{teacher}}(h,a)=\mu_Q(h,a)-0.5U(h,a)
$$

The reported distillation run constructs a target distribution

$$
q^*(a\mid h)\propto\pi_{\mathrm{base}}(a\mid h)\exp\left(\frac{S_{\mathrm{teacher}}(h,a)}{\beta}\right),\qquad \beta=1.
$$

The base-policy factor acts as a trust region in the pretrained prior.

### E.2 Confidence gate and KL anchor

The teacher contributes only on states satisfying

$$
w(h)=\mathbf 1[\mathrm{margin}(h)\ge m_{\min}]\mathbf 1[U_{\mathrm{chosen}}(h)\le u_{\max}].
$$

The thresholds are calibrated from the teacher's training log rather than evaluation outcomes.

The loss is

$$
\mathcal L(h)=w(h)\sum_{a\in C(h)}q^*(a\mid h)(-\log\pi_\theta(a\mid h))+\lambda\,\mathrm{KL}\left(\pi_\theta(\cdot\mid h)\|\pi_{\mathrm{base}}(\cdot\mid h)\right)
$$

with $\lambda=0.1$.

Survival actions bypass the multiplicative base-prior factor in the reported run because the base LLM assigns essentially zero probability to drinking, making ordinary multiplicative regularization unable to transfer the behavior.

### E.3 Training

The reported student uses two LoRA stages:

1. hard cross-entropy toward the teacher argmax;
2. soft distribution matching with the confidence gate and KL anchor.

The student uses the same menu policy representation, action serialization, length normalization, and temperature as PPO.

### E.4 Evidence-weighted follow-up

A second experiment replaced the hand-written survival exception with an evidence-dependent prior veto:

$$
q^*(a\mid h)\propto\max(\pi_{\mathrm{base}}(a\mid h),e^{-V})^{1-w_a}\exp\left(\frac{S_{\mathrm{teacher}}(h,a)}{\beta}\right),
$$

$$
w_a=\frac{n_{f(a),b}}{n_{f(a),b}+m}
$$

Here $f(a)$ is the action family and $b$ is the surface/dungeon bucket. The method recovered placing and suppressed unsafe dungeon sleeping, but assigned too much probability to placing, crowding out drinking and deep crafting. It remains an ablation rather than the reported student.

---

## F. Evaluation protocol

### F.1 Surface evaluation

- 60 paired development worlds, seeds 40--99;
- identical seeds for every system;
- 10,000 paired bootstrap resamples;
- two-sided confidence intervals and $p$-values;
- minimum detectable effect at 80% power and $\alpha=0.05$.

### F.2 Final test set

- 80 worlds, seeds 100--179;
- untouched during development;
- intended for one final evaluation after code and reporting are frozen.

### F.3 Floor-1 evaluation

The current floor-1 study uses 40 saved entry states. The near-zero-shot comparison reports prior exposure counts explicitly. The 400-decision adaptation experiment uses the same frontier-world distribution for adaptation and reevaluation; it therefore measures rapid adaptation on encountered frontier conditions rather than held-out target-floor generalization.

A future version will separate:

- adaptation snapshots;
- validation snapshots;
- final held-out floor-1 snapshots.

---

## G. Reproducibility and known limitations

The repository records:

- data and menu versions;
- policy, model, and actor checkpoint versions;
- intended and realized interaction budgets;
- server-side and training-side action probabilities;
- ensemble predictions and uncertainties;
- all action menus and selected actions;
- evaluation seeds and paired per-world results.

Important limitations include:

- two teacher training seeds but one PPO and one reported distillation seed;
- known domain structure in the teacher's state, action interface, and reward assembly;
- no raw-pixel or primitive-action learning;
- one-step model prediction rather than recursive multi-step model rollout;
- a measurable vLLM versus Hugging Face policy-score mismatch;
- surface-focused training and limited dungeon exposure;
- development-set results pending one-shot test evaluation.
