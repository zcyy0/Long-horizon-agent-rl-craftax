# craftax-rl — long-horizon RL for hierarchical LLM agents on full Craftax

A research program on **long-horizon credit assignment**: can an agent learn to execute
plans whose payoff arrives hundreds of primitive steps later — and what should the
source of that competence be, a pretrained prior or a learned world model?

The testbed is **full [Craftax](https://github.com/MichaelTMatthews/Craftax)** (not the
simplified Crafter or the 1-floor variants): a 9-floor dungeon crawler with a deep tech
tree (wood → stone → iron → diamond and beyond), survival meters (food, water, energy,
health), day/night cycles, hostile mobs, and 67 one-time achievements. Reaching the
deep floors requires chaining dozens of macro-decisions — gear up, keep meters alive,
descend, re-gear under harsher rules — which makes it one of the harder open
credit-assignment benchmarks that still runs fast enough for controlled experiments.

## Project scope

The agent is hierarchical: a planner (an LLM, or a learned world model — that choice is
the experiment) picks one **grounded macro-action** per decision from a menu built by
the environment harness; a scripted controller expands it into primitive steps. This
turns an episode from thousands of primitive actions into a short sequence of
meaningful decisions — and makes the process semi-Markov, so all credit assignment is
duration-aware (discounting by elapsed game time, not decision count).

The program, end to end:

1. **Matched-budget comparison of knowledge sources** *(complete — results below)*:
   frozen LLM planner vs. PPO+GAE-finetuned LLM vs. from-scratch world-model planner
   vs. teacher-distilled LLM, each given exactly 3,200 environment decisions.
2. **Floor transfer and few-shot adaptation** *(complete for floor 1)*: how does each
   system's knowledge survive a distribution shift it never trained on, and how fast
   does each convert new experience into behavior change?
3. **Knowledge-union distillation** *(designed — [docs/DISTILL_DESIGN.md](docs/DISTILL_DESIGN.md))*:
   merge the LLM's tech-tree prior with the world model's learned survival competence
   without importing either side's failure modes.
4. **Full-depth training and evaluation** *(ahead)*: extend training beyond the surface
   via the snapshot-bank curriculum (floor-entry states are banked and restorable
   exactly, RNG and all), toward training and evaluating across all nine floors.
5. **Long-horizon credit assignment beyond GAE** *(ahead — the program's namesake)*:
   once deep, diverse trajectories exist, use the exact branch-and-replay machinery to
   compute counterfactual credit — "which decision 200 steps ago made this diamond
   possible?" — and test whether it improves on standard advantage estimation. The
   infrastructure (exact state/RNG/memory snapshotting, paired branching) is built and
   already powers the transfer studies.

Full design rationale: [REVISED_RESEARCH_PLAN.md](REVISED_RESEARCH_PLAN.md).
Living lab notebook: [PROGRESS.md](PROGRESS.md).
Narrative report of results so far: [project_report.md](project_report.md).

## Current results

### 1. Under a matched budget, the from-scratch world model beats the LLM — on its training distribution

60 paired evaluation worlds · matched 3,200-decision training budgets (audited exactly) ·
identical grounded action space · one shared evaluator · paired bootstrap.

| system | reward | achievements | survival | vs FROZEN |
|---|---:|---:|---:|---|
| **World-model planner** (from scratch, no LLM, ~0 GPU) | **10.61** | **10.6** | **90%** | **+3.32, p<0.001** |
| PPO+GAE-finetuned LLM (Qwen3-4B + LoRA, 3.2 GPU-h) | 7.66 | 8.5 | 52% | +0.36, p=0.22 n.s. |
| Frozen LLM planner (Qwen3-4B) | 7.29 | 8.1 | 48% | — |
| Distilled (teacher → LLM) | *designed, not yet run* | | | |

PPO on this budget is statistically indistinguishable from not training at all — a
carefully-measured null result (n=60, MDE ≈ 0.9).

### 2. The systems know different things — and die where their knowledge comes from elsewhere

- The LLM planners execute the deep tech chain out of the box — wood → stone → **iron**
  (14/60 worlds) — because the pretrained prior already contains Minecraft's tech tree.
  The from-scratch planner never discovers iron on the surface: its exploration never
  completes the chain to reach ore.
- The LLM planners **die of thirst** (55 dehydration deaths across two arms, ~zero
  `drink` actions) — Minecraft has no thirst meter, and 3,200 decisions of PPO did not
  patch the hole in the prior. The world model *learned* the meters and survives 90%.
- One floor down, in a dungeon neither trained on, the roles invert exactly: the world
  model's surface-learned habit (sleep at low energy — safe above ground) becomes its
  leading cause of death, while the LLM prior, which lacks the sleep mechanic entirely,
  transfers better. **Each system dies precisely where its knowledge's origin diverges
  from the world it is standing in.**

### 3. Adaptation speed is itself a world-model advantage

Given 400 dungeon decisions to adapt (few-shot, matched across arms), the world-model
planner unlearns the lethal sleep habit (deaths 22→8), acquires the whole dungeon
economy — coal, furnaces, potions, bows, **iron in 7/40 and diamond in 3/40 worlds** —
and nearly triples its floor-1 reward (4.81 → 13.46, p<0.001), with zero forgetting on
the surface. One KL-bounded PPO step on the same 400 decisions leaves the LLM's
behavior almost unchanged. **Per decision of experience, the model-based system
converts evidence into behavior change roughly an order of magnitude faster** — model
refit vs. trust-region policy step is a structural difference between the schools.

The floor-1 result also resolves *why* the surface iron gap exists: with ore actually
exposed one floor down, 400 decisions sufficed — so the gap is about the long chain to
**reach** ore, not about valuing it. That is a credit-assignment/exploration finding,
and it is what stages 4–5 of the program attack.

## Why the comparison is fair (the part that took the longest)

Most "A beats B" agent comparisons die on inspection. The fairness contract
([docs/METRICS.md](docs/METRICS.md)) holds constant: interaction budget (3,200
decisions per arm, audited to the unit from each run's own transition files), action
space (one menu constructor for all systems, with the invariant "offered ⟹ doable"),
evaluation worlds (paired seeds), evaluator code (one script for every system), and
the exploration floor (ε=0.05 mirrored across learned arms). Version stamps on every
data file make cross-version comparisons a **fatal preflight error**.

Deliberately *not* matched, and reported instead: mechanism and compute. The LLM arms
inherit an internet-scale prior and spend ~376M tokens + 3.2 GPU-h; the world model
trains from zeros in CPU minutes. The from-scratch system winning *despite* that
asymmetry is what makes the result interesting.

Measurement discipline that turned out to be load-bearing:

- **Powered evals** — at n=10 worlds, four separate "trends" appeared and died at n=60;
  every reported comparison carries its minimum detectable effect.
- **Pre-registered readouts** before each run; negative results (an honest reward head
  made things *worse* — the report's bias-variance section) reported with the same
  prominence as wins.
- **Gates before compute** — every mechanism (menu grounding, reward-head exactness,
  PPO gradient path, evaluator consistency, model-exploitation detection) has an
  executable gate that must pass before budget is spent on it.

## What's in the box

```
harness/        Craftax as a text game: verified renderer, fog-of-war, 17 scripted
                skills, executor with shared affordance predicates, grounded menu
                constructor ("offered ⟹ doable")
rl/             Semi-MDP returns/GAE (duration-aware discounting), PPO core,
                budget ledger — unit-tested against hand-computed references
models/         Typed-head macro world model, 3-member bootstrap ensemble,
                compositional reward head (hard achievement masking + empirical-
                Bayes shrinkage), continuation-value network
planner/        Teacher planner: Thompson sampling to collect, LCB (μ−κσ) to deploy
exploration/    Snapshot bank: exact save/restore of env+RNG+memory — the substrate
                for floor curricula and counterfactual credit assignment
scripts/        Drivers for every arm, 12+ gate scripts, verdict/budget/exploitation
                auditors
eval/           Paired-bootstrap statistics; one evaluator for all systems
docs/           METRICS.md (measurement contract), LESSONS.md (15 engineering rules
                this project paid for), DISTILL_DESIGN.md, ACTION_INTERFACE.md
```

## Reproducing

```bash
python -m venv env && env/bin/pip install -r requirements.txt   # CPU env (game, RL, WM)
# GPU env additionally needs: vllm, peft, transformers (LLM arms only)

# gates (seconds–minutes each, CPU):
python scripts/preflight.py --help          # comparability preflight
python scripts/reward_head_demo.py          # reward-head exactness gate

# the arms:
bash scripts/run_s3_ppo.sh                  # PPO loop (GPU, ~8 h)
bash scripts/run_s6_teacher.sh              # teacher loop (CPU, ~2 h)
bash scripts/s11_fewshot_teacher.sh         # floor-1 few-shot arm (CPU, ~1 h)
python scripts/m5_verdict.py --teacher ... --frozen ...   # paired verdict
```

## Honest caveats

- Dev-set results (seeds 40–99). A pristine 80-world test set (seeds 100–179) is
  reserved and unspent pending the final roster.
- One training seed per arm so far; training-seed variance is unmeasured. The headline
  teacher-vs-FROZEN gap is p<0.001; one internal ablation rung sits at p=0.050.
- Training so far is surface-floor; floor-1 results are transfer/few-shot studies.
  Full-depth training (stage 4) and counterfactual credit assignment (stage 5) are the
  program's open half.
- Single environment, single model size (4B), 40-turn episodes. Claims are scoped to
  this regime.
