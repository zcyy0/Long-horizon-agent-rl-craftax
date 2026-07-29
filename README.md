**If a pretrained LLM and a from-scratch world model get exactly the same amount of
game experience, who plays better?** This repo runs that comparison properly — same
budget, same action space, same evaluator — and follows the answer somewhere
unexpected.

The game is [Craftax](https://github.com/MichaelTMatthews/Craftax) (full version): a
2D survival game in the Minecraft family, with a tech tree from wood to diamond,
hunger/thirst/energy meters, monsters, and a nine-floor dungeon. 

Everything here — environment harness, agent, RL training, world model, evaluation
statistics — was built and run end-to-end as one project.

## The program

The agent picks one **macro-action** per turn from a menu the harness builds (things
like `craft(stone_pickaxe)`, `fight(zombie)`, `drink_water`, `descend`). A scripted
controller executes the choice. The question is what should do the picking, and the
plan runs in five stages:

1. **The matched-budget comparison** *(done — results below)*. Four pickers, 3,200
   training decisions each: a frozen LLM, a PPO-fine-tuned LLM, a from-scratch world
   model, and the world model distilled back into the LLM.
2. **Floor transfer and few-shot adaptation** *(done for floor 1)*. Drop everyone in
   the dungeon. Who transfers? Who adapts fastest with a little local experience?
3. **Distillation** *(running)*. Merge the LLM's deep game knowledge with the world
   model's learned survival skills into one policy.
4. **Full-depth training** *(ahead)*. Extend training beyond the surface floor using
   the saved-state curriculum, toward all nine floors.
5. **Long-horizon credit assignment** *(ahead — the reason this project exists)*.
   Once trajectories run deep, use exact save-and-branch replay to answer "which
   decision 200 steps ago made this diamond possible?" — and test whether that beats
   standard advantage estimation.

Full details and rationale: [project_report.md](project_report.md) (the readable
story)

## Results so far

### 1. Given equal experience, the from-scratch world model wins

60 paired evaluation worlds, 3,200 training decisions per system, identical menus,
one shared evaluator:

| system | reward | achievements | survival | vs frozen |
|---|---:|---:|---:|---|
| **World-model planner** (from scratch, no LLM, ~0 GPU) | **10.61** | **10.6** | **90%** | **+3.32, p<0.001** |
| PPO-fine-tuned LLM (Qwen3-4B, 3.2 GPU-h) | 7.66 | 8.5 | 52% | +0.36, not significant |
| Frozen LLM (Qwen3-4B) | 7.29 | 8.1 | 48% | — |

A small model with random-weight beginnings beats a 4-billion-parameter pretrained
planner on every metric — and PPO on this budget is statistically indistinguishable
from not training at all.

### 2. But they know different things — and each dies of what it doesn't know

- The LLMs execute the deep tech chain out of the box (iron in 14 of 60 worlds)
  because Minecraft folklore contains that chain. The world model never found iron on
  the surface: its exploration never got there.
- The LLMs also **die of thirst** — Minecraft has no thirst meter, and 3,200
  decisions of PPO didn't fix the blind spot. The world model learned the meters this
  game actually has, and survives 90% of episodes.
- One floor down, in a dungeon nobody trained on, the pattern flips exactly. The
  world model's surface habit — sleep when tired, safe up there — becomes its top
  cause of death in a monster-filled dungeon. The LLM prior has no sleep habit and so
  cannot make that mistake. **Each system dies exactly where its knowledge came from
  somewhere else.**

### 3. Given a little dungeon experience, the world model adapts ~10× faster

Both learners got 400 dungeon decisions and one standard update. The world model
nearly tripled its dungeon score (4.81 → 13.46, p<0.001), unlearned the fatal sleep
habit, and picked up the whole dungeon economy — coal, furnaces, potions, bows,
**iron, diamond** — with zero forgetting of its surface skills. The same 400
decisions moved the LLM barely at all: PPO's trust region (the rule that limits how
far the policy can move per update) tripped after a single 16-state step. The data
was equally good; the difference is how much of it each system can *absorb*.
Learning facts and re-planning has no speed limit; a policy gradient must move
slowly to stay valid.

## Why you can trust the comparison

Most "A beats B" agent results fall apart when you look closely. The guardrails here:

- **Matched budgets, audited.** 3,200 decisions per arm, counted from each run's own
  log files.
- **One action menu for everyone**, with the rule "if it's offered, it can be done."
  Menu versions are stamped into every data file; comparing across versions is a
  hard error, not a footnote.
- **One evaluator, paired worlds, real statistics.** Paired bootstrap on 60 worlds;
  every comparison reports the smallest effect it could detect. Four early "trends"
  died when the evaluation was properly powered — they are documented, not buried.
- **Predictions written down before each run.** Two of the most useful results in
  the report are negative ones that pre-registration forced into view.
- **Checks before compute.** A dozen-plus gate scripts (menu correctness, reward
  exactness, gradient correctness, evaluator consistency) must pass before a run is
  allowed to spend budget.
- **A test set nobody has touched.** All results are development-set; 80 held-out
  worlds get spent once, at the end.

What is deliberately *not* matched: starting knowledge and compute. The LLM begins
with an internet-scale prior and spends GPU-hours; the world model begins with random
weights and trains in CPU minutes. That asymmetry is the subject of the study.

## What's in the repo

```
harness/        Craftax as a text game: verified renderer, fog of war, 17 scripted
                skills, and the grounded menu constructor
rl/             semi-Markov returns and GAE (time-aware discounting), PPO core,
                budget ledger — unit-tested against hand-computed references
models/         the typed world model, ensemble, reward head with achievement
                masking and shrinkage, value network
planner/        the teacher: Thompson sampling to explore, pessimistic scoring to act
exploration/    snapshot bank — exact save/restore of game+RNG+memory, the substrate
                for the dungeon studies and future counterfactual credit assignment
scripts/        drivers for every experiment, gate scripts, verdict and audit tools
eval/           paired-bootstrap statistics; one evaluator for all systems
docs/           METRICS.md (the measurement contract), LESSONS.md (15 hard-won
                engineering rules), DISTILL_DESIGN.md, ACTION_INTERFACE.md
```

## Running it

```bash
python -m venv env && env/bin/pip install -r requirements.txt   # CPU env (game, RL, world model)
# the LLM arms additionally need a GPU env with: vllm, peft, transformers

# quick checks (CPU, seconds to minutes):
python scripts/preflight.py --help          # comparability preflight
python scripts/reward_head_demo.py          # reward-head exactness gate

# the main experiments:
bash scripts/run_s6_teacher.sh              # world-model teacher (CPU, ~2 h)
bash scripts/run_s3_ppo.sh                  # PPO loop (GPU, ~8 h)
bash scripts/s11_fewshot_teacher.sh         # dungeon few-shot arm (CPU, ~1 h)
python scripts/m5_verdict.py --teacher ... --frozen ...   # paired statistics
```

## Honest caveats

- Development-set results (seeds 40–99); the test set (100–179) is reserved and
  unspent.
- One training seed per arm so far. The headline is p<0.001, but one internal
  ablation rung sits at p=0.05; a second seed comes before any strong claim.
- Training so far is surface-floor only; the dungeon results are transfer and
  few-shot studies. Stages 4–5 are the open half of the program.
- One game, one model size (4B), 40-turn episodes. Findings here are hypotheses
  anywhere else.

## Where to read next

1. [project_report.md](project_report.md) — the full story: motivation, design,
   results, lessons
2. [PROGRESS.md](PROGRESS.md) — the running lab notebook (denser, has every ablation)
3. [docs/METRICS.md](docs/METRICS.md) — the pre-registered measurement contract
4. [docs/LESSONS.md](docs/LESSONS.md) — 15 engineering rules this project paid for
