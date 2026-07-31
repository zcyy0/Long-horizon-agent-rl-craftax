# Long Horizon Agent RL in Craftax

**If a pretrained language model and a world model built from scratch get exactly the
same amount of game experience, which one plays better?**

This repo runs the comparison inside one
harness, and then follows the result somewhere more interesting than the scoreboard.

The game is [Craftax](https://github.com/MichaelTMatthews/Craftax) (full version): a
2D survival game in the Minecraft family, with a tech tree running from wood to
diamond, meters for hunger, thirst and energy, monsters, and a nine-floor dungeon
underneath. 
Everything in this repo — the environment harness, the agent, the RL training, the
world model, and the evaluation statistics — was built and run end to end as one
project.

## How the agent plays

Rather than emitting raw keypresses, the agent chooses one **macro-action** per turn
from a menu the harness builds for the current situation — things like
`craft(stone_pickaxe)`, `fight(zombie)`, `drink_water`, `descend`. A scripted
controller then carries that choice out over however many game steps it needs. One
turn of thinking becomes roughly seven game steps of doing.

That design puts a clean question in the middle of the project: *what should do the
choosing?* Four candidates get exactly the same 3,200 turns of experience to become
good at it, and everything else about them is held equal.

## The program, in five stages

1. **The matched-budget comparison** *(done — results below)*. Four systems, 3,200
   training decisions each: a frozen LLM, the same LLM fine-tuned with PPO, a
   from-scratch world model that plans, and that planner distilled back into the LLM.
2. **Floor transfer and few-shot adaptation** *(done for floor 1)*. Drop everyone into
   the dungeon, which nobody trained on. Whose knowledge transfers? Who adapts
   fastest when given a little local experience?
3. **Distillation** *(done — result 4 below)*. Combine the LLM's deep game knowledge
   with the world model's learned survival instincts into a single policy.
4. **Full-depth training** *(ahead)*. Push training below the surface floor using a
   bank of saved game states as a curriculum, working toward all nine floors.
5. **Long-horizon credit assignment** *(ahead)*.
   Once trajectories run deep, use exact save-and-replay branching to ask "which
   decision two hundred steps ago made this diamond possible?", and test whether
   answering it properly beats standard advantage estimation.

## Results so far

### 1. Given equal experience, the from-scratch world model wins

60 evaluation worlds, played by every system so the comparison is paired
world-by-world rather than averaged across different games. 3,200 training decisions
each, identical menus, one shared evaluator:

| system | reward | achievements | survival | vs. frozen |
|---|---:|---:|---:|---|
| **World-model planner** (from scratch, no LLM, ~0 GPU) | **10.61** | **10.6** | **90%** | **+3.32, p<0.001** |
| PPO-fine-tuned LLM (Qwen3-4B, 3.2 GPU-hours) | 7.66 | 8.5 | 52% | +0.36, not significant |
| Frozen LLM (Qwen3-4B) | 7.29 | 8.1 | 48% | — |

A small model that starts from random weights and trains in CPU minutes beats a
4-billion-parameter pretrained planner on reward, achievements and survival
simultaneously — and the result replicates: a second teacher trained from a
different random seed scored 11.75 reward with the same 90% survival, beating the
frozen LLM at p<0.001 on all three metrics again. And PPO, on this budget, is
statistically indistinguishable from not training at all.

### 2. But they know different things — and each dies of what it doesn't know

- The LLMs walk the deep tech chain straight out of the box (iron in 14 of 60 worlds). The world model never found iron on
  the surface: its exploration never reached the ore.
- The LLMs also **die of thirst**. The LLM prior has no
  such habit — and 3,200 decisions of PPO did not patch the hole. The world model
  learned the meters this game actually has and survives 90% of its episodes.
- One floor down, in a dungeon nobody trained on, the pattern inverts precisely. The
  world model's surface habit — sleep when tired, which is perfectly safe up there —
  becomes its leading cause of death in a monster-filled dungeon. The LLM prior has no
  sleep habit and therefore cannot make that particular mistake. **Each system dies
  exactly where its knowledge came from somewhere else.**

### 3. Given a little dungeon experience, the world model adapts about 10× faster

Both learning systems got 400 dungeon decisions and one standard update apiece. The
world model nearly tripled its dungeon score (4.81 → 13.46, p<0.001), unlearned the
fatal sleep habit, and picked up the entire dungeon economy — coal, furnaces, potions,
bows, **iron and diamond** — while forgetting none of its surface skills.

The same 400 decisions barely moved the LLM. The reason is not data quality: PPO's
batch was just as rich, and its own internal estimates pointed at the right lessons.
The bottleneck is absorption. PPO's trust region — the safety rule that stops a policy
from moving too far in one update — tripped after a single 16-state step, because
unfamiliar states make the policy hypersensitive. The world model simply refits on
everything it has ever seen and re-plans.

A follow-up A/B closed the loophole in that explanation: with that rule switched off the update ran 22× further and still produced no
adaptation — PPO scored the same as not training at all. So the rule tripping was a
symptom, not the cause: the update was free to run and had nothing to absorb. A
policy gradient can only nudge the probabilities of actions it happened to take,
while the world model turns the same experience into updated beliefs and replans
every decision. The 10× adaptation gap looks structural to the algorithm class rather
than to the safety setting first blamed.

### 4. Distilling the world model back into the LLM mostly works — and its two failures were predicted in advance

The last arm teaches the world model's decisions back to the LLM. Both rank the same
menu, so every decision the world model made during its own training is already a
labelled example and no new game experience is needed. Two guards stop the lesson
from overwriting what the LLM already knows: the teacher's preferences are damped on
actions the LLM considers absurd, and a penalty holds the student at its original
behaviour wherever the teacher is unsure. Survival actions are exempt from the first
guard — "the LLM thinks drinking is pointless" is the disease being treated.

The result is the second-best system measured here: **9.75 reward — far above the
frozen LLM (7.29, p<0.001) and statistically tied with its own search-based teacher —
at one forward pass per decision** instead of a search over an ensemble. It drinks
339 times and **never once dies of thirst** (frozen: zero drinks, 29 thirst deaths):
the mechanic RL couldn't install, delivered by distillation from the same experience.
Five predictions were written down before the run and two failed, both reported as
failures — deep-tier reach shrank (iron in 8 worlds, was 14) because 28% of its turns
now go to upkeep and a 40-turn episode can't fit everything, and the teacher's risky
dungeon sleep habit came along with the rest.

## Why you can trust the comparison
The guardrails here, each of which exists because something went wrong first:

- **Matched budgets, audited after the fact.** 3,200 decisions per system, counted
  from each run's own log files rather than from what the driver script intended.
- **One action menu for everyone**, built on the rule *if an action is offered, it can
  actually be performed*. 
- **Paired statistics with stated power.** Every system plays the *same* 60 worlds;
  verdicts come from a paired bootstrap over worlds and each one reports its minimum
  detectable effect, so a null result is labelled underpowered or null rather than
  read as "equal". 
- **Predictions written down before the results exist.** Before an experiment runs,
  what it is expected to show is written into the script that launches it. 
- **Checks before compute.** More than a dozen gate scripts — menu correctness,
  reward-head exactness, gradient correctness, evaluator consistency — must pass
  before a run is allowed to spend any budget.
- **A test set nobody has touched.** Every number here is from development worlds.
  Eighty held-out worlds are reserved and will be spent exactly once, at the end.

What is deliberately **not** held equal: starting knowledge and compute. The LLM
arms begin with an internet-scale prior and spend GPU-hours; the world model begins
with random weights and trains in CPU minutes. That asymmetry is not a flaw in the
study — it is the subject of the study, and it is reported rather than equalised.

## What's in the repo

```
harness/        Craftax as a text game: a renderer verified against the native
                observation, real fog of war, 17 scripted skills, and the grounded
                menu constructor that defines the shared action space
models/         the world model: typed prediction heads, the ensemble, the reward
                head with achievement masking and shrinkage, and the value network
planner/        the world-model planner ("teacher") — Thompson sampling to explore,
                pessimistic scoring to act
rl/             semi-Markov returns and GAE (discounting by elapsed game time, not by
                decision count), the PPO core, and the budget ledger — all unit-tested
                against hand-computed references
data/*.py       feature builder, transition schema, and target construction
                (the .py files; collected rollouts are not published — see below)
exploration/    the snapshot bank: exact save/restore of game state, RNG and agent
                memory, which is what makes branch-and-replay possible
counterfactual/ branch-replay machinery for the long-horizon credit-assignment stage
eval/           paired-bootstrap statistics and the single shared evaluator
scripts/        drivers for every experiment, the gate scripts, and the verdict and
                audit tools
configs/        the frozen hyperparameters for the world model, PPO and distillation
docs/           METRICS.md (the measurement contract), LESSONS.md (the engineering
                rules this project paid for), ACTION_INTERFACE.md, DISTILL_DESIGN.md
```

Collected rollouts (2.7 GB) and model checkpoints (116 GB) are **not** in the repo.
Everything they contain is regenerable from the drivers, and every finding derived
from them lives in the reports.

## Running it

Two environments, deliberately isolated so that neither can accidentally depend on
the other: the game, the RL core and the world model run on CPU, while the language
model runs on a GPU behind a serving stack. They communicate only through files.

```bash
# CPU environment — the game, the RL core, the world model, all evaluation
python3.10 -m venv envs/craftax
envs/craftax/bin/pip install -r requirements.txt

# GPU environment — only needed for the LLM arms (frozen, PPO, distillation)
python3.10 -m venv envs/vllm
envs/vllm/bin/pip install vllm peft transformers torch
```

Quick checks, which run on CPU in seconds to minutes and are the fastest way to see
what this project considers a correctness test:

```bash
PY=envs/craftax/bin/python
$PY scripts/preflight.py --help        # the comparability gate that blocks bad runs
$PY scripts/reward_head_demo.py        # reward-head exactness
$PY scripts/ppo_core_demo.py           # semi-Markov returns, GAE and the PPO ratio
```

The experiments themselves:

```bash
bash scripts/run_s6_teacher.sh         # world-model planner, CPU, ~2 h, no GPU at all
bash scripts/run_s3_ppo.sh             # PPO loop, GPU, ~8 h
bash scripts/s11_fewshot_teacher.sh    # the dungeon few-shot arm, CPU, ~1 h

# paired statistics for any two result files
$PY scripts/m5_verdict.py --teacher <a.jsonl> --frozen <b.jsonl>
```

Every driver is resumable: rerunning one skips the stages whose outputs already exist,
so an interrupted run continues rather than restarting.

## Honest caveats

- All results are from development worlds (seeds 40–99). The test set (seeds 100–179)
  is reserved and unspent.
- The world-model headline is replicated across two training seeds; PPO still has
  one. The two teacher seeds differ from each other by ~1.1 reward — a first
  estimate of training-seed variance, and a reason to read internal ablation steps
  of similar size (one sits right at p=0.05) with caution.
- Training so far happens on the surface floor only; the dungeon results are transfer
  and few-shot studies. Stages 4 and 5 are the open half of the program.
- One game, one model size (4B), 40-turn episodes. What is a result here is a
  hypothesis anywhere else.

## Where to read next

1. **[project_report.md](project_report.md)** — the full story: motivation, design
   decisions, results, and what it took to measure them. Its appendix has the precise
   contracts and formulas for the action constructor, the world model, and the
   semi-Markov PPO implementation.
2. [PROGRESS.md](PROGRESS.md) — the running lab notebook: denser, with every ablation
   and every negative result in the order they happened.
3. [docs/METRICS.md](docs/METRICS.md) — the pre-registered measurement contract.
4. [docs/LESSONS.md](docs/LESSONS.md) — the engineering rules this project paid for,
   each written against the incident that motivated it.
