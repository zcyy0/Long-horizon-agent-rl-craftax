"""Hierarchical agent loop over CraftaxTextEnv + Executor (full Craftax).

The research design (see HANDOFF §1): an LLM **planner** picks one *subgoal* per
turn from a fixed **skill menu**; the scripted **Executor** expands that subgoal
into primitives and returns a `SkillResult`. This collapses ~100k primitive steps
into a few hundred subgoal decisions (a semi-MDP), which is what makes long-horizon
credit assignment tractable. The per-turn record (think + subgoal + SkillResult)
accumulates into a structured **ledger** — that ledger *is* the CA substrate.

This module is the spine of that loop, policy-agnostic:
  - SKILLS         : the menu (single source of truth, mirrors Executor methods).
  - dispatch()     : {name, args} -> validated Executor call -> SkillResult
                     (never raises; a bad subgoal becomes a `failure` the planner
                     can read and correct).
  - build_system_prompt() / build_turn_prompt() : the cached contract + per-turn
                     context an LLM policy consumes.
  - Ledger entries + HierarchicalAgent : the driver.
  - Policy protocol + ScriptedPolicy : a deterministic stand-in so the loop can be
                     tested end-to-end with no model (HANDOFF §7.1, step 1). An
                     APIPolicy / local-model policy drops into the same slot next.
"""
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from executor import CRAFTABLES, PLACEABLES, RESOURCES, Executor, SkillResult

DIRECTIONS = ["up", "down", "left", "right"]
FIGHT_TYPES = ["melee", "ranged", "passive"]
EAT_TARGETS = ["cow", "plant", "auto"]


# --- skill menu (the planner's action space; mirrors Executor) ---------------
@dataclass
class Arg:
    name: str
    kind: str                       # "int" | "str" | "enum"
    required: bool = False
    default: Any = None
    choices: Optional[List[str]] = None
    desc: str = ""


@dataclass
class Skill:
    name: str
    method: str                     # Executor method to call
    desc: str
    args: List[Arg] = field(default_factory=list)


SKILLS: Dict[str, Skill] = {
    "explore": Skill(
        "explore", "explore",
        "Reveal new map by walking to the frontier of seen space. The fallback "
        "when a target isn't visible yet; stops on first notable discovery.",
        [Arg("direction", "enum", False, None, DIRECTIONS,
             "bias the search; omit to head to the nearest frontier")],
    ),
    "mine": Skill(
        "mine", "mine",
        "Gather a resource from the nearest SEEN deposit (navigate→face→do). "
        "Requires the right pickaxe tier; does NOT gather prerequisites.",
        [Arg("resource", "enum", True, None, list(RESOURCES), "what to mine"),
         Arg("count", "int", False, 1, None, "how many units")],
    ),
    "craft": Skill(
        "craft", "craft",
        "Craft one item, placing a table/furnace if the recipe needs one and none "
        "is adjacent. Fails (does not gather) if materials are short.",
        [Arg("item", "enum", True, None, list(CRAFTABLES), "what to craft")],
    ),
    "descend": Skill(
        "descend", "descend",
        "Go down one floor via the nearest seen down-ladder. Requires the floor "
        "cleared (>=8 kills).",
        [],
    ),
    "fight": Skill(
        "fight", "fight",
        "Kill nearby visible hostile mobs (approach→face→do); bails at health_floor.",
        [Arg("count", "int", False, 1, None, "how many to kill"),
         Arg("types", "enum", False, None, FIGHT_TYPES,
             "restrict to one mob class; omit for melee+ranged"),
         Arg("health_floor", "int", False, 2, None, "retreat threshold")],
    ),
    "drink_water": Skill(
        "drink_water", "drink_water",
        "Restore thirst: go to the nearest seen water/fountain and drink. Omit "
        "count to drink until full. Do this before drink runs out (you take damage "
        "when a necessity hits 0).",
        [Arg("count", "int", False, None, None, "sips to take; omit to fill up")],
    ),
    "eat": Skill(
        "eat", "eat",
        "Restore food: eat a ripe plant (+4) or kill&eat a cow (+6). Do this before "
        "food runs out. `auto` prefers a reachable plant, else a cow.",
        [Arg("target", "enum", False, "auto", EAT_TARGETS, "food source"),
         Arg("count", "int", False, 1, None, "how many to eat")],
    ),
    "sleep": Skill(
        "sleep", "sleep",
        "Restore energy by sleeping until full (unlocks wake_up). Only when energy "
        "is low, and somewhere SAFE — sleeping multiplies melee damage taken.",
        [],
    ),
    "place": Skill(
        "place", "place",
        "Place an item on a faced tile: torch (light up dark floors 2/5/7/8 so you "
        "can explore them), stone (bridge water/block mobs), table, furnace, or "
        "plant (a sapling that grows into food). Needs the material; does not gather.",
        [Arg("item", "enum", True, None, list(PLACEABLES), "what to place"),
         Arg("direction", "enum", False, None, DIRECTIONS,
             "face this way first; omit to use the current facing")],
    ),
    "ascend": Skill(
        "ascend", "ascend",
        "Go up one floor via the nearest seen up-ladder (no clear requirement).",
        [],
    ),
}


class DispatchError(ValueError):
    """A subgoal that can't be turned into a valid Executor call."""


def _coerce(skill: Skill, args: Dict[str, Any]) -> Dict[str, Any]:
    """Validate/coerce planner args against a skill's schema -> method kwargs."""
    args = dict(args or {})
    known = {a.name for a in skill.args}
    unknown = set(args) - known
    if unknown:
        raise DispatchError(f"{skill.name}: unknown arg(s) {sorted(unknown)}; "
                            f"valid: {sorted(known) or 'none'}")
    kwargs: Dict[str, Any] = {}
    for a in skill.args:
        if a.name not in args:
            if a.required:
                raise DispatchError(f"{skill.name}: missing required arg '{a.name}'")
            continue
        v = args[a.name]
        if v is None:
            continue
        if a.kind == "int":
            try:
                v = int(v)
            except (TypeError, ValueError):
                raise DispatchError(f"{skill.name}.{a.name}: expected int, got {v!r}")
        elif a.kind == "enum":
            v = str(v).lower().strip()
            if v not in a.choices:
                raise DispatchError(f"{skill.name}.{a.name}: {v!r} not in {a.choices}")
        else:
            v = str(v)
        kwargs[a.name] = v
    # fight.types takes a tuple; planner gives a single class name
    if skill.name == "fight" and "types" in kwargs:
        kwargs["types"] = (kwargs["types"],)
    return kwargs


def dispatch(ex: Executor, subgoal: Dict[str, Any]) -> SkillResult:
    """Run a {name, args} subgoal on the Executor. Always returns a SkillResult —
    invalid subgoals come back as `failure` so the planner gets readable feedback
    instead of an exception."""
    name = (subgoal or {}).get("name")
    if name not in SKILLS:
        return SkillResult("failure",
                           f"unknown subgoal {name!r}; choose from {sorted(SKILLS)}",
                           0, 0.0, [], [], ex.env.obs())
    skill = SKILLS[name]
    try:
        kwargs = _coerce(skill, subgoal.get("args", {}))
    except DispatchError as e:
        return SkillResult("failure", str(e), 0, 0.0, [], [], ex.env.obs())
    return getattr(ex, skill.method)(**kwargs)


# --- prompt construction -----------------------------------------------------
def _menu_text() -> str:
    lines = []
    for s in SKILLS.values():
        if s.args:
            sig = ", ".join(
                (f"{a.name}" if a.required else f"{a.name}?")
                + (f"={{{'|'.join(a.choices)}}}" if a.kind == "enum" else f":{a.kind}")
                for a in s.args
            )
        else:
            sig = ""
        lines.append(f"- {s.name}({sig}): {s.desc}")
    return "\n".join(lines)


SYSTEM_RULES = """\
You are the planner for an agent playing Craftax (a long-horizon survival/crafting
game over 9 floors). You do NOT control the joystick. Each turn you choose ONE
subgoal from the skill menu; a scripted executor carries it out and reports back.

Key rules:
- You only know tiles you have SEEN. If a skill reports "explore first", the target
  is not yet in view — choose `explore` (optionally toward a direction) to reveal it.
- Skills are single-purpose: `craft` will NOT gather materials. If it reports a
  shortfall, YOU must mine/collect the inputs first, then retry.
- Progress is rewarded by first-time achievements (collect wood, make pickaxe,
  defeat a mob, descend, ...). Plan a tech tree: wood -> table -> wood pickaxe ->
  stone -> stone pickaxe -> coal/iron -> furnace -> iron gear -> descend.
- A floor's down-ladder unlocks only after clearing it (>=8 kills).
- SURVIVE: watch food, drink, and energy in the progress header. When any gets
  low, `drink_water` / `eat` / `sleep` to refill it — if one hits 0 you lose health
  and the episode can end. Dark floors (2,5,7,8) need `place` torch before you can
  explore them.

Respond with ONE JSON object and nothing else:
  {"think": "<one or two sentences of reasoning>",
   "subgoal": {"name": "<skill>", "args": {<args>}}}"""


def build_system_prompt() -> str:
    """Static contract (role + rules + menu). Cache this once across the episode."""
    return f"{SYSTEM_RULES}\n\n# Skill menu\n{_menu_text()}"


def build_header(env, ledger: List["LedgerEntry"]) -> str:
    """Compact running progress summary (distinct from the raw obs)."""
    inv = env.state.inventory
    s = env.state
    unlocked = sorted(env._unlocked)
    names = [a for e in ledger for a in e.achievements]
    last = ledger[-1] if ledger else None
    return "\n".join([
        "# Progress",
        f"  turn={len(ledger)} env_step={env.t} floor={int(s.player_level)}",
        f"  achievements_unlocked={len(unlocked)}: {names if names else '(none yet)'}",
        f"  vitals: health={int(s.player_health)} food={int(s.player_food)} "
        f"drink={int(s.player_drink)} energy={int(s.player_energy)}"
        + ("  [SLEEPING]" if bool(s.is_sleeping) else ""),
        f"  pickaxe={int(inv.pickaxe)} sword={int(inv.sword)} | "
        f"wood={int(inv.wood)} stone={int(inv.stone)} coal={int(inv.coal)} "
        f"iron={int(inv.iron)} diamond={int(inv.diamond)}",
        f"  last_subgoal={last.subgoal if last else None} -> "
        f"{last.status + ': ' + last.reason if last else '(start)'}",
    ])


def render_ledger(ledger: List["LedgerEntry"], k: int = 8) -> str:
    if not ledger:
        return "# Recent subgoals\n  (none)"
    rows = ["# Recent subgoals (most recent last)"]
    for e in ledger[-k:]:
        rows.append(f"  t{e.turn}: {e.subgoal} -> {e.status} ({e.reason}) "
                    f"[+{e.reward:.2f}r, {e.steps} steps"
                    + (f", unlocked {e.achievements}" if e.achievements else "") + "]")
    return "\n".join(rows)


def build_turn_prompt(header: str, obs_text: str, ledger: List["LedgerEntry"]) -> str:
    return "\n\n".join([
        header,
        render_ledger(ledger),
        obs_text,
        'Choose the next subgoal. Reply with the JSON object only.',
    ])


def extract_decision(text: str) -> Dict[str, Any]:
    """Pull the first balanced JSON object out of an LLM response (tolerating code
    fences / surrounding prose). Raises ValueError if none parses."""
    depth, start = 0, None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    start = None  # keep scanning for the next candidate
    raise ValueError("no JSON object found in policy response")


# --- ledger + driver ---------------------------------------------------------
@dataclass
class LedgerEntry:
    turn: int
    think: str
    subgoal: Dict[str, Any]
    status: str
    reason: str
    steps: int
    reward: float
    achievements: List[str]
    floor: int
    env_step: int


@dataclass
class TurnContext:
    """Everything a policy might need. An LLM policy uses the prompt strings; a
    scripted policy may peek at the live env/executor (a test stand-in only)."""
    turn: int
    env: Any
    executor: Executor
    obs_text: str
    header: str
    system_prompt: str
    turn_prompt: str
    ledger: List[LedgerEntry]


class HierarchicalAgent:
    def __init__(self, env, executor: Executor, policy, max_turns: int = 40,
                 max_env_steps: Optional[int] = None):
        self.env = env
        self.ex = executor
        self.policy = policy
        self.max_turns = max_turns
        self.max_env_steps = max_env_steps
        self.system_prompt = build_system_prompt()
        self.ledger: List[LedgerEntry] = []
        self.total_reward = 0.0
        self.done = False

    def step_turn(self) -> LedgerEntry:
        turn = len(self.ledger) + 1
        obs = self.env.obs()
        header = build_header(self.env, self.ledger)
        turn_prompt = build_turn_prompt(header, obs, self.ledger)
        ctx = TurnContext(turn, self.env, self.ex, obs, header,
                          self.system_prompt, turn_prompt, self.ledger)
        decision = self.policy.act(ctx) or {}
        subgoal = decision.get("subgoal") or {}
        result = dispatch(self.ex, subgoal)
        self.total_reward += result.reward
        entry = LedgerEntry(
            turn, str(decision.get("think", "")), subgoal, result.status,
            result.reason, result.steps, result.reward, list(result.achievements),
            int(self.env.state.player_level), self.env.t,
        )
        self.ledger.append(entry)
        if "episode ended" in result.reason:
            self.done = True
        if self.max_env_steps is not None and self.env.t >= self.max_env_steps:
            self.done = True
        return entry

    def run(self) -> List[LedgerEntry]:
        while not self.done and len(self.ledger) < self.max_turns:
            self.step_turn()
        return self.ledger


# --- policies ----------------------------------------------------------------
class ScriptedPolicy:
    """A deterministic stand-in (no model): a reactive early-game tech tree that
    also resolves "explore first" interruptions by exploring. Exercises the loop's
    explore<->mine<->craft chaining and SkillResult feedback, not intelligence."""

    def __init__(self):
        self._i = 0

    def _explore(self, why: str) -> Dict[str, Any]:
        d = DIRECTIONS[self._i % len(DIRECTIONS)]
        self._i += 1
        return {"think": why, "subgoal": {"name": "explore", "args": {"direction": d}}}

    def act(self, ctx: TurnContext) -> Dict[str, Any]:
        inv = ctx.env.state.inventory
        wood, stone, pick = int(inv.wood), int(inv.stone), int(inv.pickaxe)
        last = ctx.ledger[-1] if ctx.ledger else None
        if last and last.status == "interrupted" and "explore first" in last.reason:
            return self._explore("target not in view; reveal more map")
        if pick < 1 and wood >= 3:
            return {"think": "have wood; make a table+wood pickaxe",
                    "subgoal": {"name": "craft", "args": {"item": "wood_pickaxe"}}}
        if wood < 3:
            return {"think": "need wood for a table and pickaxe",
                    "subgoal": {"name": "mine", "args": {"resource": "wood", "count": 3}}}
        if pick >= 1 and stone < 1:
            return {"think": "have a pickaxe; mine stone",
                    "subgoal": {"name": "mine", "args": {"resource": "stone", "count": 1}}}
        if pick < 2 and wood >= 1 and stone >= 1:
            return {"think": "upgrade to a stone pickaxe",
                    "subgoal": {"name": "craft", "args": {"item": "stone_pickaxe"}}}
        return self._explore("tech tree satisfied for now; scout for more")
