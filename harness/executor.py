"""Scripted skill executor over CraftaxTextEnv (full Craftax).

The first skill is navigate_to. Skills are macro-actions (options) that expand
into primitive moves; the executor turns a subgoal into primitives via BFS +
precondition checks and returns a closed-loop status contract.

Observation discipline: the executor plans ONLY over tiles it has actually
observed. `SeenMemory` accumulates the visible window each step (via
craftax_text.visible_block_window, which exposes only light-lit in-window tiles —
never the raw, unobserved state.map). BFS treats unseen tiles as non-walkable, so
navigate_to never routes through fog. Where the target is unknown/unreachable
through seen space it returns "interrupted" (the caller can explore).

Movement rules (verified against craftax core):
  - walkable = in bounds, not a SOLID_BLOCK, not water, not lava
    (COLLISION_LAND_CREATURE); mobs block dynamically and are handled reactively.
  - one direction action moves AND faces in a single step.
"""
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from craftax.craftax.constants import (
    BlockType,
    ItemType,
    MONSTERS_KILLED_TO_CLEAR_LEVEL,
    OBS_DIM,
    SOLID_BLOCK_MAPPING,
    SOLID_BLOCKS,
)
from craftax.craftax.util.game_logic_utils import (get_max_drink, get_max_energy,
                                                   get_max_food)

from craftax_env import ACTION_TO_INT
from craftax_text import light_window, visible_block_window, visible_item_window

# hostile mob classes (state attr) for fight
_HOSTILE_CLASSES = {"melee": "melee_mobs", "ranged": "ranged_mobs", "passive": "passive_mobs"}

UNKNOWN = -1
# terrain the player cannot stand on
_BLOCKED = (
    {int(b) for b in SOLID_BLOCKS}
    | {BlockType.WATER.value, BlockType.LAVA.value,
       BlockType.OUT_OF_BOUNDS.value, BlockType.INVALID.value}
)
_DELTA_TO_ACTION = {(-1, 0): "up", (1, 0): "down", (0, -1): "left", (0, 1): "right"}
_ACTION_TO_DELTA = {v: k for k, v in _DELTA_TO_ACTION.items()}
_NEIGHBORS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# resource -> (mineable block ids, required pickaxe level, inventory field).
# All targets are SOLID, so navigate_to(adjacent) + face + do works. One `do`
# collects one unit (verified against do_action in craftax core).
RESOURCES = {
    "wood": ((BlockType.TREE.value,), 0, "wood"),
    "stone": ((BlockType.STONE.value,), 1, "stone"),
    "coal": ((BlockType.COAL.value,), 1, "coal"),
    "iron": ((BlockType.IRON.value,), 2, "iron"),
    "diamond": ((BlockType.DIAMOND.value,), 3, "diamond"),
    "sapphire": ((BlockType.SAPPHIRE.value,), 4, "sapphire"),
    "ruby": ((BlockType.RUBY.value,), 4, "ruby"),
}
PICKAXE_NAME = {0: "none", 1: "wood", 2: "stone", 3: "iron", 4: "diamond"}

# player_direction (1-4) -> (dr, dc)
_PDIR = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
# the 8 CLOSE_BLOCKS offsets (is_near_block neighborhood)
_CLOSE = [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]
_SOLID = {int(b) for b in SOLID_BLOCKS}

# blocks/items worth surfacing to the planner the moment explore first reveals one
_INTEREST_BLOCKS = {
    BlockType.TREE.value, BlockType.STONE.value, BlockType.COAL.value,
    BlockType.IRON.value, BlockType.DIAMOND.value, BlockType.SAPPHIRE.value,
    BlockType.RUBY.value, BlockType.WATER.value, BlockType.CRAFTING_TABLE.value,
    BlockType.FURNACE.value, BlockType.CHEST.value, BlockType.FOUNTAIN.value,
    BlockType.ENCHANTMENT_TABLE_FIRE.value, BlockType.ENCHANTMENT_TABLE_ICE.value,
}
_INTEREST_ITEMS = {
    ItemType.LADDER_DOWN.value, ItemType.LADDER_UP.value,
    ItemType.LADDER_DOWN_BLOCKED.value,
}

# craft item -> recipe. mats: field->qty; table/furnace: adjacency needed;
# action: the make_* primitive; probe: inventory field that must increase on success.
CRAFTABLES = {
    "wood_pickaxe": dict(action="make_wood_pickaxe", mats={"wood": 1}, table=True, furnace=False, probe="pickaxe"),
    "stone_pickaxe": dict(action="make_stone_pickaxe", mats={"wood": 1, "stone": 1}, table=True, furnace=False, probe="pickaxe"),
    "iron_pickaxe": dict(action="make_iron_pickaxe", mats={"wood": 1, "stone": 1, "iron": 1, "coal": 1}, table=True, furnace=True, probe="pickaxe"),
    "diamond_pickaxe": dict(action="make_diamond_pickaxe", mats={"wood": 1, "diamond": 3}, table=True, furnace=False, probe="pickaxe"),
    "wood_sword": dict(action="make_wood_sword", mats={"wood": 1}, table=True, furnace=False, probe="sword"),
    "stone_sword": dict(action="make_stone_sword", mats={"wood": 1, "stone": 1}, table=True, furnace=False, probe="sword"),
    "iron_sword": dict(action="make_iron_sword", mats={"wood": 1, "stone": 1, "iron": 1, "coal": 1}, table=True, furnace=True, probe="sword"),
    "diamond_sword": dict(action="make_diamond_sword", mats={"wood": 1, "diamond": 2}, table=True, furnace=False, probe="sword"),
    "iron_armour": dict(action="make_iron_armour", mats={"iron": 3, "coal": 3}, table=True, furnace=True, probe="armour"),
    "diamond_armour": dict(action="make_diamond_armour", mats={"diamond": 3}, table=True, furnace=False, probe="armour"),
    "arrow": dict(action="make_arrow", mats={"wood": 1, "stone": 1}, table=True, furnace=False, probe="arrows"),
    "torch": dict(action="make_torch", mats={"coal": 1, "wood": 1}, table=True, furnace=False, probe="torches"),
}

# place item -> (place_* primitive, inventory field consumed, units per placement).
# Placement lands on the FACED tile (verified against place_block in craftax core);
# success is detected by the material being consumed, so we don't have to mirror
# every per-tile validity predicate.
PLACEABLES = {
    "stone": dict(action="place_stone", mat="stone", qty=1),
    "table": dict(action="place_table", mat="wood", qty=2),
    "furnace": dict(action="place_furnace", mat="stone", qty=1),
    "torch": dict(action="place_torch", mat="torches", qty=1),
    "plant": dict(action="place_plant", mat="sapling", qty=1),
}
# food/water sources for the block-interaction skills (a `do` on the faced tile)
_WATER_BLOCKS = (BlockType.WATER.value, BlockType.FOUNTAIN.value)


def craft_requirements(item: str, near_table: bool, near_furnace: bool) -> Dict[str, int]:
    """Total materials needed to craft `item` FROM HERE — recipe PLUS any station this
    call will have to place first.

    SINGLE SOURCE OF TRUTH, and it exists because there were two.
    `candidate_actions` decides what to OFFER; `Executor.craft` decides what SUCCEEDS.
    Until 2026-07-26 the constructor charged only `recipe["mats"]` while the skill also
    charged the table's 2 wood, so `craft` was offered in states where it could not
    possibly succeed. Measured on the frozen policy over 10 dev worlds: 118 of 394
    decisions (30%) were failed crafts, 101 of them "insufficient materials … missing
    {'wood': 1 or 2}" — exactly the uncharged table.

    Note the requirement is CONTEXT-DEPENDENT: a wood_pickaxe needs wood>=1 standing next
    to a table, but wood>=3 with no table adjacent. Callers must pass the adjacency they
    actually observe, not a default.
    """
    rec = CRAFTABLES[item]
    req = dict(rec["mats"])
    if rec["table"] and not near_table:
        sp = PLACEABLES["table"]
        req[sp["mat"]] = req.get(sp["mat"], 0) + sp["qty"]
    if rec["furnace"] and not near_furnace:
        sp = PLACEABLES["furnace"]
        req[sp["mat"]] = req.get(sp["mat"], 0) + sp["qty"]
    return req


# Stations `go_to` can walk to. Only fixed, re-usable structures belong here: walking to
# a resource deposit is already `mine`'s job (navigate->face->do in one decision).
_GOTO_TARGETS = {
    "crafting_table": (BlockType.CRAFTING_TABLE.value,),
    "furnace": (BlockType.FURNACE.value,),
}

# Tools are TIERED, not counted: the inventory stores one `pickaxe`/`sword` level, so
# make_wood_pickaxe is a silent no-op once you hold any pickaxe.
_CRAFT_TIER = {"wood": 1, "stone": 2, "iron": 3}


def craft_would_upgrade(item: str, probe_value: int) -> bool:
    """Would crafting `item` change anything, given the current probe reading?

    The other half of the craft precondition, and the residual after the materials fix:
    with wood in hand and a table adjacent, `craft wood_pickaxe` still fails as
    "did not take effect (precondition unmet)" when a pickaxe is already held. The
    executor only discovers this AFTER spending the decision (it probes before/after);
    the constructor can know it up front.

    Tiered probes (`pickaxe`, `sword`) upgrade only strictly upward. Counted probes
    (`torches`, `arrows`) stack, so crafting always has an effect.
    """
    probe = CRAFTABLES[item]["probe"]
    if probe not in ("pickaxe", "sword"):
        return True
    tier = _CRAFT_TIER.get(item.split("_")[0])
    return tier is None or int(probe_value) < tier


# --- satiation preconditions -------------------------------------------------
# SINGLE SOURCE OF TRUTH for "would this survival skill actually do anything?", for
# exactly the reason `craft_requirements` exists: the precondition lived only in the
# skill, so `candidate_actions` offered actions the executor then refused.
#
# Cost of the omission, measured on S6's 3,200-decision TEACHER stream (2026-07-28):
# 1,964 decisions (61.4%) consumed ZERO env steps — `sleep` 1,062/1,082 (98.2%)
# "energy already full", `drink_water` 834/863 (96.6%) "drink already full".
#
# A zero-step action is not merely wasted, it is PREFERRED: the planner scores
# `Q = r_hat + gamma^tau * V`, and tau=0 with an unchanged successor state makes
# `Q(no-op) = V(s)` undiscounted — the only action that escapes the duration discount.
# Every real action must first cover `(1 - gamma^tau) * V(s)`, so the no-op becomes an
# absorbing self-loop. Mean mu_q on that stream: sleep 1.407, drink 1.600 vs mine 0.918,
# while realised reward ran the other way (mine 0.149, craft 1.154 vs sleep 0.011).
#
# These take the STATE, not an Executor, so the constructor and the skill can both call
# them without either owning the other. Gated by `scripts/noop_grounding_demo.py`
# (NOOPGATE_OK), which branch-and-replays every offered candidate and asserts none
# consumes zero steps.
def sleep_would_help(state) -> bool:
    """Would `sleep` restore anything? False at full energy, where the skill returns
    `success` in 0 steps for 0 reward."""
    return int(state.player_energy) < int(get_max_energy(state))


def drink_would_help(state) -> bool:
    """Would `drink_water` restore anything? False at full drink."""
    return int(state.player_drink) < int(get_max_drink(state))


def eat_would_help(state) -> bool:
    """Would `eat` restore anything? False at full food.

    Included for symmetry and because the same silent no-op is reachable, though it cost
    far less than the other two (55 zero-step `eat`s in the S6 stream, all `interrupted`
    rather than `success`).
    """
    return int(state.player_food) < int(get_max_food(state))


@dataclass
class SkillResult:
    status: str            # "success" | "failure" | "interrupted"
    reason: str = ""
    steps: int = 0
    reward: float = 0.0
    achievements: List[str] = field(default_factory=list)
    events: List[str] = field(default_factory=list)
    obs_text: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "success"


class SeenMemory:
    """Per-floor memory of observed block ids (UNKNOWN where never seen)."""

    def __init__(self, n_levels: int, h: int, w: int):
        self.known = np.full((n_levels, h, w), UNKNOWN, dtype=np.int32)
        self.known_items = np.full((n_levels, h, w), UNKNOWN, dtype=np.int32)

    def update(self, state) -> None:
        vbw = visible_block_window(state)  # (9,11): block id where visible, else -1
        viw = visible_item_window(state)   # (9,11): item id where visible, else -1
        px, py = int(state.player_position[0]), int(state.player_position[1])
        cr, cc = OBS_DIM[0] // 2, OBS_DIM[1] // 2
        lvl = int(state.player_level)
        h, w = self.known.shape[1:]
        for r in range(OBS_DIM[0]):
            for c in range(OBS_DIM[1]):
                b = int(vbw[r, c])
                if b == UNKNOWN:
                    continue
                x, y = px + r - cr, py + c - cc
                if 0 <= x < h and 0 <= y < w:
                    self.known[lvl, x, y] = b
                    self.known_items[lvl, x, y] = int(viw[r, c])

    def walkable(self, lvl: int, x: int, y: int) -> bool:
        """Is (x, y) a known, non-blocking tile? OFF-MAP counts as not walkable.

        The bounds check is not defensive padding — without it this is wrong in two
        different ways, and both bit on 2026-07-28:
          * high edge: `known[lvl, 48, y]` on a 48-wide axis raises IndexError and killed
            an S6 collection at round 5 (`place` -> `_step_to_walkable_neighbor` at the
            map edge). A crash, at least, is loud.
          * low edge: numpy accepts NEGATIVE indices, so x=-1 silently reads the tile on
            the OPPOSITE side of the map and reports it as the neighbour. No exception,
            just wrong pathfinding — the far more dangerous half.
        `SeenMemory.update` a few lines above already guards `0 <= x < h and 0 <= y < w`;
        this method simply never did, and every caller (`_bfs`, `reachable`,
        `_step_to_walkable_neighbor`, `navigate_to`) inherited the gap.
        """
        h, w = self.known.shape[1], self.known.shape[2]
        if not (0 <= x < h and 0 <= y < w):
            return False
        b = int(self.known[lvl, x, y])
        return b != UNKNOWN and b not in _BLOCKED


def _bfs(walkable, start, goal, shape, adjacent: bool) -> Optional[List[Tuple[int, int]]]:
    """Shortest path (excluding start) over walkable tiles. In adjacent mode the
    goal need not be walkable; we stop on a walkable tile next to it."""
    sr, sc = start
    gr, gc = goal
    h, w = shape

    def is_target(r, c):
        return abs(r - gr) + abs(c - gc) == 1 if adjacent else (r, c) == (gr, gc)

    if is_target(sr, sc):
        return []
    prev = {(sr, sc): None}
    q = deque([(sr, sc)])
    while q:
        r, c = q.popleft()
        for dr, dc in _NEIGHBORS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w) or (nr, nc) in prev:
                continue
            if not walkable(nr, nc):
                continue
            prev[(nr, nc)] = (r, c)
            if is_target(nr, nc):
                path = [(nr, nc)]
                p = prev[(nr, nc)]
                while p is not None and p != (sr, sc):
                    path.append(p)
                    p = prev[p]
                path.reverse()
                return path
            q.append((nr, nc))
    return None


class Executor:
    STUCK_LIMIT = 5

    def __init__(self, env):
        self.env = env
        n_levels, h, w = self.env.state.map.shape
        self.mem = SeenMemory(n_levels, h, w)
        self.mem.update(self.env.state)

    def _pos(self) -> Tuple[int, int]:
        p = self.env.state.player_position
        return int(p[0]), int(p[1])

    def _level(self) -> int:
        return int(self.env.state.player_level)

    def navigate_to(self, target, adjacent: bool = False, max_steps: int = 200) -> SkillResult:
        """Walk to `target` (absolute (x, y) on the current floor). If adjacent,
        stop on a tile next to the target (for mining/fighting solids)."""
        gx, gy = int(target[0]), int(target[1])
        shape = self.env.state.map.shape[1:]
        events, achievements = [], []
        total_reward, stuck = 0.0, 0

        for steps in range(max_steps):
            self.mem.update(self.env.state)
            lvl = self._level()
            px, py = self._pos()

            reached = (abs(px - gx) + abs(py - gy) == 1) if adjacent else ((px, py) == (gx, gy))
            if reached:
                where = "adjacent to" if adjacent else "reached"
                return SkillResult("success", f"{where} ({gx},{gy})", steps,
                                   total_reward, achievements, events, self.env.obs())

            path = _bfs(lambda r, c: self.mem.walkable(lvl, r, c),
                        (px, py), (gx, gy), shape, adjacent)
            if not path:
                return SkillResult("interrupted", "no known path to target (explore first)",
                                   steps, total_reward, achievements, events, self.env.obs())

            nr, nc = path[0]
            action = _DELTA_TO_ACTION[(nr - px, nc - py)]
            res = self.env.step(action)
            total_reward += res.reward
            if res.achievements_unlocked:
                achievements += res.achievements_unlocked
                events.append(f"step{steps + 1}: unlocked {res.achievements_unlocked}")
            if res.done:
                return SkillResult("interrupted", "episode ended", steps + 1,
                                   total_reward, achievements, events, res.obs_text)

            if self._pos() == (px, py):  # didn't move -> blocked (e.g., a mob)
                stuck += 1
                events.append(f"step{steps + 1}: blocked moving {action} at ({px},{py})")
                if stuck >= self.STUCK_LIMIT:
                    return SkillResult("interrupted", "stuck (blocked repeatedly)", steps + 1,
                                       total_reward, achievements, events, self.env.obs())
            else:
                stuck = 0

        return SkillResult("interrupted", "max_steps reached", max_steps,
                           total_reward, achievements, events, self.env.obs())

    def explore(self, direction: Optional[str] = None, max_steps: int = 120,
                reveal_target: int = 40, stop_on_discovery: bool = True) -> SkillResult:
        """Expand SeenMemory by walking to the frontier of seen space (optionally
        biased toward `direction` in {up, down, left, right}). The fallback skill
        when a target isn't visible yet: it reveals new tiles and, by default,
        stops as soon as something notable (resource, station, ladder, chest)
        first comes into view, so the planner gets a fresh decision point.

        A frontier is a reachable, seen-walkable tile bordering an UNKNOWN tile;
        stepping onto it pulls the 9x11 window over the fog beyond. Observation
        discipline is preserved — it only ever moves over already-seen tiles and
        learns new ones through the normal windowed update (never the raw map).

        Returns `success` if any new tiles were revealed (reason names what was
        found), else `interrupted` (the reachable area is fully explored / boxed
        in / dark — on dark floors moving can't reveal unlit tiles; place torches)."""
        if direction is not None and direction not in _ACTION_TO_DELTA:
            return SkillResult("failure",
                               f"unknown direction {direction!r} "
                               f"(use one of {sorted(_ACTION_TO_DELTA)})",
                               0, 0.0, [], [], self.env.obs())

        self.mem.update(self.env.state)
        lvl = self._level()
        n_known0 = self._n_known(lvl)
        blocks0, items0 = self._seen_block_ids(lvl), self._seen_item_ids(lvl)
        start_t = self.env.t
        events, achievements, total = [], [], 0.0
        exhausted, discovered = set(), []

        while (self.env.t - start_t) < max_steps:
            self.mem.update(self.env.state)
            if self._n_known(lvl) - n_known0 >= reveal_target:
                break
            target = self._nearest_frontier(lvl, direction, exhausted)
            if target is None:
                break  # no reachable frontier left -> region fully explored

            before = self._n_known(lvl)
            nav = self.navigate_to(target, adjacent=False,
                                   max_steps=max_steps - (self.env.t - start_t))
            total += nav.reward
            achievements += nav.achievements
            events += nav.events
            self.mem.update(self.env.state)
            if self._n_known(lvl) <= before:
                exhausted.add(target)  # revealed nothing (window edge / darkness)

            if stop_on_discovery:
                discovered = self._new_interesting(lvl, blocks0, items0)
                if discovered:
                    events.append("found: " + ", ".join(discovered))
                    break
            if not nav.ok:
                if "episode ended" in nav.reason:
                    return SkillResult("interrupted", "episode ended",
                                       self.env.t - start_t, total, achievements,
                                       events, self.env.obs())
                events.append(f"leg interrupted: {nav.reason}")
                break

        gained = self._n_known(lvl) - n_known0
        if gained > 0:
            reason = f"revealed {gained} new tiles"
            if discovered:
                reason += f"; found {', '.join(discovered)}"
            status = "success"
        else:
            status = "interrupted"
            reason = ("no new tiles revealed (reachable area fully explored / "
                      "boxed in / dark — place torches to see unlit tiles)")
        return SkillResult(status, reason, self.env.t - start_t, total,
                           achievements, events, self.env.obs())

    def _n_known(self, lvl: int) -> int:
        return int((self.mem.known[lvl] != UNKNOWN).sum())

    def _seen_block_ids(self, lvl: int) -> set:
        k = self.mem.known[lvl]
        return {int(b) for b in np.unique(k) if int(b) != UNKNOWN}

    def _seen_item_ids(self, lvl: int) -> set:
        k = self.mem.known_items[lvl]
        return {int(b) for b in np.unique(k)
                if int(b) not in (UNKNOWN, ItemType.NONE.value)}

    def _new_interesting(self, lvl: int, blocks0: set, items0: set) -> List[str]:
        """Human labels for interesting block/item types seen now but not at entry."""
        new_b = (self._seen_block_ids(lvl) & _INTEREST_BLOCKS) - blocks0
        new_i = (self._seen_item_ids(lvl) & _INTEREST_ITEMS) - items0
        return ([BlockType(b).name.lower() for b in sorted(new_b)]
                + [ItemType(i).name.lower() for i in sorted(new_i)])

    def _nearest_frontier(self, lvl: int, direction, exclude) -> Optional[Tuple[int, int]]:
        """Reachable seen-walkable tile bordering UNKNOWN. With no direction, the
        nearest one; with a direction, the one advancing furthest that way (falling
        back to nearest if none lie in that direction)."""
        known = self.mem.known[lvl]
        h, w = known.shape
        dist = self.reachable()
        px, py = self._pos()
        ddr, ddc = _ACTION_TO_DELTA[direction] if direction is not None else (0, 0)

        cands = []  # (progress_in_direction, distance, (x, y))
        for (x, y), d in dist.items():
            if (x, y) in exclude:
                continue
            for dr, dc in _NEIGHBORS:
                nx, ny = x + dr, y + dc
                if 0 <= nx < h and 0 <= ny < w and int(known[nx, ny]) == UNKNOWN:
                    cands.append(((x - px) * ddr + (y - py) * ddc, d, (x, y)))
                    break
        if not cands:
            return None
        if direction is not None:
            pool = [c for c in cands if c[0] > 0] or cands
            pool.sort(key=lambda c: (-c[0], c[1]))  # furthest in direction, then nearest
            return pool[0][2]
        cands.sort(key=lambda c: c[1])  # nearest frontier
        return cands[0][2]

    def _interact_with_nearest(self, block_ids, probe, count, label,
                               max_steps) -> SkillResult:
        """Repeatedly go to the nearest SEEN target block (id in `block_ids`) that
        has a reachable neighbor, face it, and `do` it, until `probe` has risen
        `count` times or nothing reachable remains. `probe` is a 0-arg callable
        returning an int that strictly increases on a successful interaction (an
        inventory unit for mine, player_drink for drink, player_food for eat-plant).
        This is the navigate->face->do loop shared by mine/drink_water/eat."""
        events, achievements = [], []
        total_reward, steps, done_n = 0.0, 0, 0
        tried = set()  # targets where `do` did nothing -> skip

        def acc(res):
            nonlocal total_reward, achievements
            total_reward += res.reward
            achievements += res.achievements_unlocked

        while done_n < count and steps < max_steps:
            self.mem.update(self.env.state)
            target = self._nearest_block(block_ids, exclude=tried)
            if target is None:
                status = "success" if done_n else "interrupted"
                reason = (f"{label} {done_n}/{count}" if done_n
                          else f"no reachable {label} target in view (explore first)")
                return SkillResult(status, reason, steps, total_reward, achievements,
                                   events, self.env.obs())

            nav = self.navigate_to(target, adjacent=True, max_steps=max_steps - steps)
            steps += nav.steps
            total_reward += nav.reward
            achievements += nav.achievements
            events += nav.events
            if not nav.ok:
                return SkillResult("interrupted",
                                   f"could not reach {label} target at {target}: "
                                   f"{nav.reason} ({label} {done_n}/{count})",
                                   steps, total_reward, achievements, events, self.env.obs())

            # face the target (walkable ones aside, most are solid so this only turns)
            px, py = self._pos()
            face = _DELTA_TO_ACTION[(target[0] - px, target[1] - py)]
            if int(self.env.state.player_direction) != ACTION_TO_INT[face]:
                r = self.env.step(face)
                steps += 1
                acc(r)
                if r.done:
                    return SkillResult("interrupted", "episode ended", steps, total_reward,
                                       achievements, events, r.obs_text)

            before = probe()
            r = self.env.step("do")
            steps += 1
            acc(r)
            if r.done:
                return SkillResult("interrupted", "episode ended", steps, total_reward,
                                   achievements, events, r.obs_text)
            if probe() > before:
                done_n += 1
                events.append(f"{label} ({done_n}/{count}) at {target}")
                self.mem.update(self.env.state)  # target may have changed/vanished
            else:
                tried.add(target)
                events.append(f"do at {target} yielded nothing; skipping")

        return SkillResult("success" if done_n >= count else "interrupted",
                           f"{label} {done_n}/{count}",
                           steps, total_reward, achievements, events, self.env.obs())

    def mine(self, resource: str, count: int = 1, max_steps: int = 300) -> SkillResult:
        """Mine `count` of `resource` from the nearest SEEN deposits: for each,
        navigate adjacent, face it, and `do`. Precondition: the right pickaxe."""
        if resource not in RESOURCES:
            return SkillResult("failure", f"unknown resource {resource!r}", 0, 0.0, [], [],
                               self.env.obs())
        blocks, req_pick, inv_field = RESOURCES[resource]
        have = int(self.env.state.inventory.pickaxe)
        if have < req_pick:
            return SkillResult("failure",
                               f"need {PICKAXE_NAME[req_pick]} pickaxe to mine {resource} "
                               f"(have {PICKAXE_NAME[have]})",
                               0, 0.0, [], [], self.env.obs())
        return self._interact_with_nearest(
            blocks, lambda: int(getattr(self.env.state.inventory, inv_field)),
            count, resource, max_steps)

    def drink_water(self, count: Optional[int] = None, max_steps: int = 200) -> SkillResult:
        """Drink from the nearest seen water/fountain tile. Each `do` on water
        restores +1 drink and resets thirst (the tile is not consumed); unlocks
        collect_drink. With `count` omitted, drinks until full."""
        max_drink = int(get_max_drink(self.env.state))
        cur = int(self.env.state.player_drink)
        # `drink_would_help` is the SAME function `candidate_actions` gates the offer on,
        # so the menu and the skill cannot disagree about this again (LESSONS R1). With
        # `count` omitted — the only form the constructor offers — the two are identical.
        n = (max_drink - cur) if count is None else int(count)
        if n <= 0 or (count is None and not drink_would_help(self.env.state)):
            return SkillResult("success", f"drink already full ({cur}/{max_drink})",
                               0, 0.0, [], [], self.env.obs())
        return self._interact_with_nearest(
            _WATER_BLOCKS, lambda: int(self.env.state.player_drink),
            n, "drink_water", max_steps)

    def eat(self, target: str = "auto", count: int = 1,
            health_floor: int = 0, max_steps: int = 200) -> SkillResult:
        """Eat to restore food. `target`: 'plant' (`do` a ripe plant, +4 food),
        'cow'/'bat'/'snail' (kill the nearest passive mob, +6 food), or 'auto' (a
        reachable ripe plant if any, else a passive mob). Which passive mob is present
        depends on the floor (surface=cow, dungeon=snail, mines=bat), and the unlocked
        achievement (eat_cow/eat_bat/eat_snail) follows the mob actually eaten.

        `health_floor` was 2 until 2026-07-28, matching `fight`'s old default — and since
        `_eat_cow` passes it straight into `fight`, `eat` went on silently reimposing the
        veto that was deliberately removed from `fight` on 2026-07-27. The owner's rule:
        low-health combat is PERMITTED by the game, so a skill that retreats on the
        agent's behalf is a strategy veto, not a rule. Measured cost: 4/70 `eat` offers
        returned 0 steps as "low health (2)" (NOOPGATE check 1), and since a no-op has
        tau=0 it is the discount-free action the planner prefers. It is also a death
        spiral — starving AND hurt meant the agent could not eat at all."""
        target = str(target).lower().strip()
        if target not in ("auto", "plant", "cow", "bat", "snail"):
            return SkillResult("failure", f"unknown eat target {target!r} "
                               "(choose from cow/plant/bat/snail/auto)", 0, 0.0, [], [], self.env.obs())
        if target in ("plant", "auto"):
            res = self._interact_with_nearest(
                (BlockType.RIPE_PLANT.value,), lambda: int(self.env.state.player_food),
                count, "eat_plant", max_steps)
            if target == "plant" or res.status == "success":
                return res
        return self._eat_cow(count, health_floor, max_steps)

    def _eat_cow(self, count: int, health_floor: int, max_steps: int) -> SkillResult:
        """Eat cows by killing the nearest visible passive mobs (each grants +6
        food and unlocks eat_cow). Reuses the fight approach loop, but success is
        read from FOOD GAINED, not `killed`: passive kills don't bump
        monsters_killed (fight's own kill counter), so fight under-reports here."""
        f0 = int(self.env.state.player_food)
        res = self.fight(count=count, types=("passive",),
                         health_floor=health_floor, max_steps=max_steps)
        gained = int(self.env.state.player_food) - f0
        eaten = next((a.split("_", 1)[1] for a in res.achievements
                      if a in ("eat_cow", "eat_bat", "eat_snail")), None)
        if gained > 0 or eaten is not None:
            res.status = "success"
            res.reason = f"ate {eaten or 'a passive mob'} (+{gained} food)"
        elif "no hostile" in res.reason:
            res.status = "interrupted"
            res.reason = "no reachable passive mob in view (explore first)"
        return res

    def sleep(self, max_steps: int = 200) -> SkillResult:
        """Sleep to restore energy. Issues `sleep`; the engine then forces the
        player to keep sleeping (auto-noop) until energy is full, at which point it
        wakes and unlocks wake_up. No-op if energy is already full. NOTE: sleeping
        multiplies incoming melee damage, so do it somewhere safe."""
        max_e = int(get_max_energy(self.env.state))
        e0 = int(self.env.state.player_energy)
        # Shared with `candidate_actions` — see `sleep_would_help` (LESSONS R1).
        if not sleep_would_help(self.env.state):
            return SkillResult("success", f"energy already full ({e0}/{max_e})",
                               0, 0.0, [], [], self.env.obs())
        events, achievements, total = [], [], 0.0
        start_t = self.env.t
        r = self.env.step("sleep")
        total += r.reward
        achievements += r.achievements_unlocked
        if r.done:
            return SkillResult("interrupted", "episode ended", 1, total,
                               achievements, events, r.obs_text)
        # keep sleeping (any action is forced to noop) until the engine wakes us
        while bool(self.env.state.is_sleeping) and (self.env.t - start_t) < max_steps:
            r = self.env.step("noop")
            total += r.reward
            achievements += r.achievements_unlocked
            if r.done:
                return SkillResult("interrupted", "episode ended", self.env.t - start_t,
                                   total, achievements, events, self.env.obs())
        e1 = int(self.env.state.player_energy)
        woke = not bool(self.env.state.is_sleeping)
        steps = self.env.t - start_t
        if woke:
            events.append(f"slept and woke (energy {e0}->{e1}/{max_e})")
            return SkillResult("success", f"slept, energy {e0}->{e1}/{max_e} (woke up)",
                               steps, total, achievements, events, self.env.obs())
        return SkillResult("interrupted",
                           f"still sleeping at max_steps (energy {e0}->{e1}/{max_e})",
                           steps, total, achievements, events, self.env.obs())

    def place(self, item: str, direction: Optional[str] = None,
              max_tries: int = 6) -> SkillResult:
        """Place `item` (stone/table/furnace/torch/plant) on a faced tile. If
        `direction` is given, orient that way first; otherwise place on the current
        facing and, if that tile isn't valid, reface a fresh tile and retry.
        Precondition: enough of the material (does not gather it). Placement success
        is read from the material being consumed."""
        if item not in PLACEABLES:
            return SkillResult("failure",
                               f"unknown placeable {item!r} (choose from {sorted(PLACEABLES)})",
                               0, 0.0, [], [], self.env.obs())
        if direction is not None and direction not in _ACTION_TO_DELTA:
            return SkillResult("failure", f"unknown direction {direction!r} "
                               f"(use one of {sorted(_ACTION_TO_DELTA)})",
                               0, 0.0, [], [], self.env.obs())
        rec = PLACEABLES[item]
        have = self._inv(rec["mat"])
        if have < rec["qty"]:
            return SkillResult("failure",
                               f"insufficient {rec['mat']} to place {item}: "
                               f"have {have}, need {rec['qty']}",
                               0, 0.0, [], [], self.env.obs())

        events, achievements, total = [], [], [0.0]
        start_t = self.env.t

        def step(action):
            r = self.env.step(action)
            total[0] += r.reward
            achievements.extend(r.achievements_unlocked)
            return r

        if direction is not None and \
                int(self.env.state.player_direction) != ACTION_TO_INT[direction]:
            if step(direction).done:
                return SkillResult("interrupted", "episode ended", self.env.t - start_t,
                                   total[0], achievements, events, self.env.obs())

        for _ in range(max_tries):
            before = self._inv(rec["mat"])
            r = step(rec["action"])
            if r.done:
                return SkillResult("interrupted", "episode ended", self.env.t - start_t,
                                   total[0], achievements, events, self.env.obs())
            if self._inv(rec["mat"]) < before:
                f = self._front()
                where = f"({f[0]},{f[1]})" if f else "ahead"
                events.append(f"placed {item} at {where}")
                return SkillResult("success", f"placed {item} at {where}",
                                   self.env.t - start_t, total[0], achievements,
                                   events, self.env.obs())
            if direction is not None:
                break  # caller pinned a direction; don't wander looking for a tile
            self._step_to_walkable_neighbor(step)  # reface a fresh tile and retry

        return SkillResult("failure",
                           f"could not place {item} (no valid tile faced; "
                           "try a direction or move first)",
                           self.env.t - start_t, total[0], achievements, events,
                           self.env.obs())

    def ascend(self, max_steps: int = 200) -> SkillResult:
        """Go up one floor via the nearest seen up-ladder (no clear requirement,
        unlike descend). Precondition: not already on the top floor (0)."""
        self.mem.update(self.env.state)
        lvl = self._level()
        if lvl <= 0:
            return SkillResult("failure", "already on the top floor (0); cannot ascend",
                               0, 0.0, [], [], self.env.obs())
        ladder = self._nearest_item(lvl, (ItemType.LADDER_UP.value,))
        if ladder is None:
            return SkillResult("interrupted", "no up-ladder in view (explore first)",
                               0, 0.0, [], [], self.env.obs())
        nav = self.navigate_to(ladder, adjacent=False, max_steps=max_steps)
        if not nav.ok:
            return SkillResult("interrupted", f"could not reach up-ladder: {nav.reason}",
                               nav.steps, nav.reward, nav.achievements, nav.events,
                               self.env.obs())
        res = self.env.step("ascend")
        steps = nav.steps + 1
        reward = nav.reward + res.reward
        ach = nav.achievements + res.achievements_unlocked
        events = nav.events + [f"ascend at {ladder}"]
        if int(self.env.state.player_level) < lvl:
            return SkillResult("success", f"ascended to floor {self._level()}",
                               steps, reward, ach, events, self.env.obs())
        return SkillResult("failure", "ascend did not take effect", steps, reward,
                           ach, events, self.env.obs())

    def _nearest_block(self, blocks, exclude=frozenset()):
        """Nearest seen block (id in `blocks`) that has a reachable walkable
        neighbor; returns its (x, y) or None. Shared by mine/drink/eat."""
        lvl = self._level()
        known = self.mem.known[lvl]
        dist = self.reachable()
        best, best_d = None, None
        for x, y in np.argwhere(np.isin(known, list(blocks))):
            cell = (int(x), int(y))
            if cell in exclude:
                continue
            adj = [dist[(cell[0] + dr, cell[1] + dc)]
                   for dr, dc in _NEIGHBORS if (cell[0] + dr, cell[1] + dc) in dist]
            if adj and (best_d is None or min(adj) < best_d):
                best_d, best = min(adj), cell
        return best

    def craft(self, item: str, max_steps: int = 60) -> SkillResult:
        """Craft `item`: ensure materials + adjacency to a crafting table (and a
        furnace if the recipe needs one, placing them if absent), then issue the
        make_* primitive. Single-purpose: it does NOT gather materials — if they
        are short it returns a `failure` listing the shortfall."""
        if item not in CRAFTABLES:
            return SkillResult("failure", f"unknown craftable {item!r}", 0, 0.0, [], [],
                               self.env.obs())
        rec = CRAFTABLES[item]
        events, achievements, total = [], [], [0.0]

        def step(action):
            r = self.env.step(action)
            total[0] += r.reward
            achievements.extend(r.achievements_unlocked)
            if r.achievements_unlocked:
                events.append(f"unlocked {r.achievements_unlocked}")
            return r

        self.mem.update(self.env.state)
        need_table = rec["table"] and not self._near(BlockType.CRAFTING_TABLE.value)
        need_furnace = rec["furnace"] and not self._near(BlockType.FURNACE.value)

        # material precondition (recipe + any placement costs) — SHARED with
        # candidate_actions so the menu cannot offer what this would reject.
        req = craft_requirements(item, not need_table, not need_furnace)
        short = {f: req[f] - self._inv(f) for f in req if self._inv(f) < req[f]}
        if short:
            return SkillResult("failure",
                               f"insufficient materials for {item}: missing {short}",
                               0, 0.0, [], [], self.env.obs())

        # ensure stations
        steps0 = self.env.t
        if need_table and need_furnace:
            placed = self._place_two(step)
        elif need_table:
            placed = self._place_one(BlockType.CRAFTING_TABLE.value, "place_table", step)
        elif need_furnace:
            placed = self._place_one(BlockType.FURNACE.value, "place_furnace", step)
        else:
            placed = True
        if not placed:
            return SkillResult("interrupted", "could not position next to crafting station(s)",
                               self.env.t - steps0, total[0], achievements, events, self.env.obs())

        # craft
        before = self._probe(rec["probe"])
        r = step(rec["action"])
        after = self._probe(rec["probe"])
        steps = self.env.t - steps0
        if r.done:
            return SkillResult("interrupted", "episode ended", steps, total[0],
                               achievements, events, r.obs_text)
        if after > before:
            events.append(f"crafted {item}")
            return SkillResult("success", f"crafted {item}", steps, total[0],
                               achievements, events, self.env.obs())
        return SkillResult("failure", f"craft {item} did not take effect (precondition unmet)",
                           steps, total[0], achievements, events, self.env.obs())

    # --- crafting helpers ------------------------------------------------
    def _inv(self, field: str) -> int:
        return int(getattr(self.env.state.inventory, field))

    def _probe(self, field: str) -> int:
        if field == "armour":
            return int(np.asarray(self.env.state.inventory.armour).sum())
        return self._inv(field)

    def _near(self, block_value: int) -> bool:
        s = self.env.state
        lvl = int(s.player_level)
        m = np.asarray(s.map[lvl])
        px, py = self._pos()
        h, w = m.shape
        for dr, dc in _CLOSE:
            x, y = px + dr, py + dc
            if 0 <= x < h and 0 <= y < w and int(m[x, y]) == block_value:
                return True
        return False

    def _front(self):
        """The currently-faced adjacent tile (always observed): (x, y, block, item)."""
        s = self.env.state
        lvl = int(s.player_level)
        dr, dc = _PDIR[int(s.player_direction)]
        px, py = self._pos()
        x, y = px + dr, py + dc
        h, w = np.asarray(s.map[lvl]).shape
        if not (0 <= x < h and 0 <= y < w):
            return None
        return (x, y, int(np.asarray(s.map[lvl])[x, y]), int(np.asarray(s.item_map[lvl])[x, y]))

    def _front_placeable(self) -> bool:
        f = self._front()
        return f is not None and f[2] not in _SOLID and f[3] == ItemType.NONE.value

    def _place_one(self, block_value, place_action, step, max_tries=6) -> bool:
        """Place `block_value` on a faced tile so it ends up adjacent (near)."""
        if self._near(block_value):
            return True
        for _ in range(max_tries):
            if self._front_placeable():
                r = step(place_action)
                if r.done:
                    return False
                f = self._front()
                if f is not None and f[2] == block_value:
                    return True
            # reface: move onto a walkable neighbour so a fresh tile is ahead
            self._step_to_walkable_neighbor(step)
            if self._near(block_value):
                return True
        return self._near(block_value)

    def _place_two(self, step, max_tries=8) -> bool:
        """Place a table and a furnace both within the 8-neighborhood (for iron
        recipes). Place the table, then step to a tile that keeps it near and
        place the furnace ahead."""
        if not self._place_one(BlockType.CRAFTING_TABLE.value, "place_table", step):
            return False
        table_pos = self._find_near(BlockType.CRAFTING_TABLE.value)
        for _ in range(max_tries):
            if self._near(BlockType.FURNACE.value) and self._near(BlockType.CRAFTING_TABLE.value):
                return True
            if self._near(BlockType.CRAFTING_TABLE.value) and self._front_placeable():
                r = step("place_furnace")
                if r.done:
                    return False
                f = self._front()
                if f is not None and f[2] == BlockType.FURNACE.value and \
                        self._near(BlockType.CRAFTING_TABLE.value):
                    return True
            # move to a walkable neighbour that keeps the table within reach
            self._step_to_walkable_neighbor(step, keep_near=table_pos)
        return self._near(BlockType.FURNACE.value) and self._near(BlockType.CRAFTING_TABLE.value)

    def _find_near(self, block_value):
        s = self.env.state
        m = np.asarray(s.map[int(s.player_level)])
        px, py = self._pos()
        for dr, dc in _CLOSE:
            x, y = px + dr, py + dc
            if 0 <= x < m.shape[0] and 0 <= y < m.shape[1] and int(m[x, y]) == block_value:
                return (x, y)
        return None

    def _step_to_walkable_neighbor(self, step, keep_near=None):
        lvl = self._level()
        px, py = self._pos()
        for d, (dr, dc) in _PDIR.items():
            nx, ny = px + dr, py + dc
            if not self.mem.walkable(lvl, nx, ny):
                continue
            if keep_near is not None and max(abs(nx - keep_near[0]), abs(ny - keep_near[1])) > 1:
                continue
            step({1: "left", 2: "right", 3: "up", 4: "down"}[d])
            return True
        # nothing ideal: just turn to change the faced tile
        step("up")
        return False

    def descend(self, max_steps: int = 200) -> SkillResult:
        """Go down one floor: navigate onto the nearest seen down-ladder and
        issue `descend`. Precondition: the floor must be cleared (>= 8 monsters
        killed), else a `failure` with the kill count."""
        self.mem.update(self.env.state)
        lvl = self._level()
        ladder = self._nearest_item(
            lvl, (ItemType.LADDER_DOWN.value, ItemType.LADDER_DOWN_BLOCKED.value)
        )
        if ladder is None:
            return SkillResult("interrupted", "no down-ladder in view (explore first)",
                               0, 0.0, [], [], self.env.obs())

        nav = self.navigate_to(ladder, adjacent=False, max_steps=max_steps)
        if not nav.ok:
            return SkillResult("interrupted", f"could not reach down-ladder: {nav.reason}",
                               nav.steps, nav.reward, nav.achievements, nav.events,
                               self.env.obs())

        killed = int(np.asarray(self.env.state.monsters_killed)[lvl])
        if killed < MONSTERS_KILLED_TO_CLEAR_LEVEL:
            return SkillResult("failure",
                               f"floor {lvl} not cleared: {killed}/{MONSTERS_KILLED_TO_CLEAR_LEVEL} "
                               f"monsters killed (down-ladder blocked)",
                               nav.steps, nav.reward, nav.achievements, nav.events,
                               self.env.obs())

        res = self.env.step("descend")
        steps = nav.steps + 1
        reward = nav.reward + res.reward
        ach = nav.achievements + res.achievements_unlocked
        events = nav.events + [f"descend at {ladder}"]
        if int(self.env.state.player_level) > lvl:
            return SkillResult("success", f"descended to floor {self._level()}",
                               steps, reward, ach, events, self.env.obs())
        return SkillResult("failure", "descend did not take effect", steps, reward,
                           ach, events, self.env.obs())

    def fight(self, count: int = 1, types=("melee", "ranged"),
              health_floor: int = 0, max_steps: int = 200) -> SkillResult:
        """Kill `count` of the nearest visible hostile mobs. Approaches (re-planning
        each step, since mobs move), faces, and `do`-attacks. Uses LIVE mob positions,
        not SeenMemory.

        `health_floor` was 2 until 2026-07-27, which made the SKILL retreat on the
        agent's behalf: 68 of 122 fight non-successes were "low health (1 or 2)". The
        game permits attacking at any health — with a better sword or a potion it can be
        the right call — so this was the executor VETOING a legal action and taking a
        strategic decision away from the policy. Worse for learning: the agent could
        never experience the outcome, so PPO could not learn when fighting hurt is worth
        it, and the aborted attempts injected uninformative failures into the gradient.
        Now 0: only actual death stops the skill. Contrast `descend`, which is gated in
        the constructor because Craftax's own MONSTERS_KILLED_TO_CLEAR_LEVEL makes it
        genuinely impossible — remove what the GAME forbids, keep what the game permits
        and the agent should judge.
        """
        events, achievements, total = [], [], [0.0]
        shape = self.env.state.map.shape[1:]
        killed, stuck = 0, 0

        def step(action):
            r = self.env.step(action)
            total[0] += r.reward
            achievements.extend(r.achievements_unlocked)
            if r.achievements_unlocked:
                events.append(f"unlocked {r.achievements_unlocked}")
            return r

        start_t = self.env.t
        while killed < count and (self.env.t - start_t) < max_steps:
            self.mem.update(self.env.state)
            s = self.env.state
            lvl = self._level()
            # float, NOT int: player_health is float32, so int() TRUNCATES — at the
            # default health_floor=0 a living agent on 0.1..0.9 health read as 0 and the
            # skill refused to fight in 0 steps (4/92 offers, NOOPGATE 2026-07-28). The
            # same truncation made the old health_floor=2 actually block health < 3.
            if float(s.player_health) <= health_floor:
                return SkillResult("interrupted",
                                   f"low health ({int(s.player_health)}); killed {killed}/{count}",
                                   self.env.t - start_t, total[0], achievements, events, self.env.obs())
            mobs = self._visible_hostiles(types)
            if not mobs:
                status = "success" if killed else "interrupted"
                reason = (f"killed {killed}/{count}" if killed
                          else "no hostile mob in view (explore first)")
                return SkillResult(status, reason, self.env.t - start_t, total[0],
                                   achievements, events, self.env.obs())

            px, py = self._pos()
            tx, ty, ttype = min(mobs, key=lambda m: abs(m[0] - px) + abs(m[1] - py))
            if abs(tx - px) + abs(ty - py) == 1:  # cardinally adjacent -> attack
                face = _DELTA_TO_ACTION[(tx - px, ty - py)]
                if int(s.player_direction) != ACTION_TO_INT[face]:
                    if step(face).done:
                        break
                mk_before = int(np.asarray(self.env.state.monsters_killed)[lvl])
                if step("do").done:
                    break
                mk_after = int(np.asarray(self.env.state.monsters_killed)[lvl])
                if mk_after > mk_before:
                    killed += mk_after - mk_before
                    events.append(f"killed {ttype} mob ({killed}/{count})")
                stuck = 0
            else:  # approach (re-plan each step; the mob moves)
                path = _bfs(lambda r, c: self.mem.walkable(lvl, r, c),
                            (px, py), (tx, ty), shape, adjacent=True)
                moved = False
                if path:
                    nr, nc = path[0]
                    if step(_DELTA_TO_ACTION[(nr - px, nc - py)]).done:
                        break
                    moved = self._pos() != (px, py)
                else:  # greedy fallback toward the mob
                    for d in ([(int(np.sign(tx - px)), 0)] if tx != px else []) + \
                             ([(0, int(np.sign(ty - py)))] if ty != py else []):
                        if self.mem.walkable(lvl, px + d[0], py + d[1]):
                            if step(_DELTA_TO_ACTION[d]).done:
                                return SkillResult("interrupted", "episode ended",
                                                   self.env.t - start_t, total[0], achievements,
                                                   events, self.env.obs())
                            moved = self._pos() != (px, py)
                            break
                stuck = 0 if moved else stuck + 1
                if stuck >= self.STUCK_LIMIT:
                    return SkillResult("interrupted",
                                       f"stuck approaching mob; killed {killed}/{count}",
                                       self.env.t - start_t, total[0], achievements, events,
                                       self.env.obs())

        status = "success" if killed >= count else "interrupted"
        return SkillResult(status, f"killed {killed}/{count}", self.env.t - start_t,
                           total[0], achievements, events, self.env.obs())

    def place_would_succeed(self, item: str) -> bool:
        """Would `place(item)` find a valid target tile from where the agent stands?

        The constructor offered `place` on material-held alone, so 28/229 offers (12.2%)
        came back "no valid tile faced" — 11 of 301 real decisions in the v4 probe, `plant`
        being 7 of them because a sapling needs GRASS specifically, not merely a free tile.

        The rules are craftax's own (`game_logic.place_block`), transcribed once here
        rather than in `candidate_actions`, so the constructor cannot hold a second,
        drifting copy (LESSONS R1):
          table, furnace : target not solid AND no item already on it
          stone, torch   : as above, OR the target is WATER (you may bridge)
          plant          : target block is GRASS

        Only the four CARDINAL NEIGHBOURS are considered, which is what the skill can face
        without moving. It may also step and retry (`max_tries=6`), so this is mildly
        conservative — it can withhold a `place` that walking would have enabled. That
        direction is the safe one: the alternative is offering an action that cannot work.

        Observation discipline is intact: adjacent tiles are always inside the visible
        window, so reading them is not god-mode.
        """
        if item not in PLACEABLES:
            return False
        rec = PLACEABLES[item]
        if self._inv(rec["mat"]) < rec["qty"]:
            return False
        s = self.env.state
        lvl = self._level()
        amap = np.asarray(s.map[lvl])
        imap = np.asarray(s.item_map[lvl])
        px, py = self._pos()
        h, w = amap.shape
        for dr, dc in _NEIGHBORS:
            x, y = px + dr, py + dc
            if not (0 <= x < h and 0 <= y < w):
                continue
            blk = int(amap[x, y])
            occupied = (bool(SOLID_BLOCK_MAPPING[blk])
                        or int(imap[x, y]) != ItemType.NONE.value)
            if item == "plant":
                ok = blk == BlockType.GRASS.value
            elif item in ("stone", "torch"):
                ok = (blk == BlockType.WATER.value) or not occupied
            else:                                   # table, furnace
                ok = not occupied
            if ok:
                return True
        return False

    def enchant_would_succeed(self, target: str = "sword") -> bool:
        """Would `enchant(target)` have an effect from here?

        The constructor offered `enchant` on enchantment-table reachability alone, while
        the skill needs FOUR more things — the item, 9 mana, and the gem matching THAT
        table (fire consumes a ruby, ice a sapphire), each of which returns 0 steps.
        NOOPGATE never caught it because enchantment tables only appear on deeper floors
        and its population starts on the surface; found by reading the 0-step paths.

        Shared with `candidate_actions` (LESSONS R1).
        """
        target = str(target).lower().strip()
        if target not in ("sword", "armour", "bow"):
            return False
        if target == "armour":
            if int(np.asarray(self.env.state.inventory.armour).sum()) < 1:
                return False
        elif self._inv(target) < 1:
            return False
        if int(self.env.state.player_mana) < 9:
            return False
        tgt = self._nearest_block((BlockType.ENCHANTMENT_TABLE_FIRE.value,
                                   BlockType.ENCHANTMENT_TABLE_ICE.value))
        if tgt is None:
            return False
        table = int(np.asarray(self.env.state.map[self._level()])[tgt[0], tgt[1]])
        gem = "ruby" if table == BlockType.ENCHANTMENT_TABLE_FIRE.value else "sapphire"
        return self._inv(gem) >= 1

    def reachable_hostiles(self, types, dist=None):
        """Visible hostiles of `types` the approach loop can actually reach.

        `_visible_hostiles` answers "can I SEE it". `fight` (and `_eat_cow` through it)
        additionally needs a walkable route: it walks toward the nearest mob and gives up
        after STUCK_LIMIT non-moves, returning "stuck approaching mob" — in 0 steps when
        the very first move is blocked. Offering fight/eat on visibility alone therefore
        put actions on the menu that spend a decision and do nothing: 8/2521 offers
        measured by NOOPGATE (eat 6/72, fight 2/98), confirmed still zero-step at the
        executor's own 200-step cap.

        Shared with `candidate_actions` so the menu and the skill agree (LESSONS R1).
        NOTE this is a REACHABILITY rule, not a strategy veto — it is the same call as
        `descend`'s kill quota, and deliberately unlike `fight`'s health_floor (2 -> 0),
        which was removed precisely because fighting at low health is permitted by the
        game and is therefore the agent's decision to get wrong.
        """
        mobs = self._visible_hostiles(types)
        if not mobs:
            return []
        if dist is None:
            dist = self.reachable()
        return [m for m in mobs
                if any((int(m[0]) + dr, int(m[1]) + dc) in dist for dr, dc in _NEIGHBORS)]

    def _visible_hostiles(self, types):
        """Live (pos_x, pos_y, type) for mobs of the given classes that are within
        the window and lit (observation discipline — only visible mobs)."""
        s = self.env.state
        lvl = int(s.player_level)
        px, py = self._pos()
        half = np.array([OBS_DIM[0] // 2, OBS_DIM[1] // 2])
        lit = light_window(s)
        out = []
        for t in types:
            m = getattr(s, _HOSTILE_CLASSES[t])
            pos = np.asarray(m.position)[lvl]
            mask = np.asarray(m.mask)[lvl]
            for i in range(len(mask)):
                if not bool(mask[i]):
                    continue
                loc = pos[i] - np.array([px, py]) + half
                if (loc >= 0).all() and (loc < np.array(OBS_DIM)).all() and bool(lit[loc[0], loc[1]]):
                    out.append((int(pos[i][0]), int(pos[i][1]), t))
        return out

    def _nearest_item(self, lvl, item_values):
        """Nearest reachable (standable) seen tile whose item is in `item_values`."""
        dist = self.reachable()
        known_items = self.mem.known_items[lvl]
        best, best_d = None, None
        for x, y in np.argwhere(np.isin(known_items, list(item_values))):
            cell = (int(x), int(y))
            if cell in dist and (best_d is None or dist[cell] < best_d):
                best_d, best = dist[cell], cell
        return best

    # --- extended skills: chest / sapling / ranged / spells / potions / enchant ---
    def known_stations(self) -> Dict[str, Optional[int]]:
        """{station: 0 if adjacent, else path-steps to the nearest remembered one, else
        None if none is remembered on this floor}.

        Feeds the prompt. Without it `go_to(target=crafting_table)` appears on the menu
        while the observation — a pure function of the VISIBLE window — says nothing
        about a table the agent built and walked away from, so the planner is asked to
        choose an action it has no basis to evaluate. Floor-scoped for free: `mem.known`
        is (n_levels, h, w) and `_nearest_block` reads only the current level.
        """
        out: Dict[str, Optional[int]] = {}
        dist = None
        for name, blocks in _GOTO_TARGETS.items():
            if self._near(blocks[0]):
                out[name] = 0
                continue
            cell = self._nearest_block(blocks)
            if cell is None:
                out[name] = None
                continue
            if dist is None:
                dist = self.reachable()
            adj = [dist[(cell[0] + dr, cell[1] + dc)]
                   for dr, dc in _NEIGHBORS if (cell[0] + dr, cell[1] + dc) in dist]
            out[name] = int(min(adj)) if adj else None
        return out

    def go_to(self, target: str, max_steps: int = 200) -> SkillResult:
        """Walk until ADJACENT to the nearest remembered `target` station.

        Exists because a crafting station the agent has already built is otherwise
        unreachable as a *choice*: `craft` places a NEW table whenever one is not
        adjacent (see `_place_one`), so with 1 wood and a table ten tiles away the agent
        had no way to craft at all — it needed 3 wood to build a redundant second table.
        After `go_to`, the station is adjacent and the recipe costs its materials alone
        (`craft_requirements(..., near_table=True)`).

        Deliberately NOT folded into `craft`: place-vs-walk is a genuine trade-off
        (2 wood against N steps of hunger and time) and §4.2 assigns trade-offs to the
        policy, keeping only impossibilities in the constructor.
        """
        blocks = _GOTO_TARGETS.get(target)
        if blocks is None:
            return SkillResult("failure",
                               f"unknown go_to target {target!r}; "
                               f"choose from {sorted(_GOTO_TARGETS)}",
                               0, 0.0, [], [], self.env.obs())
        self.mem.update(self.env.state)
        if self._near(blocks[0]):
            return SkillResult("success", f"already adjacent to {target}", 0, 0.0, [], [],
                               self.env.obs())
        cell = self._nearest_block(blocks)
        if cell is None:
            return SkillResult("interrupted",
                               f"no reachable {target} remembered (build or find one first)",
                               0, 0.0, [], [], self.env.obs())
        nav = self.navigate_to(cell, adjacent=True, max_steps=max_steps)
        if not nav.ok:
            return SkillResult("interrupted",
                               f"could not reach {target} at {cell}: {nav.reason}",
                               nav.steps, nav.reward, nav.achievements, nav.events,
                               self.env.obs())
        return SkillResult("success", f"adjacent to {target} at {cell}",
                           nav.steps, nav.reward, nav.achievements, nav.events,
                           self.env.obs())

    def open_chest(self, max_steps: int = 200) -> SkillResult:
        """Open the nearest SEEN chest (navigate adjacent -> face -> do). A chest
        yields loot (torches / ore / a potion / arrows / tools; a bow from the FIRST
        floor-1 chest, a book from the first floor-3/4 chest) and unlocks open_chest;
        the chest tile becomes PATH. Single-purpose: it does not gather."""
        self.mem.update(self.env.state)
        target = self._nearest_block((BlockType.CHEST.value,))
        if target is None:
            return SkillResult("interrupted", "no reachable chest in view (explore first)",
                               0, 0.0, [], [], self.env.obs())
        events, achievements, total = [], [], 0.0
        nav = self.navigate_to(target, adjacent=True, max_steps=max_steps)
        total += nav.reward; achievements += nav.achievements; events += nav.events
        steps = nav.steps
        if not nav.ok:
            return SkillResult("interrupted", f"could not reach chest at {target}: {nav.reason}",
                               steps, total, achievements, events, self.env.obs())
        px, py = self._pos()
        face = _DELTA_TO_ACTION.get((target[0] - px, target[1] - py))
        if face is not None and int(self.env.state.player_direction) != ACTION_TO_INT[face]:
            r = self.env.step(face); steps += 1; total += r.reward
            achievements += r.achievements_unlocked
            if r.done:
                return SkillResult("interrupted", "episode ended", steps, total,
                                   achievements, events, r.obs_text)
        lvl = self._level()
        was = int(np.asarray(self.env.state.map[lvl])[target[0], target[1]])
        r = self.env.step("do"); steps += 1; total += r.reward
        achievements += r.achievements_unlocked
        if r.done:
            return SkillResult("interrupted", "episode ended", steps, total,
                               achievements, events, r.obs_text)
        now = int(np.asarray(self.env.state.map[lvl])[target[0], target[1]])
        self.mem.update(self.env.state)
        if was == BlockType.CHEST.value and now != BlockType.CHEST.value:
            events.append(f"opened chest at {target}")
            return SkillResult("success", f"opened chest at {target}", steps, total,
                               achievements, events, self.env.obs())
        return SkillResult("interrupted", f"could not open chest at {target} (mob on it?)",
                           steps, total, achievements, events, self.env.obs())

    def collect_sapling(self, count: int = 1, max_steps: int = 120) -> SkillResult:
        """Collect `count` saplings by `do`-ing on grass (10% chance per try; grass
        is NOT consumed, so we retry the same tile). Unlocks collect_sapling; the
        sapling can later be placed as a plant that grows into food."""
        self.mem.update(self.env.state)
        events, achievements, total, steps = [], [], 0.0, 0
        start = self._inv("sapling")
        f = self._front()
        if f is None or f[2] != BlockType.GRASS.value:
            target = self._nearest_block((BlockType.GRASS.value,))
            if target is None:
                return SkillResult("interrupted", "no reachable grass in view (explore first)",
                                   0, 0.0, [], [], self.env.obs())
            nav = self.navigate_to(target, adjacent=True, max_steps=max_steps)
            total += nav.reward; achievements += nav.achievements; events += nav.events
            steps = nav.steps
            if not nav.ok:
                return SkillResult("interrupted", f"could not reach grass at {target}: {nav.reason}",
                                   steps, total, achievements, events, self.env.obs())
            px, py = self._pos()
            face = _DELTA_TO_ACTION.get((target[0] - px, target[1] - py))
            if face is not None and int(self.env.state.player_direction) != ACTION_TO_INT[face]:
                r = self.env.step(face); steps += 1; total += r.reward
                achievements += r.achievements_unlocked
                if r.done:
                    return SkillResult("interrupted", "episode ended", steps, total,
                                       achievements, events, r.obs_text)
        while (self._inv("sapling") - start) < count and steps < max_steps:
            r = self.env.step("do"); steps += 1; total += r.reward
            achievements += r.achievements_unlocked
            if r.done:
                return SkillResult("interrupted", "episode ended", steps, total,
                                   achievements, events, r.obs_text)
        got = self._inv("sapling") - start
        if got >= count:
            events.append(f"collected {got} sapling(s)")
            return SkillResult("success", f"collected {got} sapling(s)", steps, total,
                               achievements, events, self.env.obs())
        return SkillResult("interrupted", f"collected {got}/{count} saplings (out of tries)",
                           steps, total, achievements, events, self.env.obs())

    def _aim(self, direction, step):
        """Face `direction` (given, else toward the nearest visible hostile). Returns
        the StepResult of the facing move, or None if no facing was needed/possible."""
        if direction is None:
            mobs = self._visible_hostiles(("melee", "ranged"))
            if mobs:
                px, py = self._pos()
                tx, ty, _ = min(mobs, key=lambda m: abs(m[0] - px) + abs(m[1] - py))
                dr, dc = int(np.sign(tx - px)), int(np.sign(ty - py))
                if abs(tx - px) >= abs(ty - py) and dr != 0:
                    direction = _DELTA_TO_ACTION[(dr, 0)]
                elif dc != 0:
                    direction = _DELTA_TO_ACTION[(0, dc)]
        if direction is not None and direction in _ACTION_TO_DELTA and \
                int(self.env.state.player_direction) != ACTION_TO_INT[direction]:
            return step(direction)
        return None

    def shoot(self, direction: Optional[str] = None, max_steps: int = 6) -> SkillResult:
        """Fire an arrow with the bow in the faced (or given) direction; auto-aims at
        the nearest visible hostile if no direction is given. Consumes one arrow and
        unlocks fire_bow. Precondition: a bow and >=1 arrow (does not craft them)."""
        if self._inv("bow") < 1:
            return SkillResult("failure", "no bow (find one in a chest first)",
                               0, 0.0, [], [], self.env.obs())
        if self._inv("arrows") < 1:
            return SkillResult("failure", "no arrows (craft arrows first)",
                               0, 0.0, [], [], self.env.obs())
        events, achievements, total, steps = [], [], [0.0], 0

        def step(a):
            r = self.env.step(a); total[0] += r.reward
            achievements.extend(r.achievements_unlocked)
            return r
        r = self._aim(direction, step)
        if r is not None:
            steps += 1
            if r.done:
                return SkillResult("interrupted", "episode ended", steps, total[0],
                                   achievements, events, r.obs_text)
        a0 = self._inv("arrows")
        r = step("shoot_arrow"); steps += 1
        if r.done:
            return SkillResult("interrupted", "episode ended", steps, total[0],
                               achievements, events, r.obs_text)
        if self._inv("arrows") < a0 or "fire_bow" in achievements:
            events.append("fired an arrow")
            return SkillResult("success", "fired an arrow", steps, total[0],
                               achievements, events, self.env.obs())
        return SkillResult("interrupted", "arrow not fired (projectile slots full?)",
                           steps, total[0], achievements, events, self.env.obs())

    def cast(self, spell: str, direction: Optional[str] = None,
             max_steps: int = 6) -> SkillResult:
        """Cast a learned spell (fireball/iceball) in the faced (or given) direction;
        auto-aims if no direction. Costs 2 mana and unlocks cast_fireball/cast_iceball.
        Precondition: the spell is learned (read a book) and mana >= 2."""
        spell = str(spell).lower().strip()
        idx = {"fireball": 0, "iceball": 1}.get(spell)
        if idx is None:
            return SkillResult("failure", f"unknown spell {spell!r} (fireball/iceball)",
                               0, 0.0, [], [], self.env.obs())
        if not bool(np.asarray(self.env.state.learned_spells)[idx]):
            return SkillResult("failure", f"{spell} not learned (read a book first)",
                               0, 0.0, [], [], self.env.obs())
        mana = int(self.env.state.player_mana)
        if mana < 2:
            return SkillResult("failure", f"not enough mana to cast {spell} "
                               f"(need 2, have {mana})", 0, 0.0, [], [], self.env.obs())
        events, achievements, total, steps = [], [], [0.0], 0

        def step(a):
            r = self.env.step(a); total[0] += r.reward
            achievements.extend(r.achievements_unlocked)
            return r
        r = self._aim(direction, step)
        if r is not None:
            steps += 1
            if r.done:
                return SkillResult("interrupted", "episode ended", steps, total[0],
                                   achievements, events, r.obs_text)
        m0 = int(self.env.state.player_mana)
        r = step("cast_fireball" if idx == 0 else "cast_iceball"); steps += 1
        if r.done:
            return SkillResult("interrupted", "episode ended", steps, total[0],
                               achievements, events, r.obs_text)
        if int(self.env.state.player_mana) < m0 or f"cast_{spell}" in achievements:
            events.append(f"cast {spell}")
            return SkillResult("success", f"cast {spell}", steps, total[0],
                               achievements, events, self.env.obs())
        return SkillResult("interrupted", f"{spell} not cast (projectile slots full?)",
                           steps, total[0], achievements, events, self.env.obs())

    def read_book(self, max_steps: int = 2) -> SkillResult:
        """Read a book to learn a random not-yet-learned spell (fireball/iceball),
        unlocking learn_fireball/learn_iceball. Precondition: a book in inventory
        (books drop from chests on the deeper floors)."""
        if self._inv("books") < 1:
            return SkillResult("failure", "no book to read (find one in a chest)",
                               0, 0.0, [], [], self.env.obs())
        learned0 = np.asarray(self.env.state.learned_spells).astype(bool)
        if learned0.all():
            return SkillResult("failure", "both spells already learned",
                               0, 0.0, [], [], self.env.obs())
        r = self.env.step("read_book")
        ach = list(r.achievements_unlocked)
        if r.done:
            return SkillResult("interrupted", "episode ended", 1, r.reward, ach, [], r.obs_text)
        learned1 = np.asarray(self.env.state.learned_spells).astype(bool)
        if learned1.sum() > learned0.sum() or any(a.startswith("learn_") for a in ach):
            spell = "fireball" if (learned1[0] and not learned0[0]) else "iceball"
            return SkillResult("success", f"learned {spell}", 1, r.reward, ach,
                               [f"read book -> learned {spell}"], self.env.obs())
        return SkillResult("interrupted", "book read but no spell learned",
                           1, r.reward, ach, [], self.env.obs())

    def drink_potion(self, color: Optional[str] = None, max_steps: int = 2) -> SkillResult:
        """Drink a potion from inventory (color auto-picked if omitted). Unlocks
        drink_potion. Effects are RANDOMIZED per world (each color helps or harms
        health/mana/energy), so the result is unknown in advance. Needs a potion."""
        colors = ["red", "green", "blue", "pink", "cyan", "yellow"]
        potions = np.asarray(self.env.state.inventory.potions).astype(int)
        if color is not None:
            color = str(color).lower().strip()
            if color not in colors:
                return SkillResult("failure", f"unknown potion color {color!r}",
                                   0, 0.0, [], [], self.env.obs())
            slot = colors.index(color)
            if potions[slot] < 1:
                return SkillResult("failure", f"no {color} potion", 0, 0.0, [], [], self.env.obs())
        else:
            avail = [i for i in range(6) if potions[i] > 0]
            if not avail:
                return SkillResult("failure", "no potions to drink", 0, 0.0, [], [], self.env.obs())
            slot = avail[0]; color = colors[slot]
        p0 = int(potions[slot])
        r = self.env.step(f"drink_potion_{color}")
        ach = list(r.achievements_unlocked)
        if r.done:
            return SkillResult("interrupted", "episode ended", 1, r.reward, ach, [], r.obs_text)
        if int(np.asarray(self.env.state.inventory.potions)[slot]) < p0 or "drink_potion" in ach:
            return SkillResult("success", f"drank {color} potion", 1, r.reward, ach,
                               [f"drank {color} potion"], self.env.obs())
        return SkillResult("interrupted", f"{color} potion not drunk",
                           1, r.reward, ach, [], self.env.obs())

    def enchant(self, target: str = "sword", max_steps: int = 200) -> SkillResult:
        """Enchant your sword/armour/bow at the nearest SEEN enchantment table (a fire
        table consumes a ruby -> fire enchant; an ice table a sapphire -> ice enchant).
        Costs 9 mana + 1 gem. Precondition: a table in view, the gem, the item, and
        mana >= 9. Only sword/armour enchants unlock an achievement (not bow)."""
        target = str(target).lower().strip()
        if target not in ("sword", "armour", "bow"):
            return SkillResult("failure", f"unknown enchant target {target!r} (sword/armour/bow)",
                               0, 0.0, [], [], self.env.obs())
        if target == "sword" and self._inv("sword") < 1:
            return SkillResult("failure", "no sword to enchant", 0, 0.0, [], [], self.env.obs())
        if target == "armour" and int(np.asarray(self.env.state.inventory.armour).sum()) < 1:
            return SkillResult("failure", "no armour to enchant", 0, 0.0, [], [], self.env.obs())
        if target == "bow" and self._inv("bow") < 1:
            return SkillResult("failure", "no bow to enchant", 0, 0.0, [], [], self.env.obs())
        mana = int(self.env.state.player_mana)
        if mana < 9:
            return SkillResult("failure", f"not enough mana to enchant (need 9, have {mana})",
                               0, 0.0, [], [], self.env.obs())
        self.mem.update(self.env.state)
        tgt = self._nearest_block((BlockType.ENCHANTMENT_TABLE_FIRE.value,
                                   BlockType.ENCHANTMENT_TABLE_ICE.value))
        if tgt is None:
            return SkillResult("interrupted", "no enchantment table in view (explore first)",
                               0, 0.0, [], [], self.env.obs())
        lvl = self._level()
        table = int(np.asarray(self.env.state.map[lvl])[tgt[0], tgt[1]])
        gem = "ruby" if table == BlockType.ENCHANTMENT_TABLE_FIRE.value else "sapphire"
        if self._inv(gem) < 1:
            return SkillResult("failure", f"need a {gem} for this enchantment table "
                               f"(have {self._inv(gem)})", 0, 0.0, [], [], self.env.obs())
        events, achievements, total = [], [], 0.0
        nav = self.navigate_to(tgt, adjacent=True, max_steps=max_steps)
        total += nav.reward; achievements += nav.achievements; events += nav.events
        steps = nav.steps
        if not nav.ok:
            return SkillResult("interrupted", f"could not reach enchantment table at {tgt}: "
                               f"{nav.reason}", steps, total, achievements, events, self.env.obs())
        px, py = self._pos()
        face = _DELTA_TO_ACTION.get((tgt[0] - px, tgt[1] - py))
        if face is not None and int(self.env.state.player_direction) != ACTION_TO_INT[face]:
            r = self.env.step(face); steps += 1; total += r.reward
            achievements += r.achievements_unlocked
            if r.done:
                return SkillResult("interrupted", "episode ended", steps, total,
                                   achievements, events, r.obs_text)
        g0 = self._inv(gem)
        r = self.env.step(f"enchant_{target}"); steps += 1; total += r.reward
        achievements += r.achievements_unlocked
        if r.done:
            return SkillResult("interrupted", "episode ended", steps, total,
                               achievements, events, r.obs_text)
        if self._inv(gem) < g0 or any(a.startswith("enchant_") for a in achievements):
            events.append(f"enchanted {target} ({gem})")
            return SkillResult("success", f"enchanted {target} ({gem})", steps, total,
                               achievements, events, self.env.obs())
        return SkillResult("interrupted", f"enchant did not take effect on {target}",
                           steps, total, achievements, events, self.env.obs())

    # --- helpers for tests / higher-level skills -------------------------
    def reachable(self) -> dict:
        """BFS-reachable known walkable tiles from the player -> distance."""
        lvl = self._level()
        h, w = self.env.state.map.shape[1:]
        start = self._pos()
        dist = {start: 0}
        q = deque([start])
        while q:
            r, c = q.popleft()
            for dr, dc in _NEIGHBORS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in dist \
                        and self.mem.walkable(lvl, nr, nc):
                    dist[(nr, nc)] = dist[(r, c)] + 1
                    q.append((nr, nc))
        return dist
