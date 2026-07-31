"""Grounded candidate-action constructor (REVISED_RESEARCH_PLAN §4.2–§4.3).

The shared action space for **every** compared system — FROZEN, PPO, TEACHER, DISTILL.
The plan's core fairness requirement (§4.2) is that the LLM's own generations must not
define the action space: "a reranker or planner cannot select an action that is absent."
So the menu is built from the *public skill schema* plus *observable state* only, never
from the model.

Two disciplines this module must not break:

  1. **Observation discipline.** Availability is decided from the executor's SeenMemory
     and visible-window helpers (`ex.mem`, `ex._visible_hostiles`, `ex.reachable`) — the
     same sources `data/feature_builder.py` uses — never from god-mode `state.map`
     contents. (`reachable()` touches `state.map.shape` for array dimensions only.)

  2. **Availability is not value.** §4.2: "The constructor may expose an action without
     revealing whether it is valuable." Offering `open_chest` says a chest is visible and
     reachable; it says nothing about the loot. This is what makes the novel-affordance
     experiment (§11) meaningful — the action is *available* long before the agent knows
     it is *good*, so learning-to-use is measured, not learning-to-see.

Grounding predicates mirror the executor's own targeting helpers, so the invariant
**offered ⇒ dispatchable** holds: `_nearest_block` for solids you stand next to (chest,
water, ore, plant), `_nearest_item` for standable tiles (ladders). Where a skill has a
*dynamic* precondition the agent is supposed to learn (descend needs the floor cleared),
the action stays on the menu and the precondition is recorded as a diagnostic instead —
hiding it would remove exactly the decision we want the agent to get right.

Public API:
    build_candidates(env, ex, cap=...)           -> [{"name":..., "args":...}, ...]
    build_candidates_detailed(env, ex, cap=...)  -> [Candidate, ...]  (with provenance)
    candidate_diagnostics(cands)                 -> §14.3 availability/coverage dict
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from craftax.craftax.constants import BlockType, ItemType  # noqa: E402

from craftax.craftax.util.game_logic_utils import (get_max_drink,  # noqa: E402
                                                   get_max_energy, get_max_food)

from executor import (CRAFTABLES, MONSTERS_KILLED_TO_CLEAR_LEVEL,  # noqa: E402
                      PLACEABLES, RESOURCES, _GOTO_TARGETS,
                      craft_requirements, craft_would_upgrade,
                      drink_would_help, eat_would_help, sleep_would_help,
                      _NEIGHBORS, _WATER_BLOCKS)

from data.transition_schema import canonical_action  # noqa: E402

# 2 (2026-07-26): craft is offered only when the FULL cost is affordable (recipe + any
#   station this call must place) AND the craft would actually have an effect (tools are
#   tiered, so re-crafting a held tier is a no-op). `go_to` added for remembered stations.
#   Menu size fell 15.43 -> 12.16 on dev worlds, so the S1 temperature calibration
#   (T=0.1378, tuned to 0.5*ln|C|) DOES NOT TRANSFER and must be redone.
# 3 (2026-07-27): `descend` is offered only once the floor's kill quota is met. Craftax
#   REFUSES the descend below MONSTERS_KILLED_TO_CLEAR_LEVEL kills, so offering it earlier
#   put an impossible action on the menu and confounded credit assignment — the agent
#   cannot distinguish "descending here was a bad idea" from "descending was blocked".
#   This is a GAME RULE, not a strategy veto: `fight`'s health_floor went the other way
#   (executor default 2 -> 0) because low-health combat is permitted by the game and is
#   therefore the agent's decision to get wrong and learn from.
#   Data collected before this stamp (S1b, S3b) carries version 2 and is NOT re-run;
#   the writeup reports it under the v2 action space.
# 4 (2026-07-28): the survival skills are offered only when their meter is not already
#   full. `sleep` was offered unconditionally and `drink_water` on water-reachability
#   alone, but both return `success` in 0 steps for 0 reward at a full meter. On S6's
#   3,200-decision TEACHER stream that was 1,964 wasted decisions (61.4%) — sleep
#   1,062/1,082, drink 834/863. Worse than waste: `Q = r_hat + gamma^tau * V` with tau=0
#   and an unchanged successor makes a no-op the ONLY undiscounted action, so it is an
#   absorbing self-loop the planner actively prefers (mean mu_q sleep 1.407, drink 1.600
#   vs mine 0.918, against realised rewards 0.011 / 0.019 / 0.149). On dev seeds 40/41/45
#   TEACHER chose sleep 39/40, 37/40, 38/40 times — those three worlds are the whole M5
#   gap. Same class as the v2 craft fix (LESSONS R1), now gated by NOOPGATE_OK.
CANDIDATE_SCHEMA_VERSION = 4
DEFAULT_CAP = 20  # REVISED_RESEARCH_EXECUTION §1: "grounded-only, cap ~20"

# Category labels for the §14.3 candidate-availability / coverage diagnostics.
CATEGORIES = ("explore", "chest", "floor", "survival", "combat", "gather",
              "craft", "place", "item")

# Fixed public craft shortlist (§4.2 "legal craft ... actions"). Deliberately the
# progression spine rather than all of CRAFTABLES, so one satisfiable recipe family
# cannot crowd the menu. Offered only when the recipe's materials are already held —
# the skills explicitly do NOT gather (`executor.craft` docstring).
CRAFT_SHORTLIST = ("wood_pickaxe", "stone_pickaxe", "iron_pickaxe",
                   "wood_sword", "stone_sword", "iron_sword", "torch", "arrow")
# Fixed public place shortlist, offered when the material is in inventory.
PLACE_SHORTLIST = ("table", "torch", "stone", "plant", "furnace")

_DOWN_LADDER = (ItemType.LADDER_DOWN.value, ItemType.LADDER_DOWN_BLOCKED.value)
_UP_LADDER = (ItemType.LADDER_UP.value,)
_ENCHANT_TABLES = (BlockType.ENCHANTMENT_TABLE_FIRE.value,
                   BlockType.ENCHANTMENT_TABLE_ICE.value)
_GRASS = (BlockType.GRASS.value,)
_RIPE_PLANT = (BlockType.RIPE_PLANT.value,)

# Pickaxe tier required per resource, straight from the executor's own table.
_RES_TIER = {name: spec[1] for name, spec in RESOURCES.items()}
_RES_BLOCKS = {name: spec[0] for name, spec in RESOURCES.items()}

# Priority tiers — lower survives the cap. Tier 0 is the guaranteed core: the
# undirected-explore fallback plus the two floor-specific affordances the study
# measures (§11 chest, descent). Directional explores are the designated ballast.
_P_CORE, _P_SURVIVAL, _P_COMBAT = 0, 1, 2
_P_GATHER, _P_CRAFT, _P_PLACE, _P_ITEM, _P_EXPLORE_DIR = 3, 4, 5, 6, 7


@dataclass
class Candidate:
    """One grounded macro-action, with the observable fact that licensed it."""
    name: str
    args: Dict[str, Any] = field(default_factory=dict)
    category: str = ""
    reason: str = ""          # provenance — why this action is on the menu
    priority: int = 9

    @property
    def canonical(self) -> str:
        return canonical_action({"name": self.name, "args": self.args})

    def to_subgoal(self) -> Dict[str, Any]:
        return {"name": self.name, "args": dict(self.args)}


def _reachable_block(known: np.ndarray, blocks, dist: Dict[Tuple[int, int], int]
                     ) -> Optional[Tuple[int, int]]:
    """Nearest SEEN block in `blocks` with a reachable walkable neighbour.

    Mirrors `Executor._nearest_block` exactly (same adjacency + reachability test) so
    that "on the menu" implies "the skill can find its target". Shares one prebuilt
    `dist` map instead of re-running BFS per lookup, which `_nearest_block` would do.
    """
    best, best_d = None, None
    for x, y in np.argwhere(np.isin(known, list(blocks))):
        cell = (int(x), int(y))
        adj = [dist[(cell[0] + dr, cell[1] + dc)]
               for dr, dc in _NEIGHBORS if (cell[0] + dr, cell[1] + dc) in dist]
        if adj and (best_d is None or min(adj) < best_d):
            best_d, best = min(adj), cell
    return best


def _reachable_item(known_items: np.ndarray, item_values,
                    dist: Dict[Tuple[int, int], int]) -> Optional[Tuple[int, int]]:
    """Nearest reachable STANDABLE tile carrying one of `item_values` (ladders).
    Mirrors `Executor._nearest_item`."""
    best, best_d = None, None
    for x, y in np.argwhere(np.isin(known_items, list(item_values))):
        cell = (int(x), int(y))
        if cell in dist and (best_d is None or dist[cell] < best_d):
            best_d, best = dist[cell], cell
    return best


def build_candidates_detailed(env, ex, cap: int = DEFAULT_CAP) -> List[Candidate]:
    """Grounded menu for the current (env, ex), richest-first, capped at `cap`."""
    ex.mem.update(env.state)          # refresh seen-memory from the visible window only
    s = env.state
    lvl = int(s.player_level)
    inv = s.inventory
    known = ex.mem.known[lvl]
    known_items = ex.mem.known_items[lvl]
    dist = ex.reachable()             # one BFS, shared by every predicate below

    def inv_count(field_name: str) -> int:
        return int(np.asarray(getattr(inv, field_name)).sum())

    out: List[Candidate] = []

    def add(name, args, category, reason, priority):
        out.append(Candidate(name, dict(args), category, reason, priority))

    # --- core: guaranteed fallback (§4.2 "a small set of fallback exploration actions")
    add("explore", {}, "explore", "always available (frontier fallback)", _P_CORE)

    # --- chest: the §11 novel-affordance probe. Availability only — never loot value.
    if _reachable_block(known, (BlockType.CHEST.value,), dist) is not None:
        add("open_chest", {}, "chest", "chest seen and reachable", _P_CORE)

    # --- floor transitions when a ladder is visible (§4.2)
    if _reachable_item(known_items, _DOWN_LADDER, dist) is not None:
        # REVISED 2026-07-27. This previously stayed on the menu even with the floor
        # uncleared, on the rationale that the >=8-kill rule is "a dynamic precondition
        # the agent should learn" and hiding it "would remove exactly the decision we
        # want the agent to get right". Both halves fail on inspection:
        #   * The decision it protects is `fight` ("should I clear this floor?"), which
        #     is already on the menu. Gating `descend` removes nothing.
        #   * The INFORMATION is already in the observation — `env.obs()` renders both
        #     the ladder and the kill/cleared status — so hiding an UNDISPATCHABLE
        #     action loses nothing the planner could have used.
        # Measured before the change: descend offered in 258/3200 decisions, 7 of them
        # (3%) with the precondition unmet. Harmless only because P(descend|offered) is
        # ~0.01 so the agent never picked those; the whole point of the exploration work
        # is to raise that, at which point 3% of attempts would start failing.
        # Same rule as craft: the constructor rejects the IMPOSSIBLE, the policy judges
        # the UNWISE. `floor_cleared` stays exported as a diagnostic.
        killed = int(np.asarray(s.monsters_killed)[lvl])
        if killed >= MONSTERS_KILLED_TO_CLEAR_LEVEL:
            add("descend", {}, "floor",
                f"down-ladder reachable, floor cleared ({killed}/"
                f"{MONSTERS_KILLED_TO_CLEAR_LEVEL} kills)", _P_CORE)
    if _reachable_item(known_items, _UP_LADDER, dist) is not None:
        add("ascend", {}, "floor", "up-ladder seen and reachable", _P_SURVIVAL)

    # --- survival (§4.2 "applicable survival actions")
    # Each of these needs TWO preconditions, and until 2026-07-28 only the first was
    # checked: a reachable SOURCE, and a meter that is not already full. A survival skill
    # at a full meter returns `success` in 0 steps for 0 reward — the same silent no-op
    # `craft_would_upgrade` exists to prevent, and by far the more expensive one. It is
    # the game's own rule (drinking at full drink does nothing), so it belongs here under
    # the §4.2 line the descend gate cites: the constructor rejects the IMPOSSIBLE, the
    # policy judges the UNWISE. The `*_would_help` predicates are the executor's, shared.
    # `eat` is PLANT-ONLY on the menu (2026-07-28). Eating a cow in Craftax means killing
    # it — the +6 food arrives on the killing blow, there is no corpse and no separate
    # eat-the-corpse action — so `eat(target=auto|cow)` was a COMBAT action wearing a
    # survival name. That hid the low-health lesson in the wrong place: dying to a cow hunt
    # taught "eating is bad" instead of "fighting at 2 health is bad". Hunting is now
    # `fight(types=passive)` in the combat block below, so food from mobs arrives as a
    # CONSEQUENCE of a fight the agent chose, and `eat` is the genuinely atomic action it
    # claims to be (one `do` on a ripe plant tile).
    food_ok = eat_would_help(s)
    plant = _reachable_block(known, _RIPE_PLANT, dist)
    if food_ok and plant is not None:
        add("eat", {"target": "plant"}, "survival",
            f"ripe plant reachable, food {int(s.player_food)}/{int(get_max_food(s))}",
            _P_SURVIVAL)
    if (drink_would_help(s)
            and _reachable_block(known, _WATER_BLOCKS, dist) is not None):
        add("drink_water", {}, "survival",
            f"water/fountain reachable, drink {int(s.player_drink)}/"
            f"{int(get_max_drink(s))}", _P_SURVIVAL)
    # sleep has no WORLD precondition, but it does have a state one: at full energy it is
    # a 0-step no-op. It was offered unconditionally, and TEACHER took it 1,082 times in
    # S6 (98.2% no-ops) because tau=0 makes a no-op the only undiscounted action.
    if sleep_would_help(s):
        add("sleep", {}, "survival",
            f"energy {int(s.player_energy)}/{int(get_max_energy(s))}", _P_SURVIVAL)

    # --- combat, one option per visible hostile class (§4.2 "actions targeting
    # visible enemies"). The skill schema takes a mob CLASS, not an instance id.
    # `reachable_hostiles`, not `_visible_hostiles`: a mob behind a wall is visible but the
    # approach loop cannot get to it and returns 0 steps (see NOOPGATE / R1).
    melee = ex.reachable_hostiles(("melee",), dist)
    ranged = ex.reachable_hostiles(("ranged",), dist)
    # Passive mobs (cow/bat/snail) are HUNTING, not eating — see the survival block. Not
    # gated on food: the kill is legal at any food level, the first one unlocks an
    # achievement regardless, and "was that worth a decision?" is the policy's call.
    passive = ex.reachable_hostiles(("passive",), dist)
    if passive:
        add("fight", {"count": 1, "types": "passive"}, "combat",
            f"{len(passive)} passive mob(s) reachable (+food on kill)", _P_COMBAT)
    if melee or ranged:
        add("fight", {"count": 1}, "combat",
            f"{len(melee)} melee + {len(ranged)} ranged visible", _P_COMBAT)
    if melee:
        add("fight", {"count": 1, "types": "melee"}, "combat",
            f"{len(melee)} melee mob(s) visible", _P_COMBAT)
    if ranged:
        add("fight", {"count": 1, "types": "ranged"}, "combat",
            f"{len(ranged)} ranged mob(s) visible", _P_COMBAT)

    # --- gathering: seen deposit + a pickaxe that can actually break it
    pick = int(inv.pickaxe)
    for res, blocks in _RES_BLOCKS.items():
        if _RES_TIER[res] > pick:
            continue                                  # legality, not value
        if _reachable_block(known, blocks, dist) is not None:
            add("mine", {"resource": res, "count": 1}, "gather",
                f"{res} deposit reachable, pickaxe {pick}>={_RES_TIER[res]}", _P_GATHER)

    # --- walking to a remembered crafting station -------------------------------
    # Offered whenever a station is remembered and reachable but NOT adjacent, and
    # deliberately NOT gated on whether it would enable a craft right now. §4.2 keeps
    # availability separate from value — the same rule that puts a chest on the menu
    # "long before the agent has any idea whether opening it is worthwhile". Walking to
    # a table with an empty inventory is legitimate positioning (go there, mine nearby,
    # then craft at recipe cost instead of recipe + 2 wood for a second table).
    for _tgt, _blocks in _GOTO_TARGETS.items():
        if ex._near(_blocks[0]):
            continue                      # already there; nothing to walk to
        _cell = ex._nearest_block(_blocks)
        if _cell is not None:
            add("go_to", {"target": _tgt}, "craft",
                f"{_tgt} remembered at {_cell}, reachable, not adjacent", _P_CRAFT)

    # --- crafting from the fixed shortlist, when the FULL cost is affordable ---
    # Charges the station this craft would have to PLACE, via the same function the
    # executor uses to decide success (executor.craft_requirements). Offering on
    # recipe["mats"] alone made 30% of all decisions unsatisfiable crafts — the menu
    # promised an action the skill then rejected for materials.
    near_table = ex._near(BlockType.CRAFTING_TABLE.value)
    near_furnace = ex._near(BlockType.FURNACE.value)
    for item in CRAFT_SHORTLIST:
        # Two preconditions, both previously unchecked: can we AFFORD it (incl. the
        # station we would have to place), and would it DO anything (tools are tiered,
        # so re-crafting a held tier is a silent no-op).
        if not craft_would_upgrade(item, ex._probe(CRAFTABLES[item]["probe"])):
            continue
        req = craft_requirements(item, near_table, near_furnace)
        if all(inv_count(mat) >= qty for mat, qty in req.items()):
            mats = ", ".join(f"{m}x{q}" for m, q in sorted(req.items()))
            station = ("" if near_table or not CRAFTABLES[item]["table"]
                       else ", incl. table to place")
            add("craft", {"item": item}, "craft",
                f"materials held ({mats}{station})", _P_CRAFT)

    # --- placing from the fixed shortlist: material held AND a valid target tile.
    # Material-held alone made 12.2% of `place` offers fail with "no valid tile faced"
    # (plant needs GRASS, not merely a free tile). `place_would_succeed` is the executor's,
    # transcribed from craftax's own `place_block` in ONE place.
    for item in PLACE_SHORTLIST:
        spec = PLACEABLES[item]
        if ex.place_would_succeed(item):
            add("place", {"item": item}, "place",
                f"{spec['mat']}>={spec['qty']} held, valid tile adjacent", _P_PLACE)

    # --- inventory-gated extras
    if _reachable_block(known, _GRASS, dist) is not None:
        add("collect_sapling", {"count": 1}, "item", "grass reachable", _P_ITEM)
    if inv_count("bow") > 0 and inv_count("arrows") > 0:
        add("shoot", {}, "item", "bow + arrows held", _P_ITEM)
    if inv_count("books") > 0:
        add("read_book", {}, "item", "book held", _P_ITEM)
    if inv_count("potions") > 0:
        add("drink_potion", {}, "item", "potion held", _P_ITEM)
    # enchant needs the table AND the item AND 9 mana AND the gem matching that table —
    # four preconditions the skill checked and the constructor did not (all 0-step).
    for _et in ("sword", "armour", "bow"):
        if ex.enchant_would_succeed(_et):
            add("enchant", {"target": _et}, "item",
                f"enchantment table reachable, {_et} + gem + mana>=9 held", _P_ITEM)

    # --- directional explores: deliberate ballast, trimmed first under the cap
    for d in ("up", "down", "left", "right"):
        add("explore", {"direction": d}, "explore", "directional frontier search",
            _P_EXPLORE_DIR)

    # Dedupe by canonical form (first occurrence wins), then trim by priority.
    # Python's sort is stable and actions are already appended in priority order, so
    # ties keep insertion order and the result is deterministic for a given state.
    seen, uniq = set(), []
    for c in out:
        if c.canonical in seen:
            continue
        seen.add(c.canonical)
        uniq.append(c)
    return sorted(uniq, key=lambda c: c.priority)[:cap]


def build_candidates(env, ex, cap: int = DEFAULT_CAP) -> List[Dict[str, Any]]:
    """Grounded menu as plain {name, args} subgoals — the form the policy consumes."""
    return [c.to_subgoal() for c in build_candidates_detailed(env, ex, cap)]


def candidate_diagnostics(cands: List[Candidate]) -> Dict[str, Any]:
    """§14.3 candidate/policy metrics for one decision."""
    cats = [c.category for c in cands]
    return {
        "n_candidates": len(cands),
        "categories": sorted(set(cats)),
        "n_categories": len(set(cats)),
        "category_counts": {k: cats.count(k) for k in sorted(set(cats))},
        "has_chest": any(c.name == "open_chest" for c in cands),
        "has_descend": any(c.name == "descend" for c in cands),
        "has_survival": any(c.category == "survival" for c in cands),
        "has_explore": any(c.name == "explore" for c in cands),
        "canonical": [c.canonical for c in cands],
    }
