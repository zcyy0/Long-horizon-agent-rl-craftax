"""Credit-assignment evaluation scaffolding (RESEARCH_PLAN Phase 0.2, §7.3).

Three pieces:
  1. build_prereq_dag()      — the Craftax tech-tree PREREQUISITE DAG, derived
                               programmatically from the executor's recipe/resource
                               tables (CRAFTABLES / RESOURCES / PLACEABLES) plus a few
                               tool-free roots. Node & edge names are validated against
                               the real Achievement vocabulary.
  2. necessary_decisions()   — for each achievement a rollout unlocks, which EARLIER
                               subgoal turns were causally necessary (produced its
                               transitive prerequisites). Redundancy convention: each
                               prerequisite is an achievement that fires once, so the
                               producing turn is unambiguous = the last (only)
                               prerequisite-satisfying occurrence before the unlock.
  3. ca_auc()                — threshold-free rank statistic: does a method's
                               per-decision credit rank DAG-necessary turns above
                               filler? (Mann-Whitney AUC.) Scale-free, so it compares
                               across methods whose "credit" lives on different scales.

Torch-free; pure harness. The DAG covers the well-defined crafting/mining/placing
tech tree; combat/dungeon/magic achievements are left unmodeled (no prerequisites) —
the labeler flags them via `modeled=False` so they can be excluded from the metric.
"""
from typing import Dict, Iterable, List, Optional, Set

from craftax_env import ACHIEVEMENT_NAMES
from executor import CRAFTABLES, PLACEABLES, RESOURCES

_ACH: Set[str] = set(ACHIEVEMENT_NAMES)

# mined material -> the achievement that produces it
MATERIAL_ACHIEVEMENT = {
    "wood": "collect_wood", "stone": "collect_stone", "coal": "collect_coal",
    "iron": "collect_iron", "diamond": "collect_diamond",
    "sapphire": "collect_sapphire", "ruby": "collect_ruby",
    "sapling": "collect_sapling",
}
# pickaxe tier required to mine -> the achievement that yields that pickaxe
PICKAXE_ACHIEVEMENT = {1: "make_wood_pickaxe", 2: "make_stone_pickaxe",
                       3: "make_iron_pickaxe", 4: "make_diamond_pickaxe"}
# producer of a placeable's material (torches are crafted, not mined)
_PRODUCER = {**MATERIAL_ACHIEVEMENT, "torches": "make_torch"}

# tool-free achievements the recipe tables don't cover (roots with no prereqs)
_ROOTS = {
    "collect_wood": set(), "collect_drink": set(), "collect_sapling": set(),
    "eat_cow": set(), "eat_plant": set(), "wake_up": set(),
}


def build_prereq_dag() -> Dict[str, Set[str]]:
    """Achievement -> set of DIRECT prerequisite achievements."""
    dag: Dict[str, Set[str]] = {}

    # mining collects: gated by pickaxe tier
    for res, (_blocks, req_pick, _field) in RESOURCES.items():
        dag[MATERIAL_ACHIEVEMENT[res]] = (
            {PICKAXE_ACHIEVEMENT[req_pick]} if req_pick >= 1 else set()
        )

    # crafting: material collects + station placements
    for _item, rec in CRAFTABLES.items():
        pre = {MATERIAL_ACHIEVEMENT[f] for f in rec["mats"]}
        if rec["table"]:
            pre.add("place_table")
        if rec["furnace"]:
            pre.add("place_furnace")
        dag[rec["action"]] = pre  # rec["action"] == the make_* achievement name

    # placing: the producer of the placed material
    for _item, rec in PLACEABLES.items():
        dag[rec["action"]] = {_PRODUCER[rec["mat"]]}  # place_* achievement name

    # tool-free roots (don't clobber an entry the tables already set)
    for node, pre in _ROOTS.items():
        dag.setdefault(node, set())
        dag[node] |= pre

    bad = ({n for n in dag if n not in _ACH}
           | {p for pre in dag.values() for p in pre if p not in _ACH})
    if bad:
        raise ValueError(f"DAG references unknown achievement names: {sorted(bad)}")
    return dag


def transitive_prereqs(dag: Dict[str, Set[str]], node: str) -> Set[str]:
    """All upstream prerequisite achievements of `node` (transitive closure)."""
    seen: Set[str] = set()
    stack = list(dag.get(node, set()))
    while stack:
        p = stack.pop()
        if p in seen:
            continue
        seen.add(p)
        stack.extend(dag.get(p, set()))
    return seen


def necessary_decisions(rollout: Dict, dag: Optional[Dict[str, Set[str]]] = None
                        ) -> Dict[str, Dict]:
    """For each achievement unlocked in `rollout`, the earlier turns that produced
    its transitive prerequisites (`necessary_turns`) vs. the rest (`filler_turns`).
    Ground truth for the CA-quality metric."""
    if dag is None:
        dag = build_prereq_dag()
    turns = rollout["turns"]

    unlock_turn: Dict[str, int] = {}
    for t in turns:
        for a in t["achievements"]:
            unlock_turn.setdefault(a, t["turn"])  # first (only) firing

    out: Dict[str, Dict] = {}
    for a, t_a in unlock_turn.items():
        prereqs = transitive_prereqs(dag, a)
        nec: List[int] = []
        missing: List[str] = []
        for p in sorted(prereqs):
            tp = unlock_turn.get(p)
            if tp is None:
                missing.append(p)             # prereq never seen (world-found, or gap)
            elif tp < t_a:
                nec.append(tp)
        prior = {t["turn"] for t in turns if t["turn"] < t_a}
        nec_set = set(nec)
        out[a] = {
            "unlock_turn": t_a,
            "prereqs": sorted(prereqs),
            "necessary_turns": sorted(nec_set),
            "filler_turns": sorted(prior - nec_set),
            "missing_prereqs": missing,
            "modeled": a in dag,
        }
    return out


def ca_auc(credit_by_turn: Dict[int, float], positive_turns: Iterable[int],
           candidate_turns: Iterable[int]) -> Optional[float]:
    """Mann-Whitney AUC: probability a DAG-necessary (positive) turn outranks a
    filler turn by assigned credit. `credit_by_turn` maps turn -> score;
    `candidate_turns` is the pool (positives + fillers, typically all turns before
    the unlock). Ties score 0.5. Returns None if either group is empty."""
    positives = set(positive_turns)
    pos = [credit_by_turn[t] for t in positives if t in credit_by_turn]
    neg = [credit_by_turn[t] for t in set(candidate_turns) - positives
           if t in credit_by_turn]
    if not pos or not neg:
        return None
    wins = 0.0
    for a in pos:
        for b in neg:
            wins += 1.0 if a > b else (0.5 if a == b else 0.0)
    return wins / (len(pos) * len(neg))
