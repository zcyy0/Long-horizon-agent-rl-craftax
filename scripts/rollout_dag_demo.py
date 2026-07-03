"""Phase 0 validation: rollout logger + prerequisite DAG + necessity labeler + AUC.

Proves the CA scaffolding end-to-end on CPU (no GPU / no LLM needed): record a
ScriptedPolicy episode, log→reload it losslessly, build the tech-tree DAG (with a
spot-check on a known recipe), label the causally-necessary decisions for each
unlocked achievement, and verify the AUC rank statistic behaves (perfect credit→1,
reversed→0, tie→0.5) both synthetically and on a real achievement from the rollout.

Run:
    /workspace/envs/craftax/bin/python scripts/rollout_dag_demo.py
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "harness"))

from agent import ScriptedPolicy  # noqa: E402
from credit_eval import (build_prereq_dag, ca_auc, necessary_decisions,  # noqa: E402
                         transitive_prereqs)
from rollout_log import load_rollouts, record_rollout, write_rollouts  # noqa: E402


def main():
    checks = []

    # (1) record + log + reload a rollout
    rec = record_rollout(ScriptedPolicy(), seed=0, max_turns=40)
    s = rec["summary"]
    print(f"rollout: policy={s['policy']} turns={s['n_turns']} env_steps={s['env_steps']} "
          f"reward={s['total_reward']:.2f} floor<= {s['max_floor']} done={s['done']}")
    print(f"  achievements ({s['n_achievements']}): {s['achievements']}")

    path = os.path.join(tempfile.gettempdir(), "rollout_dag_demo.jsonl")
    write_rollouts([rec], path)
    loaded = load_rollouts(path)[0]

    checks.append(("rollout has turns", s["n_turns"] >= 1))
    checks.append(("jsonl round-trips losslessly", loaded == rec))
    checks.append(("turns carry vitals+inventory",
                   all("vitals_post" in t and "inventory_post" in t for t in loaded["turns"])))
    checks.append(("some achievements unlocked", s["n_achievements"] >= 1))

    # (2) DAG build + spot-check + transitive closure
    dag = build_prereq_dag()
    sp = dag.get("make_stone_pickaxe", set())
    print(f"\nDAG: {len(dag)} nodes")
    print(f"  make_stone_pickaxe -> {sorted(sp)}")
    print(f"  transitive(make_iron_pickaxe) -> {sorted(transitive_prereqs(dag, 'make_iron_pickaxe'))}")
    checks.append(("DAG non-empty", len(dag) > 10))
    checks.append(("stone_pickaxe direct prereqs correct",
                   sp == {"collect_wood", "collect_stone", "place_table"}))
    checks.append(("iron_pickaxe transitively needs wood-pickaxe",
                   "make_wood_pickaxe" in transitive_prereqs(dag, "make_iron_pickaxe")))

    # (3) necessity labeler on the logged rollout
    nd = necessary_decisions(loaded, dag)
    print("\nnecessary decisions per unlocked achievement:")
    for a, info in sorted(nd.items(), key=lambda kv: kv[1]["unlock_turn"]):
        print(f"  {a}@t{info['unlock_turn']}: necessary={info['necessary_turns']} "
              f"filler={info['filler_turns']} missing={info['missing_prereqs']} modeled={info['modeled']}")
    checks.append(("necessary turns precede the unlock",
                   all(all(nt < info["unlock_turn"] for nt in info["necessary_turns"])
                       for info in nd.values())))

    # (4a) AUC correctness on a synthetic case
    pos, cand = {1, 2}, {1, 2, 3, 4}
    perfect = ca_auc({1: 10, 2: 9, 3: 1, 4: 0}, pos, cand)
    reverse = ca_auc({1: 0, 2: 1, 3: 9, 4: 10}, pos, cand)
    tie = ca_auc({1: 5, 2: 5, 3: 5, 4: 5}, pos, cand)
    print(f"\nAUC synthetic: perfect={perfect} reversed={reverse} tie={tie}")
    checks.append(("AUC perfect=1, reversed=0, tie=0.5",
                   perfect == 1.0 and reverse == 0.0 and tie == 0.5))

    # (4b) AUC on a real achievement that has both necessary and filler turns
    real = next((info for info in nd.values()
                 if info["necessary_turns"] and info["filler_turns"]), None)
    if real is not None:
        candidate = real["necessary_turns"] + real["filler_turns"]
        oracle = {t: (1.0 if t in set(real["necessary_turns"]) else 0.0) for t in candidate}
        auc_perfect = ca_auc(oracle, real["necessary_turns"], candidate)
        print(f"AUC on real achievement (unlock@t{real['unlock_turn']}, "
              f"{len(real['necessary_turns'])} necessary / {len(real['filler_turns'])} filler): "
              f"oracle-credit AUC={auc_perfect}")
        checks.append(("oracle credit scores AUC=1 on a real achievement", auc_perfect == 1.0))
    else:
        print("\n(no achievement with both necessary and filler turns this rollout — "
              "synthetic AUC check stands)")

    print("\n=== results ===")
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    ok = all(o for _, o in checks)
    print("PHASE0_OK" if ok else "PHASE0_FAIL")


if __name__ == "__main__":
    main()
