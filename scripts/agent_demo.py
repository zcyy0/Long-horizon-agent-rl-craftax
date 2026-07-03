"""Test/demo for the hierarchical agent loop (planner -> executor -> ledger).

Drives the loop with a deterministic ScriptedPolicy stand-in (no model) so the
spine can be verified end-to-end before an LLM policy drops into the same slot
(HANDOFF §7.1, step 1).

Checks:
 (1) system prompt exposes every skill in the menu;
 (2) dispatch is total — valid subgoals run; malformed ones come back as `failure`
     (unknown name / bad enum / missing-required / unknown-arg), never an exception;
 (3) extract_decision pulls JSON out of a fenced/prose LLM-style response;
 (4) end-to-end run: one ledger entry per turn; reward accounting is consistent;
     every status is valid; the agent unlocks >=1 achievement through the loop;
 (5) feedback handling: an "explore first" interruption is followed by an explore.

Run:
    /workspace/envs/craftax/bin/python scripts/agent_demo.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "harness"))

from agent import (  # noqa: E402
    SKILLS, HierarchicalAgent, ScriptedPolicy, build_system_prompt, dispatch,
    extract_decision,
)
from craftax_env import CraftaxTextEnv  # noqa: E402
from executor import Executor  # noqa: E402

VALID_STATUS = {"success", "failure", "interrupted"}


def main():
    checks = []

    # (1) system prompt lists every skill
    sysp = build_system_prompt()
    menu_ok = all(name in sysp for name in SKILLS)
    print(f"system prompt: {len(sysp)} chars | all {len(SKILLS)} skills listed: {menu_ok}")
    checks.append(("system prompt menu", menu_ok))

    # (2) dispatch is total (valid runs; malformed -> failure, no exception)
    env = CraftaxTextEnv(seed=0)
    ex = Executor(env)
    cases = [
        ("valid mine",       {"name": "mine", "args": {"resource": "wood", "count": 2}}, None),
        ("unknown name",     {"name": "frobnicate"}, "failure"),
        ("bad enum",         {"name": "mine", "args": {"resource": "unobtanium"}}, "failure"),
        ("missing required", {"name": "mine", "args": {}}, "failure"),
        ("unknown arg",      {"name": "craft", "args": {"item": "wood_pickaxe", "x": 1}}, "failure"),
        ("empty subgoal",    {}, "failure"),
    ]
    dispatch_ok = True
    for label, sg, expect in cases:
        try:
            res = dispatch(ex, sg)
        except Exception as e:  # noqa: BLE001 — dispatch must never raise
            print(f"  [{label}] RAISED {type(e).__name__}: {e}")
            dispatch_ok = False
            continue
        ok = res.status in VALID_STATUS and (expect is None or res.status == expect)
        dispatch_ok &= ok
        print(f"  [{label}] status={res.status} reason='{res.reason[:60]}'")
    checks.append(("dispatch total", dispatch_ok))

    # (3) extract_decision tolerates fences/prose
    raw = 'Sure!\n```json\n{"think": "go", "subgoal": {"name": "explore", "args": {}}}\n```\n'
    dec = extract_decision(raw)
    parse_ok = dec.get("subgoal", {}).get("name") == "explore"
    print(f"extract_decision -> {dec} | ok={parse_ok}")
    checks.append(("json extraction", parse_ok))

    # (4) end-to-end scripted run
    env = CraftaxTextEnv(seed=0)
    ex = Executor(env)
    agent = HierarchicalAgent(env, ex, ScriptedPolicy(), max_turns=30)
    ledger = agent.run()
    reward_sum = sum(e.reward for e in ledger)
    unlocked = sorted({a for e in ledger for a in e.achievements})
    status_ok = all(e.status in VALID_STATUS for e in ledger)
    reward_ok = abs(reward_sum - agent.total_reward) < 1e-6
    print(f"\nran {len(ledger)} turns | total_reward={agent.total_reward:.2f} "
          f"| final floor={ledger[-1].floor} env_step={ledger[-1].env_step}")
    print(f"achievements unlocked ({len(unlocked)}): {unlocked}")
    for e in ledger:
        print(f"  t{e.turn:2d} {str(e.subgoal):<58} -> {e.status} ({e.reason[:42]})")
    checks.append(("one entry per turn", len(ledger) >= 1 and len(ledger) <= 30))
    checks.append(("status valid", status_ok))
    checks.append(("reward accounting", reward_ok))
    checks.append(("unlocked >=1 achievement", len(unlocked) >= 1))

    # (5) feedback handling: explore-first interruption -> next turn explores
    feedback_ok = True
    for prev, nxt in zip(ledger, ledger[1:]):
        if prev.status == "interrupted" and "explore first" in prev.reason:
            if nxt.subgoal.get("name") != "explore":
                feedback_ok = False
    checks.append(("explore-first handled", feedback_ok))

    print("\n=== results ===")
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print("AGENT_OK" if all(ok for _, ok in checks) else "AGENT_FAIL")


if __name__ == "__main__":
    main()
