"""Collect a Qwen rollout batch via the vLLM server and report Phase-1 diagnostics.

Drives the hierarchical loop with `VLLMPolicy` (HTTP client to the local vLLM
OpenAI-compatible server), logs each episode to JSONL (`rollout_log`), and prints the
diagnostics the CA plan needs before fixing γ / λ / RRD horizon (RESEARCH_PLAN §7.1,
§7.4): achievement histogram, reward sparsity + delay-gap distribution, per-skill
status counts, mean option length, DAG coverage of unlocked achievements, and the
matched-compute / throughput numbers (generated tokens, tokens/s, episodes/day).

Prereq: a running server, e.g.
    bash scripts/serve_qwen.sh          # (launches `vllm serve` on :8000)
Run:
    /workspace/envs/craftax/bin/python scripts/collect_rollouts.py --n 8 --max-turns 64
"""
import argparse
import os
import statistics
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "harness"))

from credit_eval import build_prereq_dag, necessary_decisions  # noqa: E402
from rollout_log import load_rollouts, record_rollout, write_rollouts  # noqa: E402
from vllm_policy import VLLMPolicy  # noqa: E402


def rewarded_gaps(turns):
    """Turn-index gaps between consecutive reward-bearing decisions (delay proxy)."""
    idx = [t["turn"] for t in turns if t["reward"] > 0 or t["achievements"]]
    return [b - a for a, b in zip(idx, idx[1:])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8, help="episodes (seeds 0..n-1)")
    ap.add_argument("--max-turns", type=int, default=64)
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--out", default="/workspace/craftax-rl/data/rollouts.jsonl")
    args = ap.parse_args()

    policy = VLLMPolicy(model=args.model, base_url=args.base_url,
                        temperature=args.temperature)
    records = []
    for seed in range(args.n):
        rec = record_rollout(policy, seed=seed, max_turns=args.max_turns)
        records.append(rec)
        s = rec["summary"]
        print(f"seed {seed}: turns={s['n_turns']} env_steps={s['env_steps']} "
              f"floor<={s['max_floor']} reward={s['total_reward']:.2f} "
              f"ach={s['n_achievements']} {s['achievements']}")

    if policy.stats["http_error"] >= policy.stats["turns"] and policy.stats["turns"]:
        print("\n!! every request hit an HTTP error — is the vLLM server up at "
              f"{args.base_url}? (all subgoals fell back to explore)")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    write_rollouts(records, args.out)
    assert load_rollouts(args.out) == records, "JSONL round-trip mismatch"
    print(f"\nwrote {len(records)} rollouts -> {args.out}")

    # ---- aggregate diagnostics (§7.4) ----
    all_turns = [t for r in records for t in r["turns"]]
    ach_hist = Counter(a for r in records for a in r["summary"]["achievements"])
    status_hist = Counter(t["status"] for t in all_turns)
    skill_hist = Counter((t["subgoal"] or {}).get("name") for t in all_turns)
    ks = [t["steps"] for t in all_turns]
    gaps = [g for r in records for g in rewarded_gaps(r["turns"])]
    rewarded = sum(1 for t in all_turns if t["reward"] > 0 or t["achievements"])

    print("\n=== diagnostics ===")
    print(f"episodes={len(records)} turns={len(all_turns)} "
          f"mean_turns={len(all_turns)/max(1,len(records)):.1f}")
    print(f"achievements/ep: mean={statistics.mean(r['summary']['n_achievements'] for r in records):.2f} "
          f"max={max(r['summary']['n_achievements'] for r in records)}")
    print(f"achievement histogram: {dict(ach_hist.most_common())}")
    print(f"status counts: {dict(status_hist)}")
    print(f"skill usage: {dict(skill_hist.most_common())}")
    print(f"reward sparsity: {rewarded}/{len(all_turns)} turns reward-bearing "
          f"({100*rewarded/max(1,len(all_turns)):.0f}%)")
    if gaps:
        print(f"reward delay-gaps (turns between rewarded decisions): "
              f"mean={statistics.mean(gaps):.1f} median={statistics.median(gaps)} max={max(gaps)}")
    if ks:
        print(f"option length k (primitives/subgoal): mean={statistics.mean(ks):.1f} "
              f"median={statistics.median(ks)} max={max(ks)}")

    # ---- DAG coverage of what was unlocked ----
    dag = build_prereq_dag()
    unlocked = set(ach_hist)
    modeled = {a for a in unlocked if a in dag}
    print(f"DAG coverage: {len(modeled)}/{len(unlocked)} unlocked achievements modeled "
          f"(unmodeled: {sorted(unlocked - modeled)})")

    # ---- matched compute / throughput (§7.1) ----
    st = policy.stats
    secs = st["gen_seconds"] or 1e-9
    print("\n=== compute / throughput ===")
    print(f"parsed={st['parsed']}/{st['turns']} fallback={st['fallback']} "
          f"http_error={st['http_error']}")
    print(f"prompt_tokens={st['prompt_tokens']} gen_tokens={st['gen_tokens']} "
          f"gen_seconds={secs:.1f}")
    print(f"decode throughput ~{st['gen_tokens']/secs:.1f} tok/s | "
          f"~{st['turns']/secs:.2f} decisions/s")
    if st["turns"]:
        eps_per_day = 86400 / (secs / max(1, len(records)))
        print(f"~{eps_per_day:.0f} episodes/day at this batch's decisions/episode "
              f"(≈{len(all_turns)/max(1,len(records)):.0f} decisions/ep)")


if __name__ == "__main__":
    main()
