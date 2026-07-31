"""Exact save / restore of a Craftax episode (NEW_RESEARCH_PLAN §7.4, §12, §17.1).

The whole project rests on one capability: freeze the *exact* state at a planner
decision, then restore it and either (a) reproduce the same trajectory or (b) branch
a *different* first action without disturbing the source. That underwrites both the
P0 frontier snapshot bank and the Phase-II branch-and-replay runner.

A `Snapshot` captures everything mutable in the agent loop:
  - the Craftax env state pytree  (JAX -> host numpy leaves, picklable);
  - the running per-step PRNG key  (so future stochastic steps are reproducible);
  - the CraftaxTextEnv bookkeeping (action_history, achievement_log, unlocked, t);
  - the Executor's SeenMemory      (`known` / `known_items`) — controller state;
  - metadata (world seed, floor, achievements, primitive step, decision index).

Restore rebuilds a fresh CraftaxTextEnv + Executor and injects the frozen state, so
the restored objects are independent of the source (branching is side-effect free).

Torch-free; craftax-core only. Save/load uses pickle (arrays, not JSON).
"""
from __future__ import annotations

import copy
import pickle
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import jax
import jax.numpy as jnp
import numpy as np

from craftax_env import CraftaxTextEnv
from executor import Executor

SNAPSHOT_FORMAT_VERSION = 1


@dataclass
class Snapshot:
    """A self-contained, picklable freeze of an agent-loop state."""
    # --- env core ---
    seed: int
    state: Any                    # EnvState pytree with host (numpy) leaves
    step_rng: np.ndarray          # running per-step PRNG key data (uint32[2])
    action_history: List[int]
    achievement_log: List[List[str]]
    reward_history: List[float]
    unlocked: List[int]
    t: int
    # --- executor (controller) state ---
    mem_known: Optional[np.ndarray] = None
    mem_known_items: Optional[np.ndarray] = None
    # --- metadata (cheap, human/analysis readable) ---
    meta: Dict[str, Any] = field(default_factory=dict)
    format_version: int = SNAPSHOT_FORMAT_VERSION


def _host(pytree):
    """Pull a JAX pytree to host memory (numpy leaves) so it pickles and is
    decoupled from any live device buffer."""
    return jax.device_get(pytree)


def _key_data(key) -> np.ndarray:
    """Raw uint32 data of an (old-style) PRNGKey, as a plain numpy array."""
    return np.asarray(jax.device_get(key)).astype(np.uint32)


def capture(env: CraftaxTextEnv, ex: Optional[Executor] = None,
            meta: Optional[Dict[str, Any]] = None) -> Snapshot:
    """Freeze `env` (+ optional `ex`) into an independent, picklable Snapshot."""
    s = env.state
    md = {
        "floor": int(s.player_level),
        "t": int(env.t),
        "n_achievements": int(np.asarray(s.achievements).astype(bool).sum()),
        "health": int(s.player_health),
        "food": int(s.player_food),
        "drink": int(s.player_drink),
        "energy": int(s.player_energy),
    }
    if meta:
        md.update(meta)
    snap = Snapshot(
        seed=int(env.seed),
        state=_host(s),
        step_rng=_key_data(env.step_rng),
        action_history=list(env.action_history),
        achievement_log=[list(a) for a in env.achievement_log],
        reward_history=[float(r) for r in env.reward_history],
        unlocked=sorted(int(i) for i in env._unlocked),
        t=int(env.t),
        meta=md,
    )
    if ex is not None:
        snap.mem_known = np.array(ex.mem.known, copy=True)
        snap.mem_known_items = np.array(ex.mem.known_items, copy=True)
    return snap


def restore(snap: Snapshot, with_executor: bool = True):
    """Rebuild an independent (env[, ex]) at the frozen state.

    Returns `env` if `with_executor` is False, else `(env, ex)`. The returned
    objects share nothing mutable with the snapshot or with any other restore of
    the same snapshot, so branching different actions is side-effect free.
    """
    env = CraftaxTextEnv(seed=snap.seed)
    # inject frozen env state (fresh copies; keep the snapshot pristine)
    env.state = copy.deepcopy(snap.state)
    env.step_rng = jnp.asarray(np.asarray(snap.step_rng, dtype=np.uint32))
    env.action_history = list(snap.action_history)
    env.achievement_log = [list(a) for a in snap.achievement_log]
    env.reward_history = [float(r) for r in getattr(snap, "reward_history", [])]
    env._unlocked = set(int(i) for i in snap.unlocked)
    env.t = int(snap.t)
    if not with_executor:
        return env
    ex = Executor(env)  # __init__ seeds mem from current state; overwrite if frozen
    if snap.mem_known is not None:
        ex.mem.known = np.array(snap.mem_known, copy=True)
        ex.mem.known_items = np.array(snap.mem_known_items, copy=True)
    return env, ex


# --- disk persistence (pickle; snapshots hold big arrays, not JSON) ----------
def save(snap: Snapshot, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(snap, f, protocol=pickle.HIGHEST_PROTOCOL)


def load(path: str) -> Snapshot:
    with open(path, "rb") as f:
        return pickle.load(f)
