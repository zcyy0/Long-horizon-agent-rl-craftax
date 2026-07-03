"""Thin agent-computer loop over full Craftax (Craftax-Symbolic-v1).

From scratch, craftax-core only (no craftaxlm). Owns:
  - RNG: running per-step key, reset to a deterministic base on reset(), split
    every step -> stochastic transitions, replay-deterministic per seed.
  - Achievement ledger: per-step newly-unlocked achievements (CA substrate).
  - Action history + working go_back/go_forward (reset-and-replay).
  - Native text observation (craftax_text.render_text).

Action/achievement vocab come straight from the authoritative full Craftax enums.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Union

import jax
import numpy as np
from craftax.craftax.constants import Achievement, Action
from craftax.craftax_env import make_craftax_env_from_name

from craftax_text import render_text

ACTION_TO_INT = {a.name.lower(): int(a.value) for a in Action}
ACTION_NAMES = [Action(i).name.lower() for i in range(len(Action))]
# state.achievements is indexed by enum VALUE, and in full Craftax the enum's
# definition order != value order (diverges at value 25), so index by value.
ACHIEVEMENT_NAMES = [Achievement(i).name.lower() for i in range(len(Achievement))]


@dataclass
class StepResult:
    obs_text: str
    reward: float
    done: bool
    achievements_unlocked: List[str]
    step: int
    floor: int
    info: dict = field(repr=False, default_factory=dict)


class CraftaxTextEnv:
    ENV_NAME = "Craftax-Symbolic-v1"

    def __init__(self, seed: int = 0):
        self.seed = seed
        self.env = make_craftax_env_from_name(self.ENV_NAME, auto_reset=False)
        self.params = self.env.default_params
        self._base_rng = jax.random.PRNGKey(seed)
        self.reset()

    # ---- core loop -------------------------------------------------------
    def reset(self) -> str:
        reset_key, self.step_rng = jax.random.split(self._base_rng)
        _, self.state = self.env.reset(reset_key, self.params)
        self.action_history: List[int] = []
        self.achievement_log: List[List[str]] = []
        self._unlocked: set = set()
        self.t = 0
        return self.obs()

    def step(self, action: Union[str, int]) -> StepResult:
        a = self._to_int(action)
        self.step_rng, step_key = jax.random.split(self.step_rng)
        _, self.state, reward, done, info = self.env.step(
            step_key, self.state, a, self.params
        )
        self.action_history.append(a)
        self.t += 1
        newly = self._newly_unlocked()
        self.achievement_log.append(newly)
        return StepResult(
            self.obs(), float(reward), bool(done), newly, self.t,
            int(self.state.player_level), info,
        )

    def multistep(self, actions: List[Union[str, int]]) -> List[StepResult]:
        results = []
        for a in actions:
            r = self.step(a)
            results.append(r)
            if r.done:
                break
        return results

    # ---- replay / backtracking ------------------------------------------
    def go_forward(self, actions: List[Union[str, int]]) -> Optional[StepResult]:
        result = None
        for a in actions:
            result = self.step(a)
        return result

    def go_back(self, n_steps: int) -> Optional[StepResult]:
        keep = self.action_history[:-n_steps] if n_steps > 0 else list(self.action_history)
        self.reset()
        return self.go_forward(keep)

    # ---- helpers ---------------------------------------------------------
    def obs(self) -> str:
        return render_text(self.state)

    def _to_int(self, action: Union[str, int]) -> int:
        if isinstance(action, str):
            try:
                return ACTION_TO_INT[action.lower().strip()]
            except KeyError:
                raise ValueError(f"unknown action {action!r}; valid: {ACTION_NAMES}")
        return int(action)

    def _newly_unlocked(self) -> List[str]:
        ach = np.asarray(self.state.achievements).astype(bool)
        newly = []
        for i, on in enumerate(ach):
            if on and i not in self._unlocked:
                self._unlocked.add(i)
                newly.append(ACHIEVEMENT_NAMES[i])
        return newly
