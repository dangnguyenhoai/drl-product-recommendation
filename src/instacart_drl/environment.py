from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from .config import RewardConfig


@dataclass
class Episode:
    user_id: int
    history: np.ndarray
    context: np.ndarray
    targets: set[int]


class InstacartRecommendationEnv:
    """One episode asks the agent to recommend top-N products for one user."""

    def __init__(
        self,
        episodes: list[Episode],
        action_size: int,
        top_n: int = 10,
        rewards: RewardConfig | None = None,
        seed: int = 42,
    ) -> None:
        if not episodes:
            raise ValueError("episodes must not be empty")
        self.episodes = episodes
        self.action_size = action_size
        self.top_n = top_n
        self.rewards = rewards or RewardConfig()
        self.rng = random.Random(seed)
        self.current: Episode | None = None
        self.selected: set[int] = set()
        self.steps = 0

    @property
    def state_dim(self) -> int:
        return self.action_size * 2 + 4

    def reset(self, episode: Episode | None = None) -> np.ndarray:
        self.current = episode or self.rng.choice(self.episodes)
        self.selected = set()
        self.steps = 0
        return self._state()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, bool]]:
        if self.current is None:
            raise RuntimeError("Call reset() before step().")

        duplicate = action in self.selected
        hit = action in self.current.targets and not duplicate

        if duplicate:
            reward = self.rewards.duplicate
        elif hit:
            reward = self.rewards.hit
        else:
            reward = self.rewards.miss

        self.selected.add(action)
        self.steps += 1
        done = self.steps >= self.top_n
        return self._state(), float(reward), done, {"hit": hit, "duplicate": duplicate}

    def _state(self) -> np.ndarray:
        if self.current is None:
            raise RuntimeError("Call reset() before requesting state.")
        mask = np.zeros(self.action_size, dtype=np.float32)
        if self.selected:
            mask[list(self.selected)] = 1.0
        return np.concatenate([self.current.history, self.current.context, mask]).astype(np.float32)

