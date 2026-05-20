from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int = 42) -> None:
        self.data: deque[Transition] = deque(maxlen=capacity)
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.data)

    def push(self, transition: Transition) -> None:
        self.data.append(transition)

    def sample(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, ...]:
        batch = self.rng.sample(self.data, batch_size)
        states = torch.tensor(np.stack([x.state for x in batch]), dtype=torch.float32, device=device)
        actions = torch.tensor([x.action for x in batch], dtype=torch.long, device=device).unsqueeze(1)
        rewards = torch.tensor([x.reward for x in batch], dtype=torch.float32, device=device).unsqueeze(1)
        next_states = torch.tensor(np.stack([x.next_state for x in batch]), dtype=torch.float32, device=device)
        dones = torch.tensor([x.done for x in batch], dtype=torch.float32, device=device).unsqueeze(1)
        return states, actions, rewards, next_states, dones

