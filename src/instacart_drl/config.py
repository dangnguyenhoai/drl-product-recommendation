from dataclasses import dataclass


@dataclass(frozen=True)
class RewardConfig:
    hit: float = 1.0
    miss: float = -0.1
    duplicate: float = -0.5


@dataclass(frozen=True)
class DQNConfig:
    gamma: float = 0.95
    learning_rate: float = 1e-4
    replay_size: int = 50_000
    batch_size: int = 128
    min_replay: int = 1_000
    target_update_steps: int = 500
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 20_000
    hidden_dim: int = 256
