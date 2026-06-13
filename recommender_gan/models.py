from __future__ import annotations

import torch
from torch import nn


class Generator(nn.Module):
    def __init__(
        self,
        action_dim: int,
        state_size: int,
        embedding_dim: int = 32,
        hidden_dim: int = 128,
        noise_dim: int = 32,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.noise_dim = noise_dim
        self.item_embedding = nn.Embedding(action_dim, embedding_dim)
        input_dim = state_size * embedding_dim + noise_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, states: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        batch_size = states.size(0)
        embedded = self.item_embedding(states).reshape(batch_size, -1)
        if noise is None:
            noise = torch.randn(batch_size, self.noise_dim, device=states.device)
        return self.net(torch.cat([embedded, noise], dim=1))


class Discriminator(nn.Module):
    def __init__(
        self,
        action_dim: int,
        state_size: int,
        embedding_dim: int = 32,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.state_embedding = nn.Embedding(action_dim, embedding_dim)
        self.item_embedding = nn.Embedding(action_dim, embedding_dim)
        input_dim = state_size * embedding_dim + embedding_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, states: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        batch_size = states.size(0)
        state_vec = self.state_embedding(states).reshape(batch_size, -1)
        item_vec = self.item_embedding(items)
        return self.net(torch.cat([state_vec, item_vec], dim=1)).squeeze(1)
