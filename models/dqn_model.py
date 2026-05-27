import torch
import torch.nn as nn


class DQN(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        embedding_dim=32,
        hidden_dim=128,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.embedding_dim = embedding_dim

        self.item_embedding = nn.Embedding(
            num_embeddings=action_dim,
            embedding_dim=embedding_dim,
        )

        self.q_network = nn.Sequential(
            nn.Linear(state_dim * embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state):
        state = state.long()

        embedded_state = self.item_embedding(state)

        batch_size = embedded_state.size(0)
        flattened_state = embedded_state.view(batch_size, -1)

        q_values = self.q_network(flattened_state)

        return q_values