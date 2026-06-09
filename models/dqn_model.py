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
        self.hidden_dim = hidden_dim

        self.item_embedding = nn.Embedding(
            num_embeddings=action_dim,
            embedding_dim=embedding_dim,
        )

        # GRU state encoder (processes sequential state input)
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            num_layers=1,
        )

        # Dueling network streams
        # Value stream: V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Advantage stream: A(s, a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state):
        # Ensure state is long tensor
        state = state.long()

        # Handle single state vs batch of states input
        # If input has shape [state_dim], unsqueeze to [1, state_dim]
        if state.dim() == 1:
            state = state.unsqueeze(0)

        # Embed state items -> [batch_size, state_dim, embedding_dim]
        embedded_state = self.item_embedding(state)

        # Pass through GRU -> gru_out: [batch_size, state_dim, hidden_dim]
        # hn: [1, batch_size, hidden_dim]
        gru_out, hn = self.gru(embedded_state)

        # Get the representation for the sequence
        # We take the last hidden state of the GRU
        features = hn.squeeze(0)  # shape: [batch_size, hidden_dim]

        # Value and Advantage calculation
        values = self.value_stream(features)  # shape: [batch_size, 1]
        advantages = self.advantage_stream(features)  # shape: [batch_size, action_dim]

        # Combine value and advantage: Q(s, a) = V(s) + (A(s, a) - mean(A(s, a)))
        q_values = values + (advantages - advantages.mean(dim=1, keepdim=True))

        return q_values