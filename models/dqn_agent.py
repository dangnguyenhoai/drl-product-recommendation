import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.dqn_model import DQN


class DQNAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        valid_actions=None,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.999,
        learning_rate=0.0001,
        batch_size=32,
        memory_size=10000,
        top_k=5,
        target_update_freq=100,
        device="auto",
        embedding_dim=32,
        hidden_dim=128,
        recent_boost=0.0,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim

        if valid_actions is None:
            self.valid_actions = list(range(action_dim))
        else:
            self.valid_actions = sorted(set(valid_actions))

        if len(self.valid_actions) < top_k:
            raise ValueError(
                f"valid_actions has only {len(self.valid_actions)} items, "
                f"but top_k={top_k}."
            )

        max_valid_action = max(self.valid_actions)
        if max_valid_action >= action_dim:
            raise ValueError(
                f"max valid action is {max_valid_action}, "
                f"but action_dim is {action_dim}. "
                "action_dim must be greater than max item id."
            )

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.top_k = top_k
        self.target_update_freq = target_update_freq
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.recent_boost = recent_boost

        if device == "auto":
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        self.valid_actions_tensor = torch.LongTensor(
            self.valid_actions
        ).to(self.device)

        self.memory = deque(maxlen=memory_size)

        self.model = DQN(
            state_dim=state_dim,
            action_dim=action_dim,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)

        self.target_model = DQN(
            state_dim=state_dim,
            action_dim=action_dim,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)

        self.update_target_network()
        self.target_model.eval()

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
        )

        self.loss_fn = nn.SmoothL1Loss()
        self.replay_steps = 0
        self.valid_mask = torch.zeros(self.action_dim, dtype=torch.bool, device=self.device)
        self.valid_mask[self.valid_actions_tensor] = True

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append(
            (
                state,
                action,
                reward,
                next_state,
                done,
            )
        )

    def choose_action(self, state, banned_actions=None):
        available_actions = self.get_available_actions(banned_actions)

        if random.random() < self.epsilon:
            return random.sample(
                available_actions,
                self.top_k,
            )

        state = np.array(state, dtype=np.int64)
        state_tensor = torch.LongTensor(state).unsqueeze(0).to(self.device)

        available_actions_tensor = torch.LongTensor(
            available_actions
        ).to(self.device)

        with torch.no_grad():
            q_values = self.model(state_tensor)[0]

            if self.recent_boost != 0.0:
                banned_action_set = (
                    set()
                    if banned_actions is None
                    else set(int(action) for action in banned_actions)
                )
                recent_items = set(int(item) for item in state)
                boosted_actions = [
                    action
                    for action in available_actions
                    if (
                        action in recent_items
                        and action not in banned_action_set
                    )
                ]

                if boosted_actions:
                    q_values = q_values.clone()
                    boosted_actions_tensor = torch.LongTensor(
                        boosted_actions
                    ).to(self.device)
                    q_values[boosted_actions_tensor] += self.recent_boost

            available_q_values = q_values[available_actions_tensor]

            top_indices = torch.topk(
                available_q_values,
                k=self.top_k,
            ).indices

            top_actions = available_actions_tensor[top_indices]

        return top_actions.cpu().tolist()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return 0

        # Sample a batch of transitions
        batch = random.sample(self.memory, self.batch_size)

        # Decompose the batch into lists
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to numpy arrays for efficiency
        states = np.array(states, dtype=np.int64)
        next_states = np.array(next_states, dtype=np.int64)
        actions = np.array(actions, dtype=np.int64)
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)

        # Convert to PyTorch tensors and move to device
        state_tensor = torch.LongTensor(states).to(self.device)
        next_state_tensor = torch.LongTensor(next_states).to(self.device)
        action_tensor = torch.LongTensor(actions).to(self.device)
        reward_tensor = torch.FloatTensor(rewards).to(self.device)
        done_tensor = torch.FloatTensor(dones).to(self.device)

        # 1. Compute current Q-values for chosen actions: Q(s, a)
        q_values = self.model(state_tensor)
        current_q = q_values.gather(1, action_tensor.unsqueeze(1)).squeeze(1) # shape: [batch_size]

        # 2. Compute target Q-values: r + gamma * max_a' Q_target(s', a')
        with torch.no_grad():
            next_q_values = self.target_model(next_state_tensor) # shape: [batch_size, action_dim]

            # Mask out invalid actions
            masked_next_q_values = next_q_values.clone()
            masked_next_q_values[:, ~self.valid_mask] = -1e9

            # Standard Q-learning: max over actions
            max_next_q = masked_next_q_values.max(dim=1).values # shape: [batch_size]

            # Compute target: r + gamma * max_a' Q(s', a') (only if not done)
            target_q = reward_tensor + self.gamma * max_next_q * (1.0 - done_tensor)

            # Clamp target Q-values for stability
            target_q = torch.clamp(target_q, min=-10.0, max=10.0)

        # 3. Compute loss
        loss = self.loss_fn(current_q, target_q)

        # 4. Optimize
        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=1.0,
        )

        self.optimizer.step()

        self.replay_steps += 1

        if self.replay_steps % self.target_update_freq == 0:
            self.update_target_network()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            if self.epsilon < self.epsilon_min:
                self.epsilon = self.epsilon_min

        return loss.item()
    
    def get_available_actions(self, banned_actions=None):
        if banned_actions is None:
            return self.valid_actions

        banned_actions = set(int(action) for action in banned_actions)

        available_actions = [
            action
            for action in self.valid_actions
            if action not in banned_actions
        ]

        if len(available_actions) < self.top_k:
            return self.valid_actions

        return available_actions
