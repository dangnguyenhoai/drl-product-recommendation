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

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.valid_actions_tensor = torch.LongTensor(self.valid_actions).to(self.device)

        self.memory = deque(maxlen=memory_size)

        self.model = DQN(state_dim, action_dim).to(self.device)
        self.target_model = DQN(state_dim, action_dim).to(self.device)
        self.update_target_network()

        self.target_model.eval()

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
        )

        self.loss_fn = nn.MSELoss()
        self.replay_steps = 0

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

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.sample(
                self.valid_actions,
                self.top_k,
            )

        state = np.array(state, dtype=np.float32) / self.action_dim
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values = self.model(state_tensor)[0]

            valid_q_values = q_values[self.valid_actions_tensor]

            top_indices = torch.topk(
                valid_q_values,
                k=self.top_k,
            ).indices

            top_actions = self.valid_actions_tensor[top_indices]

        return top_actions.cpu().tolist()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return 0

        batch = random.sample(
            self.memory,
            self.batch_size,
        )

        total_loss = 0

        for state, actions, reward, next_state, done in batch:
            state = np.array(state, dtype=np.float32) / self.action_dim
            next_state = np.array(next_state, dtype=np.float32) / self.action_dim

            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)

            q_values = self.model(state_tensor)[0]
            action_tensor = torch.LongTensor(actions).to(self.device)

            current_q = q_values[action_tensor].mean()

            if done:
                target_q = torch.tensor(
                    reward,
                    dtype=torch.float32,
                    device=self.device,
                )
            else:
                with torch.no_grad():
                    next_q_values = self.target_model(next_state_tensor)[0]

                    valid_next_q_values = next_q_values[self.valid_actions_tensor]

                    top_next_q = torch.topk(
                        valid_next_q_values,
                        k=self.top_k,
                    ).values

                    max_next_q = top_next_q.mean()

                    target_q = reward + self.gamma * max_next_q
                    target_q = torch.clamp(
                        target_q,
                        min=-10,
                        max=10,
                    )

            loss = self.loss_fn(
                current_q,
                target_q,
            )

            self.optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=1.0,
            )

            self.optimizer.step()

            total_loss += loss.item()

        self.replay_steps += 1

        if self.replay_steps % self.target_update_freq == 0:
            self.update_target_network()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return total_loss / self.batch_size