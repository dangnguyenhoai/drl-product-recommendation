import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from collections import deque

from models.dqn_model import DQN

class DQNAgent:

    def __init__(self, state_dim, action_dim):

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.gamma = 0.99

        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999

        self.learning_rate = 0.0001

        self.batch_size = 32

        self.memory = deque(maxlen=10000)

        self.model = DQN(state_dim, action_dim)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.loss_fn = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):

        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state):

        if random.random() < self.epsilon:

            return random.randint(0, self.action_dim - 1)
        
        state = np.array(state) / self.action_dim

        state_tensor = torch.FloatTensor(
            state
        ).unsqueeze(0)

        with torch.no_grad():

            q_values = self.model(state_tensor)
        
        action = torch.argmax(
            q_values
        ).item()

        return action
    
    def replay(self):

        if len(self.memory) < self.batch_size:
            return 0

        batch = random.sample(
            self.memory,
            self.batch_size
        )

        total_loss = 0

        for (
            state,
            action,
            reward,
            next_state,
            done
        ) in batch:

            state = np.array(state) / self.action_dim

            next_state = np.array(next_state) / self.action_dim

            state_tensor = torch.FloatTensor(
                state
            ).unsqueeze(0)

            next_state_tensor = torch.FloatTensor(
                next_state
            ).unsqueeze(0)

            current_q = self.model(
                state_tensor
            )[0][action]

            if done:

                target_q = torch.tensor(
                    reward,
                    dtype=torch.float32)
            else:

                with torch.no_grad():

                    max_next_q = torch.max(
                        self.model(next_state_tensor)
                    )

                target_q = (
                    reward
                    + self.gamma * max_next_q
                )

                target_q = torch.clamp(
                    target_q,
                    -10,
                    10
                )

            loss = self.loss_fn(
                current_q,
                target_q
            )

            total_loss += loss.item()

            self.optimizer.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=1.0
            )

            self.optimizer.step()

        if self.epsilon > self.epsilon_min:

            self.epsilon *= self.epsilon_decay

        return total_loss / self.batch_size