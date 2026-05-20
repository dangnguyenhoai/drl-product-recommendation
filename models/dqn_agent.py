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
        self.epsilon_decay = 0.995

        self.learning_rate = 0.001

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
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():

            q_values = self.model(state_tensor)
        
        action = torch.argmax(
            q_values
        ).item()

        return action
    
    def replay(self):

        if len(self.memory) < self.batch_size:

            return
        
        batch = random.sample(
            self.memory, self.batch_size
        )

        for (state, action, reward, next_state, done) in batch:
            
            state_tensor = torch.FloatTensor(state).unsqueeze(0)

            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)

            current_q = self.model(state_tensor)[0][action]
            
            target_q = reward

            with torch.no_grad():

                max_next_q = torch.max(
                    self.model(next_state_tensor)
                )

            target_q = (
                reward
                + self.gamma * max_next_q
            )

            loss = self.loss_fn(
                current_q, target_q
            )

            self.optimizer.zero_grad()

            loss.backward()
            
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:

            self.epsilon *= self.epsilon_decay
            
