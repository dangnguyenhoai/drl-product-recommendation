import torch

from models.dqn_model import DQN

state_dim = 5

action_dim = 100

model = DQN(state_dim, action_dim)

sample_state = torch.randn(1, state_dim)

q_values = model(sample_state)

print("Q-values shape:", q_values.shape)

print("Q-values:", q_values)