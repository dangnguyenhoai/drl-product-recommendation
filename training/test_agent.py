from models.dqn_agent import DQNAgent
import numpy as np

state_dim = 5
action_dim = 100

agent = DQNAgent(state_dim, action_dim) 

state = [1,5,7,9,2]

action = agent.choose_action(state)

print("Chosen action:", action)

for _ in range(40):

    state = np.random.randint(
        0,
        100,
        size=5
    )

    action = np.random.randint(
        0,
        100
    )

    reward = np.random.randint(
        0,
        2
    )

    next_state = np.random.randint(
        0,
        100,
        size=5
    )

    done = False

    agent.remember(
        state,
        action,
        reward,
        next_state,
        done
    )

agent.replay()

print("Replay ran successfully")

action = agent.choose_action(state)

print("Chosen action:", action)