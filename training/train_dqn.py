import pickle

from env.recommendation_env import RecommendationEnv

from models.dqn_agent import DQNAgent

with open(
    "data/processed/indexed_history.pkl",
    "rb"
) as f:

    indexed_history = pickle.load(f)

state_dim = 5
action_dim = 100

env = RecommendationEnv(
    indexed_history
)

agent = DQNAgent(
    state_dim,
    action_dim
)

episodes = 50

for episode in range(episodes):

    state = env.reset()

    total_reward = 0

    done = False

    while not done:

        action = agent.choose_action(state)

        next_state, reward, done, _ = env.step(action)

        agent.remember(
            state,
            action,
            reward,
            next_state,
            done
        )

        agent.replay()

        state = next_state

        total_reward += reward

        print( "\n\n"
        f" | Memory Size: {len(agent.memory)}"
        f" | State: {state}"
        f" | Action: {action}")


    print(
        f"\n\nEpisode {episode + 1}"
        f" | Total Reward: {total_reward}"
        f" | Epsilon: {agent.epsilon:.3f}"
        f" | Memory Size: {len(agent.memory)}"
        f" | State: {state}"
        f" | Action: {action}"
    )

