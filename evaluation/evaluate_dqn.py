import pickle
import torch

from env.recommendation_env import (
    RecommendationEnv
)

from models.dqn_agent import (
    DQNAgent
)


def evaluate():

    # =========================
    # Load processed data
    # =========================

    with open(
        "data/processed/indexed_history.pkl",
        "rb"
    ) as f:

        indexed_history = pickle.load(f)

    # =========================
    # Create environment
    # =========================

    state_dim = 5
    action_dim = 20

    env = RecommendationEnv(
        indexed_history
    )

    # =========================
    # Create agent
    # =========================

    agent = DQNAgent(
        state_dim,
        action_dim
    )

    # =========================
    # Load trained model
    # =========================

    agent.model.load_state_dict(
        torch.load(
            "models/dqn_weights.pth"
        )
    )

    # =========================
    # Disable exploration
    # =========================

    agent.epsilon = 0

    # =========================
    # Evaluation settings
    # =========================

    episodes = 20

    total_rewards = []

    total_hits = 0

    total_recommendations = 0

    # =========================
    # Evaluation loop
    # =========================

    for episode in range(episodes):

        state = env.reset()

        done = False

        episode_reward = 0

        while not done:

            # =========================
            # Agent chooses action
            # =========================

            action = agent.choose_action(
                state
            )

            # =========================
            # Environment response
            # =========================

            next_state, reward, done, _ = env.step(
                action
            )

            # =========================
            # Metrics
            # =========================

            episode_reward += reward

            total_recommendations += 1

            if reward > 0:

                total_hits += 1

            # =========================
            # Move to next state
            # =========================

            state = next_state

        total_rewards.append(
            episode_reward
        )

        print(
            f"Episode {episode + 1}"
            f" | Reward: {episode_reward}"
        )

    # =========================
    # Final metrics
    # =========================

    average_reward = (
        sum(total_rewards)
        / len(total_rewards)
    )

    hit_rate = (
        total_hits
        / total_recommendations
    )

    # =========================
    # Results
    # =========================

    print("\n===== Evaluation Results =====")

    print(
        f"Average Reward: {average_reward:.3f}"
    )

    print(
        f"Hit Rate: {hit_rate:.3f}"
    )


if __name__ == "__main__":

    evaluate()