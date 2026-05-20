import pickle
import os

import torch

import matplotlib.pyplot as plt

from env.recommendation_env import RecommendationEnv

from models.dqn_agent import DQNAgent

def show_training_results(reward_history, loss_history, epsilon_history):

    output_dir = "training_results_visualize"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.figure(figsize=(10,5))

    plt.plot(reward_history)

    plt.title("Episode Reward")

    plt.xlabel("Episode")

    plt.ylabel("Total Reward")

    plt.savefig(os.path.join(output_dir, "episode_reward.png"))
    plt.close()

    plt.figure(figsize=(10,5))

    plt.plot(loss_history)

    plt.title("Training Loss")

    plt.xlabel("Episode")

    plt.ylabel("Loss")

    plt.savefig(os.path.join(output_dir, "training_loss.png"))
    plt.close()

    plt.figure(figsize=(10,5))

    plt.plot(epsilon_history)

    plt.title("Epsilon Decay")

    plt.xlabel("Episode")

    plt.ylabel("Epsilon")

    plt.savefig(os.path.join(output_dir, "epsilon_decay.png"))
    plt.close()

if __name__ == "__main__":
    with open(
        "data/processed/indexed_history.pkl",
        "rb"
    ) as f:

        indexed_history = pickle.load(f)

    state_dim = 5
    action_dim = 500

    env = RecommendationEnv(
        indexed_history
    )

    agent = DQNAgent(
        state_dim,
        action_dim
    )

    episodes = 500

    reward_history = []

    loss_history = []

    epsilon_history = []

    for episode in range(episodes):

        state = env.reset()

        total_reward = 0

        done = False

        episode_loss = 0

        step_count = 0

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

            step_count += 1

            loss = 0

            if (
                len(agent.memory)
                > agent.batch_size
                and step_count % 4 == 0
            ):

                loss = agent.replay()

            episode_loss += loss

            state = next_state

            total_reward += reward

        #     print( "\n\n"
        #     f" | Memory Size: {len(agent.memory)}"
        #     f" | State: {state}"
        #     f" | Action: {action}")


        # print(
        #     f"\n\nEpisode {episode + 1}"
        #     f" | Total Reward: {total_reward}"
        #     f" | Epsilon: {agent.epsilon:.3f}"
        #     f" | Memory Size: {len(agent.memory)}"
        #     f" | State: {state}"
        #     f" | Action: {action}"
        # )

        print(f"\n\nEpisode {episode + 1}")

        reward_history.append(total_reward)

        loss_history.append(episode_loss)

        epsilon_history.append(agent.epsilon)

    torch.save(
        agent.model.state_dict(),
        "models/dqn_weights.pth"
    )

    show_training_results(reward_history, loss_history, epsilon_history)
