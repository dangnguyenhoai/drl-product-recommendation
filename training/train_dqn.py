import argparse
import os
import pickle

import matplotlib.pyplot as plt
import torch

from env.recommendation_env import RecommendationEnv
from models.dqn_agent import DQNAgent


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_path",
        type=str,
        default="data/processed/indexed_history.pkl",
        help="Path to processed indexed history pickle file",
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of training episodes",
    )

    parser.add_argument(
        "--state_dim",
        type=int,
        default=5,
        help="State dimension",
    )

    parser.add_argument(
        "--action_dim",
        type=int,
        default=None,
        help="Number of possible item actions. If omitted, inferred from data.",
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default="outputs/checkpoints/dqn_weights.pth",
        help="Where to save trained model weights",
    )

    parser.add_argument(
        "--plot_dir",
        type=str,
        default="outputs/plots",
        help="Directory to save training plots",
    )

    return parser.parse_args()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def show_training_results(reward_history, loss_history, epsilon_history, plot_dir):
    ensure_dir(plot_dir)

    plt.figure(figsize=(10, 5))
    plt.plot(reward_history)
    plt.title("Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.savefig(os.path.join(plot_dir, "episode_reward.png"))
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(loss_history)
    plt.title("Training Loss")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(plot_dir, "training_loss.png"))
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(epsilon_history)
    plt.title("Epsilon Decay")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.savefig(os.path.join(plot_dir, "epsilon_decay.png"))
    plt.close()


def load_indexed_history(data_path):
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Data file not found: {data_path}. "
            "Check your --data_path. Do not pretend the file exists."
        )

    with open(data_path, "rb") as f:
        return pickle.load(f)

def get_valid_actions(indexed_history):
    valid_actions = set()

    for history in indexed_history.values():
        valid_actions.update(history)

    return sorted(valid_actions)


def infer_action_dim(valid_actions):
    return max(valid_actions) + 1


def train(args):
    indexed_history = load_indexed_history(args.data_path)

    env = RecommendationEnv(indexed_history)

    valid_actions = get_valid_actions(indexed_history)

    if args.action_dim is None:
        args.action_dim = infer_action_dim(valid_actions)

    print(f"Using action_dim: {args.action_dim}")
    print(f"Using valid actions: {len(valid_actions)}")

    agent = DQNAgent(
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        valid_actions=valid_actions,
    )

    model_dir = os.path.dirname(args.model_path)
    if model_dir:
        ensure_dir(model_dir)

    reward_history = []
    loss_history = []
    epsilon_history = []

    for episode in range(args.episodes):
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
                done,
            )

            step_count += 1
            loss = 0

            if len(agent.memory) > agent.batch_size and step_count % 4 == 0:
                loss = agent.replay()
                episode_loss += loss

            state = next_state
            total_reward += reward

        reward_history.append(total_reward)
        loss_history.append(episode_loss)
        epsilon_history.append(agent.epsilon)

        print(
            f"Episode {episode + 1}/{args.episodes}"
            f" | Reward: {total_reward}"
            f" | Loss: {episode_loss:.4f}"
            f" | Epsilon: {agent.epsilon:.4f}"
        )

        torch.save(agent.model.state_dict(), args.model_path)

    show_training_results(
        reward_history,
        loss_history,
        epsilon_history,
        args.plot_dir,
    )

    print(f"\nSaved model to: {args.model_path}")
    print(f"Saved plots to: {args.plot_dir}")


if __name__ == "__main__":
    args = parse_args()
    train(args)