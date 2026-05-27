import argparse
import csv
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

    parser.add_argument(
        "--log_path",
        type=str,
        default="outputs/logs/train_log.csv",
        help="Where to save training log CSV",
    )

    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--epsilon_min", type=float, default=0.05)
    parser.add_argument("--epsilon_decay", type=float, default=0.995)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--memory_size", type=int, default=10000)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--target_update_freq", type=int, default=100)
    parser.add_argument("--device", type=str, default="auto")

    parser.add_argument("--embedding_dim", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=128)

    return parser.parse_args()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def ensure_parent_dir(file_path):
    parent_dir = os.path.dirname(file_path)
    if parent_dir:
        ensure_dir(parent_dir)


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


def save_training_log(log_path, reward_history, loss_history, epsilon_history):
    ensure_parent_dir(log_path)

    with open(log_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "episode",
                "reward",
                "loss",
                "epsilon",
            ]
        )

        for idx, (reward, loss, epsilon) in enumerate(
            zip(reward_history, loss_history, epsilon_history),
            start=1,
        ):
            writer.writerow(
                [
                    idx,
                    reward,
                    loss,
                    epsilon,
                ]
            )


def train(args):
    indexed_history = load_indexed_history(args.data_path)

    valid_actions = get_valid_actions(indexed_history)

    if args.action_dim is None:
        args.action_dim = infer_action_dim(valid_actions)

    print(f"Using action_dim: {args.action_dim}")
    print(f"Using valid actions: {len(valid_actions)}")
    print(f"Using embedding_dim: {args.embedding_dim}")
    print(f"Using hidden_dim: {args.hidden_dim}")

    env = RecommendationEnv(
        indexed_history,
        state_size=args.state_dim,
        top_k=args.top_k,
    )

    agent = DQNAgent(
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        valid_actions=valid_actions,
        gamma=args.gamma,
        epsilon=args.epsilon,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        memory_size=args.memory_size,
        top_k=args.top_k,
        target_update_freq=args.target_update_freq,
        device=args.device,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
    )

    ensure_parent_dir(args.model_path)
    ensure_parent_dir(args.log_path)
    ensure_dir(args.plot_dir)

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
            action = agent.choose_action(
                state,
                banned_actions=env.recommended_items,
            )
            next_state, reward, done, _ = env.step(action)

            agent.remember(
                state,
                action,
                reward,
                next_state,
                done,
            )

            step_count += 1

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

    save_training_log(
        args.log_path,
        reward_history,
        loss_history,
        epsilon_history,
    )

    print(f"\nSaved model to: {args.model_path}")
    print(f"Saved plots to: {args.plot_dir}")
    print(f"Saved log to: {args.log_path}")


if __name__ == "__main__":
    args = parse_args()
    train(args)