import argparse
import csv
import json
import os
import pickle

import matplotlib.pyplot as plt
import torch

from env.recommendation_env import RecommendationEnv
from models.dqn_agent import DQNAgent
from utils.full_pass import (
    count_full_pass,
    iter_full_pass_episode_starts,
    parse_episode_count,
    uses_full_pass,
)


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
        type=str,
        default=5,
        help=(
            "Number of training episodes, or 'all' to make one full pass over "
            "all possible training windows/interactions."
        ),
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
        help="Where to save last trained model weights",
    )

    parser.add_argument(
        "--best_model_path",
        type=str,
        default=None,
        help=(
            "Where to save the best training checkpoint by episode reward. "
            "If omitted, '<model_path stem>_best.pth' is used."
        ),
    )

    parser.add_argument(
        "--best_info_path",
        type=str,
        default=None,
        help=(
            "Where to save JSON metadata for the best training checkpoint. "
            "If omitted, '<best_model_path stem>.json' is used."
        ),
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
    parser.add_argument(
        "--log_interval",
        type=int,
        default=1,
        help="Print progress every N episodes. Use a larger value for --episodes all.",
    )

    parser.add_argument("--embedding_dim", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--recent_boost", type=float, default=0.0)

    return parser.parse_args()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def ensure_parent_dir(file_path):
    parent_dir = os.path.dirname(file_path)
    if parent_dir:
        ensure_dir(parent_dir)


def default_best_model_path(model_path):
    root, ext = os.path.splitext(model_path)

    if not ext:
        ext = ".pth"

    return f"{root}_best{ext}"


def default_best_info_path(best_model_path):
    root, _ = os.path.splitext(best_model_path)
    return f"{root}.json"


def save_best_info(best_info_path, info):
    ensure_parent_dir(best_info_path)

    with open(best_info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)


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
    print(f"Using recent_boost: {args.recent_boost}")

    env = RecommendationEnv(
        indexed_history,
        state_size=args.state_dim,
        top_k=args.top_k,
    )

    full_pass = uses_full_pass(args.episodes)
    if full_pass:
        total_episodes, total_windows = count_full_pass(
            indexed_history,
            state_size=args.state_dim,
            top_k=args.top_k,
            max_steps=env.max_steps,
        )
        episode_starts = iter_full_pass_episode_starts(
            indexed_history,
            state_size=args.state_dim,
            top_k=args.top_k,
            max_steps=env.max_steps,
        )
        print("Training mode: full pass over all training windows/interactions")
        print(f"Total full-pass episodes: {total_episodes}")
        print(f"Total full-pass training windows: {total_windows}")
    else:
        total_episodes = parse_episode_count(args.episodes)
        total_windows = None
        episode_starts = (None for _ in range(total_episodes))
        print("Training mode: sampled random episodes")
        print(f"Total requested episodes: {total_episodes}")

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
        recent_boost=args.recent_boost,
    )

    ensure_parent_dir(args.model_path)
    if args.best_model_path is None:
        args.best_model_path = default_best_model_path(args.model_path)

    if args.best_info_path is None:
        args.best_info_path = default_best_info_path(args.best_model_path)

    ensure_parent_dir(args.best_model_path)
    ensure_parent_dir(args.best_info_path)
    ensure_parent_dir(args.log_path)
    ensure_dir(args.plot_dir)

    reward_history = []
    loss_history = []
    epsilon_history = []
    best_reward = float("-inf")
    best_episode = 0

    for episode, episode_start in enumerate(episode_starts, start=1):
        if full_pass:
            user_id, pointer = episode_start
            state = env.reset_at(user_id, pointer)
        else:
            state = env.reset()

        total_reward = 0
        done = False
        episode_loss = 0
        step_count = 0

        while not done:
            # Capture already recommended items before this step to identify hits
            already_recommended = set(env.recommended_items)

            action = agent.choose_action(
                state,
                banned_actions=env.recommended_items,
            )
            next_state, reward, done, info = env.step(action)

            # Decompose the multi-action recommendation into single transitions
            target_set = set(info.get("target_items", []))
            for a in action:
                if a in target_set and a not in already_recommended:
                    r_i = env.hit_reward
                    already_recommended.add(a)
                else:
                    r_i = env.miss_penalty

                agent.remember(
                    state,
                    a,
                    r_i,
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

        if (
            args.log_interval <= 1
            or episode == 1
            or episode == total_episodes
            or episode % args.log_interval == 0
        ):
            print(
                f"Episode {episode}/{total_episodes}"
                f" | Reward: {total_reward}"
                f" | Loss: {episode_loss:.4f}"
                f" | Epsilon: {agent.epsilon:.4f}"
            )

        torch.save(agent.model.state_dict(), args.model_path)

        if total_reward > best_reward:
            best_reward = total_reward
            best_episode = episode
            torch.save(agent.model.state_dict(), args.best_model_path)
            save_best_info(
                args.best_info_path,
                {
                    "best_episode": best_episode,
                    "best_reward": best_reward,
                    "model_path": args.best_model_path,
                    "last_model_path": args.model_path,
                    "data_path": args.data_path,
                    "episodes": args.episodes,
                    "resolved_episodes": total_episodes,
                    "training_windows": total_windows,
                    "training_mode": "full_pass" if full_pass else "sampled",
                    "action_dim": args.action_dim,
                    "embedding_dim": args.embedding_dim,
                    "hidden_dim": args.hidden_dim,
                    "recent_boost": args.recent_boost,
                    "selection_metric": "episode_reward",
                },
            )

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
    print(
        f"Saved best training model to: {args.best_model_path} "
        f"(episode {best_episode}, reward {best_reward})"
    )
    print(f"Saved best training metadata to: {args.best_info_path}")
    print(f"Saved plots to: {args.plot_dir}")
    print(f"Saved log to: {args.log_path}")


if __name__ == "__main__":
    args = parse_args()
    train(args)
