import argparse
import os
import pickle

import torch

from env.recommendation_env import RecommendationEnv
from evaluation.metrics import TopKMetrics, print_top_k_metrics
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
        "--model_path",
        type=str,
        default="outputs/checkpoints/dqn_weights.pth",
        help="Path to trained DQN model weights",
    )

    parser.add_argument(
        "--episodes",
        type=str,
        default=100,
        help=(
            "Number of evaluation episodes, or 'all' to evaluate all possible "
            "windows/interactions."
        ),
    )

    parser.add_argument(
        "--state_dim",
        type=int,
        default=5,
        help="State dimension / history window size",
    )

    parser.add_argument(
        "--action_dim",
        type=int,
        default=None,
        help="Number of possible item actions. If omitted, inferred from data.",
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of recommended items",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: auto, cpu, or cuda",
    )

    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=32,
        help="Item embedding dimension",
    )

    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=128,
        help="Hidden layer dimension",
    )

    parser.add_argument(
        "--recent_boost",
        type=float,
        default=0.0,
        help="Q-value boost applied to recent state items during action selection",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=1,
        help="Print progress every N episodes. Use a larger value for --episodes all.",
    )

    return parser.parse_args()


def load_indexed_history(data_path):
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Data file not found: {data_path}. "
            "Check your --data_path."
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


def load_agent(args, valid_actions):
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(
            f"Model file not found: {args.model_path}. "
            "Train the model first or check your --model_path."
        )

    agent = DQNAgent(
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        valid_actions=valid_actions,
        top_k=args.top_k,
        device=args.device,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        recent_boost=args.recent_boost,
    )

    state_dict = torch.load(
        args.model_path,
        map_location=agent.device,
    )

    agent.model.load_state_dict(state_dict)
    agent.model.eval()

    agent.epsilon = 0.0

    return agent


def evaluate(args):
    indexed_history = load_indexed_history(args.data_path)

    valid_actions = get_valid_actions(indexed_history)

    if args.action_dim is None:
        args.action_dim = infer_action_dim(valid_actions)

    print(f"Using action_dim: {args.action_dim}")
    print(f"Using valid actions: {len(valid_actions)}")
    print(f"Using embedding_dim: {args.embedding_dim}")
    print(f"Using hidden_dim: {args.hidden_dim}")
    print(f"Using top_k: {args.top_k}")
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
        print("Evaluation mode: full pass over all windows/interactions")
        print(f"Total full-pass episodes: {total_episodes}")
        print(f"Total full-pass evaluation windows: {total_windows}")
    else:
        total_episodes = parse_episode_count(args.episodes)
        episode_starts = (None for _ in range(total_episodes))
        print("Evaluation mode: sampled random episodes")
        print(f"Total requested episodes: {total_episodes}")

    agent = load_agent(
        args=args,
        valid_actions=valid_actions,
    )

    total_rewards = []
    metrics_tracker = TopKMetrics(args.top_k)

    for episode, episode_start in enumerate(episode_starts, start=1):
        if full_pass:
            user_id, pointer = episode_start
            state = env.reset_at(user_id, pointer)
        else:
            state = env.reset()

        done = False
        episode_reward = 0

        while not done:
            actions = agent.choose_action(
                state,
                banned_actions=env.recommended_items,
            )
            next_state, reward, done, info = env.step(actions)

            episode_reward += reward
            metrics_tracker.update(
                info.get("recommended_items", actions),
                info.get("target_items", []),
            )

            state = next_state

        total_rewards.append(episode_reward)

        if (
            args.log_interval <= 1
            or episode == 1
            or episode == total_episodes
            or episode % args.log_interval == 0
        ):
            print(
                f"Episode {episode}/{total_episodes}"
                f" | Reward: {episode_reward}"
            )

    average_reward = sum(total_rewards) / len(total_rewards)

    metrics = metrics_tracker.as_dict()

    print("\n===== Evaluation Results =====")
    print(f"Average Reward: {average_reward:.3f}")
    print_top_k_metrics(metrics, args.top_k)


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
