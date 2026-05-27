import argparse
import os
import pickle

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
        "--model_path",
        type=str,
        default="outputs/checkpoints/dqn_weights.pth",
        help="Path to trained DQN model weights",
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes",
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

    env = RecommendationEnv(
        indexed_history,
        state_size=args.state_dim,
        top_k=args.top_k,
    )

    agent = load_agent(
        args=args,
        valid_actions=valid_actions,
    )

    total_rewards = []
    total_hits = 0
    total_recommendations = 0

    for episode in range(args.episodes):
        state = env.reset()
        done = False
        episode_reward = 0

        while not done:
            actions = agent.choose_action(state)
            next_state, reward, done, info = env.step(actions)

            episode_reward += reward
            total_hits += info.get("hits", 0)
            total_recommendations += len(actions)

            state = next_state

        total_rewards.append(episode_reward)

        print(
            f"Episode {episode + 1}/{args.episodes}"
            f" | Reward: {episode_reward}"
        )

    average_reward = sum(total_rewards) / len(total_rewards)

    if total_recommendations == 0:
        hit_rate = 0.0
    else:
        hit_rate = total_hits / total_recommendations

    print("\n===== Evaluation Results =====")
    print(f"Average Reward: {average_reward:.3f}")
    print(f"Hit Rate@{args.top_k}: {hit_rate:.4f}")


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)