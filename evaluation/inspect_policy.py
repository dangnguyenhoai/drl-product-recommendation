import argparse
import os
import pickle
from collections import Counter

import torch

from env.recommendation_env import RecommendationEnv
from models.dqn_agent import DQNAgent


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, default="data/processed/indexed_history.pkl")
    parser.add_argument("--model_path", type=str, default="outputs/checkpoints/dqn_weights.pth")
    parser.add_argument("--episodes", type=int, default=20)

    parser.add_argument("--state_dim", type=int, default=5)
    parser.add_argument("--action_dim", type=int, default=None)
    parser.add_argument("--top_k", type=int, default=5)

    parser.add_argument("--embedding_dim", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--device", type=str, default="auto")

    return parser.parse_args()


def load_indexed_history(data_path):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

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


def main(args):
    indexed_history = load_indexed_history(args.data_path)

    valid_actions = get_valid_actions(indexed_history)

    if args.action_dim is None:
        args.action_dim = infer_action_dim(valid_actions)

    env = RecommendationEnv(
        indexed_history,
        state_size=args.state_dim,
        top_k=args.top_k,
    )

    agent = load_agent(args, valid_actions)

    action_counter = Counter()
    hit_counter = Counter()

    total_hits = 0
    total_recommendations = 0

    print("\n===== Policy Inspection =====")
    print(f"Using action_dim: {args.action_dim}")
    print(f"Using valid actions: {len(valid_actions)}")
    print(f"Using embedding_dim: {args.embedding_dim}")
    print(f"Using hidden_dim: {args.hidden_dim}")

    for episode in range(args.episodes):
        state = env.reset()
        done = False
        step = 0

        print(f"\n--- Episode {episode + 1} ---")

        while not done:
            actions = agent.choose_action(
                state,
                banned_actions=env.recommended_items,
            )
            next_state, reward, done, info = env.step(actions)

            target_items = info.get("target_items", [])
            hits = info.get("hits", 0)

            action_counter.update(actions)

            if hits > 0:
                for action in actions:
                    if action in target_items:
                        hit_counter[action] += 1

            total_hits += hits
            total_recommendations += len(actions)

            if step < 3:
                print(f"Step {step + 1}")
                print(f"  State:   {list(state)}")
                print(f"  Target:  {list(target_items)}")
                print(f"  Actions: {list(actions)}")
                print(f"  Hits:    {hits}")
                print(f"  Reward:  {reward}")

            state = next_state
            step += 1

    hit_rate = total_hits / total_recommendations if total_recommendations else 0

    print("\n===== Summary =====")
    print(f"Total hits: {total_hits}")
    print(f"Total recommendations: {total_recommendations}")
    print(f"Hit Rate@{args.top_k}: {hit_rate:.4f}")

    print("\n===== Top Recommended Items =====")
    for item, count in action_counter.most_common(20):
        print(f"Item {item}: recommended {count} times")

    print("\n===== Items That Actually Hit =====")
    if len(hit_counter) == 0:
        print("No hit items found.")
    else:
        for item, count in hit_counter.most_common(20):
            print(f"Item {item}: hit {count} times")


if __name__ == "__main__":
    args = parse_args()
    main(args)