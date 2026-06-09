import argparse
import os
import pickle
from collections import Counter

from env.recommendation_env import RecommendationEnv


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_path",
        type=str,
        default="data/processed/test_history.pkl",
        help="Path to processed indexed history pickle file",
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes",
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of recommended items",
    )

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


def get_popular_items(indexed_history):
    item_counter = Counter()

    for history in indexed_history.values():
        item_counter.update(history)

    return [
        item
        for item, _ in item_counter.most_common()
    ]


def evaluate_recent_policy(indexed_history, episodes, top_k):
    env = RecommendationEnv(
        indexed_history,
        top_k=top_k,
    )

    valid_actions = get_valid_actions(indexed_history)
    popular_items = get_popular_items(indexed_history)

    total_rewards = []
    total_step_hits = 0
    total_steps = 0

    for _ in range(episodes):
        state = env.reset()
        done = False
        episode_reward = 0

        while not done:
            actions = []

            # First: recommend recent items from the current state, newest first.
            for item in reversed(list(state)):
                item = int(item)

                if item not in actions and item not in env.recommended_items:
                    actions.append(item)

                if len(actions) >= top_k:
                    break

            # Fill remaining slots with popular items.
            for item in popular_items:
                item = int(item)

                if item not in actions and item not in env.recommended_items:
                    actions.append(item)

                if len(actions) >= top_k:
                    break

            # Final fallback: valid actions.
            for item in valid_actions:
                item = int(item)

                if item not in actions and item not in env.recommended_items:
                    actions.append(item)

                if len(actions) >= top_k:
                    break

            next_state, reward, done, info = env.step(actions)

            episode_reward += reward
            hits = info.get("hits", 0)
            if hits > 0:
                total_step_hits += 1
            total_steps += 1

            state = next_state

        total_rewards.append(episode_reward)

    average_reward = sum(total_rewards) / len(total_rewards)

    hit_rate = (
        total_step_hits / total_steps
        if total_steps > 0
        else 0.0
    )

    print("\n===== Recent-Item Baseline Results =====")
    print(f"Average Reward: {average_reward:.3f}")
    print(f"Hit Rate@{top_k}: {hit_rate:.4f}")


if __name__ == "__main__":
    args = parse_args()

    indexed_history = load_indexed_history(args.data_path)

    evaluate_recent_policy(
        indexed_history=indexed_history,
        episodes=args.episodes,
        top_k=args.top_k,
    )