import argparse
import os
import pickle
import random
from collections import Counter

from env.recommendation_env import RecommendationEnv


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
        raise FileNotFoundError(
            f"Data file not found: {data_path}"
        )

    with open(data_path, "rb") as f:
        return pickle.load(f)


def get_valid_actions(indexed_history):
    valid_actions = set()

    for history in indexed_history.values():
        valid_actions.update(history)

    return sorted(valid_actions)


def get_popular_items(indexed_history, top_k):
    item_counter = Counter()

    for history in indexed_history.values():
        item_counter.update(history)

    popular_items = [
        item
        for item, _ in item_counter.most_common(top_k)
    ]

    return popular_items


def evaluate_policy(env, episodes, policy_fn, top_k):
    total_rewards = []
    total_step_hits = 0
    total_steps = 0

    for episode in range(episodes):
        state = env.reset()
        done = False
        episode_reward = 0

        while not done:
            actions = policy_fn(state)

            next_state, reward, done, info = env.step(actions)

            episode_reward += reward
            hits = info.get("hits", 0)
            if hits > 0:
                total_step_hits += 1
            total_steps += 1

            state = next_state

        total_rewards.append(episode_reward)

    average_reward = sum(total_rewards) / len(total_rewards)

    if total_steps == 0:
        hit_rate = 0.0
    else:
        hit_rate = total_step_hits / total_steps

    return average_reward, hit_rate


def main(args):
    indexed_history = load_indexed_history(args.data_path)

    valid_actions = get_valid_actions(indexed_history)
    popular_items = get_popular_items(indexed_history, args.top_k)

    print(f"Using valid actions: {len(valid_actions)}")
    print(f"Top popular items: {popular_items}")

    random_env = RecommendationEnv(
        indexed_history,
        top_k=args.top_k,
    )

    popularity_env = RecommendationEnv(
        indexed_history,
        top_k=args.top_k,
    )

    def random_policy(_state):
        return random.sample(
            valid_actions,
            args.top_k,
        )

    def popularity_policy(_state):
        return popular_items

    random_reward, random_hit_rate = evaluate_policy(
        env=random_env,
        episodes=args.episodes,
        policy_fn=random_policy,
        top_k=args.top_k,
    )

    popularity_reward, popularity_hit_rate = evaluate_policy(
        env=popularity_env,
        episodes=args.episodes,
        policy_fn=popularity_policy,
        top_k=args.top_k,
    )

    print("\n===== Baseline Results =====")
    print(f"Random Average Reward: {random_reward:.3f}")
    print(f"Random Hit Rate@{args.top_k}: {random_hit_rate:.4f}")

    print(f"Popularity Average Reward: {popularity_reward:.3f}")
    print(f"Popularity Hit Rate@{args.top_k}: {popularity_hit_rate:.4f}")


if __name__ == "__main__":
    args = parse_args()
    main(args)