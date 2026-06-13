import argparse
import os
import pickle
import random
from collections import Counter

from env.recommendation_env import RecommendationEnv
from evaluation.metrics import TopKMetrics, print_top_k_metrics
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
        "--top_k",
        type=int,
        default=5,
        help="Number of recommended items",
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


def evaluate_policy(env, indexed_history, episodes, policy_fn, top_k, log_interval):
    total_rewards = []
    metrics_tracker = TopKMetrics(top_k)

    full_pass = uses_full_pass(episodes)
    if full_pass:
        total_episodes, total_windows = count_full_pass(
            indexed_history,
            state_size=env.state_size,
            top_k=top_k,
            max_steps=env.max_steps,
        )
        episode_starts = iter_full_pass_episode_starts(
            indexed_history,
            state_size=env.state_size,
            top_k=top_k,
            max_steps=env.max_steps,
        )
        print("Evaluation mode: full pass over all windows/interactions")
        print(f"Total full-pass episodes: {total_episodes}")
        print(f"Total full-pass evaluation windows: {total_windows}")
    else:
        total_episodes = parse_episode_count(episodes)
        episode_starts = (None for _ in range(total_episodes))
        print("Evaluation mode: sampled random episodes")
        print(f"Total requested episodes: {total_episodes}")

    for episode, episode_start in enumerate(episode_starts, start=1):
        if full_pass:
            user_id, pointer = episode_start
            state = env.reset_at(user_id, pointer)
        else:
            state = env.reset()

        done = False
        episode_reward = 0

        while not done:
            actions = policy_fn(state, env)

            next_state, reward, done, info = env.step(actions)

            episode_reward += reward
            metrics_tracker.update(
                info.get("recommended_items", actions),
                info.get("target_items", []),
            )

            state = next_state

        total_rewards.append(episode_reward)

        if (
            log_interval <= 1
            or episode == 1
            or episode == total_episodes
            or episode % log_interval == 0
        ):
            print(
                f"Episode {episode}/{total_episodes}"
                f" | Reward: {episode_reward}"
            )

    average_reward = sum(total_rewards) / len(total_rewards)

    return average_reward, metrics_tracker.as_dict()


def main(args):
    indexed_history = load_indexed_history(args.data_path)

    valid_actions = get_valid_actions(indexed_history)
    popular_items = get_popular_items(indexed_history, args.top_k)

    print(f"Using valid actions: {len(valid_actions)}")
    print(f"Top popular items: {popular_items}")

    random_env = RecommendationEnv(
        indexed_history,
        state_size=args.state_dim,
        top_k=args.top_k,
    )

    popularity_env = RecommendationEnv(
        indexed_history,
        state_size=args.state_dim,
        top_k=args.top_k,
    )

    def random_policy(_state, env):
        candidates = [
            item
            for item in valid_actions
            if item not in env.recommended_items
        ]

        if len(candidates) <= args.top_k:
            return candidates

        return random.sample(
            candidates,
            args.top_k,
        )

    def popularity_policy(_state, env):
        actions = [
            item
            for item in popular_items
            if item not in env.recommended_items
        ]

        if len(actions) < args.top_k:
            used = set(actions)
            actions.extend(
                item
                for item in valid_actions
                if item not in used and item not in env.recommended_items
            )

        return actions[: args.top_k]

    random_reward, random_metrics = evaluate_policy(
        env=random_env,
        indexed_history=indexed_history,
        episodes=args.episodes,
        policy_fn=random_policy,
        top_k=args.top_k,
        log_interval=args.log_interval,
    )

    popularity_reward, popularity_metrics = evaluate_policy(
        env=popularity_env,
        indexed_history=indexed_history,
        episodes=args.episodes,
        policy_fn=popularity_policy,
        top_k=args.top_k,
        log_interval=args.log_interval,
    )

    print("\n===== Baseline Results =====")
    print(f"Random Average Reward: {random_reward:.3f}")
    print_top_k_metrics(random_metrics, args.top_k, prefix="Random")

    print(f"Popularity Average Reward: {popularity_reward:.3f}")
    print_top_k_metrics(popularity_metrics, args.top_k, prefix="Popularity")


if __name__ == "__main__":
    args = parse_args()
    main(args)
