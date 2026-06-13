import argparse
import os
import pickle
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
        default="data/processed/test_history.pkl",
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


def evaluate_recent_policy(indexed_history, episodes, state_dim, top_k, log_interval):
    env = RecommendationEnv(
        indexed_history,
        state_size=state_dim,
        top_k=top_k,
    )

    valid_actions = get_valid_actions(indexed_history)
    popular_items = get_popular_items(indexed_history)

    total_rewards = []
    metrics_tracker = TopKMetrics(top_k)

    full_pass = uses_full_pass(episodes)
    if full_pass:
        total_episodes, total_windows = count_full_pass(
            indexed_history,
            state_size=state_dim,
            top_k=top_k,
            max_steps=env.max_steps,
        )
        episode_starts = iter_full_pass_episode_starts(
            indexed_history,
            state_size=state_dim,
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

    metrics = metrics_tracker.as_dict()

    print("\n===== Recent-Item Baseline Results =====")
    print(f"Average Reward: {average_reward:.3f}")
    print_top_k_metrics(metrics, top_k)


if __name__ == "__main__":
    args = parse_args()

    indexed_history = load_indexed_history(args.data_path)

    evaluate_recent_policy(
        indexed_history=indexed_history,
        episodes=args.episodes,
        state_dim=args.state_dim,
        top_k=args.top_k,
        log_interval=args.log_interval,
    )
