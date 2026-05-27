import argparse
import os
import pickle

from collections import Counter


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_path",
        type=str,
        default="data/processed/indexed_history.pkl",
        help="Path to indexed history pickle file",
    )

    parser.add_argument(
        "--state_size",
        type=int,
        default=5,
        help="State size used by RecommendationEnv",
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Top-k recommendation size",
    )

    return parser.parse_args()


def load_data(data_path):
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Data file not found: {data_path}"
        )

    with open(data_path, "rb") as f:
        return pickle.load(f)


def inspect_data(indexed_history, state_size, top_k):
    num_users = len(indexed_history)

    history_lengths = [
        len(history)
        for history in indexed_history.values()
    ]

    all_items = []

    for history in indexed_history.values():
        all_items.extend(history)

    unique_items = sorted(set(all_items))

    min_item_id = min(unique_items)
    max_item_id = max(unique_items)
    num_unique_items = len(unique_items)

    total_interactions = len(all_items)

    min_history_len = min(history_lengths)
    max_history_len = max(history_lengths)
    avg_history_len = sum(history_lengths) / len(history_lengths)

    required_len = state_size + top_k + 1

    eligible_users = [
        user
        for user, history in indexed_history.items()
        if len(history) >= required_len
    ]

    item_counter = Counter(all_items)
    most_common_items = item_counter.most_common(10)

    print("\n===== Data Inspection =====")
    print(f"Number of users: {num_users}")
    print(f"Total interactions: {total_interactions}")
    print(f"Number of unique items: {num_unique_items}")
    print(f"Min item id: {min_item_id}")
    print(f"Max item id: {max_item_id}")
    print(f"Suggested action_dim: {max_item_id + 1}")

    print("\n===== History Length =====")
    print(f"Min history length: {min_history_len}")
    print(f"Max history length: {max_history_len}")
    print(f"Average history length: {avg_history_len:.2f}")
    print(f"Required minimum length: {required_len}")
    print(f"Eligible users for env: {len(eligible_users)}")

    print("\n===== Top 10 Most Frequent Items =====")
    for item_id, count in most_common_items:
        print(f"Item {item_id}: {count}")

    print("\n===== Sanity Warnings =====")

    if min_item_id != 0:
        print(
            "[WARNING] Item ids do not start from 0. "
            "DQN action space currently assumes actions are item ids from 0 to action_dim - 1."
        )

    if max_item_id + 1 != num_unique_items:
        print(
            "[WARNING] Item ids are not continuous. "
            "There may be missing item ids between min and max."
        )

    if len(eligible_users) < num_users * 0.5:
        print(
            "[WARNING] Less than 50% of users have enough history for the environment."
        )

    print("\nDone.")


if __name__ == "__main__":
    args = parse_args()

    indexed_history = load_data(args.data_path)

    inspect_data(
        indexed_history=indexed_history,
        state_size=args.state_size,
        top_k=args.top_k,
    )