import argparse
import os
import pickle


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_path",
        type=str,
        default="data/processed/indexed_history.pkl",
        help="Path to indexed history pickle file",
    )

    parser.add_argument(
        "--train_output_path",
        type=str,
        default="data/processed/train_history.pkl",
        help="Path to save train history pickle file",
    )

    parser.add_argument(
        "--test_output_path",
        type=str,
        default="data/processed/test_history.pkl",
        help="Path to save test history pickle file",
    )

    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.2,
        help="Fraction of each user's history to reserve for testing",
    )

    parser.add_argument(
        "--min_history_len",
        type=int,
        default=11,
        help="Minimum history length required for each split portion",
    )

    return parser.parse_args()


def validate_args(args):
    if not 0 < args.test_ratio < 1:
        raise ValueError("--test_ratio must be between 0 and 1")

    if args.min_history_len <= 0:
        raise ValueError("--min_history_len must be positive")


def load_history(input_path):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with open(input_path, "rb") as f:
        return pickle.load(f)


def make_parent_dir(file_path):
    parent_dir = os.path.dirname(file_path)

    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)


def save_history(history, output_path):
    make_parent_dir(output_path)

    with open(output_path, "wb") as f:
        pickle.dump(history, f)


def split_history(indexed_history, test_ratio, min_history_len):
    train_history = {}
    test_history = {}
    skipped_users = 0

    for user_id, history in indexed_history.items():
        if len(history) < min_history_len:
            skipped_users += 1
            continue

        split_idx = int(len(history) * (1 - test_ratio))
        train_items = history[:split_idx]
        test_items = history[split_idx:]

        if (
            len(train_items) < min_history_len
            or len(test_items) < min_history_len
        ):
            skipped_users += 1
            continue

        train_history[user_id] = train_items
        test_history[user_id] = test_items

    return train_history, test_history, skipped_users


def count_interactions(history_by_user):
    return sum(len(history) for history in history_by_user.values())


def main():
    args = parse_args()
    validate_args(args)

    indexed_history = load_history(args.input_path)

    train_history, test_history, skipped_users = split_history(
        indexed_history=indexed_history,
        test_ratio=args.test_ratio,
        min_history_len=args.min_history_len,
    )

    save_history(train_history, args.train_output_path)
    save_history(test_history, args.test_output_path)

    print(f"Input users: {len(indexed_history)}")
    print(f"Kept users: {len(train_history)}")
    print(f"Skipped users: {skipped_users}")
    print(f"Train interactions: {count_interactions(train_history)}")
    print(f"Test interactions: {count_interactions(test_history)}")
    print(f"Train output path: {args.train_output_path}")
    print(f"Test output path: {args.test_output_path}")


if __name__ == "__main__":
    main()
