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
        "--val_output_path",
        type=str,
        default="data/processed/val_history.pkl",
        help="Path to save validation history pickle file",
    )

    parser.add_argument(
        "--test_output_path",
        type=str,
        default="data/processed/test_history.pkl",
        help="Path to save test history pickle file",
    )

    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.7,
        help="Train ratio per user history",
    )

    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.15,
        help="Validation ratio per user history",
    )

    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.15,
        help="Test ratio per user history",
    )

    parser.add_argument(
        "--min_history_len",
        type=int,
        default=11,
        help="Minimum required length for each split portion",
    )

    return parser.parse_args()


def ensure_parent_dir(file_path):
    parent_dir = os.path.dirname(file_path)

    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)


def load_indexed_history(input_path):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with open(input_path, "rb") as f:
        return pickle.load(f)


def save_pickle(data, output_path):
    ensure_parent_dir(output_path)

    with open(output_path, "wb") as f:
        pickle.dump(data, f)


def validate_ratios(train_ratio, val_ratio, test_ratio):
    total = train_ratio + val_ratio + test_ratio

    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"Ratios must sum to 1.0, got {total}. "
            f"train={train_ratio}, val={val_ratio}, test={test_ratio}"
        )


def split_user_history(
    history,
    train_ratio,
    val_ratio,
    test_ratio,
):
    n = len(history)

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_part = history[:train_end]
    val_part = history[train_end:val_end]
    test_part = history[val_end:]

    return train_part, val_part, test_part


def split_history(args):
    validate_ratios(
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
    )

    indexed_history = load_indexed_history(args.input_path)

    train_history = {}
    val_history = {}
    test_history = {}

    skipped_users = 0

    for user_id, history in indexed_history.items():
        if len(history) < args.min_history_len * 3:
            skipped_users += 1
            continue

        train_part, val_part, test_part = split_user_history(
            history=history,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
        )

        if (
            len(train_part) < args.min_history_len
            or len(val_part) < args.min_history_len
            or len(test_part) < args.min_history_len
        ):
            skipped_users += 1
            continue

        train_history[user_id] = train_part
        val_history[user_id] = val_part
        test_history[user_id] = test_part

    save_pickle(train_history, args.train_output_path)
    save_pickle(val_history, args.val_output_path)
    save_pickle(test_history, args.test_output_path)

    train_interactions = sum(len(history) for history in train_history.values())
    val_interactions = sum(len(history) for history in val_history.values())
    test_interactions = sum(len(history) for history in test_history.values())

    print("\n===== Train / Val / Test Split =====")
    print(f"Input users: {len(indexed_history)}")
    print(f"Kept users: {len(train_history)}")
    print(f"Skipped users: {skipped_users}")

    print(f"\nTrain interactions: {train_interactions}")
    print(f"Val interactions: {val_interactions}")
    print(f"Test interactions: {test_interactions}")

    print(f"\nTrain output path: {args.train_output_path}")
    print(f"Val output path: {args.val_output_path}")
    print(f"Test output path: {args.test_output_path}")


if __name__ == "__main__":
    args = parse_args()
    split_history(args)
