"""Summarize processed history pickles in `data/processed`.

Usage:
    python data/review.py            # summarize all .pkl files in data/processed
    python data/review.py train_history.pkl

Outputs a concise summary per file: number of users, total actions,
action stats (min/max/avg), unique items, and top-10 popular items.
"""
from pathlib import Path
import argparse
import pickle
from collections import Counter
import statistics
import json


def load_history(path: Path):
    if not path.exists():
        return {}

    with open(path, "rb") as f:
        data = pickle.load(f)

    # normalize: keys -> str, values -> list[int]
    out = {}
    for user_id, history in data.items():
        try:
            out[str(user_id)] = [int(x) for x in history]
        except Exception:
            # if history isn't iterable of ints, try to coerce
            out[str(user_id)] = list(map(int, list(history)))

    return out


def summarize_history(history: dict, top_k: int = 10):
    num_users = len(history)
    lengths = [len(h) for h in history.values()]
    total_actions = sum(lengths)
    if lengths:
        avg_actions = statistics.mean(lengths)
        median_actions = statistics.median(lengths)
        min_actions = min(lengths)
        max_actions = max(lengths)
    else:
        avg_actions = median_actions = min_actions = max_actions = 0

    pop = Counter()
    unique_items = set()
    for h in history.values():
        for item in h:
            iid = int(item)
            pop[iid] += 1
            unique_items.add(iid)

    top = pop.most_common(top_k)

    return {
        "num_users": int(num_users),
        "total_actions": int(total_actions),
        "avg_actions_per_user": float(avg_actions),
        "median_actions_per_user": float(median_actions),
        "min_actions_per_user": int(min_actions),
        "max_actions_per_user": int(max_actions),
        "num_unique_items": int(len(unique_items)),
        "top_items": [(int(i), int(c)) for i, c in top],
    }


def find_processed_files(processed_dir: Path):
    if not processed_dir.exists():
        return []
    return sorted(p for p in processed_dir.iterdir() if p.suffix == ".pkl")


def main():
    parser = argparse.ArgumentParser(description="Summarize processed history pickles")
    parser.add_argument("paths", nargs="*", help="Files in data/processed to summarize (defaults to all .pkl)")
    parser.add_argument("--top-k", type=int, default=10, help="Top-K popular items to show")
    parser.add_argument("--json", action="store_true", help="Output JSON (one object per file)")
    parser.add_argument(
        "--out-file",
        type=str,
        default="data/processed_summary.json",
        help="Path to write consolidated JSON summary (defaults to data/processed_summary.json)",
    )

    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    processed_dir = repo_root / "data" / "processed"

    if args.paths:
        files = [Path(p) if Path(p).is_absolute() else processed_dir / p for p in args.paths]
    else:
        files = find_processed_files(processed_dir)

    results = {}
    for path in files:
        name = path.name
        history = load_history(path)
        stats = summarize_history(history, top_k=args.top_k)
        results[name] = stats

        if not args.json:
            print("File:", name)
            print("  Users:", stats["num_users"])
            print("  Actions:", stats["total_actions"])
            print("  Actions/user: avg={:.2f} median={:.2f} min={} max={}".format(
                stats["avg_actions_per_user"],
                stats["median_actions_per_user"],
                stats["min_actions_per_user"],
                stats["max_actions_per_user"],
            ))
            print("  Unique items:", stats["num_unique_items"])
            print("  Top-{} items:".format(args.top_k))
            for iid, cnt in stats["top_items"]:
                print("    {:>8}  {:>8}".format(iid, cnt))
            print()

    if args.json:
        print(json.dumps(results, indent=2))
    # Always write consolidated JSON summary to output file (defaults to data/processed_summary.json)
    out_path = Path(args.out_file)
    if not out_path.is_absolute():
        out_path = repo_root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote summary JSON to: {out_path}")


if __name__ == "__main__":
    main()
