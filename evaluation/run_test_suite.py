import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_path",
        type=str,
        default="data/processed/test_history.pkl",
        help="Path to test history pickle file",
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=300,
        help="Number of evaluation episodes",
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Top-K recommendation size",
    )

    parser.add_argument(
        "--action_dim",
        type=int,
        required=True,
        help="Global action dimension used during training",
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

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device: cuda or cpu",
    )

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="outputs/checkpoints",
        help="Directory containing DQN checkpoints",
    )

    parser.add_argument(
        "--pure_model",
        type=str,
        default="dqn_C_pure.pth",
        help="Checkpoint filename for pure DQN",
    )

    parser.add_argument(
        "--boost2_model",
        type=str,
        default="dqn_C_boost2.pth",
        help="Checkpoint filename for DQN with recent_boost=2",
    )

    parser.add_argument(
        "--boost5_model",
        type=str,
        default="dqn_C_boost5.pth",
        help="Checkpoint filename for DQN with recent_boost=5",
    )

    parser.add_argument(
        "--output_csv",
        type=str,
        default="outputs/logs/test_suite_results.csv",
        help="Where to save evaluation CSV",
    )

    parser.add_argument(
        "--output_md",
        type=str,
        default="outputs/logs/test_suite_report.md",
        help="Where to save Markdown report",
    )

    return parser.parse_args()


def run_command(command):
    print("\n" + "=" * 100)
    print("Running:")
    print(" ".join(command))
    print("=" * 100)

    result = subprocess.run(
        command,
        text=True,
        capture_output=True,
    )

    if result.stdout:
        print("\nSTDOUT:")
        print(result.stdout)

    if result.stderr:
        print("\nSTDERR:")
        print(result.stderr)

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with return code {result.returncode}: {' '.join(command)}"
        )

    return result.stdout


def parse_average_reward(output):
    match = re.search(r"Average Reward:\s*([-+]?\d*\.?\d+)", output)

    if match:
        return float(match.group(1))

    return None


def parse_hit_rate(output):
    match = re.search(r"(?:Hit Rate|HitRate)@\d+:\s*([-+]?\d*\.?\d+)", output)

    if match:
        return float(match.group(1))

    return None


def parse_metric_at_k(output, metric_name):
    match = re.search(rf"{metric_name}@\d+:\s*([-+]?\d*\.?\d+)", output)

    if match:
        return float(match.group(1))

    return None


def parse_named_metric(output, name):
    match = re.search(rf"{name}:\s*([-+]?\d*\.?\d+)", output)

    if match:
        return float(match.group(1))

    return None


def evaluate_baselines(args):
    command = [
        sys.executable,
        "-m",
        "evaluation.evaluate_baselines",
        "--data_path",
        args.data_path,
        "--episodes",
        str(args.episodes),
        "--top_k",
        str(args.top_k),
    ]

    output = run_command(command)

    rows = []

    rows.append(
        {
            "method": "Random baseline",
            "average_reward": parse_named_metric(output, "Random Average Reward"),
            "hit_rate_at_5": parse_named_metric(output, r"Random Hit Rate@\d+"),
            "precision_at_5": parse_named_metric(output, r"Random Precision@\d+"),
            "recall_at_5": parse_named_metric(output, r"Random Recall@\d+"),
            "ndcg_at_5": parse_named_metric(output, r"Random NDCG@\d+"),
            "episodes": args.episodes,
            "model_path": "",
            "recent_boost": "",
            "type": "baseline",
        }
    )

    rows.append(
        {
            "method": "Popularity baseline",
            "average_reward": parse_named_metric(output, "Popularity Average Reward"),
            "hit_rate_at_5": parse_named_metric(output, r"Popularity Hit Rate@\d+"),
            "precision_at_5": parse_named_metric(output, r"Popularity Precision@\d+"),
            "recall_at_5": parse_named_metric(output, r"Popularity Recall@\d+"),
            "ndcg_at_5": parse_named_metric(output, r"Popularity NDCG@\d+"),
            "episodes": args.episodes,
            "model_path": "",
            "recent_boost": "",
            "type": "baseline",
        }
    )

    return rows


def evaluate_recent_baseline(args):
    command = [
        sys.executable,
        "-m",
        "evaluation.evaluate_recent_baseline",
        "--data_path",
        args.data_path,
        "--episodes",
        str(args.episodes),
        "--top_k",
        str(args.top_k),
    ]

    output = run_command(command)

    return [
        {
            "method": "Recent-item baseline",
            "average_reward": parse_average_reward(output),
            "hit_rate_at_5": parse_hit_rate(output),
            "precision_at_5": parse_metric_at_k(output, "Precision"),
            "recall_at_5": parse_metric_at_k(output, "Recall"),
            "ndcg_at_5": parse_metric_at_k(output, "NDCG"),
            "episodes": args.episodes,
            "model_path": "",
            "recent_boost": "",
            "type": "baseline",
        }
    ]


def resolve_checkpoint(checkpoint_dir, filename):
    path = Path(filename)

    if path.is_absolute():
        return path

    return Path(checkpoint_dir) / filename


def evaluate_dqn(args, method, checkpoint_path, recent_boost):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    command = [
        sys.executable,
        "-m",
        "evaluation.evaluate_dqn",
        "--data_path",
        args.data_path,
        "--model_path",
        str(checkpoint_path),
        "--episodes",
        str(args.episodes),
        "--top_k",
        str(args.top_k),
        "--action_dim",
        str(args.action_dim),
        "--embedding_dim",
        str(args.embedding_dim),
        "--hidden_dim",
        str(args.hidden_dim),
        "--recent_boost",
        str(recent_boost),
        "--device",
        args.device,
    ]

    output = run_command(command)

    return {
        "method": method,
        "average_reward": parse_average_reward(output),
        "hit_rate_at_5": parse_hit_rate(output),
        "precision_at_5": parse_metric_at_k(output, "Precision"),
        "recall_at_5": parse_metric_at_k(output, "Recall"),
        "ndcg_at_5": parse_metric_at_k(output, "NDCG"),
        "episodes": args.episodes,
        "model_path": str(checkpoint_path),
        "recent_boost": recent_boost,
        "type": "dqn",
    }


def save_csv(rows, output_csv):
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "method",
        "average_reward",
        "hit_rate_at_5",
        "precision_at_5",
        "recall_at_5",
        "ndcg_at_5",
        "episodes",
        "model_path",
        "recent_boost",
        "type",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved CSV: {output_path}")


def save_markdown(rows, output_md):
    output_path = Path(output_md)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sorted_rows = sorted(
        rows,
        key=lambda row: row["hit_rate_at_5"] if row["hit_rate_at_5"] is not None else -1,
        reverse=True,
    )

    lines = []

    lines.append("# Test Suite Report\n")
    lines.append("## Results\n")
    lines.append(
        "| Method | Avg Reward | HitRate@5 | Precision@5 | Recall@5 | NDCG@5 | Type | Recent Boost |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---|---:|")

    for row in sorted_rows:
        avg_reward = row["average_reward"]
        hit_rate = row["hit_rate_at_5"]
        precision = row["precision_at_5"]
        recall = row["recall_at_5"]
        ndcg = row["ndcg_at_5"]

        avg_reward_text = f"{avg_reward:.3f}" if avg_reward is not None else "N/A"
        hit_rate_text = f"{hit_rate:.4f}" if hit_rate is not None else "N/A"
        precision_text = f"{precision:.4f}" if precision is not None else "N/A"
        recall_text = f"{recall:.4f}" if recall is not None else "N/A"
        ndcg_text = f"{ndcg:.4f}" if ndcg is not None else "N/A"

        lines.append(
            f"| {row['method']} | {avg_reward_text} | {hit_rate_text} | "
            f"{precision_text} | {recall_text} | {ndcg_text} | "
            f"{row['type']} | {row['recent_boost']} |"
        )

    best = sorted_rows[0] if sorted_rows else None

    lines.append("\n## Interpretation\n")

    if best:
        lines.append(
            f"- Best method by HitRate@5: **{best['method']}** "
            f"with HitRate@5 = **{best['hit_rate_at_5']:.4f}**."
        )

    recent = next(
        (row for row in rows if row["method"] == "Recent-item baseline"),
        None,
    )

    pure_dqn = next(
        (row for row in rows if row["method"] == "DQN C pure"),
        None,
    )

    boost5 = next(
        (row for row in rows if row["method"] == "DQN C + recent_boost=5"),
        None,
    )

    if pure_dqn and recent:
        if pure_dqn["hit_rate_at_5"] >= recent["hit_rate_at_5"]:
            lines.append("- Pure DQN outperforms the Recent-item baseline.")
        else:
            lines.append(
                "- Pure DQN does **not** outperform the Recent-item baseline. "
                "Do not claim pure DQN is the best model."
            )

    if boost5 and recent:
        if boost5["hit_rate_at_5"] >= recent["hit_rate_at_5"]:
            lines.append(
                "- DQN + recent_boost=5 outperforms the Recent-item baseline. "
                "Report this as **Hybrid DQN + recency prior**, not pure DQN."
            )
        else:
            lines.append(
                "- DQN + recent_boost=5 improves over pure DQN but still does not "
                "outperform the Recent-item baseline."
            )

    lines.append(
        "\nConclusion: if a model uses `recent_boost`, it must be reported as "
        "**DQN + recency prior** or **Hybrid DQN**, not pure DQN."
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Saved Markdown report: {output_path}")


def main():
    args = parse_args()

    rows = []

    rows.extend(evaluate_baselines(args))
    rows.extend(evaluate_recent_baseline(args))

    pure_checkpoint = resolve_checkpoint(
        args.checkpoint_dir,
        args.pure_model,
    )

    boost2_checkpoint = resolve_checkpoint(
        args.checkpoint_dir,
        args.boost2_model,
    )

    boost5_checkpoint = resolve_checkpoint(
        args.checkpoint_dir,
        args.boost5_model,
    )

    rows.append(
        evaluate_dqn(
            args=args,
            method="DQN C pure",
            checkpoint_path=pure_checkpoint,
            recent_boost=0.0,
        )
    )

    rows.append(
        evaluate_dqn(
            args=args,
            method="DQN C + recent_boost=2",
            checkpoint_path=boost2_checkpoint,
            recent_boost=2.0,
        )
    )

    rows.append(
        evaluate_dqn(
            args=args,
            method="DQN C + recent_boost=5",
            checkpoint_path=boost5_checkpoint,
            recent_boost=5.0,
        )
    )

    print("\n===== Final Ranking by HitRate@5 =====")

    sorted_rows = sorted(
        rows,
        key=lambda row: row["hit_rate_at_5"] if row["hit_rate_at_5"] is not None else -1,
        reverse=True,
    )

    for idx, row in enumerate(sorted_rows, start=1):
        print(
            f"{idx}. {row['method']}"
            f" | Avg Reward: {row['average_reward']:.3f}"
            f" | HitRate@5: {row['hit_rate_at_5']:.4f}"
            f" | Precision@5: {row['precision_at_5']:.4f}"
            f" | Recall@5: {row['recall_at_5']:.4f}"
            f" | NDCG@5: {row['ndcg_at_5']:.4f}"
        )

    save_csv(
        rows,
        args.output_csv,
    )

    save_markdown(
        rows,
        args.output_md,
    )


if __name__ == "__main__":
    main()
