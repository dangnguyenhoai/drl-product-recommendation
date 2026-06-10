import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--val_data_path",
        type=str,
        default="data/processed/val_history.pkl",
        help="Path to validation history pickle file",
    )

    parser.add_argument(
        "--test_data_path",
        type=str,
        default="data/processed/test_history.pkl",
        help="Path to test history pickle file",
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=1000,
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
        default="dqn_pure_stable_best.pth",
        help="Checkpoint filename for pure DQN",
    )

    parser.add_argument(
        "--boost2_model",
        type=str,
        default="dqn_recency2_stable_best.pth",
        help="Checkpoint filename for DQN with recent_boost=2",
    )

    parser.add_argument(
        "--boost5_model",
        type=str,
        default="dqn_recency5_stable_best.pth",
        help="Checkpoint filename for DQN with recent_boost=5",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/logs",
        help="Directory to save validation/test reports",
    )

    parser.add_argument(
        "--selected_model_path",
        type=str,
        default=None,
        help=(
            "Where to copy the model selected by validation. "
            "If omitted, '<checkpoint_dir>/best_selected_by_validation.pth' is used."
        ),
    )

    parser.add_argument(
        "--selected_metadata_path",
        type=str,
        default=None,
        help=(
            "Where to save JSON metadata for the selected model. "
            "If omitted, '<output_dir>/best_selected_by_validation.json' is used."
        ),
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


def resolve_checkpoint(checkpoint_dir, filename):
    path = Path(filename)

    if path.is_absolute():
        return path

    return Path(checkpoint_dir) / filename


def save_selected_model(best_row, selected_model_path, selected_metadata_path):
    source_model_path = Path(best_row["model_path"])

    if not source_model_path.exists():
        raise FileNotFoundError(
            f"Selected model checkpoint not found: {source_model_path}"
        )

    selected_model_path.parent.mkdir(parents=True, exist_ok=True)
    selected_metadata_path.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy2(source_model_path, selected_model_path)

    metadata = {
        "selection_metric": "validation_hit_rate_at_5",
        "selected_method": best_row["method"],
        "source_model_path": str(source_model_path),
        "selected_model_path": str(selected_model_path),
        "split": best_row["split"],
        "average_reward": best_row["average_reward"],
        "hit_rate_at_5": best_row["hit_rate_at_5"],
        "precision_at_5": best_row["precision_at_5"],
        "recall_at_5": best_row["recall_at_5"],
        "ndcg_at_5": best_row["ndcg_at_5"],
        "episodes": best_row["episodes"],
        "recent_boost": best_row["recent_boost"],
        "type": best_row["type"],
    }

    with open(selected_metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved selected validation model: {selected_model_path}")
    print(f"Saved selected validation metadata: {selected_metadata_path}")


def evaluate_dqn(
    data_path,
    action_dim,
    embedding_dim,
    hidden_dim,
    device,
    episodes,
    top_k,
    checkpoint_path,
    recent_boost,
    method,
    split,
):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    command = [
        sys.executable,
        "-m",
        "evaluation.evaluate_dqn",
        "--data_path",
        data_path,
        "--model_path",
        str(checkpoint_path),
        "--episodes",
        str(episodes),
        "--top_k",
        str(top_k),
        "--action_dim",
        str(action_dim),
        "--embedding_dim",
        str(embedding_dim),
        "--hidden_dim",
        str(hidden_dim),
        "--recent_boost",
        str(recent_boost),
        "--device",
        device,
    ]

    output = run_command(command)

    return {
        "split": split,
        "method": method,
        "average_reward": parse_average_reward(output),
        "hit_rate_at_5": parse_hit_rate(output),
        "precision_at_5": parse_metric_at_k(output, "Precision"),
        "recall_at_5": parse_metric_at_k(output, "Recall"),
        "ndcg_at_5": parse_metric_at_k(output, "NDCG"),
        "episodes": episodes,
        "model_path": str(checkpoint_path),
        "recent_boost": recent_boost,
        "type": "dqn",
    }


def evaluate_baselines(data_path, episodes, top_k, split):
    command = [
        sys.executable,
        "-m",
        "evaluation.evaluate_baselines",
        "--data_path",
        data_path,
        "--episodes",
        str(episodes),
        "--top_k",
        str(top_k),
    ]

    output = run_command(command)

    rows = []

    rows.append(
        {
            "split": split,
            "method": "Random baseline",
            "average_reward": parse_named_metric(output, "Random Average Reward"),
            "hit_rate_at_5": parse_named_metric(output, r"Random Hit Rate@\d+"),
            "precision_at_5": parse_named_metric(output, r"Random Precision@\d+"),
            "recall_at_5": parse_named_metric(output, r"Random Recall@\d+"),
            "ndcg_at_5": parse_named_metric(output, r"Random NDCG@\d+"),
            "episodes": episodes,
            "model_path": "",
            "recent_boost": "",
            "type": "baseline",
        }
    )

    rows.append(
        {
            "split": split,
            "method": "Popularity baseline",
            "average_reward": parse_named_metric(output, "Popularity Average Reward"),
            "hit_rate_at_5": parse_named_metric(output, r"Popularity Hit Rate@\d+"),
            "precision_at_5": parse_named_metric(output, r"Popularity Precision@\d+"),
            "recall_at_5": parse_named_metric(output, r"Popularity Recall@\d+"),
            "ndcg_at_5": parse_named_metric(output, r"Popularity NDCG@\d+"),
            "episodes": episodes,
            "model_path": "",
            "recent_boost": "",
            "type": "baseline",
        }
    )

    return rows


def evaluate_recent_baseline(data_path, episodes, top_k, split):
    command = [
        sys.executable,
        "-m",
        "evaluation.evaluate_recent_baseline",
        "--data_path",
        data_path,
        "--episodes",
        str(episodes),
        "--top_k",
        str(top_k),
    ]

    output = run_command(command)

    return [
        {
            "split": split,
            "method": "Recent-item baseline",
            "average_reward": parse_average_reward(output),
            "hit_rate_at_5": parse_hit_rate(output),
            "precision_at_5": parse_metric_at_k(output, "Precision"),
            "recall_at_5": parse_metric_at_k(output, "Recall"),
            "ndcg_at_5": parse_metric_at_k(output, "NDCG"),
            "episodes": episodes,
            "model_path": "",
            "recent_boost": "",
            "type": "baseline",
        }
    ]


def save_csv(rows, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "split",
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

    print(f"Saved CSV: {output_path}")


def save_markdown(val_rows, test_rows, best_row, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []

    lines.append("# Validation/Test Evaluation Report\n")

    lines.append("## Validation Results\n")
    lines.append(
        "| Method | Avg Reward | HitRate@5 | Precision@5 | Recall@5 | NDCG@5 | Recent Boost |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")

    sorted_val_rows = sorted(
        val_rows,
        key=lambda row: row["hit_rate_at_5"] if row["hit_rate_at_5"] is not None else -1,
        reverse=True,
    )

    for row in sorted_val_rows:
        lines.append(
            f"| {row['method']} | {row['average_reward']:.3f} | "
            f"{row['hit_rate_at_5']:.4f} | {row['precision_at_5']:.4f} | "
            f"{row['recall_at_5']:.4f} | {row['ndcg_at_5']:.4f} | "
            f"{row['recent_boost']} |"
        )

    lines.append("\n## Selected Model\n")
    lines.append(
        f"- Selected by validation HitRate@5: **{best_row['method']}**"
    )
    lines.append(f"- Model path: `{best_row['model_path']}`")
    lines.append(f"- Recent boost: `{best_row['recent_boost']}`")

    lines.append("\n## Final Test Results\n")
    lines.append(
        "| Method | Avg Reward | HitRate@5 | Precision@5 | Recall@5 | NDCG@5 | Type | Recent Boost |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---|---:|")

    sorted_test_rows = sorted(
        test_rows,
        key=lambda row: row["hit_rate_at_5"] if row["hit_rate_at_5"] is not None else -1,
        reverse=True,
    )

    for row in sorted_test_rows:
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

    lines.append("\n## Interpretation\n")
    lines.append(
        "- Models are selected using validation HitRate@5, not test HitRate@5."
    )
    lines.append(
        "- The test split is used only for final reporting after model selection."
    )
    lines.append(
        "- If the selected model uses recent_boost, report it as "
        "**DQN + recency prior**, not pure DQN."
    )
    lines.append(
        "- If Recent-item baseline is still best on test, the current DQN setup "
        "does not outperform the strongest heuristic baseline."
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Saved Markdown report: {output_path}")


def main():
    args = parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir)

    candidates = [
        {
            "method": "DQN pure stable",
            "checkpoint_path": resolve_checkpoint(checkpoint_dir, args.pure_model),
            "recent_boost": 0.0,
        },
        {
            "method": "DQN + recency prior boost=2",
            "checkpoint_path": resolve_checkpoint(checkpoint_dir, args.boost2_model),
            "recent_boost": 2.0,
        },
        {
            "method": "DQN + recency prior boost=5",
            "checkpoint_path": resolve_checkpoint(checkpoint_dir, args.boost5_model),
            "recent_boost": 5.0,
        },
    ]

    val_rows = []

    for candidate in candidates:
        row = evaluate_dqn(
            data_path=args.val_data_path,
            action_dim=args.action_dim,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            device=args.device,
            episodes=args.episodes,
            top_k=args.top_k,
            checkpoint_path=candidate["checkpoint_path"],
            recent_boost=candidate["recent_boost"],
            method=candidate["method"],
            split="validation",
        )

        val_rows.append(row)

    best_row = max(
        val_rows,
        key=lambda row: row["hit_rate_at_5"] if row["hit_rate_at_5"] is not None else -1,
    )

    print("\n===== Best Model Selected By Validation =====")
    print(f"Method: {best_row['method']}")
    print(f"Model path: {best_row['model_path']}")
    print(f"Recent boost: {best_row['recent_boost']}")
    print(f"Val HitRate@5: {best_row['hit_rate_at_5']:.4f}")

    selected_model_path = (
        Path(args.selected_model_path)
        if args.selected_model_path is not None
        else checkpoint_dir / "best_selected_by_validation.pth"
    )
    selected_metadata_path = (
        Path(args.selected_metadata_path)
        if args.selected_metadata_path is not None
        else output_dir / "best_selected_by_validation.json"
    )
    save_selected_model(
        best_row=best_row,
        selected_model_path=selected_model_path,
        selected_metadata_path=selected_metadata_path,
    )

    test_rows = []

    test_rows.extend(
        evaluate_baselines(
            data_path=args.test_data_path,
            episodes=args.episodes,
            top_k=args.top_k,
            split="test",
        )
    )

    test_rows.extend(
        evaluate_recent_baseline(
            data_path=args.test_data_path,
            episodes=args.episodes,
            top_k=args.top_k,
            split="test",
        )
    )

    best_test_row = evaluate_dqn(
        data_path=args.test_data_path,
        action_dim=args.action_dim,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        device=args.device,
        episodes=args.episodes,
        top_k=args.top_k,
        checkpoint_path=Path(best_row["model_path"]),
        recent_boost=best_row["recent_boost"],
        method=f"{best_row['method']} (selected by validation)",
        split="test",
    )

    test_rows.append(best_test_row)

    save_csv(
        val_rows,
        output_dir / "validation_model_selection.csv",
    )

    save_csv(
        test_rows,
        output_dir / "final_test_results.csv",
    )

    save_markdown(
        val_rows=val_rows,
        test_rows=test_rows,
        best_row=best_row,
        output_path=output_dir / "train_val_test_report.md",
    )

    print("\n===== Final Test Ranking By HitRate@5 =====")

    sorted_test_rows = sorted(
        test_rows,
        key=lambda row: row["hit_rate_at_5"] if row["hit_rate_at_5"] is not None else -1,
        reverse=True,
    )

    for idx, row in enumerate(sorted_test_rows, start=1):
        print(
            f"{idx}. {row['method']}"
            f" | Avg Reward: {row['average_reward']:.3f}"
            f" | HitRate@5: {row['hit_rate_at_5']:.4f}"
            f" | Precision@5: {row['precision_at_5']:.4f}"
            f" | Recall@5: {row['recall_at_5']:.4f}"
            f" | NDCG@5: {row['ndcg_at_5']:.4f}"
        )


if __name__ == "__main__":
    main()
