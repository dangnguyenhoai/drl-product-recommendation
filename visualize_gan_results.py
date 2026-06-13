from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize GAN recommender results.")
    parser.add_argument("--train_log", default="outputs/logs/gan_train_log.csv")
    parser.add_argument("--test_results", default="outputs/logs/gan_test_results.csv")
    parser.add_argument("--comparison_csv", default="outputs/logs/gan_vs_friend_dqn_comparison.csv")
    parser.add_argument("--output_dir", default="outputs/plots")
    parser.add_argument("--top_k", type=int, default=5)
    return parser.parse_args()


def save_training_loss(train_df: pd.DataFrame, output_dir: Path) -> Path:
    path = output_dir / "gan_training_loss.png"
    plt.figure(figsize=(10, 5))
    plt.plot(train_df["epoch"], train_df["generator_loss"], label="Generator loss", linewidth=2)
    plt.plot(train_df["epoch"], train_df["discriminator_loss"], label="Discriminator loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("GAN Training Loss")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def save_validation_metrics(train_df: pd.DataFrame, output_dir: Path, top_k: int) -> Path:
    path = output_dir / "gan_validation_metrics.png"
    metric_cols = [
        f"val_hit_rate_at_{top_k}",
        f"val_precision_at_{top_k}",
        f"val_recall_at_{top_k}",
        f"val_ndcg_at_{top_k}",
        f"val_mrr_at_{top_k}",
    ]
    available = [col for col in metric_cols if col in train_df.columns]

    plt.figure(figsize=(10, 5))
    for col in available:
        label = col.replace(f"val_", "").replace("_", " ").title()
        plt.plot(train_df["epoch"], train_df[col], label=label, linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("GAN Validation Metrics")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def save_test_summary(test_df: pd.DataFrame, output_dir: Path, top_k: int) -> Path:
    path = output_dir / "gan_test_metrics.png"
    row = test_df.iloc[0]
    metric_cols = [
        f"hit_rate_at_{top_k}",
        f"precision_at_{top_k}",
        f"recall_at_{top_k}",
        f"ndcg_at_{top_k}",
        f"mrr_at_{top_k}",
    ]
    available = [col for col in metric_cols if col in test_df.columns]
    labels = [col.replace("_", " ").title() for col in available]
    values = [float(row[col]) for col in available]

    plt.figure(figsize=(9, 5))
    bars = plt.bar(labels, values, color=["#2563eb", "#16a34a", "#7c3aed", "#ea580c", "#0891b2"])
    plt.ylabel("Score")
    plt.title("GAN Final Test Metrics")
    plt.ylim(0, max(values) * 1.25 if values else 1)
    plt.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=20, ha="right")
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.4f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def save_comparison(comparison_df: pd.DataFrame, output_dir: Path, top_k: int) -> Path:
    path = output_dir / "gan_vs_dqn_metrics.png"
    metric_cols = [
        f"hit_rate_at_{top_k}",
        f"precision_at_{top_k}",
        f"recall_at_{top_k}",
        f"ndcg_at_{top_k}",
    ]
    available = [col for col in metric_cols if col in comparison_df.columns]
    name_col = "method" if "method" in comparison_df.columns else "model"
    plot_df = comparison_df[[name_col] + available].copy()
    plot_df = plot_df.set_index(name_col)

    ax = plot_df.plot(kind="bar", figsize=(12, 6), width=0.78)
    ax.set_title("GAN vs DQN/Baseline Test Metrics")
    ax.set_ylabel("Score")
    ax.set_xlabel("")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(0.05, float(plot_df.max().max()) * 1.25))
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(args.train_log)
    test_df = pd.read_csv(args.test_results)

    generated = [
        save_training_loss(train_df, output_dir),
        save_validation_metrics(train_df, output_dir, args.top_k),
        save_test_summary(test_df, output_dir, args.top_k),
    ]

    comparison_path = Path(args.comparison_csv)
    if comparison_path.exists():
        comparison_df = pd.read_csv(comparison_path)
        generated.append(save_comparison(comparison_df, output_dir, args.top_k))
    else:
        print(f"Skip comparison plot, file not found: {comparison_path}")

    print("Generated plots:")
    for path in generated:
        print("-", path)


if __name__ == "__main__":
    main()
