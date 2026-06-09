import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RECOMMENDATION_METRICS = [
    ("hit_rate_at_5", "HitRate@5"),
    ("precision_at_5", "Precision@5"),
    ("recall_at_5", "Recall@5"),
    ("ndcg_at_5", "NDCG@5"),
]

TRAINING_COLUMNS = ["episode", "reward", "loss", "epsilon"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot recommendation quality and DQN training curves.",
    )
    parser.add_argument(
        "--metrics_csv",
        type=Path,
        default=Path("outputs/logs/final_test_results.csv"),
        help="CSV with model recommendation metrics.",
    )
    parser.add_argument(
        "--train_logs",
        type=Path,
        nargs="+",
        default=[
            Path("outputs/logs/train_dqn_pure_stable.csv"),
            Path("outputs/logs/train_dqn_recency2_stable.csv"),
            Path("outputs/logs/train_dqn_recency5_stable.csv"),
        ],
        help="One or more DQN training log CSV files.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/plots"),
        help="Directory for generated visualization PNG files.",
    )
    return parser.parse_args()


def require_file(path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")


def require_columns(df, columns, source):
    missing = [column for column in columns if column not in df.columns]

    if missing:
        raise ValueError(
            f"{source} is missing columns: {', '.join(missing)}.\n"
            "Run the updated evaluation suite again to regenerate this CSV "
            "with HitRate@5, Precision@5, Recall@5, and NDCG@5."
        )


def clean_method_name(name):
    return (
        str(name)
        .replace(" baseline", "")
        .replace("DQN + recency prior ", "DQN ")
        .replace(" (selected by validation)", "")
    )


def plot_recommendation_metrics(metrics_csv, output_dir):
    require_file(metrics_csv)

    df = pd.read_csv(metrics_csv)
    required_columns = ["method"] + [column for column, _ in RECOMMENDATION_METRICS]
    require_columns(df, required_columns, str(metrics_csv))

    plot_df = df.copy()
    if "split" in plot_df.columns:
        test_rows = plot_df[plot_df["split"].astype(str).str.lower() == "test"]
        if not test_rows.empty:
            plot_df = test_rows

    plot_df = plot_df.sort_values("hit_rate_at_5", ascending=False)
    labels = [clean_method_name(name) for name in plot_df["method"]]

    fig_width = max(10, len(labels) * 1.6)
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    x_positions = range(len(labels))
    bar_width = 0.18
    offsets = [-1.5, -0.5, 0.5, 1.5]

    for offset, (column, label) in zip(offsets, RECOMMENDATION_METRICS):
        values = plot_df[column].astype(float).tolist()
        ax.bar(
            [x + offset * bar_width for x in x_positions],
            values,
            width=bar_width,
            label=label,
        )

    ax.set_title("Recommendation Quality Metrics")
    ax.set_ylabel("Score")
    ax.set_ylim(0, max(0.35, plot_df[[c for c, _ in RECOMMENDATION_METRICS]].max().max() * 1.2))
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(ncol=4, loc="upper right")

    fig.tight_layout()
    output_path = output_dir / "recommendation_metrics.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    return output_path


def train_log_label(path):
    name = path.stem
    return (
        name.replace("train_", "")
        .replace("dqn_", "DQN ")
        .replace("_stable", "")
        .replace("_", " ")
    )


def plot_dqn_training(train_logs, output_dir):
    existing_logs = [path for path in train_logs if path.exists()]
    if not existing_logs:
        raise FileNotFoundError(
            "No DQN training logs found. Expected at least one CSV with "
            "episode,reward,loss,epsilon columns."
        )

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    series = [
        ("reward", "Episode Reward"),
        ("loss", "Training Loss"),
        ("epsilon", "Epsilon Decay"),
    ]

    for log_path in existing_logs:
        df = pd.read_csv(log_path)
        require_columns(df, TRAINING_COLUMNS, str(log_path))
        label = train_log_label(log_path)

        for ax, (column, title) in zip(axes, series):
            ax.plot(df["episode"], df[column], linewidth=1.25, label=label)
            ax.set_title(title)
            ax.set_ylabel(column.capitalize())
            ax.grid(axis="y", linestyle="--", alpha=0.35)

    axes[-1].set_xlabel("Episode")
    for ax in axes:
        ax.legend(loc="best")

    fig.suptitle("DQN Training Curves", y=0.995)
    fig.tight_layout()
    output_path = output_dir / "dqn_training.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    return output_path


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    recommendation_plot = plot_recommendation_metrics(
        args.metrics_csv,
        args.output_dir,
    )
    training_plot = plot_dqn_training(
        args.train_logs,
        args.output_dir,
    )

    print("Saved recommendation metrics plot:", recommendation_plot)
    print("Saved DQN training plot:", training_plot)


if __name__ == "__main__":
    main()
