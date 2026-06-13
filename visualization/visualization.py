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
        help="CSV with model recommendation metrics (test results).",
    )
    parser.add_argument(
        "--val_csv",
        type=Path,
        default=Path("outputs/logs/validation_model_selection.csv"),
        help="CSV with validation model selection results.",
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
            "Ensure the data is generated correctly before running visualization."
        )


def clean_val_method_name(name):
    name_str = str(name).strip()
    if "pure" in name_str.lower():
        return "DQN pure"
    elif "boost=2" in name_str.lower() or "recency2" in name_str.lower():
        return "DQN + recency=2"
    elif "boost=5" in name_str.lower() or "recency5" in name_str.lower():
        return "DQN + recency=5"
    return name_str


def clean_test_method_name(name):
    name_str = str(name).strip()
    if "recent-item" in name_str.lower():
        return "Recent-item"
    elif "popularity" in name_str.lower():
        return "Popularity"
    elif "random" in name_str.lower():
        return "Random"
    elif "pure" in name_str.lower():
        return "DQN pure"
    elif "boost=2" in name_str.lower() or "recency2" in name_str.lower():
        return "DQN + recency=2"
    elif "boost=5" in name_str.lower() or "recency5" in name_str.lower():
        return "DQN + recency=5"
    return name_str


def train_log_label(path):
    name = path.stem.lower()
    if "pure" in name:
        return "DQN pure"
    elif "recency2" in name or "boost2" in name:
        return "DQN + recency=2"
    elif "recency5" in name or "boost5" in name:
        return "DQN + recency=5"
    return path.stem


def plot_validation_metrics(val_csv, output_dir):
    require_file(val_csv)
    df = pd.read_csv(val_csv)

    required_cols = ["method"] + [col for col, _ in RECOMMENDATION_METRICS]
    require_columns(df, required_cols, str(val_csv))

    plot_df = df.copy()
    if "split" in plot_df.columns:
        plot_df = plot_df[plot_df["split"].astype(str).str.lower() == "validation"]

    plot_df["method_clean"] = plot_df["method"].apply(clean_val_method_name)

    order = ["DQN pure", "DQN + recency=2", "DQN + recency=5"]
    plot_df["method_clean"] = pd.Categorical(plot_df["method_clean"], categories=order, ordered=True)
    plot_df = plot_df.sort_values("method_clean").dropna(subset=["method_clean"])

    labels = plot_df["method_clean"].tolist()

    fig, ax = plt.subplots(figsize=(10, 5.5))

    x_positions = range(len(labels))
    bar_width = 0.18
    offsets = [-1.5, -0.5, 0.5, 1.5]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    for offset, (column, label), color in zip(offsets, RECOMMENDATION_METRICS, colors):
        values = (plot_df[column].astype(float) * 100).tolist()
        bars = ax.bar(
            [x + offset * bar_width for x in x_positions],
            values,
            width=bar_width,
            label=label,
            color=color,
            edgecolor="white",
            linewidth=0.5,
        )

        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.2f}%",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )

    ax.set_title("So sánh kết quả validation của các biến thể DQN", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("Giá trị (%)", fontsize=11)

    max_val = max([plot_df[col].astype(float).max() * 100 for col, _ in RECOMMENDATION_METRICS])
    ax.set_ylim(0, max_val + 3.5)

    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(labels, fontsize=10)

    ax.grid(axis="y", linestyle="-", color="lightgray", alpha=0.3)
    ax.set_axisbelow(True)

    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=True)

    fig.tight_layout()
    output_path = output_dir / "validation_metrics.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return output_path


def plot_dqn_training(train_logs, output_dir):
    existing_logs = [path for path in train_logs if path.exists()]
    if not existing_logs:
        raise FileNotFoundError(
            "No DQN training logs found. Expected at least one CSV with "
            "episode,reward,loss,epsilon columns."
        )

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

    series = [
        ("reward", "Episode Reward - trung bình trượt 100 episode", "Reward"),
        ("loss", "Training Loss - trung bình trượt 100 episode", "Loss"),
        ("epsilon", "Epsilon Decay", "Epsilon"),
    ]

    # Sort logs so that they always appear in a standard order in the legends
    legend_order = ["DQN pure", "DQN + recency=2", "DQN + recency=5"]
    existing_logs = sorted(
        existing_logs,
        key=lambda p: legend_order.index(train_log_label(p))
        if train_log_label(p) in legend_order else 99,
    )

    for log_path in existing_logs:
        df = pd.read_csv(log_path)
        require_columns(df, TRAINING_COLUMNS, str(log_path))
        label = train_log_label(log_path)

        episodes = df["episode"]

        for ax, (column, title, ylabel) in zip(axes, series):
            if column == "epsilon":
                y_values = df[column]
            else:
                y_values = df[column].rolling(window=100, min_periods=1).mean()

            ax.plot(episodes, y_values, linewidth=1.25, label=label)
            ax.set_title(title, fontsize=11, fontweight="bold" if column == "epsilon" else "normal")
            ax.set_ylabel(ylabel, fontsize=10)
            ax.grid(True, linestyle="-", color="lightgray", alpha=0.3)

    axes[-1].set_xlabel("Episode", fontsize=10)

    for ax in axes:
        ax.legend(loc="upper right", fontsize=9)

    fig.suptitle("Đường cong huấn luyện DQN sau khi làm mượt", fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout()
    fig.subplots_adjust(top=0.92)

    output_path = output_dir / "dqn_training.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return output_path


def plot_test_average_reward(metrics_csv, output_dir):
    require_file(metrics_csv)
    df = pd.read_csv(metrics_csv)

    required_cols = ["method", "average_reward"]
    require_columns(df, required_cols, str(metrics_csv))

    plot_df = df.copy()
    if "split" in plot_df.columns:
        plot_df = plot_df[plot_df["split"].astype(str).str.lower() == "test"]

    plot_df["method_clean"] = plot_df["method"].apply(clean_test_method_name)

    if "hit_rate_at_5" in plot_df.columns:
        plot_df = plot_df.sort_values("hit_rate_at_5", ascending=False)
    else:
        custom_order = ["Recent-item", "Popularity", "DQN + recency=2", "DQN + recency=5", "DQN pure", "Random"]
        plot_df["method_clean"] = pd.Categorical(plot_df["method_clean"], categories=custom_order, ordered=True)
        plot_df = plot_df.sort_values("method_clean").dropna(subset=["method_clean"])

    labels = plot_df["method_clean"].tolist()
    rewards = plot_df["average_reward"].astype(float).tolist()

    fig, ax = plt.subplots(figsize=(10, 5))

    x_positions = range(len(labels))

    bars = ax.bar(
        x_positions,
        rewards,
        width=0.6,
        color="tab:blue",
        edgecolor="white",
        linewidth=0.5,
    )

    for bar in bars:
        height = bar.get_height()
        if height >= 0:
            va = "bottom"
            xytext = (0, 3)
        else:
            va = "top"
            xytext = (0, -12)

        ax.annotate(
            f"{height:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=xytext,
            textcoords="offset points",
            ha="center",
            va=va,
            fontsize=9.5,
        )

    ax.set_title("So sánh Average Reward trên tập test", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("Average Reward", fontsize=11)

    min_reward = min(rewards)
    ax.set_ylim(min_reward * 1.15, 0 if min_reward < 0 else min_reward * 1.15)

    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(labels, fontsize=10)

    ax.grid(axis="y", linestyle="-", color="lightgray", alpha=0.3)
    ax.set_axisbelow(True)

    fig.tight_layout()
    output_path = output_dir / "test_average_reward.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return output_path


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        val_plot = plot_validation_metrics(args.val_csv, args.output_dir)
        print("Saved validation metrics comparison plot:", val_plot)
    except Exception as e:
        print(f"Skipping validation metrics plot: {e}")

    try:
        training_plot = plot_dqn_training(args.train_logs, args.output_dir)
        print("Saved DQN training plot:", training_plot)
    except Exception as e:
        print(f"Skipping training curves plot: {e}")

    try:
        test_plot = plot_test_average_reward(args.metrics_csv, args.output_dir)
        print("Saved test average reward plot:", test_plot)
    except Exception as e:
        print(f"Skipping test average reward plot: {e}")


if __name__ == "__main__":
    main()

