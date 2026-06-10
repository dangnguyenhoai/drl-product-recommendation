from __future__ import annotations

import argparse
import csv
from itertools import islice
from pathlib import Path

import torch

from evaluation.evaluate_models_common import (
    empty_totals,
    finalize_metrics,
    get_valid_actions,
    iter_batches,
    iter_test_windows,
    load_histories,
    load_mlp,
    print_results,
    resolve_device,
    select_top_k,
    update_totals,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate an MLP baseline on the deterministic test windows and metric "
            "definitions used by evaluation.evaluate_models_common."
        )
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/processed/test_history.pkl"),
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--state-size", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hit-reward", type=float, default=5.0)
    parser.add_argument("--miss-penalty", type=float, default=-2.0)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--max-windows",
        type=int,
        default=None,
        help="Optional deterministic window cap for a quick smoke test.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/logs/mlp_common_test_results.csv"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("outputs/logs/mlp_common_test_report.md"),
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for path in (args.data_path, args.model_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path}")

    for name in ("state_size", "top_k", "batch_size"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if args.max_windows is not None and args.max_windows <= 0:
        raise ValueError("--max-windows must be positive when provided.")


def validate_shared_space(
    model,
    valid_actions: list[int],
    state_size: int,
    top_k: int,
) -> None:
    if model.state_size != state_size:
        raise ValueError(
            f"MLP checkpoint uses state_size={model.state_size}, "
            f"but --state-size={state_size}."
        )
    if max(valid_actions) >= model.action_dim:
        raise ValueError(
            f"Test item id {max(valid_actions)} is outside action_dim={model.action_dim}."
        )
    if len(valid_actions) < top_k:
        raise ValueError(
            f"Test data has only {len(valid_actions)} valid actions, but top_k={top_k}."
        )


@torch.no_grad()
def evaluate_mlp(
    model,
    histories,
    valid_actions: list[int],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float | str]:
    valid_mask = torch.zeros(model.action_dim, dtype=torch.bool, device=device)
    valid_mask[torch.as_tensor(valid_actions, device=device)] = True

    windows = iter_test_windows(histories, args.state_size, args.top_k)
    if args.max_windows is not None:
        windows = islice(windows, args.max_windows)

    totals = empty_totals()
    for states, targets in iter_batches(windows, args.batch_size):
        states = states.to(device)
        targets = targets.to(device)
        recommendations = select_top_k(
            model(states),
            states,
            valid_mask,
            args.top_k,
        )
        update_totals(
            totals,
            recommendations,
            targets,
            args.hit_reward,
            args.miss_penalty,
        )

    row = finalize_metrics("MLP baseline", totals, args.top_k)
    row.update(
        {
            "model_path": str(args.model_path),
            "recent_boost": 0.0,
        }
    )
    return row


def save_csv(row: dict[str, float | str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def save_markdown(
    row: dict[str, float | str],
    path: Path,
    data_path: Path,
    top_k: int,
    max_windows: int | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metric_names = [
        "average_window_reward",
        f"next_item_hitrate@{top_k}",
        f"next_item_ndcg@{top_k}",
        f"window_hitrate@{top_k}",
        f"window_precision@{top_k}",
        f"window_recall@{top_k}",
        f"window_ndcg@{top_k}",
    ]
    cap_text = "all available windows" if max_windows is None else f"first {max_windows} windows"
    lines = [
        "# MLP Common Test Evaluation",
        "",
        f"- Test data: `{data_path}`",
        f"- Evaluated windows: {row['cases']} ({cap_text})",
        f"- Top-K: {top_k}",
        "- Uses the same deterministic windows, valid-action mask, reward, and metrics as "
        "`evaluation.evaluate_models_common`.",
        "",
        "| Model | " + " | ".join(metric_names) + " |",
        "|---|" + "|".join("---:" for _ in metric_names) + "|",
        "| "
        + str(row["model"])
        + " | "
        + " | ".join(f"{float(row[name]):.6f}" for name in metric_names)
        + " |",
        "",
        "## Metric Definitions",
        "",
        f"- `next_item_hitrate@{top_k}`: Top-{top_k} contains the immediate next item.",
        f"- `window_hitrate@{top_k}`: Top-{top_k} contains at least one of the next "
        f"{top_k} items.",
        f"- `window_precision@{top_k}`, `window_recall@{top_k}`, and "
        f"`window_ndcg@{top_k}` use the next {top_k} items as the relevant set.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    validate_args(args)
    device = resolve_device(args.device)

    print(f"Using device: {device}")
    print(f"Loading test data: {args.data_path}")
    histories = load_histories(args.data_path)
    valid_actions = get_valid_actions(histories)

    print(f"Loading MLP checkpoint: {args.model_path}")
    model = load_mlp(args.model_path, device)
    validate_shared_space(model, valid_actions, args.state_size, args.top_k)
    print(f"Shared action_dim: {model.action_dim}")
    print(f"Valid test actions: {len(valid_actions)}")

    row = evaluate_mlp(model, histories, valid_actions, args, device)
    print_results([row], args.top_k)
    save_csv(row, args.output_csv)
    save_markdown(row, args.output_md, args.data_path, args.top_k, args.max_windows)
    print(f"\nSaved CSV: {args.output_csv}")
    print(f"Saved Markdown: {args.output_md}")


if __name__ == "__main__":
    main()
