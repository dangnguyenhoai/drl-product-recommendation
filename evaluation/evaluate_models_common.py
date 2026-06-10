from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

from baseline.baseline_train import HistoryMLPRecommender
from models.dqn_model import DQN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare an MLP baseline checkpoint and a DQN checkpoint on the exact "
            "same test windows and metric definitions."
        )
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/processed/test_history.pkl"),
    )
    parser.add_argument("--mlp-model-path", type=Path, required=True)
    parser.add_argument("--dqn-model-path", type=Path, required=True)
    parser.add_argument("--state-size", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--recent-boost",
        type=float,
        default=0.0,
        help="Optional recency-prior boost applied only to DQN state items.",
    )
    parser.add_argument("--hit-reward", type=float, default=5.0)
    parser.add_argument("--miss-penalty", type=float, default=-2.0)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/logs/common_model_comparison.csv"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("outputs/logs/common_model_comparison.md"),
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for path in (args.data_path, args.mlp_model_path, args.dqn_model_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path}")

    if args.state_size <= 0:
        raise ValueError("--state-size must be positive.")
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA is unavailable. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_arg)


def load_histories(path: Path) -> list[np.ndarray]:
    with path.open("rb") as file:
        raw = pickle.load(file)

    values = raw.values() if isinstance(raw, dict) else raw
    if not isinstance(raw, (dict, list)):
        raise TypeError(
            f"{path} must contain dict[user_id, history] or list[history], "
            f"got {type(raw)!r}."
        )

    histories = []
    for items in values:
        history = np.asarray(items, dtype=np.int64)
        if history.ndim != 1:
            raise ValueError("Every history must be one-dimensional.")
        if len(history) > 0:
            histories.append(history)

    if not histories:
        raise ValueError(f"No non-empty histories found in {path}.")
    return histories


def get_valid_actions(histories: list[np.ndarray]) -> list[int]:
    return sorted({int(item) for history in histories for item in history})


def iter_test_windows(
    histories: list[np.ndarray],
    state_size: int,
    top_k: int,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    window_size = state_size + top_k
    for history in histories:
        for start in range(len(history) - window_size + 1):
            state_end = start + state_size
            yield history[start:state_end], history[state_end : state_end + top_k]


def iter_batches(
    windows: Iterator[tuple[np.ndarray, np.ndarray]],
    batch_size: int,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    states: list[np.ndarray] = []
    targets: list[np.ndarray] = []

    for state, target in windows:
        states.append(state)
        targets.append(target)
        if len(states) == batch_size:
            yield torch.as_tensor(np.asarray(states)), torch.as_tensor(np.asarray(targets))
            states.clear()
            targets.clear()

    if states:
        yield torch.as_tensor(np.asarray(states)), torch.as_tensor(np.asarray(targets))


def unwrap_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError("Model checkpoint must be a dictionary.")
    if "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def load_mlp(path: Path, device: torch.device) -> HistoryMLPRecommender:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or "model_config" not in checkpoint:
        raise ValueError(
            "MLP checkpoint must be created by baseline/baseline_train.py and "
            "contain model_config."
        )

    model = HistoryMLPRecommender(**checkpoint["model_config"]).to(device)
    model.load_state_dict(unwrap_state_dict(checkpoint))
    model.eval()
    return model


def infer_dqn_config(state_dict: dict[str, torch.Tensor]) -> tuple[int, int, int]:
    try:
        action_dim, embedding_dim = state_dict["item_embedding.weight"].shape
        hidden_dim = state_dict["gru.weight_hh_l0"].shape[1]
    except KeyError as exc:
        raise ValueError(f"Unsupported DQN checkpoint; missing key: {exc}") from exc
    return int(action_dim), int(embedding_dim), int(hidden_dim)


def load_dqn(
    path: Path,
    state_size: int,
    device: torch.device,
) -> DQN:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state_dict = unwrap_state_dict(checkpoint)
    action_dim, embedding_dim, hidden_dim = infer_dqn_config(state_dict)

    model = DQN(
        state_dim=state_size,
        action_dim=action_dim,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def validate_shared_space(
    mlp: HistoryMLPRecommender,
    dqn: DQN,
    valid_actions: list[int],
    state_size: int,
    top_k: int,
) -> None:
    if mlp.action_dim != dqn.action_dim:
        raise ValueError(
            "The models use different action dimensions: "
            f"MLP={mlp.action_dim}, DQN={dqn.action_dim}."
        )
    if mlp.state_size != state_size:
        raise ValueError(
            f"MLP checkpoint uses state_size={mlp.state_size}, "
            f"but --state-size={state_size}."
        )
    if max(valid_actions) >= mlp.action_dim:
        raise ValueError(
            f"Test item id {max(valid_actions)} is outside action_dim={mlp.action_dim}."
        )
    if len(valid_actions) < top_k:
        raise ValueError(
            f"Test data has only {len(valid_actions)} valid actions, but top_k={top_k}."
        )


def select_top_k(
    scores: torch.Tensor,
    states: torch.Tensor,
    valid_mask: torch.Tensor,
    top_k: int,
    recent_boost: float = 0.0,
) -> torch.Tensor:
    ranking_scores = scores.clone()
    ranking_scores[:, ~valid_mask] = -torch.inf

    if recent_boost != 0.0:
        recent_mask = torch.zeros_like(ranking_scores, dtype=torch.bool)
        recent_mask.scatter_(1, states, True)
        ranking_scores[recent_mask] += recent_boost

    return torch.topk(ranking_scores, k=top_k, dim=1).indices


def unique_target_counts(targets: torch.Tensor) -> torch.Tensor:
    target_matches = targets.unsqueeze(2).eq(targets.unsqueeze(1))
    previous_positions = torch.tril(
        torch.ones(
            targets.size(1),
            targets.size(1),
            dtype=torch.bool,
            device=targets.device,
        ),
        diagonal=-1,
    )
    has_previous_match = (target_matches & previous_positions).any(dim=2)
    return (~has_previous_match).sum(dim=1)


def empty_totals() -> dict[str, float]:
    return {
        "cases": 0.0,
        "reward": 0.0,
        "next_item_hits": 0.0,
        "next_item_ndcg": 0.0,
        "window_hits": 0.0,
        "window_hit_cases": 0.0,
        "window_recall": 0.0,
        "window_ndcg": 0.0,
    }


def update_totals(
    totals: dict[str, float],
    recommendations: torch.Tensor,
    targets: torch.Tensor,
    hit_reward: float,
    miss_penalty: float,
) -> None:
    batch_size, top_k = recommendations.shape
    discounts = 1.0 / torch.log2(
        torch.arange(2, top_k + 2, dtype=torch.float32, device=targets.device)
    )

    next_matches = recommendations.eq(targets[:, :1])
    next_hits = next_matches.any(dim=1)
    next_dcg = (next_matches.float() * discounts).sum(dim=1)

    window_matches = recommendations.unsqueeze(2).eq(targets.unsqueeze(1)).any(dim=2)
    hit_counts = window_matches.sum(dim=1)
    hit_cases = hit_counts > 0
    target_counts = unique_target_counts(targets)

    window_dcg = (window_matches.float() * discounts).sum(dim=1)
    cumulative_discounts = torch.cumsum(discounts, dim=0)
    ideal_counts = torch.clamp(target_counts, max=top_k)
    window_idcg = cumulative_discounts[ideal_counts - 1]

    rewards = torch.where(
        hit_cases,
        hit_counts.float() * hit_reward,
        torch.full_like(hit_counts, miss_penalty, dtype=torch.float32),
    )

    totals["cases"] += batch_size
    totals["reward"] += rewards.sum().item()
    totals["next_item_hits"] += next_hits.sum().item()
    totals["next_item_ndcg"] += next_dcg.sum().item()
    totals["window_hits"] += hit_counts.sum().item()
    totals["window_hit_cases"] += hit_cases.sum().item()
    totals["window_recall"] += (hit_counts / target_counts).sum().item()
    totals["window_ndcg"] += (window_dcg / window_idcg).sum().item()


def finalize_metrics(
    model_name: str,
    totals: dict[str, float],
    top_k: int,
) -> dict[str, float | str]:
    cases = totals["cases"]
    if cases == 0:
        raise ValueError("Test data does not contain any evaluable windows.")

    return {
        "model": model_name,
        "cases": int(cases),
        "average_window_reward": totals["reward"] / cases,
        f"next_item_hitrate@{top_k}": totals["next_item_hits"] / cases,
        f"next_item_ndcg@{top_k}": totals["next_item_ndcg"] / cases,
        f"window_hitrate@{top_k}": totals["window_hit_cases"] / cases,
        f"window_precision@{top_k}": totals["window_hits"] / (cases * top_k),
        f"window_recall@{top_k}": totals["window_recall"] / cases,
        f"window_ndcg@{top_k}": totals["window_ndcg"] / cases,
    }


@torch.no_grad()
def evaluate_models(
    mlp: HistoryMLPRecommender,
    dqn: DQN,
    histories: list[np.ndarray],
    valid_actions: list[int],
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, float | str]]:
    valid_mask = torch.zeros(mlp.action_dim, dtype=torch.bool, device=device)
    valid_mask[torch.as_tensor(valid_actions, device=device)] = True

    totals = {"MLP baseline": empty_totals(), "DQN": empty_totals()}
    windows = iter_test_windows(histories, args.state_size, args.top_k)

    for states, targets in iter_batches(windows, args.batch_size):
        states = states.to(device)
        targets = targets.to(device)

        mlp_recommendations = select_top_k(
            mlp(states),
            states,
            valid_mask,
            args.top_k,
        )
        dqn_recommendations = select_top_k(
            dqn(states),
            states,
            valid_mask,
            args.top_k,
            recent_boost=args.recent_boost,
        )

        update_totals(
            totals["MLP baseline"],
            mlp_recommendations,
            targets,
            args.hit_reward,
            args.miss_penalty,
        )
        update_totals(
            totals["DQN"],
            dqn_recommendations,
            targets,
            args.hit_reward,
            args.miss_penalty,
        )

    dqn_name = "DQN" if args.recent_boost == 0.0 else f"DQN + recency boost={args.recent_boost:g}"
    rows = [
        finalize_metrics("MLP baseline", totals["MLP baseline"], args.top_k),
        finalize_metrics(dqn_name, totals["DQN"], args.top_k),
    ]
    rows[0].update(
        {
            "model_path": str(args.mlp_model_path),
            "recent_boost": 0.0,
        }
    )
    rows[1].update(
        {
            "model_path": str(args.dqn_model_path),
            "recent_boost": args.recent_boost,
        }
    )
    return rows


def save_csv(rows: list[dict[str, float | str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_markdown(
    rows: list[dict[str, float | str]],
    path: Path,
    data_path: Path,
    top_k: int,
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

    lines = [
        "# Common MLP/DQN Test Comparison",
        "",
        f"- Test data: `{data_path}`",
        f"- Shared test windows: {rows[0]['cases']}",
        f"- Top-K: {top_k}",
        "- Both models use the same valid-action mask and the same state/target windows.",
        "",
        "| Model | " + " | ".join(metric_names) + " |",
        "|---|" + "|".join("---:" for _ in metric_names) + "|",
    ]
    for row in rows:
        values = " | ".join(f"{float(row[name]):.6f}" for name in metric_names)
        lines.append(f"| {row['model']} | {values} |")

    lines.extend(
        [
            "",
            "## Metric Definitions",
            "",
            f"- `next_item_hitrate@{top_k}`: Top-{top_k} contains the immediate next item.",
            f"- `window_hitrate@{top_k}`: Top-{top_k} contains at least one of the next {top_k} items.",
            f"- `window_precision@{top_k}`, `window_recall@{top_k}`, and "
            f"`window_ndcg@{top_k}` use the next {top_k} items as the relevant set.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_results(rows: list[dict[str, float | str]], top_k: int) -> None:
    print("\n===== Common Test Comparison =====")
    for row in rows:
        print(f"\n{row['model']} ({row['cases']:,} shared windows)")
        print(f"Average Window Reward: {row['average_window_reward']:.6f}")
        print(f"Next-item HitRate@{top_k}: {row[f'next_item_hitrate@{top_k}']:.6f}")
        print(f"Next-item NDCG@{top_k}: {row[f'next_item_ndcg@{top_k}']:.6f}")
        print(f"Window HitRate@{top_k}: {row[f'window_hitrate@{top_k}']:.6f}")
        print(f"Window Precision@{top_k}: {row[f'window_precision@{top_k}']:.6f}")
        print(f"Window Recall@{top_k}: {row[f'window_recall@{top_k}']:.6f}")
        print(f"Window NDCG@{top_k}: {row[f'window_ndcg@{top_k}']:.6f}")


def main() -> None:
    args = parse_args()
    validate_args(args)
    device = resolve_device(args.device)

    print(f"Using device: {device}")
    print(f"Loading test data: {args.data_path}")
    histories = load_histories(args.data_path)
    valid_actions = get_valid_actions(histories)

    print(f"Loading MLP checkpoint: {args.mlp_model_path}")
    mlp = load_mlp(args.mlp_model_path, device)
    print(f"Loading DQN checkpoint: {args.dqn_model_path}")
    dqn = load_dqn(args.dqn_model_path, args.state_size, device)

    validate_shared_space(mlp, dqn, valid_actions, args.state_size, args.top_k)
    print(f"Shared action_dim: {mlp.action_dim}")
    print(f"Valid test actions: {len(valid_actions)}")

    rows = evaluate_models(mlp, dqn, histories, valid_actions, args, device)
    print_results(rows, args.top_k)

    save_csv(rows, args.output_csv)
    save_markdown(rows, args.output_md, args.data_path, args.top_k)
    print(f"\nSaved CSV: {args.output_csv}")
    print(f"Saved Markdown: {args.output_md}")


if __name__ == "__main__":
    main()
