from __future__ import annotations

import argparse
import gc
import math
import pickle
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


# Đồng bộ với notebook drl_product_recommendation_train_val_test_colab:
# STATE_SIZE = 5, TOP_K = 5, embedding_dim = 32, hidden_dim = 128,
# batch_size = 64, lr = 0.0001, chọn mô hình theo validation HitRate@5.
DEFAULT_STATE_SIZE = 5
DEFAULT_TOP_K = 5
DEFAULT_EMBEDDING_DIM = 32
DEFAULT_HIDDEN_DIMS = (128,)
DEFAULT_BATCH_SIZE = 64
DEFAULT_LR = 1e-4
EARLY_STOP_MODE = "max"


def set_seed(seed: int) -> None:
    """Cố định seed để kết quả dễ tái lập hơn."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Ưu tiên tái lập kết quả hơn tốc độ.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def parse_hidden_dims(value: str) -> tuple[int, ...]:
    dims = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not dims:
        raise argparse.ArgumentTypeError("hidden dims cannot be empty.")
    if any(dim <= 0 for dim in dims):
        raise argparse.ArgumentTypeError("hidden dims must be positive integers.")
    return dims


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train baseline MLP recommender using the same history input format as "
            "the DQN notebook: train_history.pkl / val_history.pkl / test_history.pkl."
        )
    )

    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--train-data-path", type=Path, default=None)
    parser.add_argument("--val-data-path", type=Path, default=None)
    parser.add_argument("--test-data-path", type=Path, default=None)

    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--model-name", type=str, default="mlp_history_baseline.pth")

    parser.add_argument("--state-size", type=int, default=DEFAULT_STATE_SIZE)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--action-dim", type=int, default=None)

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--embedding-dim", type=int, default=DEFAULT_EMBEDDING_DIM)
    parser.add_argument("--hidden-dims", type=parse_hidden_dims, default=DEFAULT_HIDDEN_DIMS)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=0.0)

    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--log-every-batches",
        type=int,
        default=1000,
        help="Print in-epoch training progress every N batches. Use 0 to disable.",
    )
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=("cuda", "cpu", "auto"),
        help="Use 'cuda' to match the DQN notebook. If CUDA is unavailable, the script falls back to CPU.",
    )
    parser.add_argument(
        "--mask-seen-items",
        action="store_true",
        help=(
            "Mask items already appearing in the current state during evaluation. "
            "Default is False to stay closer to the DQN/recent-item setup."
        ),
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Optional cap for quick debugging. Leave empty for full training data.",
    )
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=None,
        help="Optional cap for quick debugging validation/test. Leave empty for full evaluation data.",
    )

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.state_size <= 0:
        raise ValueError("--state-size must be positive.")
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive.")
    if args.action_dim is not None and args.action_dim <= 0:
        raise ValueError("--action-dim must be positive when provided.")
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.embedding_dim <= 0:
        raise ValueError("--embedding-dim must be positive.")
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("--dropout must be in [0, 1).")
    if args.lr <= 0.0:
        raise ValueError("--lr must be positive.")
    if args.weight_decay < 0.0:
        raise ValueError("--weight-decay cannot be negative.")
    if args.patience <= 0:
        raise ValueError("--patience must be positive.")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative.")
    if args.log_every_batches < 0:
        raise ValueError("--log-every-batches cannot be negative.")
    if args.max_train_samples is not None and args.max_train_samples <= 0:
        raise ValueError("--max-train-samples must be positive when provided.")
    if args.max_eval_samples is not None and args.max_eval_samples <= 0:
        raise ValueError("--max-eval-samples must be positive when provided.")


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    if path.is_dir():
        raise FileNotFoundError(f"Expected a file, got directory: {path}")


def resolve_data_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    train_path = args.train_data_path or (args.processed_dir / "train_history.pkl")
    val_path = args.val_data_path or (args.processed_dir / "val_history.pkl")
    test_path = args.test_data_path or (args.processed_dir / "test_history.pkl")

    for path in (train_path, val_path, test_path):
        require_file(path)

    return train_path, val_path, test_path


def load_history_pickle(path: Path) -> list[np.ndarray]:
    """Load history pickle created by the DRL notebook.

    Expected format:
        dict[user_id, list[item_idx]]
    Also accepts:
        list[list[item_idx]]
    """
    require_file(path)
    with path.open("rb") as f:
        raw = pickle.load(f)

    if isinstance(raw, dict):
        values = list(raw.values())
    elif isinstance(raw, list):
        values = raw
    else:
        raise TypeError(
            f"{path.name} must be a dict[user_id, history] or list[history], got {type(raw)!r}."
        )

    histories: list[np.ndarray] = []
    for idx, items in enumerate(values):
        if items is None:
            continue

        arr = np.asarray(items, dtype=np.int64)
        if arr.ndim != 1:
            raise ValueError(f"{path.name}: history at index {idx} is not 1-dimensional.")
        if len(arr) == 0:
            continue
        if np.any(arr < 0):
            raise ValueError(f"{path.name}: history at index {idx} contains negative item ids.")

        histories.append(arr)

    if not histories:
        raise ValueError(f"{path.name} does not contain any non-empty histories.")

    return histories


def filter_histories(
    histories: list[np.ndarray],
    state_size: int,
    split_name: str,
) -> list[np.ndarray]:
    min_len = state_size + 1
    kept = [hist for hist in histories if len(hist) >= min_len]
    dropped = len(histories) - len(kept)

    if not kept:
        raise ValueError(
            f"{split_name} has no histories with length >= {min_len}. "
            f"Increase data size or reduce --state-size."
        )

    if dropped > 0:
        print(f"{split_name}: dropped {dropped:,} histories shorter than {min_len}.")

    return kept


def infer_action_dim_from_histories(histories: list[np.ndarray]) -> int:
    max_item = max(int(hist.max()) for hist in histories if len(hist) > 0)
    return max_item + 1


def count_samples(histories: list[np.ndarray], state_size: int) -> int:
    return int(sum(max(0, len(hist) - state_size) for hist in histories))


class HistoryNextItemDataset(Dataset):
    """Convert each user history into supervised next-item examples.

    Với mỗi lịch sử:
        [i1, i2, i3, i4, i5, i6, ...]

    Nếu state_size = 5 thì một mẫu huấn luyện là:
        state  = [i1, i2, i3, i4, i5]
        target = i6

    Cách này giúp MLP dùng đúng đầu vào dạng state 5 item gần nhất giống DQN.
    """

    def __init__(
        self,
        histories: list[np.ndarray],
        state_size: int,
        action_dim: int,
        max_samples: int | None = None,
        seed: int = 42,
        split_name: str = "dataset",
    ) -> None:
        self.histories = histories
        self.state_size = state_size
        self.action_dim = action_dim
        self.split_name = split_name

        history_indices: list[int] = []
        target_positions: list[int] = []

        for hist_idx, hist in enumerate(histories):
            if np.any(hist >= action_dim):
                raise ValueError(
                    f"{split_name}: item id out of action space. "
                    f"Max item in history={int(hist.max())}, action_dim={action_dim}."
                )

            # target position starts from state_size because previous state_size items are input state.
            n_samples = len(hist) - state_size
            if n_samples <= 0:
                continue

            history_indices.extend([hist_idx] * n_samples)
            target_positions.extend(range(state_size, len(hist)))

        if not history_indices:
            raise ValueError(f"{split_name}: no samples can be created with state_size={state_size}.")

        self.history_indices = np.asarray(history_indices, dtype=np.int32)
        self.target_positions = np.asarray(target_positions, dtype=np.int32)

        if max_samples is not None and max_samples < len(self.history_indices):
            rng = np.random.default_rng(seed)
            selected = rng.choice(len(self.history_indices), size=max_samples, replace=False)
            selected.sort()
            self.history_indices = self.history_indices[selected]
            self.target_positions = self.target_positions[selected]
            print(f"{split_name}: using {max_samples:,} sampled examples for quick debugging.")

    def __len__(self) -> int:
        return int(len(self.history_indices))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        hist = self.histories[int(self.history_indices[idx])]
        target_pos = int(self.target_positions[idx])

        state_np = hist[target_pos - self.state_size : target_pos]
        target = int(hist[target_pos])

        state = torch.as_tensor(state_np, dtype=torch.long)
        target_tensor = torch.tensor(target, dtype=torch.long)
        return state, target_tensor


class HistoryMLPRecommender(nn.Module):
    """MLP baseline dùng cùng state item gần nhất như DQN.

    Input:
        state: Tensor(batch_size, state_size), mỗi phần tử là item_idx.

    Output:
        logits: Tensor(batch_size, action_dim), mỗi cột là điểm dự đoán cho một item.
    """

    def __init__(
        self,
        action_dim: int,
        state_size: int = DEFAULT_STATE_SIZE,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        hidden_dims: tuple[int, ...] = DEFAULT_HIDDEN_DIMS,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.action_dim = action_dim
        self.state_size = state_size
        self.embedding_dim = embedding_dim
        self.hidden_dims = tuple(hidden_dims)
        self.dropout = dropout

        self.item_embedding = nn.Embedding(action_dim, embedding_dim)

        layers: list[nn.Module] = []
        input_dim = state_size * embedding_dim

        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, action_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        embedded = self.item_embedding(states)
        flattened = embedded.reshape(embedded.size(0), -1)
        return self.mlp(flattened)


@dataclass
class EarlyStopping:
    patience: int
    min_delta: float
    best_score: float = -math.inf
    bad_epochs: int = 0

    def step(self, score: float) -> tuple[bool, bool]:
        improved = score > self.best_score + self.min_delta

        if improved:
            self.best_score = score
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1

        should_stop = self.bad_epochs >= self.patience
        return improved, should_stop


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device_arg == "cuda" and not torch.cuda.is_available():
        print("WARNING: --device cuda was requested but CUDA is unavailable. Falling back to CPU.")
        return torch.device("cpu")

    return torch.device(device_arg)


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


def train_one_epoch(
    model: HistoryMLPRecommender,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    log_every_batches: int = 0,
) -> float:
    model.train()

    total_loss = 0.0
    total_rows = 0
    total_batches = len(loader)
    started_at = time.perf_counter()

    for batch_idx, (states, targets) in enumerate(loader, start=1):
        states = states.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = model(states)
        loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        total_loss += float(loss.item()) * batch_size
        total_rows += batch_size

        if log_every_batches and (
            batch_idx % log_every_batches == 0 or batch_idx == total_batches
        ):
            elapsed = time.perf_counter() - started_at
            batches_per_second = batch_idx / max(elapsed, 1e-9)
            remaining_seconds = (total_batches - batch_idx) / max(
                batches_per_second, 1e-9
            )
            print(
                f"  batch {batch_idx:,}/{total_batches:,} "
                f"| loss={total_loss / total_rows:.5f} "
                f"| elapsed={elapsed / 60:.1f}m "
                f"| eta={remaining_seconds / 60:.1f}m",
                flush=True,
            )

    return total_loss / max(total_rows, 1)


@torch.no_grad()
def evaluate_model(
    model: HistoryMLPRecommender,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    top_k: int,
    mask_seen_items: bool = False,
) -> dict[str, float]:
    model.eval()

    total_loss = 0.0
    total_rows = 0
    total_hits = 0.0
    total_ndcg = 0.0

    effective_k = min(top_k, model.action_dim)

    for states, targets in loader:
        states = states.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(states)
        loss = criterion(logits, targets)

        if mask_seen_items:
            # Không mask target nếu target tình cờ nằm trong state vì nhiều bài toán mua lại sản phẩm cũ là hợp lệ.
            masked_logits = logits.clone()
            masked_logits.scatter_(1, states, -torch.inf)
            target_logits = logits.gather(1, targets.view(-1, 1))
            masked_logits.scatter_(1, targets.view(-1, 1), target_logits)
            ranking_logits = masked_logits
        else:
            ranking_logits = logits

        top_indices = torch.topk(ranking_logits, k=effective_k, dim=1).indices
        matches = top_indices.eq(targets.view(-1, 1))
        hits = matches.any(dim=1)

        # Với mỗi mẫu chỉ có 1 item đúng nên Recall@K = HitRate@K.
        hit_count = hits.float().sum().item()

        rank_positions = matches.float().argmax(dim=1).float()
        discounts = 1.0 / torch.log2(rank_positions + 2.0)
        ndcg = torch.where(hits, discounts, torch.zeros_like(discounts)).sum().item()

        batch_size = targets.size(0)
        total_loss += float(loss.item()) * batch_size
        total_rows += batch_size
        total_hits += hit_count
        total_ndcg += ndcg

    if total_rows == 0:
        raise ValueError("Evaluation loader has no rows.")

    hitrate = total_hits / total_rows
    return {
        "loss": total_loss / total_rows,
        f"recall@{top_k}": hitrate,
        f"hitrate@{top_k}": hitrate,
        f"ndcg@{top_k}": total_ndcg / total_rows,
    }


def checkpoint_payload(
    model: HistoryMLPRecommender,
    args: argparse.Namespace,
    epoch: int,
    best_score: float,
    train_path: Path,
    val_path: Path,
    test_path: Path,
) -> dict[str, Any]:
    training_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }

    return {
        "model_state_dict": model.state_dict(),
        "model_config": {
            "action_dim": model.action_dim,
            "state_size": model.state_size,
            "embedding_dim": model.embedding_dim,
            "hidden_dims": model.hidden_dims,
            "dropout": model.dropout,
        },
        "epoch": epoch,
        f"best_val_hitrate@{args.top_k}": best_score,
        "training_args": training_args,
        "data_paths": {
            "train": str(train_path),
            "val": str(val_path),
            "test": str(test_path),
        },
    }


def save_checkpoint(
    path: Path,
    model: HistoryMLPRecommender,
    args: argparse.Namespace,
    epoch: int,
    best_score: float,
    train_path: Path,
    val_path: Path,
    test_path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        checkpoint_payload(model, args, epoch, best_score, train_path, val_path, test_path),
        path,
    )


def load_best_model(path: Path, device: torch.device) -> HistoryMLPRecommender:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = HistoryMLPRecommender(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def save_training_plots(history: pd.DataFrame, plot_dir: Path, top_k: int) -> dict[str, Path]:
    plot_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = {
        "loss": plot_dir / "mlp_loss_curve.png",
        "hitrate": plot_dir / f"mlp_hitrate_at_{top_k}_curve.png",
        "recall": plot_dir / f"mlp_recall_at_{top_k}_curve.png",
        "ndcg": plot_dir / f"mlp_ndcg_at_{top_k}_curve.png",
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history["epoch"], history["train_loss"], marker="o", label="train_loss")
    ax.plot(history["epoch"], history["val_loss"], marker="o", label="val_loss")
    ax.set_title("MLP Loss Curve")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("CrossEntropy Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(saved_paths["loss"], dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history["epoch"], history[f"val_hitrate@{top_k}"], marker="o", label=f"val_hitrate@{top_k}")
    ax.set_title(f"Validation HitRate@{top_k}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(f"HitRate@{top_k}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(saved_paths["hitrate"], dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history["epoch"], history[f"val_recall@{top_k}"], marker="o", label=f"val_recall@{top_k}")
    ax.set_title(f"Validation Recall@{top_k}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(f"Recall@{top_k}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(saved_paths["recall"], dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history["epoch"], history[f"val_ndcg@{top_k}"], marker="o", label=f"val_ndcg@{top_k}")
    ax.set_title(f"Validation NDCG@{top_k}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(f"NDCG@{top_k}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(saved_paths["ndcg"], dpi=150)
    plt.close(fig)

    return saved_paths


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)

    train_path, val_path, test_path = resolve_data_paths(args)

    checkpoint_dir = args.output_dir / "checkpoints"
    plot_dir = args.output_dir / "plots"
    log_dir = args.output_dir / "logs"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    model_path = checkpoint_dir / args.model_name
    history_path = log_dir / "train_mlp_history_baseline.csv"
    test_result_path = log_dir / "mlp_final_test_results.csv"

    print("===== MLP Baseline Configuration =====")
    print(f"Train data: {train_path}")
    print(f"Validation data: {val_path}")
    print(f"Test data: {test_path}")
    print(f"state_size={args.state_size}")
    print(f"top_k={args.top_k}")
    print(f"embedding_dim={args.embedding_dim}")
    print(f"hidden_dims={args.hidden_dims}")
    print(f"batch_size={args.batch_size}")
    print(f"lr={args.lr}")
    print(f"dropout={args.dropout}")
    print(f"early_stop_monitor=val_hitrate@{args.top_k}")

    print("\nLoading DRL history data...")
    train_histories = filter_histories(load_history_pickle(train_path), args.state_size, "train")
    val_histories = filter_histories(load_history_pickle(val_path), args.state_size, "validation")
    test_histories = filter_histories(load_history_pickle(test_path), args.state_size, "test")

    inferred_action_dim = max(
        infer_action_dim_from_histories(train_histories),
        infer_action_dim_from_histories(val_histories),
        infer_action_dim_from_histories(test_histories),
    )
    action_dim = args.action_dim or inferred_action_dim

    if action_dim < inferred_action_dim:
        raise ValueError(
            f"--action-dim={action_dim} is too small. Histories require at least {inferred_action_dim}."
        )

    print("\n===== Data Summary =====")
    print(f"Train users/histories: {len(train_histories):,}")
    print(f"Validation users/histories: {len(val_histories):,}")
    print(f"Test users/histories: {len(test_histories):,}")
    print(f"Train examples: {count_samples(train_histories, args.state_size):,}")
    print(f"Validation examples: {count_samples(val_histories, args.state_size):,}")
    print(f"Test examples: {count_samples(test_histories, args.state_size):,}")
    print(f"Inferred action_dim: {inferred_action_dim:,}")
    print(f"Using action_dim: {action_dim:,}")

    print("\nBuilding datasets...")
    train_dataset = HistoryNextItemDataset(
        train_histories,
        state_size=args.state_size,
        action_dim=action_dim,
        max_samples=args.max_train_samples,
        seed=args.seed,
        split_name="train",
    )
    val_dataset = HistoryNextItemDataset(
        val_histories,
        state_size=args.state_size,
        action_dim=action_dim,
        max_samples=args.max_eval_samples,
        seed=args.seed,
        split_name="validation",
    )
    test_dataset = HistoryNextItemDataset(
        test_histories,
        state_size=args.state_size,
        action_dim=action_dim,
        max_samples=args.max_eval_samples,
        seed=args.seed,
        split_name="test",
    )

    # Sau khi Dataset giữ reference histories, không cần biến đếm phụ.
    gc.collect()

    device = resolve_device(args.device)
    pin_memory = device.type == "cuda"
    print(f"\nUsing device: {device}")

    train_loader = make_loader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = make_loader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = make_loader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    model = HistoryMLPRecommender(
        action_dim=action_dim,
        state_size=args.state_size,
        embedding_dim=args.embedding_dim,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    early_stopping = EarlyStopping(patience=args.patience, min_delta=args.min_delta)

    history_rows: list[dict[str, float | int]] = []
    best_epoch = 0
    best_val_hitrate = -math.inf

    print(f"\nEarlyStopping monitor: val_hitrate@{args.top_k} (higher is better)")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            log_every_batches=args.log_every_batches,
        )
        val_metrics = evaluate_model(
            model,
            val_loader,
            criterion,
            device,
            top_k=args.top_k,
            mask_seen_items=args.mask_seen_items,
        )

        val_loss = val_metrics["loss"]
        val_recall = val_metrics[f"recall@{args.top_k}"]
        val_hitrate = val_metrics[f"hitrate@{args.top_k}"]
        val_ndcg = val_metrics[f"ndcg@{args.top_k}"]

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                f"val_recall@{args.top_k}": val_recall,
                f"val_hitrate@{args.top_k}": val_hitrate,
                f"val_ndcg@{args.top_k}": val_ndcg,
            }
        )

        improved, should_stop = early_stopping.step(val_hitrate)
        if improved:
            best_epoch = epoch
            best_val_hitrate = val_hitrate
            save_checkpoint(
                model_path,
                model,
                args,
                epoch,
                best_val_hitrate,
                train_path,
                val_path,
                test_path,
            )

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.5f} | "
            f"val_loss={val_loss:.5f} | "
            f"val_recall@{args.top_k}={val_recall:.5f} | "
            f"val_hitrate@{args.top_k}={val_hitrate:.5f} | "
            f"val_ndcg@{args.top_k}={val_ndcg:.5f} | "
            f"{'best' if improved else 'no_improve'}",
            flush=True,
        )

        if should_stop:
            print(f"Early stopping after {epoch} epochs.")
            break

    history_df = pd.DataFrame(history_rows)
    history_df.to_csv(history_path, index=False)
    plot_paths = save_training_plots(history_df, plot_dir, args.top_k)

    print("\nLoading best validation model for final test...")
    best_model = load_best_model(model_path, device)
    test_metrics = evaluate_model(
        best_model,
        test_loader,
        criterion,
        device,
        top_k=args.top_k,
        mask_seen_items=args.mask_seen_items,
    )

    result_row = {
        "model": "MLP History Baseline",
        "model_path": str(model_path),
        "state_size": args.state_size,
        "top_k": args.top_k,
        "action_dim": action_dim,
        "embedding_dim": args.embedding_dim,
        "hidden_dims": ",".join(map(str, args.hidden_dims)),
        "best_epoch": best_epoch,
        f"best_val_hitrate@{args.top_k}": best_val_hitrate,
        "test_loss": test_metrics["loss"],
        f"test_recall@{args.top_k}": test_metrics[f"recall@{args.top_k}"],
        f"test_hitrate@{args.top_k}": test_metrics[f"hitrate@{args.top_k}"],
        f"test_ndcg@{args.top_k}": test_metrics[f"ndcg@{args.top_k}"],
    }
    pd.DataFrame([result_row]).to_csv(test_result_path, index=False)

    print("\n===== Best MLP Baseline Test Results =====")
    print(f"test_loss={test_metrics['loss']:.5f}")
    print(f"test_recall@{args.top_k}={test_metrics[f'recall@{args.top_k}']:.5f}")
    print(f"test_hitrate@{args.top_k}={test_metrics[f'hitrate@{args.top_k}']:.5f}")
    print(f"test_ndcg@{args.top_k}={test_metrics[f'ndcg@{args.top_k}']:.5f}")
    print(f"Best Epoch: {best_epoch}")
    print(f"Best Validation HitRate@{args.top_k}: {best_val_hitrate:.5f}")

    print("\nSaved files:")
    print(f"- Best model: {model_path}")
    print(f"- Training history: {history_path}")
    print(f"- Final test results: {test_result_path}")
    for name, path in plot_paths.items():
        print(f"- {name} plot: {path}")


if __name__ == "__main__":
    main()
