from __future__ import annotations

import argparse
import math
import pickle
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from env.recommendation_env import RecommendationEnv


# Cấu hình mặc định đồng bộ với mô hình DQN trong project.
DEFAULT_STATE_SIZE = 5
DEFAULT_TOP_K = 5
DEFAULT_EMBEDDING_DIM = 32
DEFAULT_HIDDEN_DIMS = (128,)
DEFAULT_BATCH_SIZE = 64
DEFAULT_LR = 1e-4


def configure_utf8_output() -> None:
    """Cho phép Windows console hiển thị tiếng Việt có dấu."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def set_seed(seed: int) -> None:
    """Cố định seed để kết quả dễ tái lập hơn."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def parse_hidden_dims(value: str) -> tuple[int, ...]:
    dims = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not dims or any(dim <= 0 for dim in dims):
        raise argparse.ArgumentTypeError(
            "Kích thước tầng ẩn phải là các số nguyên dương, ví dụ: 128 hoặc 128,64."
        )
    return dims


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Huấn luyện MLP baseline và đánh giá HitRate@5 theo đúng cách của DQN."
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
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--eval-episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Thiết bị chạy mô hình. Chọn auto để tự dùng CUDA khi có.",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Giới hạn số mẫu train để chạy thử nhanh. Bỏ trống để dùng toàn bộ dữ liệu.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive_values = {
        "--state-size": args.state_size,
        "--top-k": args.top_k,
        "--epochs": args.epochs,
        "--batch-size": args.batch_size,
        "--embedding-dim": args.embedding_dim,
        "--patience": args.patience,
        "--eval-episodes": args.eval_episodes,
    }
    for name, value in positive_values.items():
        if value <= 0:
            raise ValueError(f"{name} phải lớn hơn 0.")

    if args.action_dim is not None and args.action_dim <= 0:
        raise ValueError("--action-dim phải lớn hơn 0.")
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("--dropout phải nằm trong khoảng [0, 1).")
    if args.lr <= 0.0:
        raise ValueError("--lr phải lớn hơn 0.")
    if args.max_train_samples is not None and args.max_train_samples <= 0:
        raise ValueError("--max-train-samples phải lớn hơn 0.")


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {path}")


def resolve_data_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    train_path = args.train_data_path or args.processed_dir / "train_history.pkl"
    val_path = args.val_data_path or args.processed_dir / "val_history.pkl"
    test_path = args.test_data_path or args.processed_dir / "test_history.pkl"

    for path in (train_path, val_path, test_path):
        require_file(path)
    return train_path, val_path, test_path


def load_history_pickle(path: Path) -> dict[Any, np.ndarray]:
    """Đọc lịch sử người dùng và chuẩn hóa về dạng dict[user_id, history]."""
    with path.open("rb") as file:
        raw = pickle.load(file)

    if isinstance(raw, dict):
        items = raw.items()
    elif isinstance(raw, list):
        items = enumerate(raw)
    else:
        raise TypeError(
            f"{path.name} phải chứa dict[user_id, history] hoặc list[history]."
        )

    histories: dict[Any, np.ndarray] = {}
    for user_id, values in items:
        history = np.asarray(values, dtype=np.int64)
        if history.ndim != 1:
            raise ValueError(f"Lịch sử của user {user_id!r} không phải mảng một chiều.")
        if len(history) == 0:
            continue
        if np.any(history < 0):
            raise ValueError(f"Lịch sử của user {user_id!r} chứa item id âm.")
        histories[user_id] = history

    if not histories:
        raise ValueError(f"{path.name} không có lịch sử hợp lệ.")
    return histories


def filter_histories(
    histories: dict[Any, np.ndarray],
    min_length: int,
    split_name: str,
) -> dict[Any, np.ndarray]:
    kept = {
        user_id: history
        for user_id, history in histories.items()
        if len(history) >= min_length
    }
    dropped = len(histories) - len(kept)

    if not kept:
        raise ValueError(
            f"Tập {split_name} không có lịch sử dài tối thiểu {min_length} item."
        )
    if dropped:
        print(f"Tập {split_name}: bỏ {dropped:,} lịch sử quá ngắn.")
    return kept


def infer_action_dim(*history_groups: dict[Any, np.ndarray]) -> int:
    max_item = max(
        int(history.max())
        for histories in history_groups
        for history in histories.values()
    )
    return max_item + 1


def get_valid_actions(histories: dict[Any, np.ndarray]) -> list[int]:
    return sorted({int(item) for history in histories.values() for item in history})


class HistoryNextItemDataset(Dataset):
    """Tạo mẫu train có dạng 5 item gần nhất -> item kế tiếp."""

    def __init__(
        self,
        histories: dict[Any, np.ndarray],
        state_size: int,
        action_dim: int,
        max_samples: int | None,
        seed: int,
    ) -> None:
        self.histories = list(histories.values())
        self.state_size = state_size

        history_indices: list[int] = []
        target_positions: list[int] = []
        for history_idx, history in enumerate(self.histories):
            if int(history.max()) >= action_dim:
                raise ValueError(
                    f"Item id {int(history.max())} nằm ngoài action_dim={action_dim}."
                )

            sample_count = len(history) - state_size
            history_indices.extend([history_idx] * sample_count)
            target_positions.extend(range(state_size, len(history)))

        self.history_indices = np.asarray(history_indices, dtype=np.int32)
        self.target_positions = np.asarray(target_positions, dtype=np.int32)

        if max_samples is not None and max_samples < len(self.history_indices):
            rng = np.random.default_rng(seed)
            selected = rng.choice(len(self.history_indices), size=max_samples, replace=False)
            self.history_indices = self.history_indices[selected]
            self.target_positions = self.target_positions[selected]
            print(f"Chỉ dùng {max_samples:,} mẫu train để chạy thử.")

    def __len__(self) -> int:
        return len(self.history_indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        history = self.histories[int(self.history_indices[index])]
        target_pos = int(self.target_positions[index])
        state = history[target_pos - self.state_size : target_pos]
        target = history[target_pos]
        return torch.as_tensor(state), torch.tensor(target, dtype=torch.long)


class HistoryMLPRecommender(nn.Module):
    """MLP nhận state gồm các item gần nhất và chấm điểm toàn bộ item."""

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
            layers.extend((nn.Linear(input_dim, hidden_dim), nn.ReLU()))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, action_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        embedded = self.item_embedding(states)
        return self.mlp(embedded.reshape(embedded.size(0), -1))


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
        return improved, self.bad_epochs >= self.patience


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        print("CẢNH BÁO: Không có CUDA, chuyển sang chạy bằng CPU.")
        return torch.device("cpu")
    return torch.device(device_arg)


def train_one_epoch(
    model: HistoryMLPRecommender,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0

    for states, targets in loader:
        states = states.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(states), targets)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * len(targets)
        total_samples += len(targets)

    return total_loss / total_samples


@torch.no_grad()
def choose_actions(
    model: HistoryMLPRecommender,
    state: np.ndarray,
    valid_actions: list[int],
    banned_actions: set[int],
    top_k: int,
    device: torch.device,
) -> list[int]:
    """Chọn Top-K giống DQN: chỉ dùng item hợp lệ và tránh item đã gợi ý."""
    available_actions = [
        action for action in valid_actions if action not in banned_actions
    ]
    if len(available_actions) < top_k:
        available_actions = valid_actions

    state_tensor = torch.as_tensor(state, dtype=torch.long, device=device).unsqueeze(0)
    action_tensor = torch.as_tensor(available_actions, dtype=torch.long, device=device)
    scores = model(state_tensor)[0, action_tensor]
    top_indices = torch.topk(scores, k=top_k).indices
    return action_tensor[top_indices].cpu().tolist()


@torch.no_grad()
def evaluate_hitrate_like_dqn(
    model: HistoryMLPRecommender,
    histories: dict[Any, np.ndarray],
    episodes: int,
    state_size: int,
    top_k: int,
    device: torch.device,
    seed: int,
) -> float:
    """Tính HitRate@K theo đúng cách evaluation/evaluate_dqn.py đang dùng."""
    model.eval()
    env = RecommendationEnv(histories, state_size=state_size, top_k=top_k)
    valid_actions = get_valid_actions(histories)
    total_step_hits = 0
    total_steps = 0

    # Mỗi lần đánh giá dùng lại cùng các episode để so sánh các epoch công bằng.
    previous_random_state = random.getstate()
    random.seed(seed)
    try:
        for _ in range(episodes):
            state = env.reset()
            done = False

            while not done:
                actions = choose_actions(
                    model=model,
                    state=state,
                    valid_actions=valid_actions,
                    banned_actions=env.recommended_items,
                    top_k=top_k,
                    device=device,
                )
                state, _, done, info = env.step(actions)
                total_step_hits += int(info["hits"] > 0)
                total_steps += 1
    finally:
        random.setstate(previous_random_state)

    if total_steps == 0:
        raise ValueError("Không có bước đánh giá hợp lệ.")
    return total_step_hits / total_steps


def save_checkpoint(
    path: Path,
    model: HistoryMLPRecommender,
    epoch: int,
    best_val_hitrate: float,
    top_k: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "action_dim": model.action_dim,
                "state_size": model.state_size,
                "embedding_dim": model.embedding_dim,
                "hidden_dims": model.hidden_dims,
                "dropout": model.dropout,
            },
            "epoch": epoch,
            f"best_val_hitrate@{top_k}": best_val_hitrate,
            "metric_definition": (
                "Số bước có ít nhất một gợi ý trúng trong nhóm item tiếp theo "
                "chia cho tổng số bước, giống evaluation/evaluate_dqn.py."
            ),
        },
        path,
    )


def load_best_model(path: Path, device: torch.device) -> HistoryMLPRecommender:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = HistoryMLPRecommender(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def save_report(
    path: Path,
    model_path: Path,
    best_epoch: int,
    best_val_hitrate: float,
    test_hitrate: float,
    episodes: int,
    top_k: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "KẾT QUẢ MLP BASELINE",
                "",
                (
                    f"Định nghĩa HitRate@{top_k}: số bước có ít nhất một gợi ý "
                    f"trúng trong {top_k} item tiếp theo / tổng số bước."
                ),
                f"Số episode đánh giá: {episodes}",
                f"Epoch tốt nhất: {best_epoch}",
                f"Validation HitRate@{top_k}: {best_val_hitrate:.6f}",
                f"Test HitRate@{top_k}: {test_hitrate:.6f}",
                f"File model: {model_path}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)

    train_path, val_path, test_path = resolve_data_paths(args)
    train_histories = filter_histories(
        load_history_pickle(train_path),
        args.state_size + 1,
        "train",
    )
    val_histories = filter_histories(
        load_history_pickle(val_path),
        args.state_size + args.top_k + 1,
        "validation",
    )
    test_histories = filter_histories(
        load_history_pickle(test_path),
        args.state_size + args.top_k + 1,
        "test",
    )

    inferred_action_dim = infer_action_dim(train_histories, val_histories, test_histories)
    action_dim = args.action_dim or inferred_action_dim
    if action_dim < inferred_action_dim:
        raise ValueError(
            f"--action-dim={action_dim} quá nhỏ; dữ liệu cần ít nhất {inferred_action_dim}."
        )

    train_dataset = HistoryNextItemDataset(
        histories=train_histories,
        state_size=args.state_size,
        action_dim=action_dim,
        max_samples=args.max_train_samples,
        seed=args.seed,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )

    device = resolve_device(args.device)
    model = HistoryMLPRecommender(
        action_dim=action_dim,
        state_size=args.state_size,
        embedding_dim=args.embedding_dim,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    early_stopping = EarlyStopping(args.patience, args.min_delta)

    model_path = args.output_dir / "checkpoints" / args.model_name
    report_path = args.output_dir / "logs" / "mlp_hitrate_dqn_style.txt"
    best_epoch = 0
    best_val_hitrate = -math.inf

    print("===== CẤU HÌNH MLP BASELINE =====")
    print(f"Thiết bị: {device}")
    print(f"Số item đầu ra (action_dim): {action_dim:,}")
    print(f"Số mẫu train: {len(train_dataset):,}")
    print(f"Số episode đánh giá mỗi lần: {args.eval_episodes:,}")
    print(
        f"HitRate@{args.top_k}: một bước được tính hit khi Top-{args.top_k} "
        f"trúng ít nhất một trong {args.top_k} item tiếp theo."
    )

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_hitrate = evaluate_hitrate_like_dqn(
            model=model,
            histories=val_histories,
            episodes=args.eval_episodes,
            state_size=args.state_size,
            top_k=args.top_k,
            device=device,
            seed=args.seed,
        )

        improved, should_stop = early_stopping.step(val_hitrate)
        if improved:
            best_epoch = epoch
            best_val_hitrate = val_hitrate
            save_checkpoint(
                model_path,
                model,
                epoch,
                best_val_hitrate,
                args.top_k,
            )

        status = "tốt nhất" if improved else "không cải thiện"
        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | "
            f"validation_hitrate@{args.top_k}={val_hitrate:.6f} | {status}"
        )

        if should_stop:
            print(f"Dừng sớm sau epoch {epoch}.")
            break

    print("\nĐang tải checkpoint tốt nhất để đánh giá trên tập test...")
    best_model = load_best_model(model_path, device)
    test_hitrate = evaluate_hitrate_like_dqn(
        model=best_model,
        histories=test_histories,
        episodes=args.eval_episodes,
        state_size=args.state_size,
        top_k=args.top_k,
        device=device,
        seed=args.seed,
    )
    save_report(
        report_path,
        model_path,
        best_epoch,
        best_val_hitrate,
        test_hitrate,
        args.eval_episodes,
        args.top_k,
    )

    print("\n===== KẾT QUẢ CUỐI CÙNG =====")
    print(f"Epoch tốt nhất: {best_epoch}")
    print(f"Validation HitRate@{args.top_k}: {best_val_hitrate:.6f}")
    print(f"Test HitRate@{args.top_k}: {test_hitrate:.6f}")
    print(f"Đã lưu model: {model_path}")
    print(f"Đã lưu báo cáo: {report_path}")


if __name__ == "__main__":
    configure_utf8_output()
    main()

