"""
baseline_train_improved.py
==========================

Huấn luyện và đánh giá mô hình MLP Baseline cải thiện trên dữ liệu Instacart đã xử lý.

Input từ data_preprocessing.py:
    data/processed/train.csv
    data/processed/val_candidates.csv
    data/processed/test_candidates.csv
    data/processed/mappings.pkl
    data/processed/config.json

Metrics:
    - Accuracy: chỉ dùng để tham khảo vì bài toán recommendation bị lệch nhãn.
    - Precision@10
    - Recall@10
    - HitRate@10
    - NDCG@10

Cải thiện chính:
    1. Sample 1/3 train nhưng giữ toàn bộ positive samples để không làm mất tín hiệu mua hàng.
    2. Validation chỉ forward 1 lần/epoch để lấy cả val_loss và ranking metrics.
    3. Dùng AdamW, ReduceLROnPlateau, gradient clipping và AMP nếu có CUDA.
    4. Model thêm user/item/global bias, LayerNorm và embedding dropout.
    5. Lưu đầy đủ checkpoint, scaler, history, metrics, predictions và biểu đồ.

Cách chạy:
    python baseline_train_improved.py
"""

import json
import os
import pickle
import random
import shutil
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
try:
    from torch.amp import GradScaler, autocast
    _NEW_AMP_API = True
except ImportError:
    from torch.cuda.amp import GradScaler, autocast
    _NEW_AMP_API = False
from torch.utils.data import DataLoader, Dataset

from baseline_model_improved import MLPBaseline

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_DIR = "data/processed"
MODEL_DIR = "data/models"

TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
VAL_CSV = os.path.join(DATA_DIR, "val_candidates.csv")
TEST_CSV = os.path.join(DATA_DIR, "test_candidates.csv")
MAPPING_PKL = os.path.join(DATA_DIR, "mappings.pkl")
CONFIG_JSON = os.path.join(DATA_DIR, "config.json")

EMBEDDING_DIM = 64
HIDDEN_DIMS = (256, 128, 64)
DROPOUT = 0.25
EMBEDDING_DROPOUT = 0.05

BATCH_SIZE = 2048
EVAL_BATCH_SIZE = 8192
EPOCHS = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 5.0
K_VALUES = [10]
RANDOM_SEED = 42

# Chỉ dùng 1/3 dữ liệu train để chạy nhanh hơn.
# Set None hoặc 1.0 để dùng toàn bộ train.
TRAIN_SAMPLE_RATIO: Optional[float] = 1 / 3

# Khi sample train, nên giữ toàn bộ label=1 vì positive signal rất ít.
KEEP_ALL_POSITIVES_WHEN_SAMPLING = True

# pos_weight giúp model không quá nghiêng về label 0.
# Capping tránh loss quá lớn nếu dữ liệu mất cân bằng mạnh.
POS_WEIGHT_MAX = 10.0

# EarlyStopping: chọn best model theo val_ndcg@10.
PATIENCE = 4
MIN_DELTA = 1e-6
BEST_METRIC = "val_ndcg@10"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()

# Để chạy ổn trên Windows, để 0. Nếu dùng Linux/server có thể thử 2 hoặc 4.
NUM_WORKERS = 0

DEFAULT_FEATURE_COLUMNS = [
    "history_length",
    "user_item_count",
    "user_unique_items",
    "item_popularity",
    "item_recency",
]


def make_grad_scaler() -> GradScaler:
    """Tạo GradScaler tương thích cả PyTorch mới và cũ."""
    if _NEW_AMP_API:
        return GradScaler("cuda", enabled=USE_AMP)
    return GradScaler(enabled=USE_AMP)


def amp_autocast():
    """Tạo autocast context tương thích cả PyTorch mới và cũ."""
    if _NEW_AMP_API:
        return autocast("cuda", enabled=USE_AMP)
    return autocast(enabled=USE_AMP)


# ─────────────────────────────────────────────
# UTILS
# ─────────────────────────────────────────────
def set_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def check_required_files() -> None:
    required_files = [TRAIN_CSV, VAL_CSV, TEST_CSV, MAPPING_PKL]
    missing = [path for path in required_files if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(
            "Thiếu file processed. Hãy chạy python data_preprocessing.py trước.\n"
            + "\n".join(missing)
        )


def load_feature_columns() -> List[str]:
    if os.path.exists(CONFIG_JSON):
        with open(CONFIG_JSON, "r", encoding="utf-8") as f:
            config = json.load(f)
        return config.get("dense_features", DEFAULT_FEATURE_COLUMNS)
    return DEFAULT_FEATURE_COLUMNS


def validate_columns(df: pd.DataFrame, required_cols: List[str], split_name: str) -> None:
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"{split_name} thiếu các cột: {missing_cols}")


def validate_indices(df: pd.DataFrame, num_users: int, num_items: int, split_name: str) -> None:
    """Kiểm tra user_idx/item_idx có nằm trong range của embedding hay không."""
    min_user, max_user = int(df["user_idx"].min()), int(df["user_idx"].max())
    min_item, max_item = int(df["item_idx"].min()), int(df["item_idx"].max())

    if min_user < 0 or max_user >= num_users:
        raise ValueError(
            f"{split_name}: user_idx ngoài range [0, {num_users - 1}], "
            f"thực tế min={min_user}, max={max_user}"
        )
    if min_item < 0 or max_item >= num_items:
        raise ValueError(
            f"{split_name}: item_idx ngoài range [0, {num_items - 1}], "
            f"thực tế min={min_item}, max={max_item}"
        )


def standardize_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """Chuẩn hóa dense features bằng mean/std từ train set."""
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    means = train_df[feature_cols].mean()
    stds = train_df[feature_cols].std().replace(0, 1).fillna(1)

    for df in [train_df, val_df, test_df]:
        df[feature_cols] = (df[feature_cols] - means) / stds
        df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], 0).fillna(0)

    scaler = {
        "feature_columns": feature_cols,
        "mean": {col: float(means[col]) for col in feature_cols},
        "std": {col: float(stds[col]) for col in feature_cols},
    }

    return train_df, val_df, test_df, scaler


def sample_train_dataframe(
    train_df: pd.DataFrame,
    ratio: Optional[float],
    seed: int = RANDOM_SEED,
    keep_all_positives: bool = True,
) -> pd.DataFrame:
    """
    Sample train để chạy nhanh hơn.

    Với recommendation implicit feedback, label=1 thường ít.
    Nếu random sample trực tiếp 1/3 dữ liệu, ta có thể làm mất nhiều positive samples.
    Vì vậy mặc định giữ toàn bộ positive và chỉ sample bớt negative.
    """
    if ratio is None or ratio >= 1.0:
        return train_df.reset_index(drop=True)

    if ratio <= 0.0:
        raise ValueError("TRAIN_SAMPLE_RATIO phải > 0 hoặc None")

    target_size = max(1, int(len(train_df) * ratio))

    if not keep_all_positives:
        return train_df.sample(n=target_size, random_state=seed).reset_index(drop=True)

    pos_df = train_df[train_df["label"] == 1]
    neg_df = train_df[train_df["label"] == 0]

    if len(pos_df) >= target_size:
        # Trường hợp hiếm: positive nhiều hơn target_size.
        sampled_df = pos_df.sample(n=target_size, random_state=seed)
    else:
        needed_neg = target_size - len(pos_df)
        needed_neg = min(needed_neg, len(neg_df))
        sampled_neg = neg_df.sample(n=needed_neg, random_state=seed) if needed_neg > 0 else neg_df.iloc[0:0]
        sampled_df = pd.concat([pos_df, sampled_neg], axis=0)

    return sampled_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def print_split_stats(name: str, df: pd.DataFrame) -> None:
    num_rows = len(df)
    num_pos = int((df["label"] == 1).sum())
    num_neg = int((df["label"] == 0).sum())
    pos_rate = num_pos / max(num_rows, 1)
    num_users = df["user_idx"].nunique()
    num_items = df["item_idx"].nunique()

    print(
        f"   {name:<5}: rows={num_rows:,} | pos={num_pos:,} | neg={num_neg:,} | "
        f"pos_rate={pos_rate:.4%} | users={num_users:,} | items={num_items:,}"
    )


def make_loader(
    df: pd.DataFrame,
    feature_cols: List[str],
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = RecoDataset(df, feature_cols)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=NUM_WORKERS > 0,
    )


# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
class RecoDataset(Dataset):
    """Dataset cho binary recommendation baseline."""

    def __init__(self, df: pd.DataFrame, feature_cols: List[str]):
        self.user_idx = torch.tensor(df["user_idx"].to_numpy(), dtype=torch.long)
        self.item_idx = torch.tensor(df["item_idx"].to_numpy(), dtype=torch.long)
        self.features = torch.tensor(df[feature_cols].to_numpy(), dtype=torch.float32)
        self.labels = torch.tensor(df["label"].to_numpy(), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return (
            self.user_idx[idx],
            self.item_idx[idx],
            self.features[idx],
            self.labels[idx],
        )


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────
def accuracy(labels: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> float:
    preds = (scores >= threshold).astype(np.int32)
    return float((preds == labels).mean())


def ranking_metrics_at_k(df_pred: pd.DataFrame, k: int = 10) -> Dict[str, float]:
    """
    Tính Precision@K, Recall@K, HitRate@K, NDCG@K theo từng user rồi lấy trung bình.

    df_pred cần có cột:
        user_idx, item_idx, label, score
    """
    precision_list = []
    recall_list = []
    hitrate_list = []
    ndcg_list = []

    total_users = 0
    users_without_positive = 0

    for _, group in df_pred.groupby("user_idx", sort=False):
        total_users += 1

        group = group.sort_values("score", ascending=False)
        top_k = group.head(k)

        labels_at_k = top_k["label"].to_numpy(dtype=np.float32)
        hits = float(labels_at_k.sum())
        total_relevant = float(group["label"].sum())
        denom = min(k, len(group))

        if total_relevant <= 0:
            users_without_positive += 1

        precision = hits / denom if denom > 0 else 0.0
        recall = hits / total_relevant if total_relevant > 0 else 0.0
        hitrate = 1.0 if hits > 0 else 0.0

        # DCG: item đúng ở vị trí càng cao thì điểm càng lớn.
        discounts = 1.0 / np.log2(np.arange(2, len(labels_at_k) + 2))
        dcg = float(np.sum(labels_at_k * discounts)) if len(labels_at_k) > 0 else 0.0

        # IDCG: điểm DCG tốt nhất có thể đạt được của user đó.
        ideal_hits = int(min(total_relevant, k))
        if ideal_hits > 0:
            ideal_discounts = 1.0 / np.log2(np.arange(2, ideal_hits + 2))
            idcg = float(np.sum(ideal_discounts))
            ndcg = dcg / idcg
        else:
            ndcg = 0.0

        precision_list.append(precision)
        recall_list.append(recall)
        hitrate_list.append(hitrate)
        ndcg_list.append(ndcg)

    return {
        f"precision@{k}": float(np.mean(precision_list)) if precision_list else 0.0,
        f"recall@{k}": float(np.mean(recall_list)) if recall_list else 0.0,
        f"hitrate@{k}": float(np.mean(hitrate_list)) if hitrate_list else 0.0,
        f"ndcg@{k}": float(np.mean(ndcg_list)) if ndcg_list else 0.0,
        f"eval_users@{k}": int(total_users),
        f"users_without_positive@{k}": int(users_without_positive),
    }


# ─────────────────────────────────────────────
# TRAIN / EVALUATE
# ─────────────────────────────────────────────
def train_epoch(
    model: MLPBaseline,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: GradScaler,
) -> float:
    model.train()
    total_loss = 0.0

    for user, item, features, label in loader:
        user = user.to(DEVICE, non_blocking=True)
        item = item.to(DEVICE, non_blocking=True)
        features = features.to(DEVICE, non_blocking=True)
        label = label.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with amp_autocast():
            logits = model(user, item, features)
            loss = criterion(logits, label)

        scaler.scale(loss).backward()

        if GRAD_CLIP_NORM is not None and GRAD_CLIP_NORM > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * len(label)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate_dataframe(
    model: MLPBaseline,
    df: pd.DataFrame,
    feature_cols: List[str],
    split_name: str,
    criterion: Optional[nn.Module] = None,
    batch_size: int = EVAL_BATCH_SIZE,
) -> Tuple[Dict[str, float], pd.DataFrame, Optional[float]]:
    """
    Forward đúng 1 lần trên split để lấy:
        - loss nếu có criterion
        - labels/scores
        - ranking metrics

    Cách này nhanh hơn việc tính val_loss rồi lại evaluate ranking ở 2 pass riêng.
    """
    model.eval()

    loader = make_loader(df, feature_cols, batch_size=batch_size, shuffle=False)

    all_labels = []
    all_scores = []
    total_loss = 0.0

    for user, item, features, label in loader:
        user = user.to(DEVICE, non_blocking=True)
        item = item.to(DEVICE, non_blocking=True)
        features = features.to(DEVICE, non_blocking=True)
        label_device = label.to(DEVICE, non_blocking=True)

        with amp_autocast():
            logits = model(user, item, features)
            if criterion is not None:
                loss = criterion(logits, label_device)
                total_loss += loss.item() * len(label)

        scores = torch.sigmoid(logits).detach().cpu().numpy()

        all_labels.append(label.numpy())
        all_scores.append(scores)

    labels = np.concatenate(all_labels)
    scores = np.concatenate(all_scores)

    pred_df = df[["user_idx", "item_idx", "label"]].copy()
    pred_df["score"] = scores

    metrics = {f"{split_name}_accuracy": accuracy(labels, scores)}
    for k in K_VALUES:
        k_metrics = ranking_metrics_at_k(pred_df, k=k)
        for name, value in k_metrics.items():
            metrics[f"{split_name}_{name}"] = value

    avg_loss = total_loss / len(loader.dataset) if criterion is not None else None
    return metrics, pred_df, avg_loss


# ─────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────
def plot_training_history(history: List[Dict], save_dir: str) -> None:
    """
    Vẽ biểu đồ training history:
        - Train Loss vs Val Loss
        - val_ndcg@10
        - val_recall@10
        - val_hitrate@10
    """
    if not history:
        print("Không có history để vẽ.")
        return

    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    val_ndcg = [h["val_ndcg@10"] for h in history]
    val_recall = [h["val_recall@10"] for h in history]
    val_hitrate = [h["val_hitrate@10"] for h in history]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Training History – Improved MLP Baseline (K=10)", fontsize=15, fontweight="bold")

    def _plot(ax, y_data, label, color, ylabel, title, y2_data=None, label2=None, color2=None):
        ax.plot(epochs, y_data, marker="o", color=color, linewidth=2, label=label)
        if y2_data is not None:
            ax.plot(epochs, y2_data, marker="s", color=color2, linewidth=2, linestyle="--", label=label2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_xticks(epochs)

    _plot(
        axes[0, 0],
        train_loss,
        "Train Loss",
        "#E15759",
        "Loss",
        "Train Loss vs Val Loss",
        val_loss,
        "Val Loss",
        "#9467BD",
    )
    _plot(axes[0, 1], val_ndcg, "val_ndcg@10", "#4E79A7", "NDCG@10", "Validation NDCG@10")
    _plot(axes[1, 0], val_recall, "val_recall@10", "#F28E2B", "Recall@10", "Validation Recall@10")
    _plot(axes[1, 1], val_hitrate, "val_hitrate@10", "#59A14F", "HitRate@10", "Validation HitRate@10")

    plt.tight_layout()

    chart_path = os.path.join(save_dir, "training_history_plot_improved.png")
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   {chart_path}")


def save_json(path: str, data: Dict | List) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def print_metric_block(title: str, metrics: Dict[str, float], split: str) -> None:
    print(f"\n{title}")
    print(f"Accuracy tham khảo: {metrics[f'{split}_accuracy']:.4f}")
    for k in K_VALUES:
        print(
            f"Precision@{k}: {metrics[f'{split}_precision@{k}']:.4f} | "
            f"Recall@{k}: {metrics[f'{split}_recall@{k}']:.4f} | "
            f"HitRate@{k}: {metrics[f'{split}_hitrate@{k}']:.4f} | "
            f"NDCG@{k}: {metrics[f'{split}_ndcg@{k}']:.4f} | "
            f"users_without_positive={metrics[f'{split}_users_without_positive@{k}']}"
        )


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main() -> None:
    set_seed(RANDOM_SEED)
    check_required_files()

    print("=" * 78)
    print(" IMPROVED BASELINE TRAINING – MLP Recommendation")
    print(f" Device: {DEVICE} | AMP: {USE_AMP}")
    print("=" * 78)

    feature_cols = load_feature_columns()
    print(f"\nDense features: {feature_cols}")

    with open(MAPPING_PKL, "rb") as f:
        mappings = pickle.load(f)

    num_users = int(mappings["num_users"])
    num_items = int(mappings["num_products"])

    print(f"num_users={num_users:,}, num_items={num_items:,}")

    print("\nĐang đọc processed data...")
    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)
    test_df = pd.read_csv(TEST_CSV)

    required_cols = ["user_idx", "item_idx", "label", *feature_cols]
    validate_columns(train_df, required_cols, "train.csv")
    validate_columns(val_df, required_cols, "val_candidates.csv")
    validate_columns(test_df, required_cols, "test_candidates.csv")

    validate_indices(train_df, num_users, num_items, "train.csv")
    validate_indices(val_df, num_users, num_items, "val_candidates.csv")
    validate_indices(test_df, num_users, num_items, "test_candidates.csv")

    print("\nThống kê trước khi sample:")
    print_split_stats("train", train_df)
    print_split_stats("val", val_df)
    print_split_stats("test", test_df)

    train_df, val_df, test_df, scaler = standardize_features(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        feature_cols=feature_cols,
    )

    original_train_size = len(train_df)
    train_df = sample_train_dataframe(
        train_df=train_df,
        ratio=TRAIN_SAMPLE_RATIO,
        seed=RANDOM_SEED,
        keep_all_positives=KEEP_ALL_POSITIVES_WHEN_SAMPLING,
    )

    print("\nThống kê sau khi sample train:")
    if TRAIN_SAMPLE_RATIO is not None and TRAIN_SAMPLE_RATIO < 1.0:
        print(
            f"   sampled train: {len(train_df):,}/{original_train_size:,} "
            f"({len(train_df) / max(original_train_size, 1):.2%}) | "
            f"keep_all_positives={KEEP_ALL_POSITIVES_WHEN_SAMPLING}"
        )
    print_split_stats("train", train_df)

    train_loader = make_loader(train_df, feature_cols, batch_size=BATCH_SIZE, shuffle=True)

    model = MLPBaseline(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=EMBEDDING_DIM,
        num_features=len(feature_cols),
        hidden_dims=HIDDEN_DIMS,
        dropout=DROPOUT,
        embedding_dropout=EMBEDDING_DROPOUT,
        use_bias_terms=True,
        use_layer_norm=True,
    ).to(DEVICE)

    # Với negative sampling, pos_weight giúp model không bị thiên về label 0.
    num_pos = float((train_df["label"] == 1).sum())
    num_neg = float((train_df["label"] == 0).sum())
    raw_pos_weight = num_neg / max(num_pos, 1.0)
    pos_weight_value = min(raw_pos_weight, POS_WEIGHT_MAX)
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=DEVICE)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=1,
        min_lr=1e-6,
    )

    scaler_amp = make_grad_scaler()

    print(f"\nModel:\n{model}")
    print(f"\nraw_pos_weight={raw_pos_weight:.4f} | used_pos_weight={pos_weight_value:.4f}")
    print(f"Best model metric: {BEST_METRIC}")
    print(f"EarlyStopping patience: {PATIENCE}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "mlp_baseline_improved_best.pt")
    history_path = os.path.join(MODEL_DIR, "training_history_improved.json")
    scaler_path = os.path.join(MODEL_DIR, "feature_scaler_improved.json")
    final_metrics_path = os.path.join(MODEL_DIR, "final_metrics_improved.json")
    train_config_path = os.path.join(MODEL_DIR, "train_config_improved.json")

    # Lưu bản model code đi kèm checkpoint để dễ tái lập.
    try:
        shutil.copyfile("baseline_model_improved.py", os.path.join(MODEL_DIR, "baseline_model_improved.py"))
    except OSError:
        pass

    train_config = {
        "embedding_dim": EMBEDDING_DIM,
        "hidden_dims": list(HIDDEN_DIMS),
        "dropout": DROPOUT,
        "embedding_dropout": EMBEDDING_DROPOUT,
        "batch_size": BATCH_SIZE,
        "eval_batch_size": EVAL_BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "grad_clip_norm": GRAD_CLIP_NORM,
        "k_values": K_VALUES,
        "train_sample_ratio": TRAIN_SAMPLE_RATIO,
        "keep_all_positives_when_sampling": KEEP_ALL_POSITIVES_WHEN_SAMPLING,
        "pos_weight_max": POS_WEIGHT_MAX,
        "best_metric": BEST_METRIC,
        "patience": PATIENCE,
        "min_delta": MIN_DELTA,
        "feature_cols": feature_cols,
        "device": str(DEVICE),
        "use_amp": USE_AMP,
    }
    save_json(train_config_path, train_config)

    best_val_metric = -1.0
    patience_counter = 0
    training_history = []

    print("\nBắt đầu training...")
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, scaler_amp)

        val_metrics, _, val_loss = evaluate_dataframe(
            model=model,
            df=val_df,
            feature_cols=feature_cols,
            split_name="val",
            criterion=criterion,
            batch_size=EVAL_BATCH_SIZE,
        )

        current_val_metric = float(val_metrics[BEST_METRIC])
        scheduler.step(current_val_metric)
        current_lr = optimizer.param_groups[0]["lr"]

        epoch_result = {
            "epoch": epoch,
            "lr": round(current_lr, 10),
            "train_loss": round(train_loss, 6),
            "val_loss": round(float(val_loss), 6),
            **{
                name: (round(value, 6) if isinstance(value, float) else value)
                for name, value in val_metrics.items()
            },
        }
        training_history.append(epoch_result)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"lr={current_lr:.2e} | "
            f"train_loss={train_loss:.5f} | "
            f"val_loss={val_loss:.5f} | "
            f"val_recall@10={val_metrics['val_recall@10']:.4f} | "
            f"val_hitrate@10={val_metrics['val_hitrate@10']:.4f} | "
            f"val_ndcg@10={current_val_metric:.4f}"
        )

        if current_val_metric > best_val_metric + MIN_DELTA:
            best_val_metric = current_val_metric
            patience_counter = 0

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "best_metric": BEST_METRIC,
                "best_val_metric": best_val_metric,
                "feature_cols": feature_cols,
                "train_config": train_config,
                "num_users": num_users,
                "num_items": num_items,
            }
            torch.save(checkpoint, model_path)
            print(f"   → Lưu best model theo {BEST_METRIC}={best_val_metric:.4f}")
        else:
            patience_counter += 1
            print(f"   → Không cải thiện {BEST_METRIC}. patience={patience_counter}/{PATIENCE}")

            if patience_counter >= PATIENCE:
                print(f"\nEarlyStopping: dừng tại epoch {epoch} vì {BEST_METRIC} không cải thiện.")
                break

    if not os.path.exists(model_path):
        raise RuntimeError("Không tìm thấy checkpoint best model. Có thể training đã lỗi trước khi lưu model.")

    # Load best model rồi evaluate final trên val và test.
    print("\n" + "=" * 78)
    print(" FINAL EVALUATION – BEST IMPROVED MODEL")
    print("=" * 78)

    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    val_metrics, val_pred_df, val_loss = evaluate_dataframe(
        model,
        val_df,
        feature_cols,
        split_name="val",
        criterion=criterion,
        batch_size=EVAL_BATCH_SIZE,
    )
    test_metrics, test_pred_df, test_loss = evaluate_dataframe(
        model,
        test_df,
        feature_cols,
        split_name="test",
        criterion=criterion,
        batch_size=EVAL_BATCH_SIZE,
    )

    final_metrics = {
        **{
            name: (round(value, 6) if isinstance(value, float) else value)
            for name, value in val_metrics.items()
        },
        **{
            name: (round(value, 6) if isinstance(value, float) else value)
            for name, value in test_metrics.items()
        },
        "val_loss": round(float(val_loss), 6),
        "test_loss": round(float(test_loss), 6),
        "best_metric": BEST_METRIC,
        "best_val_metric": round(float(best_val_metric), 6),
        "best_epoch": int(checkpoint["epoch"]),
    }

    print_metric_block("Validation metrics:", val_metrics, "val")
    print_metric_block("Test metrics:", test_metrics, "test")

    val_pred_path = os.path.join(MODEL_DIR, "val_predictions_improved.csv")
    test_pred_path = os.path.join(MODEL_DIR, "test_predictions_improved.csv")

    val_pred_df.to_csv(val_pred_path, index=False)
    test_pred_df.to_csv(test_pred_path, index=False)

    save_json(history_path, training_history)
    save_json(scaler_path, scaler)
    save_json(final_metrics_path, final_metrics)

    print("\nĐã lưu:")
    print(f"   {model_path}")
    print(f"   {val_pred_path}")
    print(f"   {test_pred_path}")
    print(f"   {history_path}")
    print(f"   {scaler_path}")
    print(f"   {final_metrics_path}")
    print(f"   {train_config_path}")

    print("\nVẽ biểu đồ training history...")
    plot_training_history(training_history, MODEL_DIR)

    print("\nHoàn tất huấn luyện improved baseline!")


if __name__ == "__main__":
    main()
