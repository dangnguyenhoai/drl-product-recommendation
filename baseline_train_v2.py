"""
baseline_train.py
=================
Huấn luyện và đánh giá mô hình MLP Baseline trên dữ liệu Instacart đã xử lý.

Input từ data_preprocessing.py:
    data/processed/train.csv
    data/processed/val_candidates.csv
    data/processed/test_candidates.csv
    data/processed/mappings.pkl
    data/processed/config.json

Metrics:
    - Accuracy
    - Precision@K
    - Recall@K
    - HitRate@K
    - NDCG@K

Ghi chú:
    Instacart không có click log thật, nên hành vi mua hàng được dùng như
    implicit positive feedback. Vì vậy file này bỏ CTR@K để tránh trùng ý nghĩa
    với Precision@K trong offline evaluation.

Cách chạy:
    python baseline_train.py
"""

import json
import os
import pickle
import random
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from baseline_model import MLPBaseline

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

EMBEDDING_DIM = 32
BATCH_SIZE = 1024
EPOCHS = 10
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
K_VALUES = [10]
RANDOM_SEED = 42

# Chỉ dùng 1/3 dữ liệu train để chạy nhanh hơn (set None để dùng toàn bộ)
TRAIN_SAMPLE_RATIO = 1 / 3

# EarlyStopping: nếu val_ndcg@10 không cải thiện sau PATIENCE epoch thì dừng.
PATIENCE = 3
MIN_DELTA = 1e-6
BEST_METRIC = "val_ndcg@10"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_FEATURE_COLUMNS = [
    "history_length",
    "user_item_count",
    "user_unique_items",
    "item_popularity",
    "item_recency",
]


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
    """Kiểm tra train/val/test có đủ cột cần thiết hay không."""
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"{split_name} thiếu các cột: {missing_cols}")


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


# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
class RecoDataset(Dataset):
    """Dataset cho binary recommendation baseline."""

    def __init__(self, df: pd.DataFrame, feature_cols: List[str]):
        self.user_idx = torch.tensor(df["user_idx"].values, dtype=torch.long)
        self.item_idx = torch.tensor(df["item_idx"].values, dtype=torch.long)
        self.features = torch.tensor(df[feature_cols].values, dtype=torch.float32)
        self.labels = torch.tensor(df["label"].values, dtype=torch.float32)

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

    for _, group in df_pred.groupby("user_idx", sort=False):
        group = group.sort_values("score", ascending=False)
        top_k = group.head(k)

        labels_at_k = top_k["label"].to_numpy(dtype=np.float32)
        hits = float(labels_at_k.sum())
        total_relevant = float(group["label"].sum())
        denom = min(k, len(group))

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
    }


def evaluate_ranking_dataframe(
    model: MLPBaseline,
    df: pd.DataFrame,
    feature_cols: List[str],
    batch_size: int = 4096,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    dataset = RecoDataset(df, feature_cols)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    labels, scores = evaluate_scores(model, loader)

    pred_df = df[["user_idx", "item_idx", "label"]].copy()
    pred_df["score"] = scores

    return pred_df, labels, scores


# ─────────────────────────────────────────────
# TRAIN / EVALUATE
# ─────────────────────────────────────────────
def train_epoch(
    model: MLPBaseline,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> float:
    model.train()
    total_loss = 0.0

    for user, item, features, label in loader:
        user = user.to(DEVICE)
        item = item.to(DEVICE)
        features = features.to(DEVICE)
        label = label.to(DEVICE)

        optimizer.zero_grad()
        logits = model(user, item, features)
        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(label)

    return total_loss / len(loader.dataset)



@torch.no_grad()
def compute_loss(
    model: MLPBaseline,
    loader: DataLoader,
    criterion: nn.Module,
) -> float:
    """Tính val loss trên toàn bộ loader mà không cập nhật trọng số."""
    model.eval()
    total_loss = 0.0

    for user, item, features, label in loader:
        user = user.to(DEVICE)
        item = item.to(DEVICE)
        features = features.to(DEVICE)
        label = label.to(DEVICE)

        logits = model(user, item, features)
        loss = criterion(logits, label)
        total_loss += loss.item() * len(label)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate_scores(model: MLPBaseline, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_labels = []
    all_scores = []

    for user, item, features, label in loader:
        user = user.to(DEVICE)
        item = item.to(DEVICE)
        features = features.to(DEVICE)

        logits = model(user, item, features)
        scores = torch.sigmoid(logits).detach().cpu().numpy()

        all_labels.append(label.numpy())
        all_scores.append(scores)

    return np.concatenate(all_labels), np.concatenate(all_scores)


def evaluate_all_metrics(
    model: MLPBaseline,
    df: pd.DataFrame,
    feature_cols: List[str],
    split_name: str,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    pred_df, labels, scores = evaluate_ranking_dataframe(model, df, feature_cols)

    metrics = {f"{split_name}_accuracy": accuracy(labels, scores)}

    for k in K_VALUES:
        k_metrics = ranking_metrics_at_k(pred_df, k=k)
        for name, value in k_metrics.items():
            metrics[f"{split_name}_{name}"] = value

    return metrics, pred_df


# ─────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────
def plot_training_history(history: List[Dict], save_dir: str) -> None:
    """
    Vẽ biểu đồ training history:
        - Panel 1: Train Loss vs Val Loss
        - Panel 2: val_ndcg@10
        - Panel 3: val_recall@10
        - Panel 4: val_hitrate@10
    """
    epochs      = [h["epoch"]           for h in history]
    train_loss  = [h["train_loss"]      for h in history]
    val_loss    = [h["val_loss"]        for h in history]
    val_ndcg    = [h["val_ndcg@10"]     for h in history]
    val_recall  = [h["val_recall@10"]   for h in history]
    val_hitrate = [h["val_hitrate@10"]  for h in history]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Training History – MLP Baseline (K=10)", fontsize=15, fontweight="bold")

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

    # ── Panel 1: Train Loss vs Val Loss ─────────────────────────
    _plot(axes[0, 0], train_loss, "Train Loss", "#E15759", "Loss",
          "Train Loss vs Val Loss", val_loss, "Val Loss", "#9467BD")

    # ── Panel 2: NDCG@10 ────────────────────────────────────────
    _plot(axes[0, 1], val_ndcg, "val_ndcg@10", "#4E79A7", "NDCG@10", "Validation NDCG@10")

    # ── Panel 3: Recall@10 ──────────────────────────────────────
    _plot(axes[1, 0], val_recall, "val_recall@10", "#F28E2B", "Recall@10", "Validation Recall@10")

    # ── Panel 4: HitRate@10 ─────────────────────────────────────
    _plot(axes[1, 1], val_hitrate, "val_hitrate@10", "#59A14F", "HitRate@10", "Validation HitRate@10")

    plt.tight_layout()

    chart_path = os.path.join(save_dir, "training_history_plot.png")
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   {chart_path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main() -> None:
    set_seed(RANDOM_SEED)
    check_required_files()

    print("=" * 70)
    print(" BASELINE TRAINING – MLP Recommendation")
    print(f" Device: {DEVICE}")
    print("=" * 70)

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

    train_df, val_df, test_df, scaler = standardize_features(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        feature_cols=feature_cols,
    )

    # Sample 1/3 dữ liệu train để tăng tốc độ thực nghiệm.
    if TRAIN_SAMPLE_RATIO is not None and TRAIN_SAMPLE_RATIO < 1.0:
        train_df = train_df.sample(frac=TRAIN_SAMPLE_RATIO, random_state=RANDOM_SEED).reset_index(drop=True)

    print(f"   train: {len(train_df):,}" + (f" (sampled {TRAIN_SAMPLE_RATIO:.0%})" if TRAIN_SAMPLE_RATIO else ""))
    print(f"   val  : {len(val_df):,}")
    print(f"   test : {len(test_df):,}")

    train_dataset = RecoDataset(train_df, feature_cols)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    val_dataset = RecoDataset(val_df, feature_cols)
    val_loader = DataLoader(
        val_dataset,
        batch_size=4096,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    model = MLPBaseline(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=EMBEDDING_DIM,
        num_features=len(feature_cols),
    ).to(DEVICE)

    # Với negative sampling, pos_weight giúp model không bị thiên về label 0.
    num_pos = float((train_df["label"] == 1).sum())
    num_neg = float((train_df["label"] == 0).sum())
    pos_weight_value = num_neg / max(num_pos, 1.0)
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=DEVICE)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    print(f"\nModel:\n{model}")
    print(f"\npos_weight={pos_weight_value:.4f}")
    print(f"Best model metric: {BEST_METRIC}")
    print(f"EarlyStopping patience: {PATIENCE}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "mlp_baseline.pt")
    history_path = os.path.join(MODEL_DIR, "training_history.json")
    scaler_path = os.path.join(MODEL_DIR, "feature_scaler.json")
    final_metrics_path = os.path.join(MODEL_DIR, "final_metrics.json")

    best_val_metric = -1.0
    patience_counter = 0
    training_history = []

    print("\nBắt đầu training...")
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_loss = compute_loss(model, val_loader, criterion)

        val_metrics, _ = evaluate_all_metrics(
            model=model,
            df=val_df,
            feature_cols=feature_cols,
            split_name="val",
        )

        current_val_metric = val_metrics[BEST_METRIC]

        epoch_result = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            **{name: round(value, 6) for name, value in val_metrics.items()},
        }
        training_history.append(epoch_result)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train_loss={train_loss:.5f} | "
            f"val_loss={val_loss:.5f} | "
            f"val_acc={val_metrics['val_accuracy']:.4f} | "
            f"val_recall@10={val_metrics['val_recall@10']:.4f} | "
            f"val_hitrate@10={val_metrics['val_hitrate@10']:.4f} | "
            f"val_ndcg@10={current_val_metric:.4f}"
        )

        if current_val_metric > best_val_metric + MIN_DELTA:
            best_val_metric = current_val_metric
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
            print(f"   → Lưu best model theo {BEST_METRIC}={best_val_metric:.4f}")
        else:
            patience_counter += 1
            print(f"   → Không cải thiện {BEST_METRIC}. patience={patience_counter}/{PATIENCE}")

            if patience_counter >= PATIENCE:
                print(f"\nEarlyStopping: dừng tại epoch {epoch} vì {BEST_METRIC} không cải thiện.")
                break

    # Load best model rồi evaluate final trên val và test.
    print("\n" + "=" * 70)
    print(" FINAL EVALUATION – BEST MODEL")
    print("=" * 70)

    model.load_state_dict(torch.load(model_path, map_location=DEVICE))

    val_metrics, val_pred_df = evaluate_all_metrics(model, val_df, feature_cols, split_name="val")
    test_metrics, test_pred_df = evaluate_all_metrics(model, test_df, feature_cols, split_name="test")

    final_metrics = {
        **{name: round(value, 6) for name, value in val_metrics.items()},
        **{name: round(value, 6) for name, value in test_metrics.items()},
        "best_metric": BEST_METRIC,
        "best_val_metric": round(best_val_metric, 6),
    }

    print("\nValidation metrics:")
    print(f"Accuracy: {val_metrics['val_accuracy']:.4f}")
    for k in K_VALUES:
        print(
            f"Precision@{k}: {val_metrics[f'val_precision@{k}']:.4f} | "
            f"Recall@{k}: {val_metrics[f'val_recall@{k}']:.4f} | "
            f"HitRate@{k}: {val_metrics[f'val_hitrate@{k}']:.4f} | "
            f"NDCG@{k}: {val_metrics[f'val_ndcg@{k}']:.4f}"
        )

    print("\nTest metrics:")
    print(f"Accuracy: {test_metrics['test_accuracy']:.4f}")
    for k in K_VALUES:
        print(
            f"Precision@{k}: {test_metrics[f'test_precision@{k}']:.4f} | "
            f"Recall@{k}: {test_metrics[f'test_recall@{k}']:.4f} | "
            f"HitRate@{k}: {test_metrics[f'test_hitrate@{k}']:.4f} | "
            f"NDCG@{k}: {test_metrics[f'test_ndcg@{k}']:.4f}"
        )

    # Save outputs.
    val_pred_path = os.path.join(MODEL_DIR, "val_predictions.csv")
    test_pred_path = os.path.join(MODEL_DIR, "test_predictions.csv")

    val_pred_df.to_csv(val_pred_path, index=False)
    test_pred_df.to_csv(test_pred_path, index=False)

    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(training_history, f, indent=2, ensure_ascii=False)

    with open(scaler_path, "w", encoding="utf-8") as f:
        json.dump(scaler, f, indent=2, ensure_ascii=False)

    with open(final_metrics_path, "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2, ensure_ascii=False)

    print("\nĐã lưu:")
    print(f"   {model_path}")
    print(f"   {val_pred_path}")
    print(f"   {test_pred_path}")
    print(f"   {history_path}")
    print(f"   {scaler_path}")
    print(f"   {final_metrics_path}")

    # Vẽ biểu đồ training history.
    print("\nVẽ biểu đồ training history...")
    plot_training_history(training_history, MODEL_DIR)

    print("\nHoàn tất huấn luyện baseline!")


if __name__ == "__main__":
    main()