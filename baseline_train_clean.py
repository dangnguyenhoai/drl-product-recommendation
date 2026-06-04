# %%
# baseline_train_clean.py
# A clean, notebook-style training pipeline for the Instacart recommendation baseline.
#
# How to use:
#   - Open this file in VS Code.
#   - Run cell by cell using the "Run Cell" button above each # %% block.
#   - Start with FAST_MODE=True. After the flow is clear, set FAST_MODE=False.

# %%
# 1. Imports and configuration

import json
import os
import pickle
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


RANDOM_SEED = 42

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("data/models/baseline_clean")

TRAIN_CSV = DATA_DIR / "train.csv"
VAL_CSV = DATA_DIR / "val_candidates.csv"
TEST_CSV = DATA_DIR / "test_candidates.csv"
MAPPINGS_PKL = DATA_DIR / "mappings.pkl"
CONFIG_JSON = DATA_DIR / "config.json"

FAST_MODE = True
TRAIN_SAMPLE_FRAC = 0.20 if FAST_MODE else 1.0
MAX_EVAL_USERS = 10_000 if FAST_MODE else None

EMBEDDING_DIM = 32
HIDDEN_DIMS = (128, 64)
DROPOUT = 0.20
BATCH_SIZE = 2048
EVAL_BATCH_SIZE = 8192
EPOCHS = 5 if FAST_MODE else 10
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
K = 10
USE_POS_WEIGHT = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed()
MODEL_DIR.mkdir(parents=True, exist_ok=True)

print(f"Device: {DEVICE}")
print(f"FAST_MODE={FAST_MODE}, TRAIN_SAMPLE_FRAC={TRAIN_SAMPLE_FRAC}, MAX_EVAL_USERS={MAX_EVAL_USERS}")

# %%
# 2. Load config and mappings

required_files = [TRAIN_CSV, VAL_CSV, TEST_CSV, MAPPINGS_PKL, CONFIG_JSON]
missing = [str(path) for path in required_files if not path.exists()]
if missing:
    raise FileNotFoundError("Missing processed files. Run data_preprocessing.py first:\n" + "\n".join(missing))

with CONFIG_JSON.open("r", encoding="utf-8") as f:
    config = json.load(f)

with MAPPINGS_PKL.open("rb") as f:
    mappings = pickle.load(f)

FEATURE_COLS: List[str] = config["dense_features"]
NUM_USERS = int(mappings["num_users"])
NUM_ITEMS = int(mappings["num_products"])

ALL_COLS = ["user_idx", "item_idx", "label", *FEATURE_COLS]

print("Feature columns:", FEATURE_COLS)
print(f"NUM_USERS={NUM_USERS:,}, NUM_ITEMS={NUM_ITEMS:,}")

# %%
# 3. Load processed train/validation/test data
#
# In FAST_MODE we still load the CSV, but then sample train and limit eval users.
# This keeps the code simple and makes every step visible.

train_df = pd.read_csv(TRAIN_CSV, usecols=ALL_COLS)
val_df = pd.read_csv(VAL_CSV, usecols=ALL_COLS)
test_df = pd.read_csv(TEST_CSV, usecols=ALL_COLS)

print(f"Raw train rows: {len(train_df):,}")
print(f"Raw val rows  : {len(val_df):,}")
print(f"Raw test rows : {len(test_df):,}")

# %%
# 4. Basic data checks

def print_split_stats(name: str, df: pd.DataFrame) -> None:
    num_rows = len(df)
    num_pos = int(df["label"].sum())
    num_neg = int((df["label"] == 0).sum())
    pos_rate = num_pos / max(num_rows, 1)
    num_users = df["user_idx"].nunique()
    num_items = df["item_idx"].nunique()
    print(
        f"{name:<5} rows={num_rows:,} | pos={num_pos:,} | neg={num_neg:,} | "
        f"pos_rate={pos_rate:.4%} | users={num_users:,} | items={num_items:,}"
    )


for split_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
    assert df["user_idx"].between(0, NUM_USERS - 1).all(), f"{split_name}: user_idx out of range"
    assert df["item_idx"].between(0, NUM_ITEMS - 1).all(), f"{split_name}: item_idx out of range"
    assert set(df["label"].unique()).issubset({0, 1}), f"{split_name}: label must be 0/1"
    assert not df[FEATURE_COLS].isna().any().any(), f"{split_name}: feature has NaN"

print_split_stats("train", train_df)
print_split_stats("val", val_df)
print_split_stats("test", test_df)

# %%
# 5. Keep validation/test small in FAST_MODE

def limit_eval_users(df: pd.DataFrame, max_users: Optional[int], seed: int = RANDOM_SEED) -> pd.DataFrame:
    if max_users is None:
        return df.reset_index(drop=True)

    users = df["user_idx"].drop_duplicates().to_numpy()
    if len(users) <= max_users:
        return df.reset_index(drop=True)

    rng = np.random.default_rng(seed)
    picked_users = rng.choice(users, size=max_users, replace=False)
    return df[df["user_idx"].isin(picked_users)].reset_index(drop=True)


val_eval_df = limit_eval_users(val_df, MAX_EVAL_USERS)
test_eval_df = limit_eval_users(test_df, MAX_EVAL_USERS)

print_split_stats("val*", val_eval_df)
print_split_stats("test*", test_eval_df)

# %%
# 6. Heuristic baselines
#
# This is the most important sanity check.
# If the neural model cannot beat user_item_count, the training setup needs work.

def ranking_metrics_at_k(df_pred: pd.DataFrame, k: int = 10) -> Dict[str, float]:
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

        precision = hits / k
        recall = hits / total_relevant if total_relevant > 0 else 0.0
        hitrate = 1.0 if hits > 0 else 0.0

        discounts = 1.0 / np.log2(np.arange(2, len(labels_at_k) + 2))
        dcg = float(np.sum(labels_at_k * discounts))

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
        f"precision@{k}": float(np.mean(precision_list)),
        f"recall@{k}": float(np.mean(recall_list)),
        f"hitrate@{k}": float(np.mean(hitrate_list)),
        f"ndcg@{k}": float(np.mean(ndcg_list)),
    }


def evaluate_heuristic(df: pd.DataFrame, score_col: str, k: int = K) -> Dict[str, float]:
    pred_df = df[["user_idx", "item_idx", "label", score_col]].copy()
    pred_df = pred_df.rename(columns={score_col: "score"})
    return ranking_metrics_at_k(pred_df, k=k)


heuristic_cols = ["user_item_count", "item_recency", "item_popularity"]
for col in heuristic_cols:
    metrics = evaluate_heuristic(val_eval_df, col)
    print(f"VAL heuristic={col:<16} " + " | ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

# %%
# 7. Sample train while preserving positive/negative ratio
#
# Avoid keeping all positives and dropping too many negatives.
# We want the sampled train distribution to stay close to the original train distribution.

def sample_train_preserve_ratio(
    df: pd.DataFrame,
    frac: float,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    if frac >= 1.0:
        return df.reset_index(drop=True)
    if frac <= 0.0:
        raise ValueError("frac must be positive")

    pos_df = df[df["label"] == 1]
    neg_df = df[df["label"] == 0]

    pos_sample = pos_df.sample(frac=frac, random_state=seed)
    neg_sample = neg_df.sample(frac=frac, random_state=seed)

    sampled = pd.concat([pos_sample, neg_sample], axis=0)
    sampled = sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return sampled


train_work_df = sample_train_preserve_ratio(train_df, TRAIN_SAMPLE_FRAC)
print_split_stats("train*", train_work_df)

# Free memory in notebook if desired.
if FAST_MODE:
    del train_df

# %%
# 8. Standardize dense features using train statistics only

def standardize_features(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    test_data: pd.DataFrame,
    feature_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    train_data = train_data.copy()
    val_data = val_data.copy()
    test_data = test_data.copy()

    means = train_data[feature_cols].mean()
    stds = train_data[feature_cols].std().replace(0, 1).fillna(1)

    for data in [train_data, val_data, test_data]:
        data[feature_cols] = (data[feature_cols] - means) / stds
        data[feature_cols] = data[feature_cols].replace([np.inf, -np.inf], 0).fillna(0)

    scaler = {
        "feature_columns": feature_cols,
        "mean": {col: float(means[col]) for col in feature_cols},
        "std": {col: float(stds[col]) for col in feature_cols},
    }
    return train_data, val_data, test_data, scaler


train_work_df, val_eval_df, test_eval_df, scaler = standardize_features(
    train_work_df,
    val_eval_df,
    test_eval_df,
    FEATURE_COLS,
)

print(train_work_df[FEATURE_COLS].describe().round(3))

# %%
# 9. Dataset and DataLoader

class RecoDataset(Dataset):
    def __init__(self, df: pd.DataFrame, feature_cols: List[str]):
        self.user_idx = torch.tensor(df["user_idx"].to_numpy(), dtype=torch.long)
        self.item_idx = torch.tensor(df["item_idx"].to_numpy(), dtype=torch.long)
        self.features = torch.tensor(df[feature_cols].to_numpy(), dtype=torch.float32)
        self.labels = torch.tensor(df["label"].to_numpy(), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.user_idx[idx], self.item_idx[idx], self.features[idx], self.labels[idx]


def make_loader(df: pd.DataFrame, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        RecoDataset(df, FEATURE_COLS),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


train_loader = make_loader(train_work_df, BATCH_SIZE, shuffle=True)
val_loader = make_loader(val_eval_df, EVAL_BATCH_SIZE, shuffle=False)
test_loader = make_loader(test_eval_df, EVAL_BATCH_SIZE, shuffle=False)

print(f"Train batches: {len(train_loader):,}")
print(f"Val batches  : {len(val_loader):,}")

# %%
# 10. Define a simple MLP recommender

class MLPRecommender(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        num_features: int,
        embedding_dim: int = 32,
        hidden_dims: Tuple[int, ...] = (128, 64),
        dropout: float = 0.2,
    ):
        super().__init__()

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))

        input_dim = embedding_dim * 2 + num_features
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.mlp = nn.Sequential(*layers)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.user_embedding.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, user_idx: torch.Tensor, item_idx: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        user_vec = self.user_embedding(user_idx)
        item_vec = self.item_embedding(item_idx)
        x = torch.cat([user_vec, item_vec, features], dim=1)
        logits = self.mlp(x).squeeze(1)
        logits = logits + self.user_bias(user_idx).squeeze(1)
        logits = logits + self.item_bias(item_idx).squeeze(1)
        logits = logits + self.global_bias
        return logits


model = MLPRecommender(
    num_users=NUM_USERS,
    num_items=NUM_ITEMS,
    num_features=len(FEATURE_COLS),
    embedding_dim=EMBEDDING_DIM,
    hidden_dims=HIDDEN_DIMS,
    dropout=DROPOUT,
).to(DEVICE)

print(model)

# %%
# 11. Training and evaluation helpers

def binary_accuracy(labels: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> float:
    preds = (scores >= threshold).astype(np.int32)
    return float((preds == labels).mean())


@torch.no_grad()
def predict_scores(model: nn.Module, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    labels_all = []
    scores_all = []

    for user_idx, item_idx, features, labels in loader:
        user_idx = user_idx.to(DEVICE, non_blocking=True)
        item_idx = item_idx.to(DEVICE, non_blocking=True)
        features = features.to(DEVICE, non_blocking=True)

        logits = model(user_idx, item_idx, features)
        scores = torch.sigmoid(logits).detach().cpu().numpy()

        labels_all.append(labels.numpy())
        scores_all.append(scores)

    return np.concatenate(labels_all), np.concatenate(scores_all)


def evaluate_model(model: nn.Module, df: pd.DataFrame, loader: DataLoader, split_name: str) -> Dict[str, float]:
    labels, scores = predict_scores(model, loader)

    pred_df = df[["user_idx", "item_idx", "label"]].copy()
    pred_df["score"] = scores

    metrics = {f"{split_name}_accuracy": binary_accuracy(labels, scores)}
    rank_metrics = ranking_metrics_at_k(pred_df, k=K)
    for name, value in rank_metrics.items():
        metrics[f"{split_name}_{name}"] = value
    return metrics


def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer) -> float:
    model.train()
    total_loss = 0.0

    for user_idx, item_idx, features, labels in loader:
        user_idx = user_idx.to(DEVICE, non_blocking=True)
        item_idx = item_idx.to(DEVICE, non_blocking=True)
        features = features.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(user_idx, item_idx, features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)

    return total_loss / len(loader.dataset)

# %%
# 12. Train

num_pos = float((train_work_df["label"] == 1).sum())
num_neg = float((train_work_df["label"] == 0).sum())
pos_weight_value = num_neg / max(num_pos, 1.0)

if USE_POS_WEIGHT:
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=DEVICE)
else:
    pos_weight = None

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

print(f"pos_weight={pos_weight_value:.4f} | USE_POS_WEIGHT={USE_POS_WEIGHT}")

history = []
best_val_ndcg = -1.0
best_model_path = MODEL_DIR / "mlp_clean_best.pt"

for epoch in range(1, EPOCHS + 1):
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
    val_metrics = evaluate_model(model, val_eval_df, val_loader, "val")
    val_ndcg = val_metrics[f"val_ndcg@{K}"]

    row = {
        "epoch": epoch,
        "train_loss": train_loss,
        **val_metrics,
    }
    history.append(row)

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"loss={train_loss:.5f} | "
        f"val_precision@{K}={val_metrics[f'val_precision@{K}']:.4f} | "
        f"val_recall@{K}={val_metrics[f'val_recall@{K}']:.4f} | "
        f"val_hitrate@{K}={val_metrics[f'val_hitrate@{K}']:.4f} | "
        f"val_ndcg@{K}={val_ndcg:.4f}"
    )

    if val_ndcg > best_val_ndcg:
        best_val_ndcg = val_ndcg
        torch.save(model.state_dict(), best_model_path)
        print(f"  saved best model: {best_model_path}")

# %%
# 13. Final evaluation on validation and test

model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))

val_metrics = evaluate_model(model, val_eval_df, val_loader, "val")
test_metrics = evaluate_model(model, test_eval_df, test_loader, "test")

final_metrics = {
    **{k: round(float(v), 6) for k, v in val_metrics.items()},
    **{k: round(float(v), 6) for k, v in test_metrics.items()},
    "best_val_ndcg": round(float(best_val_ndcg), 6),
    "fast_mode": FAST_MODE,
    "train_sample_frac": TRAIN_SAMPLE_FRAC,
    "max_eval_users": MAX_EVAL_USERS,
}

print(json.dumps(final_metrics, indent=2))

# %%
# 14. Save artifacts

with (MODEL_DIR / "history.json").open("w", encoding="utf-8") as f:
    json.dump([{k: round(float(v), 6) if isinstance(v, (float, np.floating)) else v for k, v in row.items()} for row in history], f, indent=2)

with (MODEL_DIR / "final_metrics.json").open("w", encoding="utf-8") as f:
    json.dump(final_metrics, f, indent=2)

with (MODEL_DIR / "feature_scaler.json").open("w", encoding="utf-8") as f:
    json.dump(scaler, f, indent=2)

run_config = {
    "embedding_dim": EMBEDDING_DIM,
    "hidden_dims": list(HIDDEN_DIMS),
    "dropout": DROPOUT,
    "batch_size": BATCH_SIZE,
    "eval_batch_size": EVAL_BATCH_SIZE,
    "epochs": EPOCHS,
    "learning_rate": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
    "feature_cols": FEATURE_COLS,
    "use_pos_weight": USE_POS_WEIGHT,
    "fast_mode": FAST_MODE,
    "train_sample_frac": TRAIN_SAMPLE_FRAC,
    "max_eval_users": MAX_EVAL_USERS,
}

with (MODEL_DIR / "run_config.json").open("w", encoding="utf-8") as f:
    json.dump(run_config, f, indent=2)

print(f"Saved artifacts to: {MODEL_DIR}")

# %%
# 15. What to check after training
#
# 1. Compare model val_ndcg@10 with heuristic user_item_count.
# 2. If the model is lower, do not increase layers first.
# 3. Improve preprocessing negative sampling:
#      - include seen-but-not-target items as hard negatives
#      - include same-aisle/same-department negatives
# 4. Then rerun this clean pipeline.

