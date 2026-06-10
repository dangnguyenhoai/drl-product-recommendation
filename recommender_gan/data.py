from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


History = Dict[int, List[int]]


@dataclass
class SplitConfig:
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    min_history_len: int = 11


def preprocess_instacart(
    raw_dir: str | Path,
    output_path: str | Path,
    top_n_items: int = 1000,
    n_users: int | None = None,
    min_history_len: int = 5,
) -> History:
    raw_dir = Path(raw_dir)
    output_path = Path(output_path)

    required = {
        "orders": raw_dir / "orders.csv",
        "products": raw_dir / "products.csv",
        "prior": raw_dir / "order_products__prior.csv",
        "train": raw_dir / "order_products__train.csv",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Thieu file raw CSV:\n"
            + "\n".join(f"- {path}" for path in missing)
            + "\n\nHay copy 4 file CSV vao data/raw hoac chay voi --raw_dir <thu_muc_chua_csv>."
        )

    orders = pd.read_csv(required["orders"])
    pd.read_csv(required["products"])  # Kept as an input check; product metadata is not needed here.
    prior = pd.read_csv(required["prior"])
    train = pd.read_csv(required["train"])

    order_products = pd.concat([prior, train], ignore_index=True)
    merged = order_products.merge(
        orders[["order_id", "user_id", "order_number"]],
        on="order_id",
        how="inner",
    )

    sort_cols = ["user_id", "order_number"]
    if "add_to_cart_order" in merged.columns:
        sort_cols.append("add_to_cart_order")
    merged = merged.sort_values(sort_cols)

    top_items = merged["product_id"].value_counts().head(top_n_items).index.tolist()
    item_to_index = {product_id: idx for idx, product_id in enumerate(top_items)}

    all_users = sorted(merged["user_id"].unique())
    selected_users = all_users[:n_users] if n_users is not None else all_users
    filtered = merged[
        merged["user_id"].isin(set(selected_users))
        & merged["product_id"].isin(set(top_items))
    ].copy()

    raw_history = filtered.groupby("user_id")["product_id"].apply(list).to_dict()
    indexed_history: History = {}
    for user_id, items in raw_history.items():
        indexed_items = [item_to_index[item] for item in items if item in item_to_index]
        if len(indexed_items) >= min_history_len:
            indexed_history[int(user_id)] = indexed_items

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(indexed_history, f)
    return indexed_history


def load_history(path: str | Path) -> History:
    with Path(path).open("rb") as f:
        return pickle.load(f)


def save_history(history: History, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(history, f)


def split_history(history: History, config: SplitConfig) -> Tuple[History, History, History]:
    if abs(config.train_ratio + config.val_ratio + config.test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio phai bang 1.0")

    train_hist: History = {}
    val_hist: History = {}
    test_hist: History = {}

    for user_id, items in history.items():
        if len(items) < config.min_history_len:
            continue
        n = len(items)
        train_end = max(1, int(n * config.train_ratio))
        val_end = max(train_end + 1, int(n * (config.train_ratio + config.val_ratio)))
        if n - val_end < 1:
            continue
        train_hist[user_id] = items[:train_end]
        val_hist[user_id] = items[:val_end]
        test_hist[user_id] = items

    return train_hist, val_hist, test_hist


def history_to_samples(history: History, state_size: int) -> Tuple[np.ndarray, np.ndarray]:
    states: List[List[int]] = []
    targets: List[int] = []
    for items in history.values():
        if len(items) <= state_size:
            continue
        for idx in range(state_size, len(items)):
            states.append(items[idx - state_size : idx])
            targets.append(items[idx])
    if not states:
        raise ValueError("Khong tao duoc sample nao. Hay giam state_size hoac kiem tra du lieu.")
    return np.asarray(states, dtype=np.int64), np.asarray(targets, dtype=np.int64)


def infer_action_dim(histories: Iterable[History]) -> int:
    max_item = -1
    for history in histories:
        for items in history.values():
            if items:
                max_item = max(max_item, max(items))
    if max_item < 0:
        raise ValueError("History rong, khong suy ra duoc action_dim.")
    return max_item + 1


class NextItemDataset(Dataset):
    def __init__(self, history: History, state_size: int):
        self.states, self.targets = history_to_samples(history, state_size)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.states[idx]), torch.tensor(self.targets[idx], dtype=torch.long)
