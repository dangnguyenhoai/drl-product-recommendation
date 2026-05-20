from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .environment import Episode
from .utils import save_json, seed_everything


def _count_top_products(prior_path: Path, action_size: int, chunksize: int) -> list[int]:
    counts: dict[int, int] = {}
    cols = ["product_id"]
    for chunk in tqdm(
        pd.read_csv(prior_path, usecols=cols, chunksize=chunksize),
        desc="Counting popular products",
    ):
        value_counts = chunk["product_id"].value_counts()
        for product_id, count in value_counts.items():
            counts[int(product_id)] = counts.get(int(product_id), 0) + int(count)
    top = sorted(counts.items(), key=lambda item: item[1], reverse=True)[:action_size]
    return [product_id for product_id, _ in top]


def _read_order_products(
    path: Path,
    order_ids: set[int],
    product_to_action: dict[int, int],
    chunksize: int,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    cols = ["order_id", "product_id", "reordered"]
    for chunk in tqdm(pd.read_csv(path, usecols=cols, chunksize=chunksize), desc=f"Reading {path.name}"):
        mask = chunk["order_id"].isin(order_ids) & chunk["product_id"].isin(product_to_action)
        if mask.any():
            frames.append(chunk.loc[mask].copy())
    if not frames:
        return pd.DataFrame(columns=cols)
    out = pd.concat(frames, ignore_index=True)
    out["action_id"] = out["product_id"].map(product_to_action).astype("int32")
    return out


def build_episodes(
    data_dir: Path,
    output: Path,
    max_users: int,
    action_size: int,
    seed: int,
    chunksize: int,
) -> None:
    seed_everything(seed)
    orders = pd.read_csv(data_dir / "orders.csv")

    train_orders = orders[orders["eval_set"] == "train"].copy()
    prior_orders = orders[orders["eval_set"] == "prior"].copy()
    eligible_users = sorted(set(train_orders["user_id"]).intersection(prior_orders["user_id"]))
    rng = np.random.default_rng(seed)
    if max_users and max_users < len(eligible_users):
        eligible_users = sorted(rng.choice(eligible_users, size=max_users, replace=False).tolist())

    user_set = set(int(x) for x in eligible_users)
    train_orders = train_orders[train_orders["user_id"].isin(user_set)].copy()
    prior_orders = prior_orders[prior_orders["user_id"].isin(user_set)].copy()

    top_products = _count_top_products(data_dir / "order_products__prior.csv", action_size, chunksize)
    product_to_action = {product_id: idx for idx, product_id in enumerate(top_products)}

    prior_items = _read_order_products(
        data_dir / "order_products__prior.csv",
        set(prior_orders["order_id"].astype(int)),
        product_to_action,
        chunksize,
    )
    train_items = _read_order_products(
        data_dir / "order_products__train.csv",
        set(train_orders["order_id"].astype(int)),
        product_to_action,
        chunksize,
    )

    prior_with_user = prior_items.merge(prior_orders[["order_id", "user_id"]], on="order_id", how="left")
    train_with_user = train_items.merge(train_orders[["order_id", "user_id"]], on="order_id", how="left")

    histories = np.zeros((len(eligible_users), action_size), dtype=np.float32)
    user_to_row = {user_id: row for row, user_id in enumerate(eligible_users)}
    for user_id, action_id in tqdm(
        prior_with_user[["user_id", "action_id"]].itertuples(index=False),
        total=len(prior_with_user),
        desc="Building histories",
    ):
        histories[user_to_row[int(user_id)], int(action_id)] += 1.0
    histories = np.log1p(histories)
    row_max = np.maximum(histories.max(axis=1, keepdims=True), 1.0)
    histories = histories / row_max

    target_by_user = (
        train_with_user.groupby("user_id")["action_id"].apply(lambda s: set(int(x) for x in s)).to_dict()
    )
    train_context = train_orders.set_index("user_id")[
        ["order_number", "order_dow", "order_hour_of_day", "days_since_prior_order"]
    ].fillna(0.0)

    episodes: list[Episode] = []
    for user_id in eligible_users:
        targets = target_by_user.get(user_id, set())
        if not targets:
            continue
        row = user_to_row[user_id]
        ctx = train_context.loc[user_id].astype("float32").to_numpy()
        ctx = np.array(
            [
                min(ctx[0] / 100.0, 1.0),
                ctx[1] / 6.0,
                ctx[2] / 23.0,
                min(ctx[3] / 30.0, 1.0),
            ],
            dtype=np.float32,
        )
        episodes.append(Episode(user_id=user_id, history=histories[row], context=ctx, targets=targets))

    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "episodes": episodes,
        "action_size": action_size,
        "top_products": top_products,
        "state_dim": action_size * 2 + 4,
    }
    torch.save(payload, output)
    save_json(
        output.with_suffix(".metadata.json"),
        {
            "episodes": len(episodes),
            "max_users": max_users,
            "action_size": action_size,
            "seed": seed,
            "output": str(output),
        },
    )
    print(f"Saved {len(episodes)} episodes to {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Instacart DQN episodes.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/episodes.pt"))
    parser.add_argument("--max-users", type=int, default=2000)
    parser.add_argument("--action-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_episodes(args.data_dir, args.output, args.max_users, args.action_size, args.seed, args.chunksize)


if __name__ == "__main__":
    main()

