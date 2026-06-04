"""
data_preprocessing.py
=====================
Pipeline tiền xử lý dữ liệu Instacart cho đồ án Recommendation System sử dụng
Baseline MLP và hỗ trợ mở rộng sang DQN/DRL.

Điểm chính:
    - Lọc dữ liệu theo:
        MIN_INTERACTIONS = 20
        MIN_ORDERS = 5
        TOP_N_PRODUCTS = 100
    - Merge đầy đủ products.csv + aisles.csv + departments.csv.
    - Encode user_id, product_id, aisle_id, department_id.
    - Tạo dữ liệu train cho MLP baseline: positive + negative samples.
    - Tạo validation/test candidates bằng cách ranking trên toàn bộ TOP_N_PRODUCTS.
    - Lưu thêm item metadata để DQN có thể dùng làm state/action metadata sau này.

Input mặc định:
    data/raw/orders.csv
    data/raw/order_products__prior.csv
    data/raw/products.csv
    data/raw/aisles.csv
    data/raw/departments.csv

Output:
    data/processed/train.csv
    data/processed/val_candidates.csv
    data/processed/test_candidates.csv
    data/processed/item_metadata.csv
    data/processed/user_history.pkl
    data/processed/order_sequence_dict.pkl
    data/processed/mappings.pkl
    data/processed/config.json
    data/processed/stats_summary.json
"""

import os
import json
import pickle
from collections import Counter

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
RAW_DATA_DIR = "data/raw"
OUTPUT_DIR = "data/processed"

MIN_INTERACTIONS = 20
MIN_ORDERS = 5
TOP_N_PRODUCTS = 100
NEGATIVE_RATIO = 3
RANDOM_SEED = 42

DENSE_FEATURES = [
    "history_length",
    "user_item_count",
    "user_unique_items",
    "item_popularity",
    "item_recency",
    "same_aisle_before",
    "same_department_before",
]

SAMPLE_COLUMNS = [
    "user_idx",
    "item_idx",
    "label",
    "history_length",
    "user_item_count",
    "user_unique_items",
    "item_popularity",
    "item_recency",
    "same_aisle_before",
    "same_department_before",
]

SAMPLE_DTYPES = {
    "user_idx": "int32",
    "item_idx": "int32",
    "label": "int8",
    "history_length": "float32",
    "user_item_count": "float32",
    "user_unique_items": "float32",
    "item_popularity": "float32",
    "item_recency": "float32",
    "same_aisle_before": "float32",
    "same_department_before": "float32",
}


# ─────────────────────────────────────────────
# UTILS
# ─────────────────────────────────────────────
def set_seed(seed: int = RANDOM_SEED) -> None:
    np.random.seed(seed)


def read_csv_with_fallback(filename: str, **read_csv_kwargs) -> pd.DataFrame:
    """Đọc file từ data/raw; nếu không có thì đọc từ thư mục hiện tại."""
    path_in_raw = os.path.join(RAW_DATA_DIR, filename)
    if os.path.exists(path_in_raw):
        return pd.read_csv(path_in_raw, **read_csv_kwargs)

    if os.path.exists(filename):
        return pd.read_csv(filename, **read_csv_kwargs)

    raise FileNotFoundError(
        f"Không tìm thấy {filename}. Hãy đặt file trong {RAW_DATA_DIR}/ hoặc cùng thư mục với script."
    )


# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
def load_data():
    """Đọc đầy đủ các bảng cần thiết của Instacart, chỉ lấy cột cần dùng để giảm RAM/I/O."""
    print("Đang đọc dữ liệu...")

    orders_df = read_csv_with_fallback(
        "orders.csv",
        usecols=[
            "order_id",
            "user_id",
            "order_number",
            "order_dow",
            "order_hour_of_day",
            "days_since_prior_order",
        ],
        dtype={
            "order_id": "int32",
            "user_id": "int32",
            "order_number": "int16",
            "order_dow": "int8",
            "order_hour_of_day": "int8",
            "days_since_prior_order": "float32",
        },
    )

    prior_df = read_csv_with_fallback(
        "order_products__prior.csv",
        usecols=["order_id", "product_id", "add_to_cart_order"],
        dtype={
            "order_id": "int32",
            "product_id": "int32",
            "add_to_cart_order": "int16",
        },
    )

    products_df = read_csv_with_fallback(
        "products.csv",
        usecols=["product_id", "product_name", "aisle_id", "department_id"],
        dtype={
            "product_id": "int32",
            "aisle_id": "int16",
            "department_id": "int16",
        },
    )
    aisles_df = read_csv_with_fallback(
        "aisles.csv",
        dtype={"aisle_id": "int16"},
    )
    departments_df = read_csv_with_fallback(
        "departments.csv",
        dtype={"department_id": "int16"},
    )

    products_df = (
        products_df
        .merge(aisles_df, on="aisle_id", how="left")
        .merge(departments_df, on="department_id", how="left")
    )

    print(f"   orders_df      : {len(orders_df):,} dòng")
    print(f"   prior_df       : {len(prior_df):,} dòng")
    print(f"   products_df    : {len(products_df):,} dòng")
    print(f"   aisles_df      : {len(aisles_df):,} dòng")
    print(f"   departments_df : {len(departments_df):,} dòng")

    return orders_df, prior_df, products_df, aisles_df, departments_df


# ─────────────────────────────────────────────
# 2. CLEAN DATA
# ─────────────────────────────────────────────
def clean_data(orders_df: pd.DataFrame, prior_df: pd.DataFrame, products_df: pd.DataFrame):
    """
    Làm sạch dữ liệu cơ bản:
        - Fill NaN days_since_prior_order.
        - Thêm is_first_order.
        - Xóa duplicate.
        - Merge product metadata gồm aisle và department.
    """
    print("\nĐang làm sạch dữ liệu...")

    orders_df = orders_df.copy()
    prior_df = prior_df.copy()
    products_df = products_df.copy()

    orders_df["is_first_order"] = orders_df["days_since_prior_order"].isna().astype(int)
    orders_df["days_since_prior_order"] = orders_df["days_since_prior_order"].fillna(0)

    dup_orders = orders_df.duplicated().sum()
    dup_prior = prior_df.duplicated().sum()
    print(f"   Duplicate orders_df : {dup_orders}")
    print(f"   Duplicate prior_df  : {dup_prior}")

    orders_df = orders_df.drop_duplicates()
    prior_df = prior_df.drop_duplicates()

    product_cols = [
        "product_id",
        "product_name",
        "aisle_id",
        "aisle",
        "department_id",
        "department",
    ]

    prior_df = prior_df.merge(
        products_df[product_cols],
        on="product_id",
        how="left",
    )

    return orders_df, prior_df, products_df


# ─────────────────────────────────────────────
# 3. FILTER SUBSET
# ─────────────────────────────────────────────
def filter_subset(
    orders_df: pd.DataFrame,
    prior_df: pd.DataFrame,
    top_n_products: int = TOP_N_PRODUCTS,
    min_orders: int = MIN_ORDERS,
    min_interactions: int = MIN_INTERACTIONS,
):
    """
    Lọc subset để train nhanh và phù hợp DQN.

    Tối ưu so với bản cũ:
        - Merge order_id -> user_id một lần.
        - Lọc lặp trực tiếp trên bảng interaction đã có user_id.
        - Tránh merge lại trong mỗi vòng while.
    """
    print("\nĐang lọc subset...")
    print(f"   MIN_INTERACTIONS = {min_interactions}")
    print(f"   MIN_ORDERS       = {min_orders}")
    print(f"   TOP_N_PRODUCTS   = {top_n_products}")

    top_products = (
        prior_df["product_id"]
        .value_counts(sort=True)
        .head(top_n_products)
        .index
    )

    prior_df = prior_df[prior_df["product_id"].isin(top_products)].copy()

    prior_with_users = prior_df.merge(
        orders_df[["order_id", "user_id"]],
        on="order_id",
        how="inner",
    )

    while True:
        user_order_count = prior_with_users.groupby("user_id", sort=False)["order_id"].nunique()
        user_interaction_count = prior_with_users.groupby("user_id", sort=False)["product_id"].size()

        valid_users = user_order_count.index[
            (user_order_count >= min_orders)
            & (user_interaction_count.reindex(user_order_count.index, fill_value=0) >= min_interactions)
        ]

        new_prior_with_users = prior_with_users[prior_with_users["user_id"].isin(valid_users)].copy()

        if len(new_prior_with_users) == len(prior_with_users):
            break

        prior_with_users = new_prior_with_users

        if prior_with_users.empty:
            break

    valid_order_ids = prior_with_users["order_id"].unique()
    valid_user_ids = prior_with_users["user_id"].unique()

    orders_prior = orders_df[
        orders_df["order_id"].isin(valid_order_ids)
        & orders_df["user_id"].isin(valid_user_ids)
    ].copy()

    prior_df = prior_with_users.drop(columns=["user_id"]).copy()

    print(f"   Số sản phẩm sau lọc : {prior_df['product_id'].nunique():,}")
    print(f"   Số user sau lọc     : {orders_prior['user_id'].nunique():,}")
    print(f"   Số đơn hàng sau lọc : {orders_prior['order_id'].nunique():,}")
    print(f"   Số tương tác sau lọc: {len(prior_df):,}")

    return orders_prior, prior_df


# ─────────────────────────────────────────────
# 4. ENCODE IDS
# ─────────────────────────────────────────────
def encode_ids(orders_df: pd.DataFrame, prior_df: pd.DataFrame):
    """Encode user/product/aisle/department thành index liên tục."""
    print("\nĐang encode IDs...")

    orders_df = orders_df.copy()
    prior_df = prior_df.copy()

    user_ids = sorted(orders_df["user_id"].unique())
    product_ids = sorted(prior_df["product_id"].unique())
    aisle_ids = sorted(prior_df["aisle_id"].dropna().unique())
    department_ids = sorted(prior_df["department_id"].dropna().unique())

    user2idx = {u: i for i, u in enumerate(user_ids)}
    idx2user = {i: u for u, i in user2idx.items()}

    prod2idx = {p: i for i, p in enumerate(product_ids)}
    idx2prod = {i: p for p, i in prod2idx.items()}

    aisle2idx = {a: i for i, a in enumerate(aisle_ids)}
    idx2aisle = {i: a for a, i in aisle2idx.items()}

    dept2idx = {d: i for i, d in enumerate(department_ids)}
    idx2dept = {i: d for d, i in dept2idx.items()}

    orders_df["user_idx"] = orders_df["user_id"].map(user2idx).astype("int32")
    prior_df["item_idx"] = prior_df["product_id"].map(prod2idx).astype("int32")
    prior_df["aisle_idx"] = prior_df["aisle_id"].map(aisle2idx).astype("int32")
    prior_df["department_idx"] = prior_df["department_id"].map(dept2idx).astype("int32")

    prior_df = prior_df.merge(
        orders_df[
            [
                "order_id",
                "user_id",
                "user_idx",
                "order_number",
                "order_dow",
                "order_hour_of_day",
                "days_since_prior_order",
                "is_first_order",
            ]
        ],
        on="order_id",
        how="left",
    )

    mappings = {
        "user2idx": user2idx,
        "idx2user": idx2user,
        "prod2idx": prod2idx,
        "idx2prod": idx2prod,
        "aisle2idx": aisle2idx,
        "idx2aisle": idx2aisle,
        "dept2idx": dept2idx,
        "idx2dept": idx2dept,
        "num_users": len(user_ids),
        "num_products": len(product_ids),
        "num_aisles": len(aisle_ids),
        "num_departments": len(department_ids),
    }

    print(f"   num_users       : {mappings['num_users']:,}")
    print(f"   num_products    : {mappings['num_products']:,}")
    print(f"   num_aisles      : {mappings['num_aisles']:,}")
    print(f"   num_departments : {mappings['num_departments']:,}")

    return orders_df, prior_df, mappings


# ─────────────────────────────────────────────
# 5. ITEM METADATA
# ─────────────────────────────────────────────
def build_item_metadata(prior_df: pd.DataFrame) -> pd.DataFrame:
    """Tạo metadata cho từng item để baseline/DQN dùng lại."""
    print("\nĐang tạo item metadata...")

    item_metadata = (
        prior_df[
            [
                "item_idx",
                "product_id",
                "product_name",
                "aisle_id",
                "aisle_idx",
                "aisle",
                "department_id",
                "department_idx",
                "department",
            ]
        ]
        .drop_duplicates("item_idx")
        .sort_values("item_idx")
        .reset_index(drop=True)
    )

    item_counts = prior_df["item_idx"].value_counts().to_dict()
    max_count = max(item_counts.values()) if item_counts else 1

    item_metadata["item_count"] = item_metadata["item_idx"].map(item_counts).fillna(0).astype(int)
    item_metadata["item_popularity"] = item_metadata["item_count"] / max_count

    print(f"   item_metadata: {len(item_metadata):,} sản phẩm")
    return item_metadata


# ─────────────────────────────────────────────
# 6. ORDER-LEVEL SEQUENCES
# ─────────────────────────────────────────────
def create_order_sequences(prior_df: pd.DataFrame):
    """
    Tạo order-level sequence:
        {user_idx: [order_1, order_2, ...]}

    Tối ưu: dùng sort_values + itertuples thay cho groupby.apply(lambda ...).
    Cách này thường nhẹ RAM hơn và nhanh hơn khi dữ liệu lớn.
    """
    print("\nĐang tạo order-level sequences...")

    cols = [
        "user_idx",
        "order_number",
        "add_to_cart_order",
        "item_idx",
        "product_id",
        "aisle_idx",
        "department_idx",
    ]
    sorted_df = prior_df[cols].sort_values(
        ["user_idx", "order_number", "add_to_cart_order"],
        kind="mergesort",
    )

    order_sequence_dict = {}
    current_user = None
    current_order = None
    current_items = []

    def flush_order():
        if current_user is not None and current_items:
            order_sequence_dict.setdefault(int(current_user), []).append(current_items.copy())

    for row in sorted_df.itertuples(index=False, name=None):
        user_idx, order_number, _add_pos, item_idx, product_id, aisle_idx, department_idx = row

        if current_user != user_idx or current_order != order_number:
            flush_order()
            current_user = user_idx
            current_order = order_number
            current_items = []

        current_items.append({
            "item_idx": int(item_idx),
            "product_id": int(product_id),
            "aisle_idx": int(aisle_idx),
            "department_idx": int(department_idx),
        })

    flush_order()

    print(f"   Số user có sequence: {len(order_sequence_dict):,}")
    return order_sequence_dict


def flatten_items(order_list):
    """Chuyển list order thành list item_idx."""
    return [item["item_idx"] for order in order_list for item in order]


def flatten_aisles(order_list):
    return [item["aisle_idx"] for order in order_list for item in order]


def flatten_departments(order_list):
    return [item["department_idx"] for order in order_list for item in order]


# ─────────────────────────────────────────────
# 7. FEATURE HELPERS
# ─────────────────────────────────────────────
def optimize_sample_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Ép dtype nhỏ hơn để giảm RAM và tăng tốc ghi/đọc file."""
    return df.astype(SAMPLE_DTYPES, copy=False)


def build_item_arrays(item_metadata: pd.DataFrame, num_items: int):
    """Tạo array lookup theo item_idx, nhanh hơn dict lookup trong vòng lặp lớn."""
    meta = item_metadata.sort_values("item_idx")
    idx = meta["item_idx"].to_numpy(dtype=np.int32)

    item_aisle_arr = np.full(num_items, -1, dtype=np.int32)
    item_dept_arr = np.full(num_items, -1, dtype=np.int32)
    item_pop_arr = np.zeros(num_items, dtype=np.float32)

    item_aisle_arr[idx] = meta["aisle_idx"].to_numpy(dtype=np.int32)
    item_dept_arr[idx] = meta["department_idx"].to_numpy(dtype=np.int32)
    item_pop_arr[idx] = meta["item_popularity"].to_numpy(dtype=np.float32)

    return item_aisle_arr, item_dept_arr, item_pop_arr


def empty_history_state() -> dict:
    return {
        "length": 0,
        "counter": Counter(),
        "item_set": set(),
        "aisle_set": set(),
        "dept_set": set(),
        "last_pos": {},
    }


def extend_history_state(state: dict, order: list) -> None:
    """Cập nhật history theo kiểu incremental, không flatten lại toàn bộ lịch sử."""
    for item in order:
        item_idx = int(item["item_idx"])
        aisle_idx = int(item["aisle_idx"])
        dept_idx = int(item["department_idx"])

        state["counter"][item_idx] += 1
        state["item_set"].add(item_idx)
        state["aisle_set"].add(aisle_idx)
        state["dept_set"].add(dept_idx)
        state["last_pos"][item_idx] = state["length"]
        state["length"] += 1


def build_history_state(order_list: list) -> dict:
    state = empty_history_state()
    for order in order_list:
        extend_history_state(state, order)
    return state


def make_rows_for_items(
    user_idx: int,
    item_indices,
    label: int,
    state: dict,
    item_aisle_arr: np.ndarray,
    item_dept_arr: np.ndarray,
    item_pop_arr: np.ndarray,
) -> list:
    """Tạo nhiều feature rows, dùng state đã cache thay vì tính Counter/set cho từng row."""
    rows = []
    history_len = int(state["length"])
    user_unique_items = len(state["item_set"])
    history_counter = state["counter"]
    aisle_set = state["aisle_set"]
    dept_set = state["dept_set"]
    last_pos = state["last_pos"]

    for item_idx in item_indices:
        item_idx = int(item_idx)
        last = last_pos.get(item_idx, -1)
        item_recency = 0.0 if last < 0 else 1.0 / (1.0 + history_len - last)

        rows.append((
            int(user_idx),
            item_idx,
            int(label),
            float(history_len),
            float(history_counter.get(item_idx, 0)),
            float(user_unique_items),
            float(item_pop_arr[item_idx]),
            float(item_recency),
            float(item_aisle_arr[item_idx] in aisle_set),
            float(item_dept_arr[item_idx] in dept_set),
        ))

    return rows


# ─────────────────────────────────────────────
# 8. TRAIN SAMPLES
# ─────────────────────────────────────────────
def create_train_samples(
    order_sequence_dict: dict,
    num_items: int,
    item_metadata: pd.DataFrame,
    negative_ratio: int = NEGATIVE_RATIO,
) -> pd.DataFrame:
    """
    Tạo train samples:
        - Positive: item xuất hiện trong target order.
        - Negative: item không xuất hiện trong target order.

    Tối ưu chính:
        - Không gọi Counter(history_items), set(history_items), set(history_aisles) cho từng item.
        - Không scan history để tìm lần xuất hiện gần nhất; dùng last_pos cache.
        - Dùng numpy mask để lấy negative pool.
    """
    print("\nĐang tạo train samples...")

    rng = np.random.default_rng(RANDOM_SEED)
    all_items_arr = np.arange(num_items, dtype=np.int32)
    samples = []

    item_aisle_arr, item_dept_arr, item_pop_arr = build_item_arrays(item_metadata, num_items)

    for user_idx, orders in order_sequence_dict.items():
        if len(orders) < MIN_ORDERS:
            continue

        state = empty_history_state()
        extend_history_state(state, orders[0])

        # Dành orders[-2] cho validation, orders[-1] cho test
        for t in range(1, len(orders) - 2):
            target_items = sorted({int(item["item_idx"]) for item in orders[t]})
            if not target_items:
                continue

            samples.extend(
                make_rows_for_items(
                    user_idx=user_idx,
                    item_indices=target_items,
                    label=1,
                    state=state,
                    item_aisle_arr=item_aisle_arr,
                    item_dept_arr=item_dept_arr,
                    item_pop_arr=item_pop_arr,
                )
            )

            neg_size = len(target_items) * negative_ratio
            target_mask = np.zeros(num_items, dtype=bool)
            target_mask[np.asarray(target_items, dtype=np.int32)] = True

            unseen_mask = ~target_mask
            if state["item_set"]:
                seen_items = np.fromiter(state["item_set"], dtype=np.int32)
                unseen_mask[seen_items] = False

            unseen_negative_pool = all_items_arr[unseen_mask]
            negative_pool = all_items_arr[~target_mask]

            if len(unseen_negative_pool) >= neg_size:
                negative_items = rng.choice(unseen_negative_pool, size=neg_size, replace=False)
            else:
                size = min(neg_size, len(negative_pool))
                negative_items = rng.choice(negative_pool, size=size, replace=False)

            samples.extend(
                make_rows_for_items(
                    user_idx=user_idx,
                    item_indices=negative_items,
                    label=0,
                    state=state,
                    item_aisle_arr=item_aisle_arr,
                    item_dept_arr=item_dept_arr,
                    item_pop_arr=item_pop_arr,
                )
            )

            extend_history_state(state, orders[t])

    df = pd.DataFrame(samples, columns=SAMPLE_COLUMNS)

    if df.empty:
        raise ValueError("train_df rỗng. Hãy giảm MIN_ORDERS/MIN_INTERACTIONS hoặc kiểm tra dữ liệu đầu vào.")

    df = optimize_sample_dtypes(df)

    positives = int(df["label"].sum())
    negatives = int((df["label"] == 0).sum())
    print(f"   Tổng train samples: {len(df):,}")
    print(f"   Positive          : {positives:,}")
    print(f"   Negative          : {negatives:,}")

    return df


# ─────────────────────────────────────────────
# 9. VALIDATION / TEST CANDIDATES
# ─────────────────────────────────────────────
def create_candidates(
    order_sequence_dict: dict,
    num_items: int,
    item_metadata: pd.DataFrame,
    split: str,
) -> pd.DataFrame:
    """
    Tạo candidates cho validation/test bằng cách ranking trên toàn bộ TOP_N_PRODUCTS.

    Tối ưu chính:
        - Với mỗi user, tạo 100 candidates bằng numpy array thay vì append dict từng dòng.
        - item_recency được tính từ last_pos cache, không scan history nhiều lần.
    """
    if split not in {"val", "test"}:
        raise ValueError("split phải là 'val' hoặc 'test'.")

    print(f"\nĐang tạo {split} candidates...")

    all_items = np.arange(num_items, dtype=np.int32)
    item_aisle_arr, item_dept_arr, item_pop_arr = build_item_arrays(item_metadata, num_items)

    col_parts = {col: [] for col in SAMPLE_COLUMNS}

    for user_idx, orders in order_sequence_dict.items():
        if len(orders) < MIN_ORDERS:
            continue

        if split == "val":
            history_orders = orders[:-2]
            target_order = orders[-2]
        else:
            history_orders = orders[:-1]
            target_order = orders[-1]

        target_items = sorted({int(item["item_idx"]) for item in target_order})
        if not target_items:
            continue

        state = build_history_state(history_orders)
        history_len = int(state["length"])
        if history_len == 0:
            continue

        label_arr = np.zeros(num_items, dtype=np.int8)
        label_arr[np.asarray(target_items, dtype=np.int32)] = 1

        count_arr = np.zeros(num_items, dtype=np.float32)
        if state["counter"]:
            idx = np.fromiter(state["counter"].keys(), dtype=np.int32)
            vals = np.fromiter(state["counter"].values(), dtype=np.float32)
            count_arr[idx] = vals

        recency_arr = np.zeros(num_items, dtype=np.float32)
        if state["last_pos"]:
            idx = np.fromiter(state["last_pos"].keys(), dtype=np.int32)
            last_vals = np.fromiter(state["last_pos"].values(), dtype=np.float32)
            recency_arr[idx] = 1.0 / (1.0 + history_len - last_vals)

        if state["aisle_set"]:
            aisle_values = np.fromiter(state["aisle_set"], dtype=np.int32)
            same_aisle_arr = np.isin(item_aisle_arr, aisle_values).astype(np.float32)
        else:
            same_aisle_arr = np.zeros(num_items, dtype=np.float32)

        if state["dept_set"]:
            dept_values = np.fromiter(state["dept_set"], dtype=np.int32)
            same_dept_arr = np.isin(item_dept_arr, dept_values).astype(np.float32)
        else:
            same_dept_arr = np.zeros(num_items, dtype=np.float32)

        col_parts["user_idx"].append(np.full(num_items, int(user_idx), dtype=np.int32))
        col_parts["item_idx"].append(all_items)
        col_parts["label"].append(label_arr)
        col_parts["history_length"].append(np.full(num_items, history_len, dtype=np.float32))
        col_parts["user_item_count"].append(count_arr)
        col_parts["user_unique_items"].append(np.full(num_items, len(state["item_set"]), dtype=np.float32))
        col_parts["item_popularity"].append(item_pop_arr)
        col_parts["item_recency"].append(recency_arr)
        col_parts["same_aisle_before"].append(same_aisle_arr)
        col_parts["same_department_before"].append(same_dept_arr)

    if not col_parts["user_idx"]:
        raise ValueError(f"{split}_df rỗng. Hãy kiểm tra điều kiện lọc hoặc dữ liệu đầu vào.")

    df = pd.DataFrame({
        col: np.concatenate(parts)
        for col, parts in col_parts.items()
    })
    df = optimize_sample_dtypes(df)

    print(f"   Tổng {split} candidates: {len(df):,}")
    print(f"   User trong {split}      : {df['user_idx'].nunique():,}")
    print(f"   Positive labels         : {int(df['label'].sum()):,}")

    return df


# ─────────────────────────────────────────────
# 10. USER HISTORY FOR DQN
# ─────────────────────────────────────────────
def build_user_history_for_dqn(order_sequence_dict: dict) -> dict:
    """
    Lưu lịch sử user theo dạng dễ dùng cho DQN Environment.
    DQN có thể dùng:
        - train_orders làm history/state ban đầu
        - val_order/test_order làm ground-truth mô phỏng reward
    """
    print("\nĐang tạo user_history cho DQN...")

    user_history = {}
    for user_idx, orders in order_sequence_dict.items():
        if len(orders) < MIN_ORDERS:
            continue

        user_history[user_idx] = {
            "train_orders": orders[:-2],
            "val_order": orders[-2],
            "test_order": orders[-1],
            "all_history_before_test": orders[:-1],
        }

    print(f"   user_history entries: {len(user_history):,}")
    return user_history


# ─────────────────────────────────────────────
# 11. SAVE
# ─────────────────────────────────────────────
def save_processed_data(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    item_metadata: pd.DataFrame,
    order_sequence_dict: dict,
    user_history: dict,
    mappings: dict,
    config: dict,
    stats: dict,
) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_path = os.path.join(OUTPUT_DIR, "train.csv")
    val_path = os.path.join(OUTPUT_DIR, "val_candidates.csv")
    test_path = os.path.join(OUTPUT_DIR, "test_candidates.csv")
    item_path = os.path.join(OUTPUT_DIR, "item_metadata.csv")
    seq_path = os.path.join(OUTPUT_DIR, "order_sequence_dict.pkl")
    history_path = os.path.join(OUTPUT_DIR, "user_history.pkl")
    mapping_path = os.path.join(OUTPUT_DIR, "mappings.pkl")
    config_path = os.path.join(OUTPUT_DIR, "config.json")
    stats_path = os.path.join(OUTPUT_DIR, "stats_summary.json")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    item_metadata.to_csv(item_path, index=False)

    with open(seq_path, "wb") as f:
        pickle.dump(order_sequence_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(history_path, "wb") as f:
        pickle.dump(user_history, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(mapping_path, "wb") as f:
        pickle.dump(mappings, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print("\nĐã lưu output:")
    print(f"   {train_path} ({len(train_df):,} dòng)")
    print(f"   {val_path} ({len(val_df):,} dòng)")
    print(f"   {test_path} ({len(test_df):,} dòng)")
    print(f"   {item_path}")
    print(f"   {seq_path}")
    print(f"   {history_path}")
    print(f"   {mapping_path}")
    print(f"   {config_path}")
    print(f"   {stats_path}")


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────
def main():
    set_seed(RANDOM_SEED)

    print("=" * 70)
    print(" DATA PREPROCESSING PIPELINE – Instacart Baseline + DQN Support")
    print("=" * 70)

    orders_df, prior_df, products_df, aisles_df, departments_df = load_data()

    orders_df, prior_df, products_df = clean_data(orders_df, prior_df, products_df)

    orders_df, prior_df = filter_subset(
        orders_df=orders_df,
        prior_df=prior_df,
        top_n_products=TOP_N_PRODUCTS,
        min_orders=MIN_ORDERS,
        min_interactions=MIN_INTERACTIONS,
    )

    orders_df, prior_df, mappings = encode_ids(orders_df, prior_df)

    item_metadata = build_item_metadata(prior_df)

    order_sequence_dict = create_order_sequences(prior_df)

    num_products = mappings["num_products"]

    train_df = create_train_samples(
        order_sequence_dict=order_sequence_dict,
        num_items=num_products,
        item_metadata=item_metadata,
        negative_ratio=NEGATIVE_RATIO,
    )

    val_df = create_candidates(
        order_sequence_dict=order_sequence_dict,
        num_items=num_products,
        item_metadata=item_metadata,
        split="val",
    )

    test_df = create_candidates(
        order_sequence_dict=order_sequence_dict,
        num_items=num_products,
        item_metadata=item_metadata,
        split="test",
    )

    user_history = build_user_history_for_dqn(order_sequence_dict)

    config = {
        "min_interactions": MIN_INTERACTIONS,
        "min_orders": MIN_ORDERS,
        "top_n_products": TOP_N_PRODUCTS,
        "negative_ratio": NEGATIVE_RATIO,
        "dense_features": DENSE_FEATURES,
        "num_users": mappings["num_users"],
        "num_products": mappings["num_products"],
        "num_aisles": mappings["num_aisles"],
        "num_departments": mappings["num_departments"],
        "notes": {
            "baseline": "MLP uses user_idx, item_idx, and dense_features.",
            "dqn_support": "item_metadata.csv and user_history.pkl can be used to build DQN state/action/reward.",
            "evaluation": "val/test candidates rank over all TOP_N_PRODUCTS items.",
        },
    }

    stats = {
        "num_users": int(mappings["num_users"]),
        "num_products": int(mappings["num_products"]),
        "num_aisles": int(mappings["num_aisles"]),
        "num_departments": int(mappings["num_departments"]),
        "min_interactions": int(MIN_INTERACTIONS),
        "min_orders": int(MIN_ORDERS),
        "top_n_products": int(TOP_N_PRODUCTS),
        "train_samples": int(len(train_df)),
        "train_positive": int(train_df["label"].sum()),
        "train_negative": int((train_df["label"] == 0).sum()),
        "val_samples": int(len(val_df)),
        "val_users": int(val_df["user_idx"].nunique()),
        "val_positive": int(val_df["label"].sum()),
        "test_samples": int(len(test_df)),
        "test_users": int(test_df["user_idx"].nunique()),
        "test_positive": int(test_df["label"].sum()),
        "dense_features": DENSE_FEATURES,
    }

    save_processed_data(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        item_metadata=item_metadata,
        order_sequence_dict=order_sequence_dict,
        user_history=user_history,
        mappings=mappings,
        config=config,
        stats=stats,
    )

    print("\nHoàn tất tiền xử lý dữ liệu!")


if __name__ == "__main__":
    main()
