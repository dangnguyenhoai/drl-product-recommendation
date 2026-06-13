import pandas as pd
import pickle
from pathlib import Path

# Định nghĩa đường dẫn tương đối từ vị trí của script
BASE_DIR = Path(__file__).parent.parent  # drl-product-recommendation folder
RAW_DIR = BASE_DIR / "data"
PROCESSED_DIR = RAW_DIR / "processed"

# Tham số xử lý
TOP_N_ITEMS = 1000  # Số lượng sản phẩm hàng đầu để giữ lại
N_USERS = None  # None = lấy tất cả users, hoặc đặt số lượng cụ thể

# Tạo folder processed nếu chưa có
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

required_files = {
    "orders": RAW_DIR / "orders.csv",
    "products": RAW_DIR / "products.csv",
    "prior": RAW_DIR / "order_products__prior.csv",
    "train": RAW_DIR / "order_products__train.csv",
}

for name, path in required_files.items():
    if not path.exists():
        raise FileNotFoundError(f"Không thấy file {name}: {path}")

print("Found raw files:")
for name, path in required_files.items():
    print(f"- {name}: {path}")

orders = pd.read_csv(required_files["orders"])
products = pd.read_csv(required_files["products"])
prior = pd.read_csv(required_files["prior"])
train = pd.read_csv(required_files["train"])

print("\nRaw shapes:")
print("orders:", orders.shape)
print("products:", products.shape)
print("prior:", prior.shape)
print("train:", train.shape)

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

top_items = (
    merged["product_id"]
    .value_counts()
    .head(TOP_N_ITEMS)
    .index
    .tolist()
)

item_to_index = {product_id: idx for idx, product_id in enumerate(top_items)}

all_users = sorted(merged["user_id"].unique())
selected_users = all_users[:N_USERS] if N_USERS is not None else all_users
selected_user_set = set(selected_users)
top_item_set = set(top_items)

filtered = merged[
    merged["user_id"].isin(selected_user_set)
    & merged["product_id"].isin(top_item_set)
].copy()

user_history_raw = (
    filtered
    .groupby("user_id")["product_id"]
    .apply(list)
    .to_dict()
)

indexed_history = {}
MIN_HISTORY_LEN = 5

for user_id, items in user_history_raw.items():
    indexed_items = [item_to_index[item] for item in items if item in item_to_index]

    if len(indexed_items) >= MIN_HISTORY_LEN:
        indexed_history[user_id] = indexed_items

# Lưu kết quả vào folder data/processed
indexed_path = PROCESSED_DIR / "indexed_history.pkl"

with open(indexed_path, "wb") as f:
    pickle.dump(indexed_history, f)

lengths = [len(v) for v in indexed_history.values()]
all_indexed_items = [item for hist in indexed_history.values() for item in hist]

print("\n===== Preprocessing Done =====")
print("Users selected:", len(selected_users))
print("Users after filtering:", len(indexed_history))
print("Top items kept:", len(top_items))
print("Total interactions:", sum(lengths))
print("Min history length:", min(lengths) if lengths else 0)
print("Max history length:", max(lengths) if lengths else 0)
print("Average history length:", sum(lengths) / len(lengths) if lengths else 0)
print("Unique indexed items:", len(set(all_indexed_items)))
print("Max indexed item id:", max(all_indexed_items) if all_indexed_items else None)
print("Saved:", indexed_path)
print("File size:", f"{indexed_path.stat().st_size / (1024 * 1024):.2f} MB")