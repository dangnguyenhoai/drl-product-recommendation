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


def save_preprocessing_charts():
    import matplotlib.pyplot as plt
    from matplotlib.ticker import StrMethodFormatter

    if not lengths:
        print("\nNo retained histories; preprocessing charts were not created.")
        return

    figures_dir = BASE_DIR / "results" / "eda"
    figures_dir.mkdir(parents=True, exist_ok=True)
    average_history_length = sum(lengths) / len(lengths)

    # This histogram is built from every retained user's actual history length.
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.hist(lengths, bins=50, color="#4C78A8", edgecolor="white", linewidth=0.5)
    ax.axvline(
        average_history_length,
        color="#E45756",
        linestyle="--",
        linewidth=2,
        label=f"Trung bình: {average_history_length:,.2f}",
    )
    ax.set_title("Phân phối độ dài lịch sử mua hàng sau tiền xử lý")
    ax.set_xlabel("Độ dài lịch sử (số tương tác)")
    ax.set_ylabel("Số người dùng")
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()

    history_chart_path = figures_dir / "history_length_distribution_after_preprocessing.png"
    fig.savefig(history_chart_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.hist(
        lengths,
        bins=50,
        range=(0, 500),
        color="#4C78A8",
        edgecolor="white",
        linewidth=0.5,
    )
    ax.axvline(
        average_history_length,
        color="#E45756",
        linestyle="--",
        linewidth=2,
        label=f"Trung bình: {average_history_length:,.2f}",
    )
    ax.set_xlim(0, 500)
    ax.set_title("Phân phối độ dài lịch sử mua hàng sau tiền xử lý (0–500)")
    ax.set_xlabel("Độ dài lịch sử (số tương tác)")
    ax.set_ylabel("Số người dùng")
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()

    limited_history_chart_path = (
        figures_dir / "history_length_distribution_after_preprocessing_0_500.png"
    )
    fig.savefig(limited_history_chart_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    before_values = [
        len(selected_users),
        len(products),
        len(order_products),
    ]
    after_values = [
        len(indexed_history),
        len(top_items),
        sum(lengths),
    ]
    metric_names = ["Người dùng", "Sản phẩm", "Tương tác"]

    # Separate panels keep the three very different scales readable.
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("So sánh quy mô dữ liệu trước và sau tiền xử lý", fontweight="bold")

    for ax, metric, before, after in zip(
        axes, metric_names, before_values, after_values
    ):
        bars = ax.bar(
            ["Trước", "Sau"],
            [before, after],
            color=["#72B7B2", "#F58518"],
            width=0.62,
        )
        retained_percent = after / before * 100 if before else 0
        largest_value = max(before, after)
        ax.set_title(f"{metric}\nGiữ lại {retained_percent:.2f}%")
        ax.set_ylim(-largest_value * 0.06, largest_value * 1.08)
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        ax.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, [before, after]):
            relative_height = value / largest_value if largest_value else 0
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() / 2,
                f"{value:,}",
                ha="center",
                va="center",
                color="white" if relative_height >= 0.08 else "#333333",
                fontweight="bold",
                fontsize=10 if relative_height >= 0.08 else 9,
            )

    fig.text(
        0.5,
        0.01,
        "Trước: toàn bộ người dùng, sản phẩm và tương tác ban đầu. "
        "Sau: dữ liệu được giữ lại sau bước lọc và mã hóa.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.92))

    scale_chart_path = figures_dir / "data_scale_before_after_preprocessing.png"
    fig.savefig(scale_chart_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("\nSaved preprocessing charts:")
    print("-", history_chart_path)
    print("-", limited_history_chart_path)
    print("-", scale_chart_path)


save_preprocessing_charts()
