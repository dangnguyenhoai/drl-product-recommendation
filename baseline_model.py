"""
baseline_model.py
=================
Định nghĩa mô hình MLP Baseline cho bài toán Recommendation System.

Kiến trúc:
    user_idx  → User Embedding
    item_idx  → Item Embedding
    features  → Dense features

    [user_vector, item_vector, dense_features]
                ↓
              MLP
                ↓
            logit score

Output của model là logits, chưa qua sigmoid.
Khi train dùng BCEWithLogitsLoss.
Khi evaluate dùng sigmoid(logits) để ra xác suất tương tác/mua hàng.
"""

from typing import Sequence

import torch
import torch.nn as nn


class MLPBaseline(nn.Module):
    """MLP baseline recommender với user/item embedding và dense features."""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 32,
        num_features: int = 5,
        hidden_dims: Sequence[int] = (128, 64),
        dropout: float = 0.2,
    ):
        super().__init__()

        if num_users <= 0:
            raise ValueError("num_users phải > 0")
        if num_items <= 0:
            raise ValueError("num_items phải > 0")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim phải > 0")
        if num_features < 0:
            raise ValueError("num_features không được âm")
        if len(hidden_dims) == 0:
            raise ValueError("hidden_dims không được rỗng, ví dụ (128, 64)")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout phải nằm trong khoảng [0, 1)")

        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_features = num_features

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        input_dim = embedding_dim * 2 + num_features

        # Build MLP linh hoạt để sau này có thể đổi số hidden layers dễ dàng.
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            if hidden_dim <= 0:
                raise ValueError("Mỗi hidden_dim phải > 0")
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.mlp = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self) -> None:
        """Khởi tạo trọng số ổn định hơn cho embedding và linear layers."""
        nn.init.normal_(self.user_embedding.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.01)

        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        user_idx: torch.Tensor,
        item_idx: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            user_idx: LongTensor shape [batch_size]
            item_idx: LongTensor shape [batch_size]
            features: FloatTensor shape [batch_size, num_features]

        Returns:
            logits: FloatTensor shape [batch_size]
        """
        if features.dim() != 2:
            raise ValueError(
                f"features phải có shape [batch_size, num_features], nhận được {tuple(features.shape)}"
            )
        if features.shape[1] != self.num_features:
            raise ValueError(
                f"Model cần {self.num_features} dense features, nhưng nhận được {features.shape[1]}"
            )

        user_vec = self.user_embedding(user_idx)
        item_vec = self.item_embedding(item_idx)

        x = torch.cat([user_vec, item_vec, features], dim=1)
        logits = self.mlp(x)

        return logits.squeeze(1)


if __name__ == "__main__":
    # Smoke test nhanh để kiểm tra model chạy được.
    model = MLPBaseline(
        num_users=1000,
        num_items=100,
        embedding_dim=32,
        num_features=5,
    )

    batch_size = 8
    user = torch.randint(0, 1000, (batch_size,))
    item = torch.randint(0, 100, (batch_size,))
    features = torch.randn(batch_size, 5)

    logits = model(user, item, features)
    print(model)
    print("Output shape:", logits.shape)
    print("Model OK")
