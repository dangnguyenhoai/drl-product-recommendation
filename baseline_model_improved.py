"""
baseline_model_improved.py
==========================

Mô hình MLP Baseline cải thiện cho Recommendation System.

Cải thiện so với bản cũ:
    1. Thêm user_bias, item_bias và global_bias để học xu hướng user/item phổ biến.
    2. Thêm embedding dropout để giảm overfitting.
    3. Thêm LayerNorm sau dense features và hidden layers để train ổn định hơn.
    4. Vẫn giữ output là logits để dùng với BCEWithLogitsLoss.

Input:
    user_idx: LongTensor [batch_size]
    item_idx: LongTensor [batch_size]
    features: FloatTensor [batch_size, num_features]

Output:
    logits: FloatTensor [batch_size]
"""

from typing import Sequence

import torch
import torch.nn as nn


class MLPBaseline(nn.Module):
    """MLP recommender với user/item embedding, dense features và bias terms."""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        num_features: int = 5,
        hidden_dims: Sequence[int] = (256, 128, 64),
        dropout: float = 0.25,
        embedding_dropout: float = 0.05,
        use_bias_terms: bool = True,
        use_layer_norm: bool = True,
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
            raise ValueError("hidden_dims không được rỗng, ví dụ (256, 128, 64)")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout phải nằm trong khoảng [0, 1)")
        if not 0.0 <= embedding_dropout < 1.0:
            raise ValueError("embedding_dropout phải nằm trong khoảng [0, 1)")

        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_features = num_features
        self.use_bias_terms = use_bias_terms

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.embedding_dropout = nn.Dropout(embedding_dropout)

        if use_bias_terms:
            self.user_bias = nn.Embedding(num_users, 1)
            self.item_bias = nn.Embedding(num_items, 1)
            self.global_bias = nn.Parameter(torch.zeros(1))
        else:
            self.user_bias = None
            self.item_bias = None
            self.global_bias = None

        self.feature_norm = (
            nn.LayerNorm(num_features)
            if use_layer_norm and num_features > 0
            else nn.Identity()
        )

        input_dim = embedding_dim * 2 + num_features

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            if hidden_dim <= 0:
                raise ValueError("Mỗi hidden_dim phải > 0")

            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.mlp = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self) -> None:
        """Khởi tạo trọng số ổn định hơn cho embedding, bias và linear layers."""
        nn.init.normal_(self.user_embedding.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.01)

        if self.use_bias_terms:
            nn.init.zeros_(self.user_bias.weight)
            nn.init.zeros_(self.item_bias.weight)
            nn.init.zeros_(self.global_bias)

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

        user_vec = self.embedding_dropout(user_vec)
        item_vec = self.embedding_dropout(item_vec)

        if self.num_features > 0:
            features = self.feature_norm(features)

        x = torch.cat([user_vec, item_vec, features], dim=1)
        logits = self.mlp(x).squeeze(1)

        if self.use_bias_terms:
            logits = (
                logits
                + self.user_bias(user_idx).squeeze(1)
                + self.item_bias(item_idx).squeeze(1)
                + self.global_bias
            )

        return logits


if __name__ == "__main__":
    # Smoke test nhanh để kiểm tra model chạy được.
    model = MLPBaseline(
        num_users=1000,
        num_items=100,
        embedding_dim=64,
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
