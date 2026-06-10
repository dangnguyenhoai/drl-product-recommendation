from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import NextItemDataset
from .models import Generator


@dataclass
class RankingMetrics:
    hit_rate_at_k: float
    precision_at_k: float
    recall_at_k: float
    ndcg_at_k: float
    mrr_at_k: float
    samples: int


@torch.no_grad()
def evaluate_generator(
    generator: Generator,
    dataset: NextItemDataset,
    top_k: int,
    batch_size: int,
    device: str,
) -> RankingMetrics:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    generator.eval()

    hits = 0
    precision = 0.0
    recall = 0.0
    ndcg = 0.0
    mrr = 0.0
    total = 0

    for states, targets in loader:
        states = states.to(device)
        targets = targets.to(device)
        logits = generator(states)
        top_items = torch.topk(logits, k=top_k, dim=1).indices
        matches = top_items.eq(targets.unsqueeze(1))

        for row in matches.detach().cpu().numpy():
            ranks = np.where(row)[0]
            if len(ranks) > 0:
                rank = int(ranks[0]) + 1
                hits += 1
                precision += 1.0 / top_k
                recall += 1.0
                ndcg += 1.0 / np.log2(rank + 1)
                mrr += 1.0 / rank
            total += 1

    if total == 0:
        return RankingMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0)
    return RankingMetrics(hits / total, precision / total, recall / total, ndcg / total, mrr / total, total)
