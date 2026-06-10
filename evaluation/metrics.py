import math


class TopKMetrics:
    def __init__(self, top_k):
        self.top_k = top_k
        self.total_steps = 0
        self.steps_with_hit = 0
        self.total_hits = 0
        self.total_recommended = 0
        self.total_targets = 0
        self.total_ndcg = 0.0

    def update(self, recommended_items, target_items):
        recommended = list(recommended_items)[: self.top_k]
        target_set = set(target_items)

        hits_by_rank = [
            1 if item in target_set else 0
            for item in recommended
        ]
        hits = sum(hits_by_rank)

        dcg = sum(
            hit / math.log2(rank + 2)
            for rank, hit in enumerate(hits_by_rank)
        )
        ideal_hits = min(len(target_set), self.top_k)
        idcg = sum(
            1 / math.log2(rank + 2)
            for rank in range(ideal_hits)
        )

        self.total_steps += 1
        self.steps_with_hit += 1 if hits > 0 else 0
        self.total_hits += hits
        self.total_recommended += len(recommended)
        self.total_targets += len(target_set)
        self.total_ndcg += dcg / idcg if idcg > 0 else 0.0

    def as_dict(self):
        if self.total_steps == 0:
            return {
                "hit_rate_at_k": 0.0,
                "precision_at_k": 0.0,
                "recall_at_k": 0.0,
                "ndcg_at_k": 0.0,
            }

        precision = (
            self.total_hits / self.total_recommended
            if self.total_recommended > 0
            else 0.0
        )
        recall = (
            self.total_hits / self.total_targets
            if self.total_targets > 0
            else 0.0
        )

        return {
            "hit_rate_at_k": self.steps_with_hit / self.total_steps,
            "precision_at_k": precision,
            "recall_at_k": recall,
            "ndcg_at_k": self.total_ndcg / self.total_steps,
        }


def print_top_k_metrics(metrics, top_k, prefix=""):
    label = f"{prefix} " if prefix else ""

    print(f"{label}Hit Rate@{top_k}: {metrics['hit_rate_at_k']:.4f}")
    print(f"{label}Precision@{top_k}: {metrics['precision_at_k']:.4f}")
    print(f"{label}Recall@{top_k}: {metrics['recall_at_k']:.4f}")
    print(f"{label}NDCG@{top_k}: {metrics['ndcg_at_k']:.4f}")
