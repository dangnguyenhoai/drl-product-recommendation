# GAN vs DQN Comparison Report

## Validation

- GAN best validation HitRate@5: 0.1467
- GAN best validation NDCG@5: 0.0987

## Final Test Results

| Method | Type | HitRate@5 | Precision@5 | Recall@5 | NDCG@5 | MRR@5 |
|---|---|---:|---:|---:|---:|---:|
| Recent-item baseline | baseline | 0.2000 | 0.0527 | 0.0541 | 0.0564 |  |
| Popularity baseline | baseline | 0.1685 | 0.0371 | 0.0380 | 0.0395 |  |
| GAN | gan | 0.1462 | 0.0292 | 0.1462 | 0.0984 | 0.0827 |
| DQN + recency prior boost=5 (selected by validation) | dqn | 0.1449 | 0.0377 | 0.0387 | 0.0450 |  |
| Random baseline | baseline | 0.0277 | 0.0056 | 0.0057 | 0.0056 |  |

## Key Findings

- GAN test HitRate@5 = 0.1462.
- Selected DQN test HitRate@5 = 0.1449.
- GAN is 0.87% higher than the selected DQN by HitRate@5.
- GAN NDCG@5 = 0.0984, while selected DQN NDCG@5 = 0.0450.
- The strongest overall test method is still Recent-item baseline with HitRate@5 = 0.2000.

## Suggested Interpretation

GAN slightly outperforms the selected DQN model on HitRate@5 and substantially improves ranking quality by NDCG@5/MRR@5. However, the recent-item heuristic remains the strongest method by HitRate@5, which suggests short-term repetition is a very strong signal in this dataset.
