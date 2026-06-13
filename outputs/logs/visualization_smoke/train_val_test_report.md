# Validation/Test Evaluation Report

## Validation Results

| Method | Avg Reward | HitRate@5 | Precision@5 | Recall@5 | NDCG@5 | Recent Boost |
|---|---:|---:|---:|---:|---:|---:|
| DQN + recency prior boost=5 | -3.992 | 0.1462 | 0.0394 | 0.0404 | 0.0477 | 5.0 |
| DQN + recency prior boost=2 | -4.325 | 0.1416 | 0.0384 | 0.0394 | 0.0465 | 2.0 |
| DQN pure stable | -8.585 | 0.0624 | 0.0127 | 0.0131 | 0.0129 | 0.0 |

## Selected Model

- Selected by validation HitRate@5: **DQN + recency prior boost=5**
- Model path: `outputs/checkpoints/dqn_recency5_stable_best.pth`
- Recent boost: `5.0`

## Final Test Results

| Method | Avg Reward | HitRate@5 | Precision@5 | Recall@5 | NDCG@5 | Type | Recent Boost |
|---|---:|---:|---:|---:|---:|---|---:|
| Recent-item baseline | -1.221 | 0.2059 | 0.0522 | 0.0538 | 0.0564 | baseline |  |
| Popularity baseline | -4.365 | 0.1668 | 0.0364 | 0.0373 | 0.0387 | baseline |  |
| DQN + recency prior boost=5 (selected by validation) | -4.511 | 0.1456 | 0.0373 | 0.0384 | 0.0459 | dqn | 5.0 |
| Random baseline | -10.649 | 0.0252 | 0.0051 | 0.0052 | 0.0054 | baseline |  |

## Interpretation

- Models are selected using validation HitRate@5, not test HitRate@5.
- The test split is used only for final reporting after model selection.
- If the selected model uses recent_boost, report it as **DQN + recency prior**, not pure DQN.
- If Recent-item baseline is still best on test, the current DQN setup does not outperform the strongest heuristic baseline.