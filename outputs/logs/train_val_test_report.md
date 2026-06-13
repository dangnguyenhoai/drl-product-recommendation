# Validation/Test Evaluation Report

## Validation Results

| Method | Avg Reward | HitRate@5 | Precision@5 | Recall@5 | NDCG@5 | Recent Boost |
|---|---:|---:|---:|---:|---:|---:|
| DQN + recency prior boost=2 | -3.118 | 0.1640 | 0.0442 | 0.0456 | 0.0540 | 2.0 |
| DQN + recency prior boost=5 | -3.736 | 0.1565 | 0.0408 | 0.0420 | 0.0497 | 5.0 |
| DQN pure stable | -9.049 | 0.0554 | 0.0113 | 0.0116 | 0.0132 | 0.0 |

## Selected Model

- Selected by validation HitRate@5: **DQN + recency prior boost=2**
- Model path: `outputs/checkpoints/dqn_recency2_stable.pth`
- Recent boost: `2.0`

## Final Test Results

| Method | Avg Reward | HitRate@5 | Precision@5 | Recall@5 | NDCG@5 | Type | Recent Boost |
|---|---:|---:|---:|---:|---:|---|---:|
| Recent-item baseline | -1.335 | 0.2029 | 0.0513 | 0.0526 | 0.0540 | baseline |  |
| Popularity baseline | -4.055 | 0.1772 | 0.0385 | 0.0394 | 0.0413 | baseline |  |
| DQN + recency prior boost=2 (selected by validation) | -4.020 | 0.1497 | 0.0400 | 0.0411 | 0.0475 | dqn | 2.0 |
| Random baseline | -9.862 | 0.0248 | 0.0050 | 0.0052 | 0.0049 | baseline |  |

## Interpretation

- Models are selected using validation HitRate@5, not test HitRate@5.
- The test split is used only for final reporting after model selection.
- If the selected model uses recent_boost, report it as **DQN + recency prior**, not pure DQN.
- If Recent-item baseline is still best on test, the current DQN setup does not outperform the strongest heuristic baseline.