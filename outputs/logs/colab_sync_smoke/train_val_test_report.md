# Validation/Test Evaluation Report

## Validation Results

| Method | Avg Reward | HitRate@5 | Precision@5 | Recall@5 | NDCG@5 | Recent Boost |
|---|---:|---:|---:|---:|---:|---:|
| DQN + recency prior boost=5 | -5.000 | 0.1667 | 0.0333 | 0.0333 | 0.0357 | 5.0 |
| DQN pure stable | -20.000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0 |
| DQN + recency prior boost=2 | -2.000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 2.0 |

## Selected Model

- Selected by validation HitRate@5: **DQN + recency prior boost=5**
- Model path: `outputs\checkpoints\dqn_recency5_stable.pth`
- Recent boost: `5.0`

## Final Test Results

| Method | Avg Reward | HitRate@5 | Precision@5 | Recall@5 | NDCG@5 | Type | Recent Boost |
|---|---:|---:|---:|---:|---:|---|---:|
| Recent-item baseline | 0.000 | 0.2857 | 0.0571 | 0.0588 | 0.0700 | baseline |  |
| Random baseline | -4.000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | baseline |  |
| Popularity baseline | -4.000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | baseline |  |
| DQN + recency prior boost=5 (selected by validation) | -8.000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | dqn | 5.0 |

## Interpretation

- Models are selected using validation HitRate@5, not test HitRate@5.
- The test split is used only for final reporting after model selection.
- If the selected model uses recent_boost, report it as **DQN + recency prior**, not pure DQN.
- If Recent-item baseline is still best on test, the current DQN setup does not outperform the strongest heuristic baseline.