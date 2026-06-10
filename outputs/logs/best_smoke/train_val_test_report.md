# Validation/Test Evaluation Report

## Validation Results

| Method | Avg Reward | HitRate@5 | Precision@5 | Recall@5 | NDCG@5 | Recent Boost |
|---|---:|---:|---:|---:|---:|---:|
| DQN + recency prior boost=5 | 3.000 | 0.5000 | 0.1000 | 0.1000 | 0.0848 | 5.0 |
| DQN + recency prior boost=2 | 4.000 | 0.2500 | 0.1000 | 0.1111 | 0.1036 | 2.0 |
| DQN pure stable | -13.000 | 0.1000 | 0.0200 | 0.0204 | 0.0131 | 0.0 |

## Selected Model

- Selected by validation HitRate@5: **DQN + recency prior boost=5**
- Model path: `outputs\checkpoints\dqn_recency5_stable.pth`
- Recent boost: `5.0`

## Final Test Results

| Method | Avg Reward | HitRate@5 | Precision@5 | Recall@5 | NDCG@5 | Type | Recent Boost |
|---|---:|---:|---:|---:|---:|---|---:|
| Recent-item baseline | 6.000 | 0.5000 | 0.1000 | 0.1538 | 0.1007 | baseline |  |
| Popularity baseline | -11.000 | 0.1111 | 0.0222 | 0.0222 | 0.0188 | baseline |  |
| Random baseline | -13.000 | 0.1000 | 0.0200 | 0.0200 | 0.0146 | baseline |  |
| DQN + recency prior boost=5 (selected by validation) | -13.000 | 0.1000 | 0.0200 | 0.0200 | 0.0131 | dqn | 5.0 |

## Interpretation

- Models are selected using validation HitRate@5, not test HitRate@5.
- The test split is used only for final reporting after model selection.
- If the selected model uses recent_boost, report it as **DQN + recency prior**, not pure DQN.
- If Recent-item baseline is still best on test, the current DQN setup does not outperform the strongest heuristic baseline.