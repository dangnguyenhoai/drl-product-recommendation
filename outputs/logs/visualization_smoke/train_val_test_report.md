# Validation/Test Evaluation Report

## Validation Results

| Method | Avg Reward | HitRate@5 | Precision@5 | Recall@5 | NDCG@5 | Recent Boost |
|---|---:|---:|---:|---:|---:|---:|
| DQN + recency prior boost=2 | 4.000 | 0.6667 | 0.1333 | 0.1333 | 0.1131 | 2.0 |
| DQN pure stable | -0.500 | 0.2500 | 0.0500 | 0.0500 | 0.0328 | 0.0 |
| DQN + recency prior boost=5 | -9.500 | 0.0769 | 0.0154 | 0.0154 | 0.0130 | 5.0 |

## Selected Model

- Selected by validation HitRate@5: **DQN + recency prior boost=2**
- Model path: `outputs\checkpoints\dqn_recency2_stable.pth`
- Recent boost: `2.0`

## Final Test Results

| Method | Avg Reward | HitRate@5 | Precision@5 | Recall@5 | NDCG@5 | Type | Recent Boost |
|---|---:|---:|---:|---:|---:|---|---:|
| Popularity baseline | -2.000 | 1.0000 | 0.2000 | 0.2195 | 0.2227 | baseline |  |
| Recent-item baseline | 8.000 | 0.3636 | 0.1091 | 0.1364 | 0.1513 | baseline |  |
| DQN + recency prior boost=2 (selected by validation) | -12.000 | 0.1053 | 0.0211 | 0.0217 | 0.0248 | dqn | 2.0 |
| Random baseline | -13.000 | 0.1000 | 0.0200 | 0.0200 | 0.0170 | baseline |  |

## Interpretation

- Models are selected using validation HitRate@5, not test HitRate@5.
- The test split is used only for final reporting after model selection.
- If the selected model uses recent_boost, report it as **DQN + recency prior**, not pure DQN.
- If Recent-item baseline is still best on test, the current DQN setup does not outperform the strongest heuristic baseline.