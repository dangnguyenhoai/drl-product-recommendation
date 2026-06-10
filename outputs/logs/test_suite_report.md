# Test Suite Report

## Results

| Method | Avg Reward | HitRate@5 | Type | Recent Boost |
|---|---:|---:|---|---:|
| Recent-item baseline | 1.863 | 0.2828 | baseline |  |
| DQN C + recent_boost=2 | 0.497 | 0.2443 | dqn | 2.0 |
| DQN C + recent_boost=5 | -1.633 | 0.2017 | dqn | 5.0 |
| Popularity baseline | -5.130 | 0.0994 | baseline |  |
| DQN C pure | -6.097 | 0.0904 | dqn | 0.0 |
| Random baseline | -6.090 | 0.0790 | baseline |  |

## Interpretation

- Best method by HitRate@5: **Recent-item baseline** with HitRate@5 = **0.2828**.
- Pure DQN does **not** outperform the Recent-item baseline. Do not claim pure DQN is the best model.
- DQN + recent_boost=5 improves over pure DQN but still does not outperform the Recent-item baseline.

Conclusion: if a model uses `recent_boost`, it must be reported as **DQN + recency prior** or **Hybrid DQN**, not pure DQN.