# Benchmark Summary

## Dataset

- Number of users: 93
- Total interactions: 5271
- Number of unique items: 466
- Min item id: 0
- Max item id: 498
- Valid actions: 466
- Action dimension: 499
- Eligible users for environment: 80

## Evaluation Setting

- State size: 5
- Top-K: 5
- Evaluation episodes: 100
- Reward:
  - Hit reward: +5
  - Miss penalty: -2
  - Duplicate recommendation penalty: -1
- Repeated recommendations inside an episode are masked.

## Results

| Method | Average Reward | HitRate@5 |
|---|---:|---:|
| Random | -12.800 | 0.0094 |
| Popularity | -12.240 | 0.0124 |
| DQN + item embedding + valid action mask + no-repeat action mask | -10.760 | 0.0149 |

## Policy Inspection

- Inspection episodes: 20
- Total hits: 8
- Total recommendations: 685
- Inspect HitRate@5: 0.0117

### Top Recommended Items

| Item | Recommended Count |
|---:|---:|
| 422 | 20 |
| 158 | 20 |
| 260 | 19 |
| 395 | 19 |
| 462 | 18 |
| 273 | 18 |
| 2 | 17 |
| 79 | 17 |
| 189 | 17 |
| 343 | 17 |

### Items That Actually Hit

| Item | Hit Count |
|---:|---:|
| 13 | 2 |
| 407 | 2 |
| 2 | 1 |
| 210 | 1 |
| 79 | 1 |
| 15 | 1 |

## Current Interpretation

The DQN model now outperforms both random and popularity baselines under the current environment. This is the first result in the project that is worth reporting. Earlier DQN variants were weaker than random, so they should not be presented as successful models.

However, the policy inspection still shows that recommendations concentrate around a limited group of items. The no-repeat action mask prevents repeated recommendations inside the same episode, but the learned policy is still not sufficiently diverse across users and states.

## Current Limitations

- Evaluation still uses the same processed history source instead of a strict train/test split.
- Reward is sparse and depends on a short future-item window.
- The model uses item id embeddings but no user embeddings or item metadata.
- The DQN replay update is still sample-by-sample, not vectorized.
- The recommendation policy still tends to concentrate on a small group of items.

## Temporal Train/Test Split Results

### Split Statistics

- Input users: 93
- Kept users: 37
- Skipped users: 56
- Train interactions: 3232
- Test interactions: 827
- Train users: 37
- Test users: 37
- Train unique items: 407
- Test unique items: 295
- Global action_dim: 499

### Test Results

| Method | Average Reward | HitRate@5 |
|---|---:|---:|
| Random | -6.530 | 0.0143 |
| Popularity | -6.240 | 0.0204 |
| DQN + item embedding + valid action mask + no-repeat action mask | -7.100 | 0.0150 |
| Recent-item baseline | 2.250 | 0.0713 |

### Interpretation

Under the stricter temporal train/test split, the DQN model slightly outperforms the random baseline in HitRate@5 but does not outperform the popularity baseline. This means the current DQN policy is not yet strong enough to claim superiority over a simple popularity-based recommender.

The previous full-history evaluation showed DQN outperforming both baselines, but that result should be treated as less reliable because training and evaluation used the same processed history source.

The recent-item baseline substantially outperforms the current DQN model. This indicates that the dataset has a strong short-term repetition signal, but the current DQN policy fails to exploit it. Therefore, the current DQN should not be claimed as the best-performing recommender under the temporal split setting.