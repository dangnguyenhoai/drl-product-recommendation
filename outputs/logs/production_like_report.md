# Production-like Experiment Report

## Results

| method                     |   average_reward |   hit_rate_at_5 |   eval_episodes |   train_episodes |       lr |   gamma |   batch_size |   memory_size |   target_update_freq |   embedding_dim |   hidden_dim |   recent_boost |
|:---------------------------|-----------------:|----------------:|----------------:|-----------------:|---------:|--------:|-------------:|--------------:|---------------------:|----------------:|-------------:|---------------:|
| Recent-item baseline       |           -0.251 |          0.0582 |            1000 |              nan | nan      |   nan   |          nan |           nan |                  nan |             nan |          nan |            nan |
| DQN + recency prior stable |           -4.785 |          0.0357 |            1000 |             5000 |   0.0001 |     0.9 |           64 |         50000 |                  200 |              32 |          128 |              5 |
| Popularity baseline        |           -9.26  |          0.0137 |            1000 |              nan | nan      |   nan   |          nan |           nan |                  nan |             nan |          nan |            nan |
| DQN pure stable            |          -10.79  |          0.0055 |            1000 |             5000 |   0.0001 |     0.9 |           64 |         50000 |                  200 |              32 |          128 |              0 |
| Random baseline            |          -10.803 |          0.0054 |            1000 |              nan | nan      |   nan   |          nan |           nan |                  nan |             nan |          nan |            nan |

## Interpretation

- Best method by HitRate@5: **Recent-item baseline** with HitRate@5 = **0.0582**.\n- If a model uses `recent_boost`, report it as **DQN + recency prior**, not pure DQN.\n- If Recent-item baseline is still best, pure DQN is not production-ready for this dataset.\n