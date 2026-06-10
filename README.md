# drl-product-recommendation

Training Deep Reinforcement Learning models to suggest personalized products on e-commerce platforms.

## Temporal Split Evaluation Workflow

The original indexed dataset uses sparse item ids:

- Global max item id: `498`
- Global action dimension: `499`
- Valid item actions: `466`

When training on `train_history.pkl` and evaluating on `test_history.pkl`, keep the DQN output size fixed with explicit `--action_dim 499`. Do not infer separate action dimensions from the train and test split files.

Generated files under `data/processed`, `outputs`, checkpoints, plots, logs, and `.pth` files should not be committed.

## Local Demo App

The demo reads project data from `data/`, writes compact demo JSON to
`data/demo/demo_data.json`, and serves a local web UI from `demo/`.

```powershell
python demo/build_demo_data.py

python -m http.server 8000 --bind 127.0.0.1
```

Open:

```text
http://127.0.0.1:8000/demo/index.html
```

Demo views:

- `Chọn sản phẩm`: choose one product and show similar recommended products with embedding similarity, co-occurrence, popularity, and final score.
- `Demo DQN`: choose a user/state case, show the 5 recent items, DQN Top-5 recommendations, real next-5 target items, hits, reward, and HitRate@5.
- `Kết quả mô hình`: show evaluation metrics from `outputs/logs/test_suite_results.csv`.

By default the demo uses `outputs/checkpoints/dqn_C_boost5.pth`. To rebuild from
another checkpoint:

```powershell
python demo/build_demo_data.py --model_path outputs/checkpoints/dqn_C_pure.pth --recent_boost 0
```

The default interactive cases come from `data/processed/test_history.pkl`, so the
demo may show only the held-out test users. To show more users during the live
demo while keeping the evaluation metrics on the test split, build cases from
the full indexed history:

```powershell
python demo/build_demo_data.py `
  --case_history_path data/processed/indexed_history.pkl `
  --max_case_users 1000 `
  --max_windows_per_user 20 `
  --max_cases 300
```

This only changes the interactive case picker. The model comparison table still
uses `outputs/logs/test_suite_results.csv`.

### 1. Inspect Original Data

```powershell
python -m utils.inspect_data `
  --data_path data/processed/indexed_history.pkl `
  --state_size 5 `
  --top_k 5
```

### 2. Create Temporal Train/Test Split

```powershell
python -m utils.split_history `
  --input_path data/processed/indexed_history.pkl `
  --train_output_path data/processed/train_history.pkl `
  --test_output_path data/processed/test_history.pkl `
  --test_ratio 0.2 `
  --min_history_len 11
```

Expected split summary:

- Input users: `93`
- Kept users: `37`
- Skipped users: `56`
- Train interactions: `3232`
- Test interactions: `827`

### 3. Inspect Train Split

```powershell
python -m utils.inspect_data `
  --data_path data/processed/train_history.pkl `
  --state_size 5 `
  --top_k 5
```

### 4. Inspect Test Split

```powershell
python -m utils.inspect_data `
  --data_path data/processed/test_history.pkl `
  --state_size 5 `
  --top_k 5
```

### 5. Train DQN On Train Split

```powershell
python -m training.train_dqn `
  --data_path data/processed/train_history.pkl `
  --episodes 200 `
  --action_dim 499 `
  --model_path outputs/checkpoints/dqn_split_weights.pth `
  --plot_dir outputs/plots `
  --log_path outputs/logs/train_split_log.csv `
  --epsilon 1.0 `
  --epsilon_min 0.05 `
  --epsilon_decay 0.98 `
  --batch_size 32 `
  --lr 0.0001 `
  --target_update_freq 50 `
  --top_k 5 `
  --embedding_dim 32 `
  --hidden_dim 128
```

### 6. Evaluate DQN On Test Split

```powershell
python -m evaluation.evaluate_dqn `
  --data_path data/processed/test_history.pkl `
  --model_path outputs/checkpoints/dqn_split_weights.pth `
  --episodes 100 `
  --action_dim 499 `
  --embedding_dim 32 `
  --hidden_dim 128
```

### 7. Evaluate Baselines On Test Split

```powershell
python -m evaluation.evaluate_baselines `
  --data_path data/processed/test_history.pkl `
  --episodes 100 `
  --top_k 5
```

### 8. Inspect Policy On Test Split

```powershell
python -m evaluation.inspect_policy `
  --data_path data/processed/test_history.pkl `
  --model_path outputs/checkpoints/dqn_split_weights.pth `
  --episodes 20 `
  --action_dim 499 `
  --embedding_dim 32 `
  --hidden_dim 128
```

### 9. Compare MLP And DQN On The Same Test Windows

Train the MLP first so its best validation checkpoint is saved:

```powershell
python -m baseline.baseline_train --action-dim 1000
```

Then load both model checkpoints into one evaluator. This reports both
immediate-next-item metrics and next-5-window metrics using exactly the same
test states, targets, valid actions, and formulas for both models:

```powershell
python -m evaluation.evaluate_models_common `
  --data-path data/processed/test_history.pkl `
  --mlp-model-path outputs/checkpoints/mlp_history_baseline.pth `
  --dqn-model-path outputs/checkpoints/dqn_pure_stable.pth `
  --state-size 5 `
  --top-k 5
```

Both checkpoints must use the same `action_dim` (`1000` for the current
train/validation/test data). To compare a DQN checkpoint trained with a recency
prior, pass its matching value with `--recent-boost`.

Results are saved to:

- `outputs/logs/common_model_comparison.csv`
- `outputs/logs/common_model_comparison.md`
