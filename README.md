# drl-product-recommendation

Training Deep Reinforcement Learning models to suggest personalized products on e-commerce platforms.

## Temporal Split Evaluation Workflow

The original indexed dataset uses sparse item ids:

- Global max item id: `498`
- Global action dimension: `499`
- Valid item actions: `466`

When training on `train_history.pkl` and evaluating on `test_history.pkl`, keep the DQN output size fixed with explicit `--action_dim 499`. Do not infer separate action dimensions from the train and test split files.

Generated files under `data/processed`, `outputs`, checkpoints, plots, logs, and `.pth` files should not be committed.

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
