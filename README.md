# Deep Reinforcement Learning Product Recommendation on Instacart

Project huan luyen DQN de goi y top-N san pham ca nhan hoa tren bo du lieu Instacart Market Basket Analysis.

## Y tuong bai toan RL

- **State**: lich su mua hang cua user tren `prior` orders, thong tin ngu canh don hang tiep theo, va mask cac san pham da duoc agent goi y trong episode.
- **Action**: chon 1 san pham trong action space gom `K` san pham pho bien nhat.
- **Reward**:
  - `+1.0` neu san pham duoc goi y nam trong gio hang thuc te cua order `train`.
  - `-0.1` neu goi y sai.
  - `-0.5` neu goi y trung san pham da chon trong cung episode.
- **Episode**: mot user co 1 order `train`; agent chon toi da `top_n` san pham.

## Cai dat

Neu may ban da co Python:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

Neu terminal cua ban dung Python launcher:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

Trong workspace nay minh da tao san `.venv`. Neu muon dung ngay:

```powershell
.\.venv\Scripts\Activate.ps1
```

Luu y: ban Torch cai mac dinh bang pip trong workspace hien la CPU. Neu muon dung GPU NVIDIA, hay cai lai Torch ban CUDA phu hop voi driver/CUDA tren may ban theo PyTorch install selector.

## Chay nhanh

Du lieu CSV da nam trong `data/`.

```powershell
python -m instacart_drl.prepare --data-dir data --output artifacts/episodes.pt --max-users 2000 --action-size 500
python -m instacart_drl.train --episodes artifacts/episodes.pt --output-dir artifacts --epochs 8
python -m instacart_drl.recommend --model artifacts/dqn_model.pt --episodes artifacts/episodes.pt --user-id 1 --top-n 10
```

Neu muon train lon hon:

```powershell
python -m instacart_drl.prepare --data-dir data --output artifacts/episodes.pt --max-users 10000 --action-size 1000
python -m instacart_drl.train --episodes artifacts/episodes.pt --output-dir artifacts --epochs 20 --batch-size 256
```

## Output

- `artifacts/episodes.pt`: tap episode da tien xu ly.
- `artifacts/dqn_model.pt`: checkpoint mo hinh DQN.
- `artifacts/training_metrics.csv`: loss va reward theo epoch.
- `artifacts/training_curves.png`: bieu do loss/reward.
