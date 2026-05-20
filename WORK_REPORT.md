# Bao cao cong viec da thuc hien

## De tai

Huấn luyện mô hình Deep Reinforcement Learning gợi ý sản phẩm cá nhân hoá trên sàn thương mại điện tử với bộ dữ liệu Instacart Market Basket Analysis.

## Pham vi da lam

### 1. Kiem tra va chuan bi du lieu

- Kiem tra folder `data/` va xac nhan cac file CSV Instacart da duoc giai nen.
- Doc schema cac file chinh: `orders.csv`, `order_products__prior.csv`, `order_products__train.csv`, `products.csv`.
- Xay dung script tien xu ly `prepare.py` de tao episode RL tu lich su mua hang.
- Ket qua tien xu ly hien tai:
  - `max_users = 2000`
  - `action_size = 500`
  - tao duoc `1722` episodes hop le

### 2. Thiet ke moi truong Reinforcement Learning

Da cai dat moi truong trong `src/instacart_drl/environment.py`.

- **State**:
  - vector lich su mua hang cua user tu prior orders
  - ngu canh order gom `order_number`, `order_dow`, `order_hour_of_day`, `days_since_prior_order`
  - mask cac san pham da duoc agent goi y trong episode
- **Action**:
  - chon 1 san pham trong action space
  - action space gom top 500 san pham pho bien nhat
- **Reward**:
  - `+1.0` neu san pham goi y nam trong gio hang that
  - `-0.1` neu goi y sai
  - `-0.5` neu goi y trung san pham da chon trong episode
- **Episode**:
  - moi episode tuong ung voi 1 user
  - agent goi y top-N san pham, mac dinh top-10

### 3. Cai dat DQN

Da cai dat cac thanh phan DQN chinh:

- Q-network bang PyTorch trong `models.py`
- epsilon-greedy action selection trong `train.py`
- replay buffer trong `replay.py`
- target network trong `train.py`
- update Q-network bang Huber loss
- gradient clipping de giam bat on training

### 4. Training pipeline

Da cai dat pipeline train trong `src/instacart_drl/train.py` theo flow:

```text
state -> action -> env.step() -> reward -> replay buffer -> train DQN
```

Trong qua trinh kiem tra, ban train dau tien bi loss tang manh. Da dieu chinh:

- khong cho agent chon lai san pham da goi y trong cung episode
- mask action da chon khi tinh target Q-value
- giam learning rate mac dinh xuong `1e-4`
- dung Matplotlib backend `Agg` de xuat PNG on dinh tren moi truong khong co GUI

### 5. Theo doi training

Da luu:

- `training_metrics.csv`
- `training_curves.png`

Ket qua train moi:

```text
epoch 1: reward -0.8416, loss 0.00456, hit_rate 1.44%
epoch 8: reward -0.2608, loss 0.00497, hit_rate 6.72%
```

Nhan xet:

- reward trung binh tang dan, cho thay agent hoc duoc tin hieu goi y
- loss duy tri nho va khong bi divergence

### 6. Goi y san pham top-N

Da cai dat `src/instacart_drl/recommend.py`.

Lenh vi du:

```powershell
python -m instacart_drl.recommend --model artifacts/dqn_model.pt --episodes artifacts/episodes.pt --user-id 252 --top-n 10
```

Script tra ve danh sach top-N san pham goi y cho user hop le trong tap episode da chuan bi.

### 7. Danh gia mo hinh

Da cai dat `src/instacart_drl/evaluate.py`.

Ket qua danh gia top-10:

```text
users: 1722
top_n: 10
total_hits: 1235
hit_rate_at_10: 50.93%
precision_at_10: 7.17%
recall_at_10: 16.41%
```

Nhan xet:

- Khoang 50.93% user co it nhat 1 san pham trong top-10 trung voi gio hang that.
- Ket qua phu hop cho ban demo/bao cao giai doan dau voi DQN, action space 500 san pham va 8 epoch train.

## Cac file chinh

```text
README.md
requirements.txt
pyproject.toml
src/instacart_drl/config.py
src/instacart_drl/environment.py
src/instacart_drl/models.py
src/instacart_drl/prepare.py
src/instacart_drl/train.py
src/instacart_drl/recommend.py
src/instacart_drl/evaluate.py
src/instacart_drl/replay.py
src/instacart_drl/utils.py
```

## Trang thai hien tai

Project hien da dap ung cac yeu cau:

- co moi truong RL ro rang `state/action/reward`
- co DQN bang PyTorch
- co replay buffer, epsilon-greedy, target network
- train duoc va xuat loss/reward curve
- recommend top-N san pham
- danh gia bang HitRate@10, Precision@10, Recall@10

