# GAN Product Recommendation - Local VS Code

Ban nay dung de so sanh mo hinh GAN voi pipeline DQN hien tai cua nhom.

## 1. Chuan bi du lieu

Dat 4 file Instacart CSV vao:

```text
data/raw/orders.csv
data/raw/products.csv
data/raw/order_products__prior.csv
data/raw/order_products__train.csv
```

## 2. Cai thu vien

```powershell
pip install -r requirements.txt
```

Neu may co CUDA, PyTorch can dung ban phu hop voi driver/GPU cua ban.

Trong VS Code, nen tao virtual environment rieng:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Neu PowerShell bao loi `running scripts is disabled`, dung 1 trong 2 cach sau:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run_gan_local.py --device cpu --n_users 500 --epochs 3
```

Hoac mo CMD thay vi PowerShell:

```cmd
.venv\Scripts\activate.bat
pip install -r requirements.txt
python run_gan_local.py --device cpu --n_users 500 --epochs 3
```

## 3. Chay toan bo pipeline GAN

Lenh nhanh de tien hanh preprocess, split, train va test:

```powershell
python run_gan_local.py --device cpu
```

Neu co GPU:

```powershell
python run_gan_local.py --device cuda
```

Mac dinh code giu top 1000 san pham pho bien nhat, `state_size=5`, `top_k=5`.
Voi full dataset, epoch dau co the lau vi tao sample va train rat nhieu batch. Terminal se in tien do theo batch.

Neu chi muon test nhanh truoc:

```powershell
python run_gan_local.py --device cpu --n_users 500 --epochs 3
```

Neu da preprocess roi va chi muon train lai:

```powershell
python run_gan_local.py --skip_preprocess --device cuda --epochs 30 --batch_size 1024
```

Neu CSV dang nam o thu muc khac:

```powershell
python run_gan_local.py --raw_dir "D:\duong_dan\toi\thu_muc_csv" --device cpu --n_users 500 --epochs 3
```

## 4. Ket qua

Sau khi chay xong, xem:

```text
outputs/logs/gan_validation_results.csv
outputs/logs/gan_test_results.csv
outputs/logs/gan_train_log.csv
outputs/checkpoints/gan_generator_best.pth
```

Metric chinh de so sanh voi DQN la `hit_rate_at_5`.

## 5. So sanh voi ket qua DQN cu

Neu ban co file tu notebook DQN cu:

```text
outputs/logs/final_test_results.csv
```

hay copy vao dung duong dan tren, roi chay:

```powershell
python compare_results.py
```

File tong hop se nam o:

```text
outputs/logs/model_comparison.csv
```

## 6. Ve bieu do GAN

Sau khi train/evaluate xong, chay:

```powershell
python visualize_gan_results.py
```

Anh PNG se nam trong:

```text
outputs/plots/gan_training_loss.png
outputs/plots/gan_validation_metrics.png
outputs/plots/gan_test_metrics.png
outputs/plots/gan_vs_dqn_metrics.png
```

## GAN trong bai nay la gi?

GAN gom 2 mang hoc cung luc:

- Generator: nhin lich su mua gan day cua user va de xuat san pham tiep theo.
- Discriminator: phan biet san pham that trong du lieu voi san pham do Generator sinh ra.

Trong recommender system, GAN giup mo hinh hoc cach sinh recommendation giong hanh vi mua that. De ket qua on dinh hon cho do an, Generator trong code nay duoc train bang ca adversarial loss va supervised cross-entropy loss.
