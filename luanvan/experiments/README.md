# FL Experiments - Hướng dẫn sử dụng

> Documentation cho Fed-M3, Fed-DGD, FedProx và FedAvg experiments.

---

## Mục lục

1. [Data Split Strategy](#data-split-strategy)
2. [Cách chạy nhanh](#cách-chạy-nhanh)
3. [run_experiment.py - Tham số chi tiết](#run_experimentpy---tham-số-chi-tiết)
4. [run_comparison.py - Tham số chi tiết](#run_comparisonpy---tham-số-chi-tiết)
5. [exp1_global_accuracy.py - So sánh Global Accuracy](#exp1_global_accuracypy---so-sánh-global-accuracy)
6. [hyperparam_search.py - Hyperparameter Grid Search](#hyperparam_searchpy---hyperparameter-grid-search)
7. [Experiment Matrix (Recommended)](#experiment-matrix-recommended)
8. [Troubleshooting](#troubleshooting)
9. [Files quan trọng](#files-quan-trọng)

---

## Data Split Strategy

```
┌─────────────────────────────────────────────────────────────────────┐
│  CIFAR-10 (Original Train: 50,000 images)                           │
│  ─────────────────────────────────────────────────────────────────  │
│  Split 80-10-10:                                                    │
│       ├── Train (40,000) → Chia cho N clients (Dirichlet/etc)      │
│       ├── Validation (5,000) → Hyperparameter tuning                │
│       └── Test (5,000) → Final evaluation (global model)            │
│                                                                      │
│  FASHION-MNIST (Original Train: 60,000 images)                      │
│  ─────────────────────────────────────────────────────────────────  │
│  Split 80-10-10:                                                    │
│       ├── Train (48,000) → Chia cho N clients                       │
│       ├── Validation (6,000) → Hyperparameter tuning                │
│       └── Test (6,000) → Final evaluation                           │
└─────────────────────────────────────────────────────────────────────┘

Tỷ lệ: Train 80% | Validation 10% | Test 10%

Seed đảm bảo:
  - Cùng seed → Cùng train/val/test split
  - Cùng seed → Cùng client data split
  → Reproducible experiments!
```

### Tại sao cần Validation set?

| Set | Mục đích |
|-----|----------|
| **Train** | Chia cho clients để train local |
| **Validation** | Tuning hyperparameters (lr, alpha, lambda, etc.) |
| **Test** | Đánh giá cuối cùng, KHÔNG dùng để tune |

**Quan trọng:** Test set chỉ dùng để báo cáo kết quả cuối cùng!

---

## Cách chạy nhanh

```bash
cd luanvan/experiments

# 1. Test isolation (kiểm tra code đúng)
python test_isolation.py

# 2. Chạy FedAvg baseline
python run_experiment.py --method fedavg --dataset cifar10

# 3. Chạy Fed-M3
python run_experiment.py --method fed_m3 --dataset cifar10

# 4. So sánh methods
python run_comparison.py --dataset cifar10 --methods fedavg fed_m3
```

---

## run_experiment.py - Tham số chi tiết

### Cú pháp
```bash
python run_experiment.py [OPTIONS]
```

### Bảng tham số

| Tham số | Kiểu | Mặc định | Mô tả |
|---------|------|----------|-------|
| `--method` | str | `fedavg` | Phương pháp FL: `fedavg`, `fed_m3`, `fed_dgd`, `fedprox` |
| `--dataset` | str | `cifar10` | Dataset: `cifar10` hoặc `fmnist` |
| `--num-clients` | int | `10` | Số lượng clients trong FL |
| `--num-rounds` | int | `100` | Số communication rounds |
| `--local-epochs` | int | `5` | Số epochs train local mỗi round |
| `--batch-size` | int | `32` | Batch size cho local training |
| `--lr` | float | `0.01` | Learning rate |
| `--non-iid` | str | `dirichlet` | Loại non-IID: `dirichlet`, `quantity`, `iid` |
| `--alpha` | float | `0.5` | Dirichlet α (nhỏ = more non-IID) |
| `--beta1` | float | `0.9` | Fed-M3: Fast momentum coefficient |
| `--beta2` | float | `0.999` | Fed-M3: Second moment coefficient |
| `--beta3` | float | `0.9` | Fed-M3: Slow momentum coefficient (server) |
| `--lam` | float | `0.3` | Fed-M3: Balance factor (local vs global) |
| `--ns-steps` | int | `5` | Fed-M3: Newton-Schulz iterations |
| `--seed` | int | `42` | Random seed cho reproducibility |
| `--device` | str | `auto` | Device: `cuda`, `cpu`, hoặc `auto` |
| `--save-dir` | str | `./results` | Thư mục lưu kết quả |

### Giải thích chi tiết

#### Non-IID Alpha (`--alpha`)
```
α = 0.1  → Very non-IID (mỗi client chỉ có 1-2 classes dominant)
α = 0.5  → Moderate non-IID (default, recommended cho main experiments)
α = 1.0  → Mild non-IID (các client có nhiều classes hơn)
α = 10.0 → Near IID (gần như uniform distribution)
```

#### Fed-M3 Lambda (`--lam`)
```
λ = 0.0  → Chỉ dùng local direction (o1), bỏ qua global
λ = 0.3  → Default: 70% local + 30% global (recommended)
λ = 0.5  → Balance: 50% local + 50% global
λ = 1.0  → Ưu tiên global direction nhiều hơn
```

#### Fed-M3 Beta values
```
β1 (beta1) = 0.9   → Fast momentum decay (local, mỗi client)
β2 (beta2) = 0.999 → Second moment decay (cho normalization)
β3 (beta3) = 0.9   → Slow momentum decay (server, global)
```

### Ví dụ

```bash
# Experiment 1: FedAvg trên CIFAR-10, moderate non-IID
python run_experiment.py \
    --method fedavg \
    --dataset cifar10 \
    --num-clients 10 \
    --num-rounds 100 \
    --alpha 0.5

# Experiment 2: Fed-M3 trên CIFAR-10, severe non-IID
python run_experiment.py \
    --method fed_m3 \
    --dataset cifar10 \
    --num-clients 10 \
    --num-rounds 100 \
    --alpha 0.1 \
    --lam 0.3 \
    --beta3 0.9

# Experiment 3: Fed-M3 trên FMNIST, IID baseline
python run_experiment.py \
    --method fed_m3 \
    --dataset fmnist \
    --non-iid iid \
    --num-rounds 50

# Experiment 4: Quick test (ít rounds)
python run_experiment.py \
    --method fedavg \
    --dataset cifar10 \
    --num-rounds 10 \
    --local-epochs 1
```

### Output mẫu (Per-Client Results)

```
──────────────────────────────────────────────────────────────────────
Round 10
──────────────────────────────────────────────────────────────────────
  Global Test Acc: 45.23% | Test Loss: 1.5432 | Avg Train Loss: 1.2345

  Per-Client Results (with global model on local data):
  Client   Samples    Train Loss   Local Acc    Global Acc
  ------------------------------------------------------
  0        5234       1.1234       52.34        48.23
  1        4521       1.2345       48.12        45.67
  2        6123       0.9876       55.67        51.23
  ...

  Summary:
    Acc Range: 42.12% - 55.67%
    Acc Std:   4.23%
    Loss Range: 0.9876 - 1.4567
```

**Giải thích các cột:**
| Cột | Ý nghĩa |
|-----|---------|
| `Client` | ID của client |
| `Samples` | Số samples local của client |
| `Train Loss` | Loss sau khi train local |
| `Local Acc` | Accuracy của LOCAL model trên local data |
| `Global Acc` | Accuracy của GLOBAL model trên local data của client |

**Tại sao cần cả Local Acc và Global Acc?**
- `Local Acc`: Cho thấy client học tốt thế nào trên data của mình
- `Global Acc`: Cho thấy global model hoạt động tốt thế nào cho từng client
- Nếu `Local Acc >> Global Acc`: Client bị overfit trên local data
- Nếu `Global Acc` variance cao: Non-IID ảnh hưởng nhiều

---

## run_comparison.py - Tham số chi tiết

### Cú pháp
```bash
python run_comparison.py [OPTIONS]
```

### Bảng tham số

| Tham số | Kiểu | Mặc định | Mô tả |
|---------|------|----------|-------|
| `--dataset` | str | `cifar10` | Dataset: `cifar10` hoặc `fmnist` |
| `--methods` | list | `fedavg fed_m3` | Các methods cần so sánh |
| `--num-clients` | int | `10` | Số lượng clients |
| `--num-rounds` | int | `100` | Số communication rounds |
| `--local-epochs` | int | `5` | Số epochs local mỗi round |
| `--batch-size` | int | `32` | Batch size |
| `--lr` | float | `0.01` | Learning rate |
| `--non-iid` | str | `dirichlet` | Loại non-IID |
| `--alpha` | float | `0.5` | Dirichlet α |
| `--seed` | int | `42` | Random seed |
| `--device` | str | `auto` | Device |
| `--save-dir` | str | `./results` | Thư mục lưu |

### Điểm quan trọng

```
┌─────────────────────────────────────────────────────────────┐
│  QUAN TRỌNG: run_comparison.py đảm bảo FAIR COMPARISON     │
│                                                              │
│  • CÙNG data split cho tất cả methods (same seed)          │
│  • CÙNG model initialization                                │
│  • CHỈ khác: optimizer/algorithm                            │
│                                                              │
│  → Sự khác biệt về accuracy chỉ do algorithm!              │
└─────────────────────────────────────────────────────────────┘
```

### Ví dụ

```bash
# So sánh FedAvg vs Fed-M3 trên CIFAR-10
python run_comparison.py \
    --dataset cifar10 \
    --methods fedavg fed_m3 \
    --alpha 0.5 \
    --num-rounds 100

# So sánh trên severe non-IID
python run_comparison.py \
    --dataset cifar10 \
    --methods fedavg fed_m3 \
    --alpha 0.1 \
    --num-rounds 150

# Quick comparison (testing)
python run_comparison.py \
    --dataset fmnist \
    --methods fedavg fed_m3 \
    --num-rounds 20 \
    --local-epochs 2
```

### Output

Sau khi chạy, sẽ có:
```
results/comparison/<dataset>_<non_iid>_a<alpha>_<timestamp>/
├── data_distribution.png    # Biểu đồ phân bố data
├── comparison_plot.png      # So sánh accuracy curves
├── summary.txt              # Tóm tắt kết quả
├── fedavg/                  # Kết quả FedAvg
│   ├── metrics_*.json
│   └── model_*.pt
└── fed_m3/                  # Kết quả Fed-M3
    ├── metrics_*.json
    └── model_*.pt
```

---

## exp1_global_accuracy.py - So sánh Global Accuracy

### Mục tiêu
So sánh **Global Accuracy** của các methods trên test set với nhiều mức độ non-IID khác nhau.

### Cú pháp
```bash
python exp1_global_accuracy.py [OPTIONS]
```

### Bảng tham số

| Tham số | Kiểu | Mặc định | Mô tả |
|---------|------|----------|-------|
| `--config` | str | `configs/exp1_config.json` | File config |
| `--methods` | list | all | Methods cần chạy: `fedavg`, `fed_m3`, `fed_dgd`, `fedprox` |
| `--alphas` | list | all | Các giá trị alpha: `0.1`, `0.5`, `1.0` |
| `--num-rounds` | int | from config | Override số rounds |
| `--dry-run` | flag | - | Xem trước config, không chạy |
| `--export-config` | str | - | Export config ra file |

### Ví dụ

```bash
# Chạy tất cả (4 methods x 3 alphas = 12 runs)
python exp1_global_accuracy.py

# Xem trước config (không chạy)
python exp1_global_accuracy.py --dry-run

# Chạy nhanh để test (10 rounds, 1 method, 1 alpha)
python exp1_global_accuracy.py --num-rounds 10 --methods fedavg --alphas 0.5

# Chỉ chạy 2 methods
python exp1_global_accuracy.py --methods fedavg fed_m3

# Chỉ chạy 1 alpha
python exp1_global_accuracy.py --alphas 0.5

# Dùng config khác
python exp1_global_accuracy.py --config configs/my_config.json

# Export config để chỉnh sửa
python exp1_global_accuracy.py --export-config my_config.json
```

### Output
```
results/exp1_global_accuracy/
├── {method}/cifar10_dirichlet_a{alpha}/
│   ├── metrics_*.json      # Metrics chi tiết
│   └── model_*.pt          # Model weights
├── config_*.json           # Config đã dùng
└── summary_*.json          # Tóm tắt kết quả
```

---

## hyperparam_search.py - Hyperparameter Grid Search

### Mục tiêu
Tìm bộ tham số tốt nhất cho mỗi method bằng cách chạy grid search.

### Cú pháp
```bash
python hyperparam_search.py [OPTIONS]
```

### Bảng tham số

| Tham số | Kiểu | Mặc định | Mô tả |
|---------|------|----------|-------|
| `--config` | str | `configs/hyperparam_search.json` | File config grid search |
| `--method` | list | all | Methods cần search: `fed_m3`, `fed_dgd`, `fedprox` |
| `--dry-run` | flag | - | Xem trước commands, không chạy |
| `--no-confirm` | flag | - | Bỏ qua xác nhận trước khi chạy |

### Config file (`configs/hyperparam_search.json`)

```json
{
  "base_config": {
    "dataset": "cifar10",
    "alpha": 0.5,
    "num_clients": 10,
    "num_rounds": 50,
    "local_epochs": 5,
    "batch_size": 64,
    "lr": 0.01,
    "seed": 42
  },
  "grid": {
    "fed_m3": {
      "beta1": [0.9, 0.99],
      "beta3": [0.9, 0.99],
      "lam": [0.1, 0.3, 0.5]
    },
    "fed_dgd": {
      "decay_strength": [0.05, 0.1, 0.2]
    },
    "fedprox": {
      "mu": [0.001, 0.01, 0.1]
    }
  },
  "output": {
    "save_dir": "./results/hyperparam_search"
  }
}
```

### Ví dụ

```bash
# Xem trước tất cả combinations (dry-run)
python hyperparam_search.py --dry-run

# Chạy full grid search
python hyperparam_search.py

# Chạy cho 1 method cụ thể
python hyperparam_search.py --method fed_m3
python hyperparam_search.py --method fed_dgd
python hyperparam_search.py --method fedprox

# Bỏ qua xác nhận
python hyperparam_search.py --no-confirm

# Dùng config khác
python hyperparam_search.py --config configs/my_grid.json
```

### Output bảng kết quả

```
================================================================================
RESULTS FOR FED_M3
================================================================================
       beta1 |        beta3 |          lam | Server Acc (Final) | Server Acc (Best) | Mean Local Acc (Final) | ...
---------------------------------------------------------------------------------------------------------------------
         0.9 |          0.9 |          0.3 |             65.42% |            67.85% |                 72.15% | ...
         0.9 |          0.9 |          0.5 |             64.18% |            66.92% |                 71.33% | ...
...

Best params (by Server Best): {'beta1': 0.9, 'beta3': 0.9, 'lam': 0.3}
  Server:     Final=65.42%  Best=67.85%
  Local Acc:  Final=72.15%  Best=73.22%
  Global Acc: Final=58.33%  Best=60.12%
```

### Giải thích các cột

| Cột | Ý nghĩa |
|-----|---------|
| **Server Acc (Final)** | Accuracy của global model trên test set ở round cuối |
| **Server Acc (Best)** | Accuracy cao nhất của global model qua tất cả rounds |
| **Mean Local Acc (Final)** | Trung bình accuracy của local model trên local data (round cuối) |
| **Mean Local Acc (Best)** | Trung bình Local Acc cao nhất qua tất cả rounds |
| **Mean Global Acc (Final)** | Trung bình accuracy của global model trên local data của clients (round cuối) |
| **Mean Global Acc (Best)** | Trung bình Global Acc cao nhất qua tất cả rounds |

### Output files
```
results/hyperparam_search/
├── {method}/cifar10_dirichlet_a{alpha}/
│   └── metrics_*.json      # Metrics cho mỗi combination
└── summary_*.json          # Tóm tắt với best params cho mỗi method
```

---

## Experiment Matrix (Recommended)

| # | Dataset | Non-IID | Alpha | Command |
|---|---------|---------|-------|---------|
| 1 | CIFAR-10 | IID | - | `--non-iid iid` |
| 2 | CIFAR-10 | Dirichlet | 1.0 | `--alpha 1.0` |
| 3 | CIFAR-10 | Dirichlet | 0.5 | `--alpha 0.5` |
| 4 | CIFAR-10 | Dirichlet | 0.1 | `--alpha 0.1` |
| 5 | CIFAR-10 | Quantity | - | `--non-iid quantity` |
| 6 | FMNIST | Dirichlet | 0.5 | `--dataset fmnist --alpha 0.5` |

### Script chạy tất cả experiments

```bash
#!/bin/bash
# run_all_experiments.sh

# CIFAR-10 experiments
for alpha in 0.1 0.5 1.0; do
    python run_comparison.py \
        --dataset cifar10 \
        --methods fedavg fed_m3 \
        --alpha $alpha \
        --num-rounds 100
done

# IID baseline
python run_comparison.py \
    --dataset cifar10 \
    --methods fedavg fed_m3 \
    --non-iid iid \
    --num-rounds 100

# Quantity skew
python run_comparison.py \
    --dataset cifar10 \
    --methods fedavg fed_m3 \
    --non-iid quantity \
    --num-rounds 100
```

---

## Troubleshooting

### CUDA out of memory
```bash
# Giảm batch size
python run_experiment.py --batch-size 16

# Hoặc dùng CPU
python run_experiment.py --device cpu
```

### Training quá chậm
```bash
# Giảm số rounds và local epochs để test
python run_experiment.py --num-rounds 10 --local-epochs 1
```

### Kết quả không reproducible
```bash
# Đảm bảo dùng cùng seed
python run_experiment.py --seed 42
python run_experiment.py --seed 42  # Phải cho kết quả giống nhau
```

---

## Files quan trọng

### Scripts chính
| File | Mô tả |
|------|-------|
| `run_experiment.py` | Chạy 1 experiment với 1 method |
| `run_comparison.py` | So sánh nhiều methods (cùng data split) |
| `exp1_global_accuracy.py` | Experiment 1: So sánh Global Accuracy |
| `hyperparam_search.py` | Grid search tìm hyperparameters tốt nhất |

### FL Framework
| File | Mô tả |
|------|-------|
| `fl/client.py` | FL Client |
| `fl/server.py` | FL Server |
| `fl/data_split.py` | Non-IID data split |
| `fl/aggregators.py` | Aggregation functions |

### Models & Optimizers
| File | Mô tả |
|------|-------|
| `models/cnn.py` | CNN models (CNNSmall, CNNMedium) |
| `optimizers/fed_m3.py` | Fed-M3 optimizer |
| `optimizers/fed_dgd.py` | Fed-DGD optimizer |
| `optimizers/fedprox.py` | FedProx optimizer |

### Utils & Configs
| File | Mô tả |
|------|-------|
| `utils/metrics.py` | MetricsTracker |
| `utils/seed.py` | set_seed() |
| `configs/exp1_config.json` | Config cho Experiment 1 |
| `configs/hyperparam_search.json` | Config cho Grid Search |

---

*Cập nhật: 2026-04-05*
