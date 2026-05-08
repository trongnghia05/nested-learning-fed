# FL Experiments - Hướng dẫn sử dụng

> Documentation cho Fed-M3, Fed-DGD, FedProx và FedAvg experiments.

---

## Mục lục

1. [Data Split Strategy](#data-split-strategy)
2. [Scripts cơ bản](#scripts-cơ-bản)
3. [Hyperparameter Search](#hyperparameter-search)
4. [Thực Nghiệm](#thực-nghiệm)
   - [Experiment 1: Global Accuracy](#experiment-1-global-accuracy)
   - [Experiment 2: Personalized Accuracy](#experiment-2-personalized-accuracy)
5. [Troubleshooting](#troubleshooting)

---

## Data Split Strategy

```
┌─────────────────────────────────────────────────────────────────────┐
│  CIFAR-10 (Original Train: 50,000 images)                           │
│       ├── Train (40,000) → Chia cho N clients (Dirichlet)           │
│       ├── Validation (5,000) → Hyperparameter tuning                │
│       └── Test (5,000) → Final evaluation (global model)            │
│                                                                      │
│  FASHION-MNIST (Original Train: 60,000 images)                      │
│       ├── Train (48,000) → Chia cho N clients                       │
│       ├── Validation (6,000) → Hyperparameter tuning                │
│       └── Test (6,000) → Final evaluation                           │
└─────────────────────────────────────────────────────────────────────┘

Tỷ lệ: Train 80% | Validation 10% | Test 10%
```

| Set | Mục đích |
|-----|----------|
| **Train** | Chia cho clients để train local |
| **Validation** | Tuning hyperparameters — KHÔNG dùng để đánh giá cuối |
| **Test** | Báo cáo kết quả cuối cùng — KHÔNG dùng để tune |

**Seed đảm bảo:** Cùng seed → cùng train/val/test split → cùng client data split → reproducible!

---

## Scripts cơ bản

Hai file này là nền tảng cho toàn bộ framework. Các script cấp cao (exp1, exp2, hyperparam_search) đều gọi lại `run_experiment.py` bên trong.

### run_experiment.py — Chạy 1 experiment đơn lẻ

Chạy 1 method với 1 bộ config cụ thể. Dùng trực tiếp khi muốn thử nhanh hoặc debug.

```bash
# Chạy FedAvg trên CIFAR-10
python run_experiment.py --method fedavg --dataset cifar10 --alpha 0.5 --num-rounds 100

# Chạy Fed-M3 với best params
python run_experiment.py --method fed_m3 --dataset cifar10 --alpha 0.1 \
    --beta1 0.9 --beta3 0.5 --lam 0.5 --num-rounds 100

# Chạy Fed-DGD
python run_experiment.py --method fed_dgd --dataset cifar10 --alpha 0.5 \
    --dgd-decay-strength 0.05 --num-rounds 100

# Chạy FedProx
python run_experiment.py --method fedprox --dataset cifar10 --alpha 0.5 \
    --fedprox-mu 0.01 --num-rounds 100

# Quick test (2 rounds, CPU)
python run_experiment.py --method fedavg --dataset cifar10 --num-rounds 2 --device cpu
```

Các tham số chính:

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `--method` | `fedavg` | `fedavg`, `fed_m3`, `fed_dgd`, `fedprox` |
| `--dataset` | `cifar10` | `cifar10` hoặc `fmnist` |
| `--alpha` | `0.5` | Dirichlet α (0.1=severe, 0.5=moderate, 1.0=mild non-IID) |
| `--num-rounds` | `100` | Số communication rounds |
| `--num-clients` | `10` | Số clients |
| `--local-epochs` | `5` | Số epochs train local mỗi round |
| `--batch-size` | `512` | Batch size |
| `--lr` | `0.01` | Learning rate |
| `--seed` | `42` | Random seed |
| `--device` | `auto` | `cuda`, `mps`, `cpu`, hoặc `auto` |
| `--save-dir` | `./results` | Thư mục lưu kết quả |

---

### run_comparison.py — So sánh nhiều methods với cùng data split

Đảm bảo **fair comparison**: cùng seed → cùng data split → sự khác biệt accuracy chỉ do algorithm.

```bash
# So sánh FedAvg vs Fed-M3
python run_comparison.py --dataset cifar10 --methods fedavg fed_m3 --alpha 0.5

# So sánh tất cả 4 methods
python run_comparison.py --dataset cifar10 --methods fedavg fedprox fed_m3 fed_dgd --alpha 0.1
```

> **Lưu ý:** Với full experiments (24 runs), nên dùng `exp1_global_accuracy.py` thay vì `run_comparison.py` vì exp1 có thêm resume và report.

---

## Hyperparameter Search

### Tại sao cần bước này?

Mỗi method (Fed-M3, Fed-DGD, FedProx) có các hyperparameter riêng ảnh hưởng lớn đến kết quả.
Nếu so sánh các methods với hyperparameters mặc định hoặc chưa được tune, kết quả sẽ **không fair**.
Hyperparameter search đảm bảo mỗi method được chạy với **bộ tham số tốt nhất** trước khi so sánh.

```
Fed-M3:   beta1 × beta3 × lam  → 27 combinations
Fed-DGD:  decay_strength        →  3 combinations
FedProx:  mu                    →  3 combinations
```

Config: `configs/hyperparam_search.json`

### Best params đã tìm được

| Method | Params | Best Server Acc |
|--------|--------|----------------|
| Fed-M3 | beta1=0.9, beta3=0.5, lam=0.5 | 84.78% |
| Fed-DGD | decay_strength=0.05 | 73.98% |
| FedProx | mu=0.01 | 74.30% |

> Kết quả lưu tại `results/hyperparam_search/`

### Cách chạy

```bash
# Xem trước (dry-run)
python hyperparam_search.py --dry-run

# Chạy cho 1 method
python hyperparam_search.py --method fed_m3 --no-confirm
python hyperparam_search.py --method fed_dgd --no-confirm
python hyperparam_search.py --method fedprox --no-confirm

# Chạy tất cả
python hyperparam_search.py --no-confirm
```

---

## Thực Nghiệm

---

### Experiment 1: Global Accuracy

### Mục tiêu
So sánh **Global Accuracy** của 4 methods trên test set với các mức độ non-IID khác nhau.

### Kịch bản
- **Dataset:** CIFAR-10, FMNIST
- **Methods:** FedAvg, FedProx, Fed-M3, Fed-DGD
- **Non-IID:** Dirichlet α = {0.1, 0.5, 1.0}
- **Setup:** 10 clients, 100 rounds, 5 local epochs
- **Tổng:** 2 datasets × 4 methods × 3 alphas = **24 runs**

Config: `configs/exp1_config.json`

### Bước 1 — Chạy experiments

```bash
# Chạy tất cả (24 runs, ~4h/run trên server GPU)
python exp1_global_accuracy.py --no-confirm

# Chạy từng dataset riêng
python exp1_global_accuracy.py --dataset cifar10 --no-confirm
python exp1_global_accuracy.py --dataset fmnist --no-confirm

# Xem trước config (không chạy)
python exp1_global_accuracy.py --dry-run

# Quick test (2 rounds)
python exp1_global_accuracy.py --num-rounds 2 --methods fedavg --alphas 0.5 --no-confirm
```

### Bước 2 — Resume nếu bị dừng giữa chừng

Script `exp1_resume.py` tự động check kết quả nào đã có, chỉ chạy cái còn thiếu.

```bash
# Xem status (done/missing)
python exp1_resume.py --dry-run

# Chạy tiếp những cái còn thiếu
python exp1_resume.py --no-confirm

# Chỉ resume 1 dataset
python exp1_resume.py --dataset cifar10 --no-confirm
python exp1_resume.py --dataset fmnist --no-confirm
```

Output mẫu:
```
=================================================================
STATUS CHECK - CIFAR10
=================================================================
Method       | a=0.1  | a=0.5  | a=1.0
-----------------------------------------
FedAvg       | DONE   | DONE   | DONE
Fed-M3       | DONE   | DONE   | MISS
Fed-DGD      | MISS   | MISS   | MISS
FedProx      | MISS   | MISS   | MISS

Done:    7/12 | Missing: 5/12
```

### Bước 3 — Xem kết quả

Script `exp1_report.py` đọc tất cả metrics JSON và in bảng kết quả cho paper.

```bash
# In tất cả bảng + hiện plot
python exp1_report.py

# Chỉ in bảng, không plot
python exp1_report.py --no-plot

# Lưu plot và CSV
python exp1_report.py --save-csv --save-plots ./figures

# Chỉ 1 dataset
python exp1_report.py --dataset cifar10 --no-plot
```

Kết quả bao gồm:
- **Table 1 (Global Accuracy):** Final server test_acc của mỗi method/dataset/alpha
- **Table 3 (Convergence Speed):** Số rounds để đạt target accuracy (CIFAR-10: 70%, FMNIST: 85%)
- **Plots:** Accuracy curves theo rounds

### Output structure

```
results/exp1_global_accuracy/
├── {method}/{dataset}_dirichlet_a{alpha}/
│   ├── metrics_*.json      # Per-round metrics
│   └── model_*.pt          # Global model checkpoint
├── config_*.json           # Config đã dùng
└── summary_*.json          # Tóm tắt kết quả
```

---

### Experiment 2: Personalized Accuracy

### Mục tiêu
Đo **Personalized Accuracy** sau khi fine-tune global model trên local data của mỗi client.

### Kịch bản
- Lấy global model từ Exp 1 (`model_*.pt`)
- Mỗi client: chia local data → **10% fine-tune / 90% test**
- Fine-tune global model → đánh giá trên 90% còn lại
- Báo cáo: Min (%) và Mean (%) accuracy across clients

> **Lưu ý:** Phải chạy **Experiment 1 trước** để có `model_*.pt`

Config: `configs/exp2_config.json`

### Bước 1 — Chạy experiments

```bash
# Xem model availability (dry-run)
python exp2_personalized_accuracy.py --dry-run

# Chạy tất cả
python exp2_personalized_accuracy.py --no-confirm

# Chạy từng dataset
python exp2_personalized_accuracy.py --dataset cifar10 --no-confirm
python exp2_personalized_accuracy.py --dataset fmnist --no-confirm

# Tùy chỉnh fine-tuning
python exp2_personalized_accuracy.py --finetune-epochs 10 --finetune-lr 0.005 --no-confirm
```

### Bước 2 — Resume nếu bị dừng

```bash
# Xem status
python exp2_resume.py --dry-run

# Chạy tiếp
python exp2_resume.py --no-confirm
python exp2_resume.py --dataset fmnist --no-confirm
```

### Bước 3 — Xem kết quả

```bash
# In Table 2 (Personalized Accuracy)
python exp2_report.py

# Lưu CSV
python exp2_report.py --save-csv

# Chỉ 1 dataset
python exp2_report.py --dataset cifar10
```

Output Table 2:
```
Dataset   α     FedAvg           FedProx          Fed-M3           Fed-DGD
               Min(%) Mean(%)   Min(%) Mean(%)   Min(%) Mean(%)   Min(%) Mean(%)
FMNIST    0.1    –      –         –      –         –      –         –      –
          0.5    –      –         –      –         –      –         –      –
          1.0    –      –         –      –         –      –         –      –
CIFAR-10  0.1    –      –         –      –         –      –         –      –
          ...
```

### Output structure

```
results/exp2_personalized_accuracy/
├── personalized_results_*.json   # Kết quả chi tiết
└── table2_personalized_*.csv     # CSV export
```

---

## Troubleshooting

### CUDA out of memory
```bash
python run_experiment.py --batch-size 64
```

### Kết quả không reproducible
```bash
# Đảm bảo dùng cùng seed (mặc định seed=42)
python exp1_global_accuracy.py --seed 42
```

### Kiểm tra code trước khi chạy
```bash
python test_isolation.py
```

---

*Cập nhật: 2026-05-08*