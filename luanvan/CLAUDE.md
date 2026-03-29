# Claude Session Memory - Luan Van Nested Learning + FL

> File nay luu lai context de Claude co the doc lai o cac session sau.
> Cap nhat: 2026-03-29

---

## RULES QUAN TRONG

```
┌─────────────────────────────────────────────────────────────┐
│  1. TAT CA kien thuc phai DUA TREN PAPER, khong suy doan   │
│  2. Neu khong chac chan, phai DOC LAI paper truoc khi noi  │
│  3. Trich dan NGUON (section, page) khi giai thich         │
│  4. Neu paper khong noi ro, phai NOI LA "paper khong noi"  │
└─────────────────────────────────────────────────────────────┘

Papers chinh:
- google_papers/Nested_Learning/Nested_Learning.md
- google_papers/TITANs/TITANs.md
```

---

## Thong tin hoc vien

- **Ho ten:** Mai Trong Nghia
- **De tai:** Nghien cuu phuong phap Federated Learning dua tren Nested Learning
- **Huong dan:** TS. Tran Trong Hieu
- **Co so:** Truong DHKHTN, DHQG Ha Noi
- **Thoi gian:** 01/2026 - 06/2026

---

## Muc tieu luan van

### Muc tieu chinh
Phat trien 2 phuong phap moi:
1. **Fed-DGD**: Federated Delta Gradient Descent
2. **Fed-M3**: Federated Multi-scale Momentum Muon

### Van de can giai quyet
1. Non-IID data trong FL
2. Client drift
3. Catastrophic forgetting
4. Communication cost

---

## TIEN DO CAP NHAT (2026-03-30)

### Da hoan thanh
- [x] Setup environment (CUDA, PyTorch)
- [x] Setup data pipeline (simple config)
- [x] Chay smoke test va training
- [x] Tao knowledge base
- [x] **Thiet ke Fed-M3 algorithm** (01_fed_m3_design.md)
- [x] **Thiet ke Fed-DGD algorithm** (02_fed_dgd_design.md)
- [x] **Thiet ke Non-IID scenarios** (04_non_iid_scenarios.md)
- [x] **Implement FL Framework** (FedAvg, Fed-M3)
- [x] **Implement Models** (CNN for CIFAR-10, FMNIST)
- [x] **Debug Fed-M3 with Newton-Schulz** → That bai (~10%)
- [x] **Implement Fed-M3 Lite** (bo NS, chi multi-scale momentum)

### Dang lam
- [x] Fed-M3 Lite: **76.14% @ Round 10** (CIFAR-10, alpha=0.5)
- [ ] So sanh Fed-M3 Lite vs FedAvg (can chay FedAvg)
- [ ] Chay them rounds (50-100)

### Chua lam
- [ ] Implement Fed-DGD
- [ ] Chay full experiments
- [ ] Ablation studies
- [ ] Viet luan van

### Bai hoc tu debug
1. Newton-Schulz output fixed magnitude (~2-3) → mat thong tin gradient size
2. Accumulation momentum (`m = m + beta*grad`) → unbounded growth
3. EMA momentum (`m = beta*m + grad`) → bounded, on dinh
4. Multi-scale momentum la core cua Nested Learning, khong phai NS

---

## CODE IMPLEMENTATION (MOI)

### Cau truc thu muc experiments

```
luanvan/experiments/
├── configs/                    # YAML configs
│   └── cifar10_dirichlet_05.yaml
├── models/                     # Neural network models
│   ├── __init__.py
│   └── cnn.py                  # CNNSmall (FMNIST), CNNMedium (CIFAR-10)
├── fl/                         # FL framework core
│   ├── __init__.py
│   ├── data_split.py           # dirichlet_split, quantity_skew_split, iid_split
│   ├── client.py               # FLClient
│   ├── server.py               # FLServer
│   └── aggregators.py          # fedavg_aggregate, weighted_aggregate
├── optimizers/                 # Fed-M3, Fed-DGD
│   ├── __init__.py
│   ├── newton_schulz.py        # Newton-Schulz orthogonalization
│   └── fed_m3.py               # FedM3Optimizer, fed_m3_aggregate
├── utils/                      # Utilities
│   ├── __init__.py
│   ├── seed.py                 # set_seed()
│   ├── metrics.py              # MetricsTracker
│   └── plotting.py             # plot_results, plot_client_distribution
├── results/                    # Output directory
├── run_experiment.py           # Main entry point
├── run_comparison.py           # Compare multiple methods
└── requirements.txt
```

### Data Split Strategy

```
Ty le: Train 80% | Validation 10% | Test 10%

CIFAR-10:  50,000 → 40,000 train + 5,000 val + 5,000 test
FMNIST:    60,000 → 48,000 train + 6,000 val + 6,000 test

Train → Chia cho N clients (Dirichlet/Quantity/IID)
Val   → Hyperparameter tuning (FUTURE USE)
Test  → Final evaluation (global model)
```

### Cach chay experiments

```bash
cd luanvan/experiments

# Chay FedAvg tren CIFAR-10
python run_experiment.py --method fedavg --dataset cifar10 --alpha 0.5 --num-rounds 100

# Chay Fed-M3 tren CIFAR-10
python run_experiment.py --method fed_m3 --dataset cifar10 --alpha 0.5 --num-rounds 100

# So sanh FedAvg vs Fed-M3 (CUNG data split)
python run_comparison.py --dataset cifar10 --methods fedavg fed_m3 --alpha 0.5
```

### Fed-M3 Lite Implementation Notes (HIEN TAI)

```python
# Fed-M3 Lite Core Components:
# 1. Fast momentum (m1): Local, per-client, RESET moi round
# 2. Slow momentum (m2): Global, server-side, KHONG reset
# 3. KHONG CO Newton-Schulz (da bo)
# 4. Update: m1 + λ*m2

# Client:
#   m1 = beta1 * m1 + grad   # EMA (bounded!)
#   update = m1 + lam * m2
#   theta = theta - lr * update

# Server:
#   theta_global = FedAvg(theta_i)
#   m2 = beta3 * m2 + buffer  # EMA (bounded!)
#   m2_normalized = m2 / ||m2|| * 0.1

# Key hyperparameters:
# - beta1 = 0.9   (fast momentum)
# - beta3 = 0.9   (slow momentum)
# - lam = 0.3     (balance local vs global)
# - lr = 0.01     (learning rate)
```

---

## EXPERIMENT DESIGNS (DA HOAN THANH)

### Knowledge Base - Experiments

```
luanvan/knowledge/06_experiments/
├── 00_experiment_overview.md   # Tong quan experiments
├── 01_fed_m3_design.md         # Fed-M3 algorithm chi tiet
├── 02_fed_dgd_design.md        # Fed-DGD algorithm chi tiet
├── 03_experiment_setup.md      # Hardware, models, hyperparameters
└── 04_non_iid_scenarios.md     # Label skew, Quantity skew
```

### Fed-M3 Design Summary

```
SERVER:
  - θ_global (NO reset)
  - m2 (slow momentum, NO reset, accumulate qua rounds)
  - o2 = Newton_Schulz(m2)

CLIENT:
  - θ_i (RESET ← θ_global)
  - m1_i (RESET ← 0)
  - v_i (RESET ← 0)
  - buffer_i (RESET ← 0)
  - o1_i = Newton_Schulz(m1_i)

Update: (o1_i + λ*o2) / sqrt(v_i + eps)
```

### Fed-DGD Design Summary

```
Preconditioner: P = α*I - η*(k ⊗ k)
Update: W = W @ P - η * ∇L

k = gradient direction (normalized accumulated gradient)
α = decay factor (0.9-0.99)
η = learning rate (same for P and gradient step)
```

### Non-IID Scenarios

| Type | Method | Values |
|------|--------|--------|
| Label Skew | Dirichlet | α = {0.1, 0.5, 1.0} |
| Quantity Skew | Power Law | Zipf-like distribution |
| IID Baseline | Uniform | - |

### Comparison Methodology

```
QUAN TRONG: Khi so sanh methods
1. CUNG data split (same seed)
2. CUNG model init (same seed)
3. CUNG hyperparameters co ban (lr, batch_size)
4. CHI KHAC: Optimizer/Algorithm
```

---

## Setup Environment (Da hoan thanh)

### Hardware
- GPU: NVIDIA GeForce GTX 1660 (6GB VRAM)
- CUDA: 12.6
- OS: Windows 10

### Software
```bash
# Python 3.11.1
# PyTorch 2.6.0+cu124

# Cai dat
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -e .
```

---

## Knowledge Base Structure

```
luanvan/knowledge/
├── 00_index.md                          # Roadmap nghien cuu
├── 01_nested_learning/
│   ├── 00_learning_path.md
│   ├── 01_core_concepts.md
│   ├── 02_mathematical_foundations.md
│   ├── 03_deep_learning_vs_nested_learning.md
│   └── 04_titan_nested_learning_hope.md
├── 02_optimizers/
│   ├── 01_delta_gradient_descent.md
│   └── 02_multi_scale_momentum_muon.md
├── 03_federated_learning/
│   └── 01_basics_and_challenges.md
├── 04_proposed_methods/
│   ├── 01_fed_dgd.md
│   └── 02_fed_m3.md
├── 05_code_implementation/
│   ├── 01_paper_to_code_mapping.md
│   ├── 02_code_explanation_insights.md
│   └── 03_critical_analysis_paper_vs_code.md
└── 06_experiments/
    ├── 00_experiment_overview.md
    ├── 01_fed_m3_design.md
    ├── 02_fed_dgd_design.md
    ├── 03_experiment_setup.md
    └── 04_non_iid_scenarios.md
```

---

## KEY DESIGN DECISIONS

### Fed-M3
- **λ** (lambda): Balance factor local vs global (KHAC voi α cua Fed-DGD)
- **Newton-Schulz**: Per-layer, KEEP SHAPE (khong flatten)
- **m1, v**: RESET moi round (theo FedAvg convention)
- **m2**: KHONG reset (accumulate long-term)
- **v formula**: v = β2*v + g² (M3 style, KHONG co 1-β2)

### Fed-DGD
- **k**: Gradient direction (normalized accumulated gradient)
- **η**: Dung cho CA preconditioner VA gradient step (theo paper)
- **Fed-DGD-Lite**: DA XOA (khong phai real DGD)

---

## Files quan trong

### Papers
- `google_papers/Nested_Learning/Nested_Learning.md`
- `google_papers/TITANs/TITANs.md`

### Code (Original)
- `src/nested_learning/optim/m3.py` - M3 implementation
- `src/nested_learning/optim/deep.py` - Deep optimizer variants

### Code (Thesis - MOI)
- `luanvan/experiments/` - FL implementation
- `luanvan/experiments/run_experiment.py` - Main entry
- `luanvan/experiments/optimizers/fed_m3.py` - Fed-M3

### De cuong
- `luanvan/decuongnghiencuu-ban2.pdf`

---

## Ghi chu cho session tiep theo

1. **Test code**: Chay thu experiments tren CIFAR-10
2. **Debug**: Fix bugs neu co
3. **Implement Fed-DGD**: Sau khi Fed-M3 work
4. **Chay full experiments**: Theo plan trong 04_non_iid_scenarios.md

---

## Cau hoi thuong gap

### Chay script .sh tren Windows?
```bash
# Option 1: WSL
wsl bash scripts/data/run_sample.sh

# Option 2: Chay tung lenh Python thu cong
```

### Loi CUDA not available?
```bash
pip uninstall torch -y
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

---

*Cap nhat: 2026-03-30*
