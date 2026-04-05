# Huong dan chay Experiments

> Tai lieu huong dan chay cac kich ban thuc nghiem cho luan van.

---

## Muc luc

1. [Cau truc thu muc](#cau-truc-thu-muc)
2. [Cai dat](#cai-dat)
3. [Experiment 1: Global Accuracy](#experiment-1-global-accuracy)
4. [Experiment 2: Personalized Accuracy](#experiment-2-personalized-accuracy)
5. [Experiment 3: Convergence Speed](#experiment-3-convergence-speed)
6. [Experiment 4: Scalability](#experiment-4-scalability)
7. [Experiment 5: Ablation Study](#experiment-5-ablation-study)
8. [Cau truc du lieu output](#cau-truc-du-lieu-output)

---

## Cau truc thu muc

```
luanvan/experiments/
├── configs/                    # Config files cho experiments
│   └── exp1_config.json        # Config cho Experiment 1
├── docs/                       # Tai lieu
│   └── README.md               # File nay
├── fl/                         # FL framework
│   ├── client.py
│   ├── server.py
│   ├── aggregators.py
│   └── data_split.py
├── models/                     # Neural network models
│   └── cnn.py
├── optimizers/                 # Fed-M3, Fed-DGD, FedProx
│   ├── fed_m3.py
│   ├── fed_dgd.py
│   └── fedprox.py
├── utils/                      # Utilities
│   └── metrics.py
├── results/                    # Output directory
├── run_experiment.py           # Script chay 1 experiment
├── exp1_global_accuracy.py     # Script cho Experiment 1
└── requirements.txt
```

---

## Cai dat

```bash
# Kich hoat virtual environment
cd C:\Users\admin\PycharmProjects\nested_learning
.venv\Scripts\activate

# Cai dat dependencies (neu chua)
pip install -r luanvan/experiments/requirements.txt

# Di chuyen vao thu muc experiments
cd luanvan/experiments
```

---

## Experiment 1: Global Accuracy

### Muc tieu
So sanh **Global Accuracy** cua cac methods tren test set.

### Methods
- FedAvg (baseline)
- Fed-M3 (proposed)
- Fed-DGD (proposed)
- FedProx (baseline)

### Kich ban
- Dataset: CIFAR-10
- Non-IID: Dirichlet alpha = {0.1, 0.5, 1.0}
- Clients: 10
- Rounds: 100

### Cach chay

```bash
# Chay tat ca (4 methods x 3 alphas = 12 runs)
python exp1_global_accuracy.py

# Xem truoc config (khong chay)
python exp1_global_accuracy.py --dry-run

# Chay nhanh de test (10 rounds, 1 method, 1 alpha)
python exp1_global_accuracy.py --num-rounds 10 --methods fedavg --alphas 0.5

# Chi chay 2 methods
python exp1_global_accuracy.py --methods fedavg fed_m3

# Chi chay 1 alpha
python exp1_global_accuracy.py --alphas 0.5

# Dung config khac
python exp1_global_accuracy.py --config configs/my_config.json

# Export config de chinh sua
python exp1_global_accuracy.py --export-config my_config.json
```

### Config file
File: `configs/exp1_config.json`

```json
{
  "methods": ["fedavg", "fed_m3", "fed_dgd", "fedprox"],
  "alphas": [0.1, 0.5, 1.0],
  "fl": {
    "num_clients": 10,
    "num_rounds": 100,
    "local_epochs": 5,
    "batch_size": 64,
    "lr": 0.01
  },
  "fed_m3": {"beta1": 0.9, "beta3": 0.9, "lam": 0.3},
  "fed_dgd": {"alpha": 1.0, "decay_strength": 0.1},
  "fedprox": {"mu": 0.01}
}
```

### Output
- `results/exp1_global_accuracy/{method}/cifar10_dirichlet_a{alpha}/metrics_*.json`
- `results/exp1_global_accuracy/config_*.json`
- `results/exp1_global_accuracy/summary_*.json`

---

## Experiment 2: Personalized Accuracy

### Muc tieu
Do **Personalized Accuracy** sau khi fine-tuning tren local data.

### Kich ban
- Sau khi global training xong
- Fine-tune moi client tren 10% local data
- Do: Min Accuracy, Mean Accuracy

### Cach chay

```bash
# CHUA IMPLEMENT
# python exp2_personalized_accuracy.py
```

### Metrics can do
- `personalized_min_acc`: Min accuracy across clients
- `personalized_mean_acc`: Mean accuracy across clients

---

## Experiment 3: Convergence Speed

### Muc tieu
Do so **rounds can thiet** de dat target accuracy.

### Kich ban
- Target accuracy: {50%, 60%, 70%}
- Dataset: CIFAR-10, alpha = 0.5

### Cach chay

```bash
# Dung ket qua tu Experiment 1
# Tinh convergence round tu metrics.json

# Hoac chay rieng
python run_experiment.py --method fedavg --dataset cifar10 --alpha 0.5 --num-rounds 100
```

### Cach tinh tu metrics

```python
from utils.metrics import MetricsTracker

metrics = MetricsTracker()
metrics.load("results/.../metrics_*.json")

# Round dat 50%
round_50 = metrics.get_convergence_round(0.50)

# Round dat 60%
round_60 = metrics.get_convergence_round(0.60)
```

---

## Experiment 4: Scalability

### Muc tieu
Do anh huong cua **so luong clients** den accuracy.

### Kich ban
- Clients: {5, 10, 20, 50}
- Dataset: CIFAR-10, alpha = 0.5

### Cach chay

```bash
# Chay voi 5 clients
python run_experiment.py --method fed_m3 --num-clients 5 --alpha 0.5

# Chay voi 10 clients
python run_experiment.py --method fed_m3 --num-clients 10 --alpha 0.5

# Chay voi 20 clients
python run_experiment.py --method fed_m3 --num-clients 20 --alpha 0.5

# Chay voi 50 clients
python run_experiment.py --method fed_m3 --num-clients 50 --alpha 0.5
```

---

## Experiment 5: Ablation Study

### Muc tieu
Phan tich dong gop cua tung component trong Fed-M3.

### Kich ban
- Fed-M3 Full: fast momentum + slow momentum
- Fed-M3 Fast only: chi fast momentum (lam = 0)
- Fed-M3 Slow only: chi slow momentum (CHUA IMPLEMENT)

### Cach chay

```bash
# Fed-M3 Full (default)
python run_experiment.py --method fed_m3 --fed-m3-lam 0.3

# Fed-M3 Fast only (lam = 0)
python run_experiment.py --method fed_m3 --fed-m3-lam 0.0

# Fed-M3 Slow only
# CHUA IMPLEMENT
```

---

## Cau truc du lieu output

### File metrics (JSON)

```jsonc
{
  "rounds": [
    {
      "round": 1,

      // === SERVER METRICS ===
      // Danh gia GLOBAL MODEL tren TEST SET (sau aggregation)
      "server": {
        "test_acc": 0.50,    // Global model accuracy tren test set
        "test_loss": 1.25    // Global model loss tren test set
      },

      // === CLIENT METRICS ===
      "clients": [
        {
          "client_id": 0,
          "num_samples": 1200,  // So samples cua client nay

          // --- Training metrics (TRONG luc train) ---
          "train_loss": 0.82,   // Avg loss DURING training (trung binh tat ca epochs)

          // --- Local Acc (SAU khi train, TRUOC aggregation) ---
          "train_acc": 0.58,    // LOCAL model accuracy tren LOCAL data
                                // = Evaluate local model sau khi train xong
                                // Thuong CAO vi local model overfit local data

          // --- Global Acc (SAU aggregation) ---
          "test_acc": 0.52,     // GLOBAL model accuracy tren LOCAL data
                                // = Evaluate global model tren data cua client nay
                                // Thuong THAP hon train_acc
          "test_loss": 0.95,    // GLOBAL model loss tren LOCAL data

          // --- Per-epoch details (TRONG luc train) ---
          "epoch_metrics": [
            {"epoch": 1, "loss": 1.2, "acc": 0.35},  // Acc tinh TRUOC weight update
            {"epoch": 2, "loss": 0.9, "acc": 0.52},  // KHAC voi train_acc (sau train)
            ...
          ]
        },
        ...
      ],

      // === AGGREGATED STATISTICS ===
      // Thong ke tong hop tu tat ca clients trong round nay
      "client_aggregated": {
        "train_acc": {
          "mean": 0.55,    // Weighted mean (theo num_samples)
          "median": 0.53   // Median (khong trong so)
        },
        "train_loss": {"mean": 0.85, "median": 0.82},
        "test_acc": {"mean": 0.50, "median": 0.48},
        "test_loss": {"mean": 1.25, "median": 1.20}
      }
    },
    ...  // Round 2, 3, ...
  ]
}
```

### Giai thich cac metrics

#### Server metrics
| Metric | Mo ta | Tinh khi nao |
|--------|-------|--------------|
| `test_acc` | Global model accuracy tren **test set** | Sau moi round aggregation |
| `test_loss` | Global model loss tren **test set** | Sau moi round aggregation |

#### Client metrics
| Metric | Mo ta | Tinh khi nao |
|--------|-------|--------------|
| `train_loss` | **Avg loss DURING training** (trung binh tat ca epochs) | Trong luc train |
| `train_acc` | **Local Acc** = Local model accuracy tren local data | Evaluate SAU KHI train xong, TRUOC aggregation |
| `test_acc` | **Global Acc** = Global model accuracy tren local data | Evaluate SAU aggregation |
| `test_loss` | Global model loss tren local data | Evaluate SAU aggregation |
| `epoch_metrics` | Chi tiet tung epoch (loss, acc TRONG luc train) | Moi epoch |

#### Phan biet Local Acc vs Global Acc
```
Local Acc (train_acc):
  - Model: LOCAL model (sau khi client train xong)
  - Data: Local data cua client do
  - Y nghia: Model local fit data local tot den dau

Global Acc (test_acc):
  - Model: GLOBAL model (sau khi server aggregate)
  - Data: Local data cua client do
  - Y nghia: Model global hoat dong tren data cua client do tot den dau
```

#### Aggregated metrics
| Metric | Mo ta |
|--------|-------|
| `mean` | Trung binh **co trong so** theo `num_samples` |
| `median` | Trung vi (khong trong so) |

#### epoch_metrics chi tiet
```json
"epoch_metrics": [
  {"epoch": 1, "loss": 1.2, "acc": 0.35},  // Accuracy TRUOC khi update weights
  {"epoch": 2, "loss": 0.9, "acc": 0.52},  // (tinh tren forward pass)
  ...
]
```
**Luu y:** `epoch_metrics.acc` khac voi `train_acc`:
- `epoch_metrics.acc`: Tinh TRONG luc train (truoc weight update)
- `train_acc`: Evaluate SAU KHI train xong (= Local Acc)

---

## Troubleshooting

### CUDA not available
```bash
pip uninstall torch -y
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

### Out of memory
```bash
# Giam batch size
python run_experiment.py --batch-size 32

# Giam so clients
python run_experiment.py --num-clients 5
```

### Ket qua khong stable
```bash
# Set seed co dinh
python run_experiment.py --seed 42

# Tang so rounds
python run_experiment.py --num-rounds 200
```

---

## Tham khao

- FedAvg: McMahan et al., 2017
- FedProx: Li et al., 2020
- Fed-M3: Proposed method (based on Nested Learning)
- Fed-DGD: Proposed method (based on Delta Gradient Descent)

---

*Cap nhat: 2026-04-05*
