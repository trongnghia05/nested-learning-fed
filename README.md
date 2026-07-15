> Source code for the paper submitted to IUKM 2026.

# Fed-M3: Multi-Scale Nested Optimization for Non-IID Federated Learning

**Authors:** Trong-Nghia Mai - Van Tham Nguyen - Trong Hieu Tran - Anh-Tu Tran

*Vietnam National University, Hanoi - Thuyloi University - VNU-UET - HUST*

---

## Abstract

Federated Learning (FL) enables collaborative model training while preserving local data privacy, but its performance often degrades under Non-IID data due to client drift and unstable aggregation. This paper investigates **Nested Learning** as a new optimization perspective for Non-IID Federated Learning. We reinterpret momentum as an inner optimization mechanism that stores and compresses historical gradient information, rather than merely an acceleration technique. Based on this view, we propose **Fed-M3**, a multi-scale nested momentum framework combining fast client-side momentum for local adaptation and slow server-side momentum for stable global aggregation. We also study **Fed-DGD** as an exploratory extension of Delta Gradient Descent to the federated setting. Experiments on Fashion-MNIST and CIFAR-10 with Dirichlet-based Non-IID partitioning show that Fed-M3 improves global accuracy, personalized accuracy, and convergence speed over standard baselines. These results suggest that Nested Learning is a promising framework for robust and efficient Federated Learning under heterogeneous data distributions.

**Keywords:** Federated Learning - Nested Learning - Non-IID Data - Multi-scale Momentum - Orthogonalization - Optimization

---

## Motivation

In practical FL applications, client data are rarely independent and identically distributed (Non-IID). Statistical heterogeneity appears in forms such as label distribution skew, feature distribution skew, concept shift, and quantity skew. Under Non-IID data, local models trained on different clients move toward inconsistent optimization directions - a phenomenon known as **client drift** - making server aggregation less stable and leading to slower convergence or degraded accuracy.

Existing methods address this in different ways:
- **FedProx** adds a proximal regularization term to keep local models close to the global model, but may limit client adaptation
- **SCAFFOLD** uses control variates to correct client drift, requiring additional communication overhead
- **FedAdam/FedYogi/FedAdagrad** apply server-side adaptive optimizers, but still rely on single-timescale momentum

These approaches do not fully exploit the **naturally nested structure** of FL, where clients perform frequent local updates and the server performs slower global aggregation across rounds. This paper addresses that gap through the lens of Nested Learning.

From the Nested Learning perspective, momentum is not only an acceleration heuristic but an *inner optimization state* that stores and compresses past gradient information. Modifying the inner optimization objective leads to new algorithms with different memory and update behaviors. Two key mechanisms from this framework are **Delta Gradient Descent (DGD)** and **Multi-scale Momentum Muon (M3)**.

---

## Proposed Methods

### Fed-M3 - Federated Multi-Scale Momentum

Fed-M3 explicitly separates two optimization timescales in FL:

- **Client-side fast momentum** (`β_f`) - initialized to zero each round, captures short-term local gradient information to help each client adapt rapidly to its own data distribution
- **Server-side slow momentum** (`β_s`) - never reset, accumulates aggregated model displacement across communication rounds to produce a stable global update direction

Since the server cannot directly access all local gradient steps, Fed-M3 uses the aggregated model displacement as a practical surrogate for accumulated client gradients. This is the key design choice that makes server-side nested momentum feasible in a federated setting. Optionally, Newton–Schulz orthogonalization can be applied to the aggregated update before updating the slow momentum, to further reduce conflicts among update directions from Non-IID clients.

### Fed-DGD - Federated Delta Gradient Descent

Fed-DGD is an exploratory extension of Delta Gradient Descent (DGD) to the federated setting. Instead of standard gradient steps, each client incorporates an adaptive decay term along the **drift direction** - the direction from the current local model to the global model - to pull local updates back toward the global solution and reduce client drift.

---

## Repository Structure

```
luanvan/experiments/
├── models/                        # CNN architectures (CNNSmall, CNNMedium)
├── fl/                            # FL framework (client, server, aggregators, data split)
├── optimizers/                    # Fed-M3, Fed-DGD, FedProx implementations
├── utils/                         # Seed, metrics, plotting
├── configs/                       # Experiment configuration files
├── run_experiment.py              # Single experiment runner
├── run_comparison.py              # Compare multiple methods on same data split
│
│   ── Paper experiments (results reported in the paper) ──
├── exp1_global_accuracy.py        # Exp 1: global accuracy comparison
├── exp1_report.py                 # Exp 1: generate report (global acc + convergence speed)
├── exp2_personalized_accuracy.py  # Exp 2: personalized accuracy after fine-tuning
├── exp2_report.py                 # Exp 2: generate report
│
│   ── Ongoing experiments (in progress, not yet reported) ──
├── exp3_num_clients.py            # Exp 3: effect of number of clients
└── exp4_ablation.py               # Exp 4: Fed-M3 ablation study
```

---

## Setup

**Requirements:** Python 3.11, PyTorch 2.11.0, CUDA 12.8

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -e .
```

---

## Running Experiments

```bash
cd luanvan/experiments

# Run a single method
python run_experiment.py --method fed_m3 --dataset cifar10 --alpha 0.5 --num-rounds 100

# Compare all methods on the same data split
python run_comparison.py --dataset cifar10 --methods fedavg fedprox fed_m3 fed_dgd --alpha 0.5
```

```bash
# Exp 1: run training, then generate report
python exp1_global_accuracy.py
python exp1_report.py

# Exp 2: fine-tune global model and evaluate personalized accuracy
python exp2_personalized_accuracy.py
python exp2_report.py
```

Available methods: `fedavg`, `fedprox`, `fed_m3`, `fed_dgd`

Available datasets: `fmnist`, `cifar10`

---

## Hardware

Experiments were conducted on:
- GPU: NVIDIA RTX 4080 24GB
- RAM: 128 GB, 64 CPU cores
- OS: Windows 10, CUDA 12.8

---

## License

Licensed under **Apache 2.0**.
