# Experiment Overview - Fed-DGD & Fed-M3

> Tài liệu này tổng hợp các kịch bản thực nghiệm cho luận văn.

---

## Mục tiêu Thực nghiệm

```
┌─────────────────────────────────────────────────────────────┐
│  RESEARCH QUESTIONS:                                        │
│                                                              │
│  1. Fed-DGD và Fed-M3 có cải thiện performance trên         │
│     non-IID data so với FedAvg không?                       │
│                                                              │
│  2. Phương pháp nào tốt hơn: Fed-DGD hay Fed-M3?           │
│                                                              │
│  3. Các hyperparameters (α, β, K) ảnh hưởng như thế nào?   │
│                                                              │
│  4. Trade-off giữa accuracy và communication cost?          │
└─────────────────────────────────────────────────────────────┘
```

---

## Các Phương pháp Đề xuất

| Method | Core Idea | Expected Benefit |
|--------|-----------|------------------|
| **Fed-DGD** | DGD preconditioner P = αI - η(k⊗k) | Adaptive decay, giảm client drift |
| **Fed-M3** | Multi-scale momentum (fast local + slow global) | Giữ long-term info, giảm conflicts |

---

## Baselines

| Method | Description | Reference |
|--------|-------------|-----------|
| FedAvg | Simple averaging | McMahan et al., 2017 |
| FedProx | FedAvg + proximal term | Li et al., 2020 |
| SCAFFOLD | Variance reduction | Karimireddy et al., 2020 |
| FedAdam | Server-side Adam | Reddi et al., 2021 |

---

## Datasets

| Dataset | Classes | Samples | Image Size |
|---------|---------|---------|------------|
| FMNIST | 10 | 70,000 | 28x28 |
| CIFAR-10 | 10 | 60,000 | 32x32 |

---

## Non-IID Scenarios

```
┌─────────────────────────────────────────────────────────────┐
│  SCENARIO 1: Label Skew (Pathological)                      │
│  - Mỗi client chỉ có 2 labels                               │
│  - VD: Client 1 có labels {0,1}, Client 2 có {2,3}          │
│  - Đây là extreme non-IID                                   │
├─────────────────────────────────────────────────────────────┤
│  SCENARIO 2: Label Skew (Dirichlet)                         │
│  - Phân phối labels theo Dirichlet(α)                       │
│  - α nhỏ = more skewed, α lớn = more uniform                │
│  - Thường dùng α = 0.1, 0.5, 1.0                            │
├─────────────────────────────────────────────────────────────┤
│  SCENARIO 3: Quantity Skew                                  │
│  - Số lượng samples khác nhau giữa các clients              │
│  - VD: Client 1 có 1000 samples, Client 2 có 100            │
├─────────────────────────────────────────────────────────────┤
│  SCENARIO 4: Label + Quantity Skew                          │
│  - Kết hợp cả hai                                           │
│  - Realistic scenario                                       │
└─────────────────────────────────────────────────────────────┘
```

---

## Experiment Files

1. `01_fed_m3_design.md` - Thiết kế chi tiết Fed-M3
2. `02_fed_dgd_design.md` - Thiết kế chi tiết Fed-DGD
3. `03_experiment_setup.md` - Setup môi trường, hyperparameters
4. `04_non_iid_scenarios.md` - Chi tiết các kịch bản non-IID
5. `05_evaluation_metrics.md` - Metrics đánh giá

---

## Timeline

| Phase | Task | Status |
|-------|------|--------|
| Design | Fed-M3 algorithm | IN PROGRESS |
| Design | Fed-DGD algorithm | TODO |
| Implement | Fed-M3 code | TODO |
| Implement | Fed-DGD code | TODO |
| Experiment | FMNIST experiments | TODO |
| Experiment | CIFAR-10 experiments | TODO |
| Analysis | Compare results | TODO |

---

*Cập nhật: 2026-03-28*
