# Non-IID Scenarios cho FL Experiments

> Chi tiết các kịch bản non-IID data để test Fed-M3 và Fed-DGD.

---

## 1. Tổng quan Non-IID trong FL

### 1.1 Định nghĩa

```
┌─────────────────────────────────────────────────────────────┐
│  IID (Independent and Identically Distributed):            │
│  - Mỗi client có data từ CÙNG distribution                  │
│  - P(x,y)_client1 = P(x,y)_client2 = ... = P(x,y)_global   │
│                                                              │
│  Non-IID:                                                   │
│  - Mỗi client có data từ KHÁC distribution                  │
│  - P(x,y)_client_i ≠ P(x,y)_client_j                       │
│  - Đây là realistic scenario trong thực tế                  │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Các loại Non-IID (Thực nghiệm)

```
┌─────────────────────────────────────────────────────────────┐
│  TRONG LUẬN VĂN NÀY, CHÚNG TA TẬP TRUNG VÀO 2 LOẠI:        │
│                                                              │
│  1. LABEL SKEW (Dirichlet distribution)                     │
│     - Class distribution khác nhau giữa các clients         │
│     - Control bằng α: α nhỏ = more skewed                   │
│                                                              │
│  2. QUANTITY SKEW                                           │
│     - Số samples khác nhau giữa các clients                 │
│     - Realistic: some users have more data                  │
└─────────────────────────────────────────────────────────────┘
```

| Type | Description | Dùng trong thực nghiệm |
|------|-------------|------------------------|
| **Label Skew** | Class distribution khác nhau | ✓ Dirichlet α ∈ {0.1, 0.5, 1.0} |
| **Quantity Skew** | Số samples khác nhau | ✓ Power law distribution |
| Feature Skew | Feature distribution khác nhau | ✗ Không test |
| Temporal Skew | Data thay đổi theo thời gian | ✗ Không test |

---

## 2. Scenario 1: Label Skew (Dirichlet Distribution)

### 3.1 Mô tả

```
┌─────────────────────────────────────────────────────────────┐
│  DIRICHLET NON-IID:                                         │
│                                                              │
│  Dùng Dirichlet distribution để control mức độ non-IID      │
│                                                              │
│  Dir(α): α nhỏ = more skewed, α lớn = more uniform          │
│                                                              │
│  α = 0.1: Rất non-IID (mỗi client dominated by 1-2 classes)│
│  α = 0.5: Moderate non-IID                                  │
│  α = 1.0: Mild non-IID                                      │
│  α = 10:  Almost IID                                        │
│  α → ∞:   Perfect IID                                       │
│                                                              │
│  Đây là cách phổ biến nhất để create controllable non-IID  │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Implementation

```python
def dirichlet_split(dataset, num_clients, alpha=0.5, seed=42):
    """
    Split dataset using Dirichlet distribution.

    Args:
        dataset: Full training dataset
        num_clients: Number of clients
        alpha: Dirichlet concentration parameter
               (smaller = more non-IID)
        seed: Random seed for reproducibility

    Returns:
        List of client datasets
    """
    np.random.seed(seed)

    # Get labels
    labels = np.array([y for _, y in dataset])
    num_classes = len(np.unique(labels))

    # Sample from Dirichlet for each class
    # proportions[class_id][client_id] = proportion of class for client
    proportions = np.random.dirichlet(
        [alpha] * num_clients,
        size=num_classes
    )  # Shape: (num_classes, num_clients)

    # Assign indices to clients
    client_indices = [[] for _ in range(num_clients)]

    for class_id in range(num_classes):
        class_indices = np.where(labels == class_id)[0]
        np.random.shuffle(class_indices)

        # Split according to proportions
        props = proportions[class_id]
        props = props / props.sum()  # Normalize

        split_points = (np.cumsum(props) * len(class_indices)).astype(int)
        split_points = np.insert(split_points, 0, 0)

        for client_id in range(num_clients):
            start = split_points[client_id]
            end = split_points[client_id + 1]
            client_indices[client_id].extend(class_indices[start:end])

    # Create subsets
    client_datasets = []
    for indices in client_indices:
        if len(indices) > 0:
            client_datasets.append(Subset(dataset, indices))
        else:
            # Empty client - should not happen with proper alpha
            client_datasets.append(Subset(dataset, [0]))  # Placeholder

    return client_datasets
```

### 3.3 Visualization

```
Dirichlet α = 0.1 (Very Non-IID):
Client 0: ████████████████████ class 0 (dominant)
          ██ class 1
          █ class 3

Client 1:                       ██████████████████ class 2 (dominant)
          ███ class 4

...

Dirichlet α = 1.0 (Mild Non-IID):
Client 0: ████████ class 0
          ██████ class 1
          ████ class 2
          █████ class 3
          ...

→ More balanced, but still heterogeneous
```

### 3.4 Comparison Table

| α | Non-IID Level | Typical Use Case |
|---|---------------|------------------|
| 0.1 | Extreme | Stress testing |
| 0.3 | Severe | Challenging scenarios |
| 0.5 | Moderate | **Default for experiments** |
| 1.0 | Mild | Easier scenarios |
| 10.0 | Near-IID | Baseline comparison |

---

## 3. Scenario 2: Quantity Skew

### 4.1 Mô tả

```
┌─────────────────────────────────────────────────────────────┐
│  QUANTITY SKEW:                                             │
│                                                              │
│  Số lượng samples KHÁC NHAU giữa các clients                │
│                                                              │
│  Ví dụ:                                                     │
│  Client 0: 10,000 samples                                   │
│  Client 1: 5,000 samples                                    │
│  Client 2: 1,000 samples                                    │
│  Client 3: 500 samples                                      │
│  Client 4: 100 samples                                      │
│                                                              │
│  Thực tế: Người dùng khác nhau có lượng data khác nhau     │
│  (power users vs casual users)                              │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Implementation

```python
def quantity_skew_split(dataset, num_clients, distribution='power_law'):
    """
    Split dataset with different quantities per client.

    Args:
        dataset: Full training dataset
        num_clients: Number of clients
        distribution: 'power_law', 'uniform', or 'exponential'

    Returns:
        List of client datasets
    """
    n = len(dataset)
    indices = list(range(n))
    np.random.shuffle(indices)

    if distribution == 'power_law':
        # Power law: few clients have many samples
        # Zipf-like distribution
        weights = 1 / np.arange(1, num_clients + 1)
        weights = weights / weights.sum()

    elif distribution == 'exponential':
        # Exponential decay
        weights = np.exp(-np.arange(num_clients) / (num_clients / 3))
        weights = weights / weights.sum()

    else:  # uniform
        weights = np.ones(num_clients) / num_clients

    # Compute split points
    split_points = (np.cumsum(weights) * n).astype(int)
    split_points = np.insert(split_points, 0, 0)

    # Create subsets
    client_datasets = []
    for i in range(num_clients):
        start = split_points[i]
        end = split_points[i + 1]
        client_datasets.append(Subset(dataset, indices[start:end]))

    return client_datasets
```

### 4.3 Sample Distribution

```
Power Law (10 clients, 60K total samples):

Client 0: ████████████████████████████████████████ 19,212 samples (32%)
Client 1: ████████████████████ 9,606 samples (16%)
Client 2: █████████████ 6,404 samples (11%)
Client 3: ██████████ 4,803 samples (8%)
Client 4: ████████ 3,842 samples (6%)
Client 5: ██████ 3,202 samples (5%)
Client 6: █████ 2,744 samples (5%)
Client 7: █████ 2,401 samples (4%)
Client 8: ████ 2,134 samples (4%)
Client 9: ████ 1,921 samples (3%)
```

---

## 4. Comparison Methodology (QUAN TRỌNG)

### 4.1 Nguyên tắc so sánh

```
┌─────────────────────────────────────────────────────────────┐
│  NGUYÊN TẮC QUAN TRỌNG:                                     │
│                                                              │
│  Khi so sánh FedAvg vs Fed-M3 vs Fed-DGD:                  │
│                                                              │
│  ★ PHẢI dùng CÙNG DATA SPLIT cho tất cả methods            │
│  ★ PHẢI dùng CÙNG SEED để reproducible                      │
│  ★ PHẢI dùng CÙNG hyperparameters cơ bản (lr, batch_size)  │
│                                                              │
│  CHỈ KHÁC: Optimizer/Algorithm                              │
│                                                              │
│  Điều này đảm bảo fair comparison:                          │
│  - Sự khác biệt về accuracy chỉ do algorithm                │
│  - Không phải do random data split                          │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Quy trình thực nghiệm

```
┌─────────────────────────────────────────────────────────────┐
│  STEP 1: Tạo data split MỘT LẦN                            │
│  ─────────────────────────────────────────────────────────  │
│                                                              │
│  seed = 42                                                  │
│  client_datasets = dirichlet_split(data, N, α, seed)       │
│                                                              │
│  → Lưu lại indices hoặc save to file                       │
│  → Dùng CÙNG split cho tất cả experiments                  │
├─────────────────────────────────────────────────────────────┤
│  STEP 2: Chạy từng method với CÙNG data split              │
│  ─────────────────────────────────────────────────────────  │
│                                                              │
│  Experiment 1: FedAvg    + client_datasets → results_1     │
│  Experiment 2: Fed-M3    + client_datasets → results_2     │
│  Experiment 3: Fed-DGD   + client_datasets → results_3     │
│  Experiment 4: FedProx   + client_datasets → results_4     │
│                                                              │
│  (Optional baselines: SCAFFOLD, FedAdam, etc.)             │
├─────────────────────────────────────────────────────────────┤
│  STEP 3: So sánh kết quả                                    │
│  ─────────────────────────────────────────────────────────  │
│                                                              │
│  Compare: accuracy, convergence speed, stability            │
│  Plot: learning curves on same graph                        │
│  Table: final metrics side-by-side                          │
└─────────────────────────────────────────────────────────────┘
```

### 4.3 Code Template

```python
def run_comparison_experiment(
    dataset,
    num_clients,
    alpha,  # Dirichlet α
    methods=['fedavg', 'fed_m3', 'fed_dgd'],
    seed=42,
    num_rounds=200,
    local_epochs=5,
    lr=0.01,
    batch_size=32
):
    """
    Chạy so sánh nhiều methods với CÙNG data split.
    """
    # ========================================
    # STEP 1: Tạo data split MỘT LẦN
    # ========================================
    set_seed(seed)
    client_datasets = dirichlet_split(dataset, num_clients, alpha, seed)

    # Log data distribution để verify
    log_data_distribution(client_datasets)

    # ========================================
    # STEP 2: Chạy từng method
    # ========================================
    all_results = {}

    for method in methods:
        print(f"Running {method}...")

        # Reset seed cho model init (fair comparison)
        set_seed(seed)

        # Create model (SAME initialization)
        model = create_model()

        # Run FL với method cụ thể
        if method == 'fedavg':
            results = run_fedavg(model, client_datasets, num_rounds, local_epochs, lr)
        elif method == 'fed_m3':
            results = run_fed_m3(model, client_datasets, num_rounds, local_epochs, lr)
        elif method == 'fed_dgd':
            results = run_fed_dgd(model, client_datasets, num_rounds, local_epochs, lr)
        elif method == 'fedprox':
            results = run_fedprox(model, client_datasets, num_rounds, local_epochs, lr)

        all_results[method] = results

    # ========================================
    # STEP 3: So sánh và visualize
    # ========================================
    compare_and_plot(all_results)

    return all_results
```

### 4.4 Bảng so sánh mẫu

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  EXPERIMENT: FMNIST, 10 clients, Dirichlet α=0.5, 200 rounds                │
│  Data split: SAME for all methods (seed=42)                                 │
├───────────────┬──────────┬──────────┬──────────┬──────────┬─────────────────┤
│    Metric     │  FedAvg  │ FedProx  │  Fed-M3  │ Fed-DGD  │ Winner          │
├───────────────┼──────────┼──────────┼──────────┼──────────┼─────────────────┤
│ Final Acc (%) │   72.3   │   74.1   │   ??.?   │   ??.?   │ ?               │
│ Rounds to 70% │   150    │   120    │   ???    │   ???    │ ?               │
│ Client Var    │   0.05   │   0.04   │   ???    │   ???    │ ?               │
│ Stability     │   Med    │   High   │   ???    │   ???    │ ?               │
└───────────────┴──────────┴──────────┴──────────┴──────────┴─────────────────┘

Notes:
- FedAvg, FedProx: Baseline numbers (sẽ đo khi chạy)
- Fed-M3, Fed-DGD: Proposed methods (cần thực nghiệm)
- Winner: Method tốt nhất cho mỗi metric
```

---

## 5. Experiment Matrix

### 5.1 Planned Experiments

```
┌─────────────────────────────────────────────────────────────┐
│  THỰC NGHIỆM CHÍNH: 2 loại Non-IID × 2 Datasets            │
│                                                              │
│  1. LABEL SKEW (Dirichlet):                                 │
│     - α = 0.1 (Severe)                                      │
│     - α = 0.5 (Moderate) ← MAIN EXPERIMENT                  │
│     - α = 1.0 (Mild)                                        │
│                                                              │
│  2. QUANTITY SKEW:                                          │
│     - Power law distribution                                │
│                                                              │
│  3. IID BASELINE:                                           │
│     - Uniform split                                         │
└─────────────────────────────────────────────────────────────┘
```

| # | Dataset | Non-IID Type | α / Distribution | Methods to Compare |
|---|---------|--------------|------------------|-------------------|
| 1 | FMNIST | IID (baseline) | Uniform | FedAvg, Fed-M3, Fed-DGD |
| 2 | FMNIST | Label Skew | α = 1.0 (Mild) | FedAvg, Fed-M3, Fed-DGD |
| 3 | FMNIST | Label Skew | α = 0.5 (Moderate) | FedAvg, FedProx, Fed-M3, Fed-DGD |
| 4 | FMNIST | Label Skew | α = 0.1 (Severe) | FedAvg, FedProx, Fed-M3, Fed-DGD |
| 5 | FMNIST | Quantity Skew | Power law | FedAvg, Fed-M3, Fed-DGD |
| 6 | CIFAR-10 | Label Skew | α = 0.5 | FedAvg, FedProx, Fed-M3, Fed-DGD |
| 7 | CIFAR-10 | Quantity Skew | Power law | FedAvg, Fed-M3, Fed-DGD |

### 5.2 Methods to Compare

| Method | Type | Description |
|--------|------|-------------|
| **FedAvg** | Baseline | Standard federated averaging |
| **FedProx** | Baseline | FedAvg + proximal term |
| **Fed-M3** | Proposed | Multi-scale momentum for FL |
| **Fed-DGD** | Proposed | Delta Gradient Descent for FL |

### 5.3 Metrics to Track

| Metric | Description | Mục đích |
|--------|-------------|----------|
| **Test Accuracy** | Accuracy trên global test set | So sánh performance |
| **Convergence Round** | Số rounds để đạt target accuracy (e.g., 70%) | So sánh tốc độ |
| **Client Variance** | Variance của accuracy giữa các clients | Đo fairness |
| **Training Loss** | Loss curve theo rounds | Đo stability |

---

## 6. Visualization Functions

```python
def plot_client_distribution(client_datasets, num_classes=10):
    """Plot class distribution for each client."""
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()

    for client_id, dataset in enumerate(client_datasets):
        labels = [y for _, y in dataset]
        counts = np.bincount(labels, minlength=num_classes)

        axes[client_id].bar(range(num_classes), counts)
        axes[client_id].set_title(f'Client {client_id}')
        axes[client_id].set_xlabel('Class')
        axes[client_id].set_ylabel('Count')

    plt.tight_layout()
    plt.savefig('client_distribution.png')
    plt.show()


def plot_quantity_distribution(client_datasets):
    """Plot number of samples per client."""
    sizes = [len(ds) for ds in client_datasets]

    plt.figure(figsize=(10, 5))
    plt.bar(range(len(sizes)), sizes)
    plt.xlabel('Client ID')
    plt.ylabel('Number of Samples')
    plt.title('Quantity Distribution across Clients')
    plt.savefig('quantity_distribution.png')
    plt.show()
```

---

## 7. Expected Behavior

### 7.1 FedAvg Baseline

```
┌─────────────────────────────────────────────────────────────┐
│  FedAvg performance degradation with non-IID:              │
│                                                              │
│  IID:           ████████████████████████████████ 90% acc    │
│  Mild (α=1.0):  ██████████████████████████ 80% acc          │
│  Moderate:      ████████████████████ 70% acc                │
│  Severe (α=0.1):██████████████ 55% acc                      │
│  Pathological:  ████████ 40% acc                            │
│                                                              │
│  → FedAvg struggles with severe non-IID                     │
└─────────────────────────────────────────────────────────────┘
```

### 7.2 Expected Fed-M3 vs Fed-DGD

```
┌─────────────────────────────────────────────────────────────┐
│  Hypothesis:                                                │
│                                                              │
│  Fed-M3:                                                    │
│  - Better when: clients have different "learning speeds"    │
│  - Slow momentum helps synchronize                          │
│  - Good for: temporal patterns, sequential learning         │
│                                                              │
│  Fed-DGD:                                                   │
│  - Better when: clear local biases to "forget"             │
│  - Decay helps remove local overfitting                     │
│  - Good for: strong label skew, distinct clusters          │
│                                                              │
│  Both should beat FedAvg on non-IID data                   │
└─────────────────────────────────────────────────────────────┘
```

---

## 8. Quick Start Code

```python
# Example: Create non-IID FMNIST split and visualize

from torchvision import datasets, transforms
from torch.utils.data import Subset
import matplotlib.pyplot as plt
import numpy as np

# Load FMNIST
transform = transforms.ToTensor()
train_data = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)

# Create non-IID split
num_clients = 10
alpha = 0.5

client_datasets = dirichlet_split(train_data, num_clients, alpha)

# Visualize
plot_client_distribution(client_datasets)
plot_quantity_distribution(client_datasets)

# Check sizes
for i, ds in enumerate(client_datasets):
    labels = [y for _, y in ds]
    print(f"Client {i}: {len(ds)} samples, classes: {set(labels)}")
```

---

*Cập nhật: 2026-03-29*
