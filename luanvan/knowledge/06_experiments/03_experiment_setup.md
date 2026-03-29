# Experiment Setup

> Chi tiết setup môi trường, model, và hyperparameters cho thực nghiệm.

---

## 1. Hardware & Software

### 1.1 Hardware (Đã có)

```
┌─────────────────────────────────────────────────────────────┐
│  GPU: NVIDIA GeForce GTX 1660 (6GB VRAM)                    │
│  CUDA: 12.6                                                 │
│  OS: Windows 10                                             │
│  RAM: (cần xác nhận)                                        │
│                                                              │
│  Lưu ý:                                                     │
│  - 6GB VRAM giới hạn batch size và model size               │
│  - Có thể cần gradient accumulation cho models lớn          │
│  - FL simulation sẽ sequential (không parallel GPUs)        │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Software

```
Python: 3.11.1
PyTorch: 2.6.0+cu124
CUDA: 12.4

# Additional packages needed:
pip install numpy scipy matplotlib
pip install tensorboard  # logging
pip install tqdm         # progress bars
```

---

## 2. FL Framework

### 2.1 Options

| Framework | Pros | Cons |
|-----------|------|------|
| **Flower** | Popular, well-documented | May be overkill |
| **PySyft** | Privacy features | Complex setup |
| **Custom** | Full control | More work |

### 2.2 Recommendation: Custom Implementation

```
┌─────────────────────────────────────────────────────────────┐
│  Lý do chọn Custom:                                         │
│                                                              │
│  1. Fed-M3 và Fed-DGD có logic đặc biệt                    │
│     - Multi-scale momentum                                  │
│     - DGD preconditioner                                    │
│     - Khó integrate vào framework có sẵn                    │
│                                                              │
│  2. Research focus, không cần production features          │
│                                                              │
│  3. Dễ debug và modify                                      │
│                                                              │
│  4. Tận dụng code từ nested_learning repo                   │
└─────────────────────────────────────────────────────────────┘
```

### 2.3 Basic FL Structure

```python
# Pseudo-structure

class FLServer:
    def __init__(self, model, num_clients):
        self.global_model = model
        self.num_clients = num_clients

    def broadcast(self):
        return copy.deepcopy(self.global_model.state_dict())

    def aggregate(self, client_updates):
        # FedAvg or custom aggregation
        pass

class FLClient:
    def __init__(self, client_id, local_data, model):
        self.id = client_id
        self.data = local_data
        self.model = model

    def local_train(self, global_weights, num_epochs):
        self.model.load_state_dict(global_weights)
        # Train locally
        # Return updates
        pass

def fl_simulation(server, clients, num_rounds):
    for r in range(num_rounds):
        # 1. Broadcast
        global_weights = server.broadcast()

        # 2. Local training
        updates = []
        for client in clients:
            update = client.local_train(global_weights, T)
            updates.append(update)

        # 3. Aggregate
        server.aggregate(updates)
```

---

## 3. Models

### 3.1 Model Choices

| Dataset | Model | Params | Notes |
|---------|-------|--------|-------|
| FMNIST | CNN-Small | ~100K | 2 conv + 2 fc |
| FMNIST | MLP | ~200K | 3 hidden layers |
| CIFAR-10 | CNN-Medium | ~500K | 4 conv + 2 fc |
| CIFAR-10 | ResNet-18 | ~11M | May need gradient accum |

### 3.2 CNN-Small (for FMNIST)

```python
class CNNSmall(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 28->14
        x = self.pool(F.relu(self.conv2(x)))  # 14->7
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

### 3.3 CNN-Medium (for CIFAR-10)

```python
class CNNMedium(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 32->16

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 16->8
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
```

---

## 4. Datasets

### 4.1 Fashion-MNIST

```python
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_data = datasets.FashionMNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

test_data = datasets.FashionMNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

# Classes: T-shirt, Trouser, Pullover, Dress, Coat,
#          Sandal, Shirt, Sneaker, Bag, Ankle boot
```

### 4.2 CIFAR-10

```python
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                        (0.2023, 0.1994, 0.2010))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                        (0.2023, 0.1994, 0.2010))
])

train_data = datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform_train
)

# Classes: airplane, automobile, bird, cat, deer,
#          dog, frog, horse, ship, truck
```

---

## 5. Hyperparameters

### 5.1 FL Common Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| num_clients | 10-100 | Start with 10 |
| participation_rate | 1.0 | All clients each round |
| num_rounds | 100-500 | Depend on convergence |
| local_epochs | 1-5 | Or local_steps |
| local_batch_size | 32-64 | Depend on GPU memory |

### 5.2 Optimizer Parameters

| Parameter | FedAvg | Fed-M3 | Fed-DGD |
|-----------|--------|--------|---------|
| lr | 0.01 | 0.01 | 0.01 |
| β1 | - | 0.9 | - |
| β2 | - | 0.999 | - |
| β3 | - | 0.9 | - |
| α | - | 0.1-0.5 | 0.9-0.99 |
| NS_steps | - | 5 | - |

### 5.3 Default Configuration

```python
config = {
    # FL settings
    'num_clients': 10,
    'num_rounds': 200,
    'local_epochs': 5,
    'batch_size': 32,

    # Model
    'model': 'cnn_small',  # or 'cnn_medium'

    # Dataset
    'dataset': 'fmnist',  # or 'cifar10'
    'non_iid': 'pathological',  # or 'dirichlet'
    'dirichlet_alpha': 0.5,

    # Optimizer (method-specific)
    'method': 'fed_m3',  # or 'fed_dgd', 'fedavg'
    'lr': 0.01,

    # Fed-M3 specific
    'm3_beta1': 0.9,
    'm3_beta2': 0.999,
    'm3_beta3': 0.9,
    'm3_alpha': 0.3,
    'm3_ns_steps': 5,

    # Fed-DGD specific
    'dgd_alpha': 0.95,

    # Logging
    'log_interval': 10,
    'save_checkpoints': True,
}
```

---

## 6. Evaluation Metrics

### 6.1 Primary Metrics

| Metric | Description | Formula |
|--------|-------------|---------|
| Test Accuracy | Global model accuracy on test set | correct / total |
| Top-5 Accuracy | For CIFAR-10 | correct_in_top5 / total |

### 6.2 Secondary Metrics

| Metric | Description |
|--------|-------------|
| Convergence Speed | Rounds to reach target accuracy |
| Final Loss | Training loss at end |
| Client Variance | Var(accuracy across clients) |
| Communication Cost | Total bytes transferred |

### 6.3 Tracking Code

```python
class MetricsTracker:
    def __init__(self):
        self.history = {
            'round': [],
            'train_loss': [],
            'test_acc': [],
            'client_accs': [],
        }

    def log(self, round, train_loss, test_acc, client_accs):
        self.history['round'].append(round)
        self.history['train_loss'].append(train_loss)
        self.history['test_acc'].append(test_acc)
        self.history['client_accs'].append(client_accs)

    def get_convergence_round(self, target_acc):
        for i, acc in enumerate(self.history['test_acc']):
            if acc >= target_acc:
                return self.history['round'][i]
        return None

    def get_client_variance(self):
        return [np.var(accs) for accs in self.history['client_accs']]
```

---

## 7. Experiment Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│  EXPERIMENT PIPELINE:                                       │
│                                                              │
│  1. Setup                                                   │
│     ├── Load config                                         │
│     ├── Create model                                        │
│     ├── Load and split data (non-IID)                      │
│     └── Initialize FL server and clients                   │
│                                                              │
│  2. Training Loop                                           │
│     For each round:                                         │
│     ├── Server broadcasts global model                      │
│     ├── Clients train locally                               │
│     ├── Server aggregates updates                           │
│     └── Evaluate and log metrics                           │
│                                                              │
│  3. Evaluation                                              │
│     ├── Final test accuracy                                 │
│     ├── Per-client accuracy                                 │
│     └── Convergence analysis                               │
│                                                              │
│  4. Save Results                                            │
│     ├── Model checkpoint                                    │
│     ├── Metrics history (JSON)                              │
│     └── Plots (accuracy curves, etc.)                       │
└─────────────────────────────────────────────────────────────┘
```

---

## 8. Directory Structure

```
experiments/
├── configs/
│   ├── fedavg_fmnist.yaml
│   ├── fedm3_fmnist.yaml
│   ├── feddgd_fmnist.yaml
│   └── ...
├── data/
│   ├── fmnist/
│   └── cifar10/
├── models/
│   ├── cnn_small.py
│   └── cnn_medium.py
├── fl/
│   ├── server.py
│   ├── client.py
│   ├── aggregators.py
│   └── data_split.py
├── optimizers/
│   ├── fed_m3.py
│   └── fed_dgd.py
├── utils/
│   ├── metrics.py
│   └── plotting.py
├── results/
│   ├── fedavg/
│   ├── fedm3/
│   └── feddgd/
└── run_experiment.py
```

---

## 9. Running Experiments

### 9.1 Single Experiment

```bash
python run_experiment.py --config configs/fedm3_fmnist.yaml
```

### 9.2 Batch Experiments

```bash
# Run all methods on FMNIST
python run_batch.py --dataset fmnist --methods fedavg fedm3 feddgd

# Run ablation study
python run_ablation.py --method fedm3 --param alpha --values 0.1 0.3 0.5 0.7
```

### 9.3 Expected Runtime

| Experiment | Estimated Time |
|------------|----------------|
| FMNIST, 10 clients, 200 rounds | ~30 min |
| CIFAR-10, 10 clients, 200 rounds | ~2 hours |
| Full ablation study | ~1 day |

---

## 10. Reproducibility

```python
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Always set seed at start
set_seed(42)
```

---

## 11. Checkpoints

```python
def save_checkpoint(model, optimizer, round, metrics, path):
    torch.save({
        'round': round,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }, path)

def load_checkpoint(path, model, optimizer):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['round'], checkpoint['metrics']
```

---

*Cập nhật: 2026-03-28*
