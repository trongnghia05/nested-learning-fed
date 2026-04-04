# FedAvg: Federated Averaging

> **Baseline algorithm** cho Federated Learning
> Paper: "Communication-Efficient Learning of Deep Networks from Decentralized Data" (McMahan et al., AISTATS 2017)

---

## 1. Tong quan

### 1.1 FedAvg la gi?

FedAvg (Federated Averaging) la thuat toan co ban nhat cho Federated Learning, duoc de xuat boi McMahan et al. (Google) nam 2017.

### 1.2 Core Idea

```
+-------------------------------------------------------------+
|  FEDAVG CORE IDEA:                                           |
|                                                              |
|  1. Server gui global model cho clients                      |
|  2. Moi client train LOCAL tren data cua minh                |
|  3. Clients gui model updates ve server                      |
|  4. Server AVERAGE cac updates (weighted by data size)       |
|  5. Lap lai                                                  |
|                                                              |
|  Don gian, hieu qua, la baseline cho moi FL research         |
+-------------------------------------------------------------+
```

---

## 2. Cong thuc

### 2.1 Notation

| Symbol | Meaning |
|--------|---------|
| K | Tong so clients |
| C | Fraction clients duoc chon moi round (0 < C <= 1) |
| B | Local batch size |
| E | So local epochs |
| eta | Learning rate |
| n_k | So samples cua client k |
| n | Tong so samples (sum of n_k) |

### 2.2 Algorithm

```
Algorithm: FedAvg (McMahan et al., 2017)

Server Initialize:
    w_0 <- random init

For round t = 0, 1, 2, ..., T-1:

    # 1. Server chon subset clients
    S_t <- random sample of max(C*K, 1) clients

    # 2. Server broadcasts global model
    For each client k in S_t (parallel):
        w_k^{t+1} <- ClientUpdate(k, w^t)

    # 3. Server aggregates (WEIGHTED AVERAGE)
    w^{t+1} <- sum_{k in S_t} (n_k / n) * w_k^{t+1}

Return w^T

---

ClientUpdate(k, w):  # Run on client k
    B <- split local data into batches of size B

    For epoch = 1, 2, ..., E:
        For batch b in B:
            w <- w - eta * gradient(loss(w; b))

    Return w
```

### 2.3 Aggregation Formula

```
Weighted Average:

    w_global = sum_{k=1}^{K} (n_k / n) * w_k

    Trong do:
    - n_k: So samples cua client k
    - n = sum(n_k): Tong so samples
    - w_k: Model weights cua client k

Neu tat ca clients co cung so samples:
    w_global = (1/K) * sum(w_k)  # Simple average
```

---

## 3. Implementation

### 3.1 Server Side

```python
def fedavg_aggregate(global_model, client_results):
    """
    FedAvg aggregation: Weighted average of client models.
    """
    total_samples = sum(r['num_samples'] for r in client_results)

    # Initialize aggregated params
    aggregated_params = {}
    for key in client_results[0]['params']:
        aggregated_params[key] = torch.zeros_like(
            client_results[0]['params'][key],
            dtype=torch.float32
        )

    # Weighted average
    for result in client_results:
        weight = result['num_samples'] / total_samples
        for key in aggregated_params:
            aggregated_params[key] += weight * result['params'][key].float()

    global_model.load_state_dict(aggregated_params)

    return aggregated_params
```

### 3.2 Client Side

```python
def client_update(model, dataloader, epochs, lr):
    """
    FedAvg client update: Standard SGD training.
    """
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for epoch in range(epochs):
        for x, y in dataloader:
            optimizer.zero_grad()
            output = model(x)
            loss = F.cross_entropy(output, y)
            loss.backward()
            optimizer.step()

    return model.state_dict()
```

---

## 4. Hyperparameters

### 4.1 Default Values

| Parameter | Value | Notes |
|-----------|-------|-------|
| C | 0.1 - 1.0 | Fraction of clients per round |
| E | 1 - 5 | Local epochs |
| B | 10 - 50 | Batch size |
| eta (lr) | 0.01 - 0.1 | Learning rate |

### 4.2 Tuning Guidelines

```
1. Local epochs (E):
   - E lon: Communication efficient, nhung co the drift
   - E nho: It drift, nhung nhieu communication

2. Batch size (B):
   - B lon: Stable gradients, nhanh hon
   - B nho: Noisy gradients, co the generalize tot hon

3. Learning rate (eta):
   - Thuong dung 0.01 cho CIFAR-10
   - Co the decay theo round

4. Client fraction (C):
   - C = 1.0: Tat ca clients (full participation)
   - C < 1.0: Partial participation (realistic)
```

---

## 5. Van de cua FedAvg

### 5.1 Client Drift (Non-IID Problem)

```
+-------------------------------------------------------------+
|  VAN DE CHINH: CLIENT DRIFT                                  |
|                                                              |
|  Khi data non-IID:                                           |
|  - Client 1 co data {0,1,2} → model bias ve {0,1,2}         |
|  - Client 2 co data {7,8,9} → model bias ve {7,8,9}         |
|                                                              |
|  Sau E local epochs:                                         |
|  - Moi client DRIFT theo local data                          |
|  - Khi average: Conflicts, slow convergence                  |
|                                                              |
|  E cang lon → Drift cang nhieu                               |
+-------------------------------------------------------------+
```

### 5.2 Minh hoa

```
Round 1:
    Global: w_0
    Client 1: w_0 → w_1 (drift ve class 0-2)
    Client 2: w_0 → w_2 (drift ve class 7-9)

    Aggregate: w_avg = (w_1 + w_2) / 2
    → w_avg co the KHONG tot cho ca 2 directions

Round 2:
    Client 1: w_avg → w_1' (tiep tuc drift)
    Client 2: w_avg → w_2' (tiep tuc drift)

    → Oscillation, slow convergence, hoac diverge
```

### 5.3 Cac giai phap

| Method | Cach giai quyet | Nhuoc diem |
|--------|-----------------|------------|
| FedProx | Them proximal term | Isotropic, can tune mu |
| SCAFFOLD | Control variates | 2x communication |
| Fed-DGD | Decay theo drift | Selective nhung phuc tap |
| Fed-M3 | Multi-scale momentum | Extra computation |

---

## 6. So sanh voi cac methods khac

### 6.1 FedAvg vs FedProx

```
FedAvg:
    w = w - lr * grad
    → Khong co regularization
    → De bi drift

FedProx:
    w = w - lr * grad - lr * mu * (w - w_global)
    → Them proximal term
    → Giam drift nhung can tune mu
```

### 6.2 FedAvg vs Fed-DGD

```
FedAvg:
    w = w - lr * grad
    → Simple SGD

Fed-DGD:
    k = normalize(w - w_global)
    w = w - lr * grad - lr * decay * (k·w) * k
    → Decay theo drift direction
    → Selective forgetting
```

### 6.3 FedAvg vs Fed-M3 Lite

```
FedAvg:
    w = w - lr * grad
    → Khong co momentum

Fed-M3 Lite:
    m1 = beta1 * m1 + grad
    update = m1 + lam * m2_global
    w = w - lr * update
    → Multi-scale momentum
    → Long-term memory (m2)
```

### 6.4 Summary Table

| Method | Drift Handling | Long-term Memory | Extra Comm | Complexity |
|--------|----------------|------------------|------------|------------|
| FedAvg | None | No | 0 | Low |
| FedProx | Proximal term | No | 0 | Low |
| Fed-DGD | Drift decay | No | k | Medium |
| Fed-M3 | Multi-scale | Yes (m2) | buffer | Medium |

---

## 7. Khi nao nen dung FedAvg?

### 7.1 Nen dung khi

```
1. IID data (hoac gan IID)
   - FedAvg hoat dong tot khi data phan bo deu

2. Baseline comparison
   - Luon dung FedAvg lam baseline de so sanh

3. Simple deployment
   - Khi can deploy nhanh, khong can toi uu

4. Communication-constrained
   - FedAvg co uu diem: 0 extra communication
```

### 7.2 Khong nen dung khi

```
1. Severe non-IID data
   - Dirichlet alpha < 0.5 → can FedProx/Fed-DGD

2. Can stability cao
   - FedAvg co the oscillate tren non-IID

3. Long training (nhieu rounds)
   - Drift accumulate, can methods khac
```

---

## 8. Usage

### 8.1 Command

```bash
cd luanvan/experiments

python run_experiment.py --method fedavg --dataset cifar10 \
    --alpha 0.5 --num-rounds 100 --local-epochs 5 --lr 0.01
```

### 8.2 Code Example

```python
from fl.server import FLServer
from fl.client import FLClient
from fl.aggregators import fedavg_aggregate

# Server
server = FLServer(model, aggregator=fedavg_aggregate)

# Clients
clients = [FLClient(model, data_k) for k in range(num_clients)]

# Training loop
for round in range(num_rounds):
    # 1. Broadcast
    global_params = server.get_params()

    # 2. Client updates
    client_results = []
    for client in clients:
        client.set_params(global_params)
        result = client.train(epochs=E, lr=lr)
        client_results.append(result)

    # 3. Aggregate
    server.aggregate(client_results)
```

---

## 9. Ket qua tham khao

### CIFAR-10, Dirichlet alpha=0.5

```
Round 10:
- FedAvg:     ~70%
- FedProx:    ~72%
- Fed-DGD:    ~73% (expected)
- Fed-M3:     ~76%
```

### CIFAR-10, IID

```
Round 10:
- FedAvg:     ~75%
- FedProx:    ~75%
- Fed-DGD:    ~75%
- Fed-M3:     ~76%

→ Khi IID, tat ca methods tuong duong
→ Non-IID moi thay su khac biet
```

---

## 10. References

```
[1] McMahan, B., Moore, E., Ramage, D., Hampson, S., & y Arcas, B. A. (2017).
    Communication-Efficient Learning of Deep Networks from Decentralized Data.
    AISTATS 2017.
    https://arxiv.org/abs/1602.05629

[2] Li, T., et al. (2020).
    Federated Learning: Challenges, Methods, and Future Directions.
    IEEE Signal Processing Magazine.

[3] Kairouz, P., et al. (2021).
    Advances and Open Problems in Federated Learning.
    Foundations and Trends in Machine Learning.
```

---

*Cap nhat: 2026-04-04*
