# Fed-DGD: Federated Delta Gradient Descent

## 1. Tong quan

**Fed-DGD** la phuong phap de xuat dau tien cua luan van, dieu chinh Delta Gradient Descent tu Nested Learning sang moi truong Federated Learning.

### Muc tieu
- Giam client drift trong FL voi non-IID data
- Xu ly tuong quan trong du lieu cuc bo
- Cai thien toc do hoi tu

## 2. Ky hieu

```
K: So clients
N_k: So samples tai client k
D_k: Dataset tai client k
W: Global model weights
W_k: Local model weights tai client k
eta: Learning rate
E: So local epochs
T: So communication rounds
```

## 3. Thuat toan Fed-DGD

### Algorithm 1: Fed-DGD (Server)

```
Input: Initial weights W_0, rounds T, clients K

For t = 0, 1, ..., T-1:
    # Select subset of clients
    S_t = sample(K, fraction=C)

    # Send global model to clients
    For each client k in S_t (parallel):
        W_k^{t+1} = ClientDGDUpdate(W^t, D_k, E, eta)

    # Aggregate
    W^{t+1} = sum(N_k * W_k^{t+1}) / sum(N_k)

Return W^T
```

### Algorithm 2: ClientDGDUpdate

```
Input: W (global weights), D_k (local data), E (epochs), eta (lr)

W_local = W

For epoch = 1, ..., E:
    For batch (x, y) in D_k:
        # Forward pass
        y_pred = model(x; W_local)

        # Compute gradient
        grad = nabla_W L(y_pred, y)

        # === DGD-specific: Compute adaptive decay ===
        # Cho moi layer l:
        For each layer l with weight matrix W_l:
            # Input features cho layer nay
            x_l = activation of previous layer

            # Adaptive decay matrix
            decay_l = I - x_l @ x_l^T / ||x_l||^2

            # Apply decay truoc khi update
            W_l = decay_l @ W_l

        # === Standard gradient update ===
        W_local = W_local - eta * grad

Return W_local
```

## 4. Phan tich chi tiet

### 4.1 Adaptive Decay Mechanism

#### Cong thuc
```
decay(x) = I - (x @ x^T) / ||x||^2
```

#### Y nghia
- **x @ x^T**: Projection matrix len huong x
- **I - x @ x^T**: Loai bo thong tin trong huong x
- **Chia ||x||^2**: Normalize de on dinh

#### Trong FL context
```
Client 1 (data nhan meo):
    x_1 co features dac trung meo
    decay_1 "xoa" features cu lien quan meo

Client 2 (data nhan cho):
    x_2 co features dac trung cho
    decay_2 "xoa" features cu lien quan cho

Sau aggregate:
    Global model "quen" di mot phan bias cuc bo
    -> Giam client drift
```

### 4.2 Hai chien luoc Aggregation

#### Chien luoc A: Implicit Decay (Recommended)
```
Server chi aggregate weights cuoi:
    W_global = average(W_1, ..., W_K)

Uu diem:
    - Khong tang communication cost
    - Decay da apply trong local training

Nhuoc diem:
    - Mat thong tin ve decay pattern
```

#### Chien luoc B: Explicit Decay
```
Client gui ca weights va decay statistics:
    Send: (W_k, decay_stats_k)

Server aggregate ca hai:
    W_global = average(W_1, ..., W_K)
    decay_global = combine(decay_stats_1, ..., decay_stats_K)

Uu diem:
    - Server co the dieu chinh update

Nhuoc diem:
    - Tang communication cost
    - Phuc tap hon
```

## 5. Phan tich ly thuyet

### 5.1 Convergence Analysis (Sketch)

#### Assumptions
1. L-smooth: ||nabla f(x) - nabla f(y)|| <= L||x - y||
2. Bounded variance: E[||nabla f_k(x) - nabla f(x)||^2] <= sigma^2
3. Bounded decay: ||decay(x)|| <= 1

#### Convergence Rate
Voi step size eta = O(1/sqrt(T)):
```
(1/T) * sum_t ||nabla f(W^t)||^2 <= O(1/sqrt(T)) + O(sigma^2/K)
```

So sanh voi FedAvg:
- FedAvg: O(1/sqrt(T)) + O(sigma^2 * E / K)
- Fed-DGD: O(1/sqrt(T)) + O(sigma^2 / K) (giam factor E nho decay)

### 5.2 Communication Complexity

```
Fed-DGD (implicit): O(d) per round (giong FedAvg)
Fed-DGD (explicit): O(d + decay_info) per round

Trong do d = so tham so model
```

## 6. Implementation

### 6.1 PyTorch Implementation

```python
import torch
import torch.nn as nn

class DGDOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=0.01, decay_strength=1.0):
        defaults = dict(lr=lr, decay_strength=decay_strength)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            lr = group['lr']
            decay_strength = group['decay_strength']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if 'input_features' in state:
                    x = state['input_features']
                    # Compute decay
                    if x.ndim == 2:  # batch x features
                        # Average over batch
                        x_mean = x.mean(dim=0, keepdim=True)
                        norm_sq = (x_mean @ x_mean.T).squeeze() + 1e-8
                        decay = torch.eye(x.shape[1], device=x.device)
                        decay -= decay_strength * (x_mean.T @ x_mean) / norm_sq

                        # Apply decay to weights
                        if p.ndim == 2:
                            p.data = decay @ p.data

                # Standard gradient update
                p.data.add_(grad, alpha=-lr)


class FedDGDClient:
    def __init__(self, model, lr=0.01, decay_strength=1.0):
        self.model = model
        self.optimizer = DGDOptimizer(
            model.parameters(),
            lr=lr,
            decay_strength=decay_strength
        )

    def local_train(self, dataloader, epochs):
        self.model.train()
        for epoch in range(epochs):
            for x, y in dataloader:
                # Store input features for decay computation
                self._store_layer_inputs(x)

                # Forward
                output = self.model(x)
                loss = nn.CrossEntropyLoss()(output, y)

                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        return self.model.state_dict()

    def _store_layer_inputs(self, x):
        # Hook to store layer inputs
        # Implementation depends on model architecture
        pass


class FedDGDServer:
    def __init__(self, model):
        self.global_model = model

    def aggregate(self, client_weights, client_sizes):
        total_size = sum(client_sizes)
        new_weights = {}

        for key in client_weights[0].keys():
            weighted_sum = sum(
                w[key] * n for w, n in zip(client_weights, client_sizes)
            )
            new_weights[key] = weighted_sum / total_size

        self.global_model.load_state_dict(new_weights)
        return new_weights
```

### 6.2 Training Loop

```python
def fed_dgd_train(
    server,
    clients,
    rounds=100,
    local_epochs=5,
    client_fraction=0.1
):
    for round_num in range(rounds):
        # Select clients
        selected = random.sample(clients, int(len(clients) * client_fraction))

        # Distribute global model
        global_weights = server.global_model.state_dict()
        for client in selected:
            client.model.load_state_dict(global_weights)

        # Local training
        client_weights = []
        client_sizes = []
        for client in selected:
            weights = client.local_train(client.dataloader, local_epochs)
            client_weights.append(weights)
            client_sizes.append(len(client.dataloader.dataset))

        # Aggregate
        server.aggregate(client_weights, client_sizes)

        # Evaluate
        if round_num % 10 == 0:
            accuracy = evaluate(server.global_model, test_loader)
            print(f"Round {round_num}: Accuracy = {accuracy:.4f}")

    return server.global_model
```

## 7. Ablation Studies

### 7.1 Experiments can thuc hien

1. **Decay strength ablation**
   - decay_strength = 0 (standard SGD)
   - decay_strength = 0.5
   - decay_strength = 1.0 (full DGD)

2. **Non-IID severity**
   - IID
   - Non-IID (2 classes per client)
   - Extreme Non-IID (1 class per client)

3. **Number of local epochs**
   - E = 1, 5, 10, 20

4. **Number of clients**
   - K = 10, 50, 100

### 7.2 Metrics

1. **Test accuracy** (global va per-client)
2. **Convergence speed** (rounds to target accuracy)
3. **Client drift** (distance giua local va global model)
4. **Communication cost** (total bytes transferred)

## 8. So sanh voi baselines

### Baselines
1. **FedAvg**: Standard federated averaging
2. **FedProx**: FedAvg + proximal term
3. **SCAFFOLD**: Control variates for drift correction
4. **FedAdam**: Adaptive server optimizer

### Expected results

| Method    | IID Acc | Non-IID Acc | Rounds to 80% | Comm Cost |
|-----------|---------|-------------|---------------|-----------|
| FedAvg    | 85%     | 70%         | 100           | 1x        |
| FedProx   | 85%     | 73%         | 90            | 1x        |
| SCAFFOLD  | 86%     | 75%         | 70            | 2x        |
| **Fed-DGD** | 86%   | **78%**     | **65**        | 1x        |

## 9. Cau hoi mo

1. Decay strength nen la hyperparameter hay adaptive?
2. Co can decay cho tat ca layers khong?
3. Lam sao estimate decay tot hon o server?

## 10. Next steps

1. Implement Fed-DGD trong PyTorch
2. Setup experiments tren FMNIST va CIFAR-10
3. Chay ablation studies
4. So sanh voi baselines
5. Viet phan ly thuyet convergence
