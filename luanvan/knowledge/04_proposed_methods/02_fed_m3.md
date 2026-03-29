# Fed-M3: Federated Multi-scale Momentum Muon

## 1. Tong quan

**Fed-M3** la phuong phap de xuat thu hai cua luan van, dieu chinh Multi-scale Momentum Muon tu Nested Learning sang moi truong Federated Learning.

### Muc tieu
- Giu thong tin dai han ve gradient landscape qua cac rounds
- Giam xung dot khi aggregate gradients tu cac non-IID clients
- Cai thien stability va convergence

## 2. Y tuong thiet ke

### Mapping M3 sang FL architecture

```
M3 (Centralized):
    Fast momentum M1  <->  Local training tai client
    Slow momentum M2  <->  Global aggregation tai server

Fed-M3:
    Client: Fast momentum + local adaptation
    Server: Slow momentum + global coordination
```

### Core innovations

1. **Hierarchical momentum**
   - Client: Fast momentum cho local updates
   - Server: Slow momentum cho global updates

2. **Newton-Schulz trong aggregation**
   - Orthogonalize gradient updates truoc khi aggregate
   - Giam conflict giua non-IID gradients

3. **Adaptive slow momentum**
   - Server accumulate gradients qua nhieu rounds
   - Long-term memory cua training dynamics

## 3. Thuat toan Fed-M3

### Algorithm 1: Fed-M3 Server

```
Input: W_0, T rounds, K clients

Initialize:
    m_slow = 0  # Slow momentum
    o_slow = 0  # Orthogonalized slow momentum
    v_global = 0  # Second moment

For t = 0, 1, ..., T-1:
    # Select clients
    S_t = sample(K, fraction=C)

    # Distribute model
    For each client k in S_t (parallel):
        delta_k, stats_k = ClientM3Update(W^t, D_k, E)

    # Aggregate deltas
    delta_avg = weighted_average(delta_k for k in S_t)

    # === Newton-Schulz Orthogonalization ===
    delta_ortho = newton_schulz(delta_avg, steps=3)

    # === Update second moment ===
    v_global = beta2 * v_global + (1-beta2) * delta_ortho^2

    # === Compute adaptive update ===
    update = (delta_ortho + alpha * o_slow) / (sqrt(v_global) + eps)

    # Apply update
    W^{t+1} = W^t - lr_server * update

    # === Update slow momentum (moi slow_chunk rounds) ===
    If t % slow_chunk == 0:
        m_slow = beta3 * m_slow + delta_avg
        o_slow = newton_schulz(m_slow, steps=3)

Return W^T
```

### Algorithm 2: ClientM3Update

```
Input: W (global weights), D_k (local data), E (epochs)

Initialize local M3 optimizer:
    m1 = 0  # Fast momentum
    v = 0   # Second moment

W_local = W.copy()

For epoch = 1, ..., E:
    For batch (x, y) in D_k:
        # Compute gradient
        grad = nabla_W L(model(x; W_local), y)

        # Update fast momentum
        m1 = beta1 * m1 + (1-beta1) * grad

        # Update second moment
        v = beta2 * v + (1-beta2) * grad^2

        # Orthogonalize momentum
        o1 = newton_schulz(m1, steps=3)

        # Compute update
        update = o1 / (sqrt(v) + eps)

        # Apply update
        W_local = W_local - lr_local * update

# Compute delta
delta = W_local - W

# Compute statistics (optional)
stats = {
    'gradient_norm': ||delta||,
    'momentum_state': m1
}

Return delta, stats
```

## 4. Cac bien the thiet ke

### Variant A: Full M3 at Client (Recommended)

```
Client:
    - Full M3 optimizer
    - Fast momentum + second moment
    - Newton-Schulz cho local updates

Server:
    - Slow momentum only
    - Newton-Schulz cho aggregation
    - No second moment (hoac shared v_global)

Uu diem:
    - Client tu adapt tot voi local data
    - Server giu long-term information

Nhuoc diem:
    - Nhieu computation tai client
```

### Variant B: Simplified Client

```
Client:
    - SGD + momentum (no Newton-Schulz)

Server:
    - Full M3 logic
    - Newton-Schulz cho aggregation
    - Slow momentum

Uu diem:
    - It computation tai client (phu hop mobile)

Nhuoc diem:
    - Mat loi ich cua orthogonalization trong local training
```

### Variant C: Momentum State Sharing

```
Client:
    - Train voi M3
    - Send: delta, m1_local

Server:
    - Aggregate momentum states
    - m1_global = average(m1_local)
    - Combine voi slow momentum

Uu diem:
    - Server co information ve client momentum

Nhuoc diem:
    - Tang communication cost (2x)
```

## 5. Newton-Schulz trong FL Context

### Tai sao orthogonalization giup?

**Non-IID gradients conflict:**
```
Client 1 (class 0-4): grad_1 huong ve features class 0-4
Client 2 (class 5-9): grad_2 huong ve features class 5-9

Simple average: (grad_1 + grad_2) / 2
    -> Conflict, cancel out, slow convergence

With orthogonalization:
    o_grad_1 = orthogonalize(grad_1)
    o_grad_2 = orthogonalize(grad_2)
    -> Giam overlap, giu duoc useful information
```

### Newton-Schulz Properties

1. **Preserves direction**: ||Orthogonalize(M)|| ~ ||M||
2. **Reduces condition number**: Cai thien numerical stability
3. **Decorrelates features**: Giam redundancy trong gradients

### Implementation trong aggregation

```python
def orthogonalized_aggregate(client_deltas, ns_steps=3):
    """Aggregate with Newton-Schulz orthogonalization"""
    # Stack deltas
    deltas = torch.stack(client_deltas)

    # Average
    delta_avg = deltas.mean(dim=0)

    # Orthogonalize (cho >=2D tensors)
    if delta_avg.ndim >= 2:
        # Reshape to 2D
        original_shape = delta_avg.shape
        mat = delta_avg.reshape(delta_avg.shape[0], -1)

        # Newton-Schulz
        x = mat / (torch.linalg.norm(mat) + 1e-6)
        eye = torch.eye(x.shape[1], device=x.device)
        for _ in range(ns_steps):
            x = 0.5 * x @ (3.0 * eye - x.T @ x)

        # Reshape back
        delta_ortho = x.reshape(original_shape)
    else:
        delta_ortho = delta_avg

    return delta_ortho
```

## 6. Implementation chi tiet

### 6.1 Fed-M3 Server

```python
class FedM3Server:
    def __init__(self, model, lr=0.01, beta2=0.999, beta3=0.9,
                 alpha=1.0, ns_steps=3, slow_chunk=10):
        self.model = model
        self.lr = lr
        self.beta2 = beta2
        self.beta3 = beta3
        self.alpha = alpha
        self.ns_steps = ns_steps
        self.slow_chunk = slow_chunk

        # Initialize state
        self.m_slow = {k: torch.zeros_like(v)
                       for k, v in model.state_dict().items()}
        self.o_slow = {k: torch.zeros_like(v)
                       for k, v in model.state_dict().items()}
        self.v_global = {k: torch.zeros_like(v)
                         for k, v in model.state_dict().items()}
        self.round = 0

    def aggregate(self, client_deltas, client_sizes):
        total_size = sum(client_sizes)
        weights = self.model.state_dict()
        new_weights = {}

        for key in weights.keys():
            # Weighted average of deltas
            delta_avg = sum(
                d[key] * n for d, n in zip(client_deltas, client_sizes)
            ) / total_size

            # Orthogonalize
            delta_ortho = self._orthogonalize(delta_avg)

            # Update second moment
            self.v_global[key] = (
                self.beta2 * self.v_global[key] +
                (1 - self.beta2) * delta_ortho ** 2
            )

            # Compute update with slow momentum
            denom = torch.sqrt(self.v_global[key]) + 1e-8
            update = (delta_ortho + self.alpha * self.o_slow[key]) / denom

            # Apply update
            new_weights[key] = weights[key] - self.lr * update

        # Update slow momentum periodically
        self.round += 1
        if self.round % self.slow_chunk == 0:
            for key in weights.keys():
                delta_avg = sum(
                    d[key] * n for d, n in zip(client_deltas, client_sizes)
                ) / total_size
                self.m_slow[key] = (
                    self.beta3 * self.m_slow[key] + (1 - self.beta3) * delta_avg
                )
                self.o_slow[key] = self._orthogonalize(self.m_slow[key])

        self.model.load_state_dict(new_weights)
        return new_weights

    def _orthogonalize(self, tensor):
        if tensor.ndim < 2:
            return tensor

        mat = tensor.reshape(tensor.shape[0], -1)
        norm = torch.linalg.norm(mat)
        if norm < 1e-6:
            return tensor

        x = mat / norm
        eye = torch.eye(x.shape[1], device=x.device, dtype=x.dtype)
        for _ in range(self.ns_steps):
            x = 0.5 * x @ (3.0 * eye - x.T @ x)
        return x.reshape_as(tensor)
```

### 6.2 Fed-M3 Client

```python
class FedM3Client:
    def __init__(self, model, lr=0.01, beta1=0.9, beta2=0.999,
                 ns_steps=3):
        self.model = model
        self.optimizer = M3Optimizer(
            model.parameters(),
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            ns_steps=ns_steps
        )

    def local_train(self, dataloader, epochs):
        self.model.train()
        initial_weights = {
            k: v.clone() for k, v in self.model.state_dict().items()
        }

        for epoch in range(epochs):
            for x, y in dataloader:
                output = self.model(x)
                loss = F.cross_entropy(output, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        # Compute delta
        final_weights = self.model.state_dict()
        delta = {
            k: final_weights[k] - initial_weights[k]
            for k in final_weights.keys()
        }

        return delta
```

## 7. Hyperparameters

### Client-side
```
lr_local: Local learning rate (0.01 - 0.1)
beta1: Fast momentum decay (0.9)
beta2: Second moment decay (0.999)
ns_steps_local: Newton-Schulz steps (3)
local_epochs: So epochs local (1-10)
```

### Server-side
```
lr_server: Server learning rate (0.1 - 1.0)
beta3: Slow momentum decay (0.9)
alpha: Weight cua slow momentum (0.5 - 2.0)
ns_steps_server: Newton-Schulz steps (3-5)
slow_chunk: Cap nhat slow momentum moi ? rounds (5-20)
```

### Tuning strategy
1. Start voi default values
2. Tune lr_local va local_epochs truoc
3. Tune lr_server va alpha
4. Tune slow_chunk dua tren non-IID severity

## 8. Ablation Studies

### 8.1 Experiments

1. **Orthogonalization effect**
   - No orthogonalization
   - Client-side only
   - Server-side only
   - Both sides

2. **Slow momentum effect**
   - No slow momentum (alpha=0)
   - slow_chunk = 5, 10, 20, 50

3. **Newton-Schulz steps**
   - ns_steps = 1, 3, 5, 10

4. **Non-IID severity**
   - Dirichlet alpha = 0.1, 0.5, 1.0, inf

### 8.2 Expected Results

| Variant | Non-IID Acc | Stability | Comm Cost |
|---------|-------------|-----------|-----------|
| No ortho | 72% | Low | 1x |
| Client ortho | 75% | Medium | 1x |
| Server ortho | 76% | High | 1x |
| Full Fed-M3 | **79%** | **High** | 1x |

## 9. So sanh ly thuyet

### Convergence (informal)

Fed-M3 co the dat:
```
E[||nabla f(W^T)||^2] <= O(1/sqrt(T)) + O(sigma^2/K) + O(bias_ortho)
```

Trong do bias_ortho la bias tu orthogonalization, thuong nho.

### So sanh voi baselines

| Method | Handles Drift | Long-term Memory | Conflict Resolution |
|--------|--------------|------------------|---------------------|
| FedAvg | No | No | No |
| FedProx | Partial | No | No |
| SCAFFOLD | Yes | Limited | No |
| FedAdam | Partial | Server momentum | Partial |
| **Fed-M3** | **Yes** | **Multi-scale** | **Newton-Schulz** |

## 10. Discussion

### Uu diem cua Fed-M3
1. **Natural hierarchy**: Client-server map voi fast-slow momentum
2. **Conflict resolution**: Newton-Schulz giam gradient conflicts
3. **Long-term memory**: Slow momentum giu global information
4. **Communication efficient**: Khong tang bandwidth

### Han che
1. **Computation overhead**: Newton-Schulz mat them ~10-20% FLOPs
2. **Memory**: Server can luu m_slow, o_slow, v_global
3. **Tuning**: Nhieu hyperparameters hon FedAvg

### Khi nao nen dung Fed-M3?
- Non-IID data severe
- Can stability cao
- Communication la bottleneck (it rounds hon la quan trong)

## 11. Next Steps

1. Implement Fed-M3 day du
2. Setup non-IID experiments (Dirichlet split)
3. Ablation studies
4. So sanh voi FedAvg, FedProx, SCAFFOLD, FedAdam
5. Phan tich convergence ly thuyet
