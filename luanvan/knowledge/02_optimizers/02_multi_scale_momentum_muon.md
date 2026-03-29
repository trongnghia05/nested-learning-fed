# Multi-scale Momentum Muon (M3)

## 1. Tong quan

**M3 (Multi-scale Momentum Muon)** la optimizer duoc gioi thieu trong Nested Learning paper, ket hop:
- **Multi-scale momentum**: Fast + Slow momentum
- **Newton-Schulz orthogonalization**: Giam xung dot gradient
- **Adaptive learning**: Second moment estimation

## 2. Muon Optimizer - Nen tang

### Muon la gi?
Muon (Jordan et al., 2024) la optimizer cho hidden layers voi:
- Momentum update
- Newton-Schulz orthogonalization cho output

### Newton-Schulz Iteration
Orthogonalize matrix M thanh orthogonal matrix:
```python
def newton_schulz(M, steps=3):
    X = M / ||M||
    for _ in range(steps):
        X = 0.5 * X @ (3*I - X^T @ X)
    return X
```

### Loi ich cua Orthogonalization
- Giam correlation giua cac gradients
- On dinh training
- Giu feature scale

## 3. Multi-scale Momentum

### Y tuong
Thay vi 1 momentum, dung 2 momentum voi **toc do khac nhau**:

```
Fast Momentum (M1): Cap nhat moi step
    -> Theo doi thay doi ngan han

Slow Momentum (M2): Cap nhat moi K steps (slow_chunk)
    -> Giu thong tin dai han ve gradient landscape
```

### Lien he voi Nested Learning
```
Level 1 (cham): Slow momentum M2
    |
    +-- Level 2 (nhanh): Fast momentum M1
        |
        +-- Level 3 (nhanh nhat): Gradient hien tai
```

## 4. Cong thuc M3 chi tiet

### State variables
```
m1: Fast momentum
m2: Slow momentum
v:  Second moment (nhu Adam)
slow_buffer: Accumulate gradients cho slow update
o1: Orthogonalized fast momentum
o2: Orthogonalized slow momentum
```

### Algorithm
```python
# Moi step t:
grad = compute_gradient()

# Update fast momentum (moi step)
m1 = m1 + beta1 * grad

# Accumulate cho slow momentum
slow_buffer = slow_buffer + grad

# Update second moment
v = v + beta2 * grad^2

# Orthogonalize fast momentum
o1 = newton_schulz(m1)

# Compute update
denom = sqrt(v) + eps
update = (o1 + alpha * o2) / denom

# Apply update
W = W - lr * update

# Neu step % slow_chunk == 0:
if step % slow_chunk == 0:
    # Update slow momentum
    m2 = m2 + beta3 * slow_buffer
    slow_buffer = 0

    # Orthogonalize slow momentum
    o2 = newton_schulz(m2)
```

### Hyperparameters
```
lr: Learning rate (default: 1e-3)
beta1: Fast momentum decay (default: 0.9)
beta2: Second moment decay (default: 0.999)
beta3: Slow momentum decay (default: 0.9)
alpha: Weight cho slow momentum trong update (default: 1.0)
eps: Numerical stability (default: 1e-8)
ns_steps: So iterations Newton-Schulz (default: 3)
slow_chunk: Cap nhat slow momentum moi bao nhieu steps (default: 100)
weight_decay: L2 regularization (default: 0.0)
```

## 5. Tai sao M3 phu hop voi FL?

### Van de trong FL

**1. Conflicting gradients**
- Gradients tu cac clients khac nhau co the conflict
- Simple averaging trong FedAvg khong xu ly tot

**2. Client drift**
- Local training lam model "drift" khoi global optimum
- Can co "anchor" de giu huong chung

**3. Non-IID data**
- Gradient variance cao
- Can smoothing tot hon

### M3 giai quyet nhu the nao

**1. Newton-Schulz Orthogonalization**
```
Truoc:
    grad_1 = [1, 2, 3]
    grad_2 = [2, 3, 1]
    -> Cos similarity cao, co the conflict

Sau orthogonalization:
    o_grad_1 = orthogonalize(grad_1)
    o_grad_2 = orthogonalize(grad_2)
    -> Giam conflict khi aggregate
```

**2. Multi-scale Momentum trong FL**
```
Client level (fast):
    - Fast momentum M1 cho local training
    - Adapt nhanh voi local data

Server level (slow):
    - Slow momentum M2 cho global aggregation
    - Giu "long-term memory" cua gradient landscape
```

**3. Adaptive scaling**
- Second moment v giup scale gradients
- Giam variance trong aggregation

## 6. Thiet ke Fed-M3

### Option 1: Separate momentum levels
```
Client:
    - Maintain fast momentum m1_local
    - Train voi orthogonalized updates
    - Send: delta_W, m1_local (optional)

Server:
    - Maintain slow momentum m2_global
    - Aggregate: W_global = aggregate(W_1, ..., W_K)
    - Update: m2_global = m2_global + beta3 * (W_global - W_global_old)
    - Orthogonalize: o2_global = newton_schulz(m2_global)
```

### Option 2: Full M3 at client, simple aggregation
```
Client:
    - Full M3 optimizer for local training
    - Send: W_local only

Server:
    - Simple averaging: W_global = average(W_1, ..., W_K)
    - Optional: Apply orthogonalization to aggregated update
```

### Option 3: Hybrid
```
Client:
    - Fast momentum + orthogonalization
    - Send: W_local, gradient_stats

Server:
    - Slow momentum tu gradient_stats
    - Orthogonalize aggregated update
```

## 7. Implementation Notes

### Newton-Schulz cho >=2D tensors
```python
def orthogonalize(tensor, steps, eps):
    if tensor.ndim < 2:
        return tensor  # Skip 1D (bias, norms)

    # Reshape to 2D
    mat = tensor.reshape(tensor.shape[0], -1)

    # Apply Newton-Schulz
    ortho = newton_schulz(mat, steps, eps)

    # Reshape back
    return ortho.reshape_as(tensor)
```

### Computational cost
- Newton-Schulz: O(n^2) per step, O(steps * n^2) total
- Them ~10-20% overhead so voi Adam
- Worthwhile neu giam so communication rounds

## 8. So sanh voi cac optimizer khac

| Optimizer | Momentum | Orthogonalization | Multi-scale | Adaptive LR |
|-----------|----------|-------------------|-------------|-------------|
| SGD+M     | Yes      | No                | No          | No          |
| Adam      | Yes      | No                | No          | Yes         |
| Muon      | Yes      | Yes               | No          | No          |
| M3        | Yes      | Yes               | Yes         | Yes         |

## 9. Cau hoi nghien cuu cho luan van

1. **Communication efficiency**:
   - Orthogonalization giam variance -> it rounds hon?

2. **Convergence**:
   - Fed-M3 co convergence rate tot hon FedAdam?

3. **Non-IID robustness**:
   - Multi-scale momentum giu knowledge dai han?

4. **Design choices**:
   - Momentum o client hay server?
   - Bao nhieu steps Newton-Schulz la du?

## 10. Pseudo-code M3 cho FL

```python
# Server
class FedM3Server:
    def __init__(self):
        self.m2_global = {}  # Slow momentum
        self.o2_global = {}  # Orthogonalized slow momentum

    def aggregate(self, client_updates, round_num):
        # Average weights
        W_global = average(client_updates)

        # Compute global gradient approximation
        delta_W = W_global - self.W_old

        # Update slow momentum
        for name, param in delta_W.items():
            if name not in self.m2_global:
                self.m2_global[name] = zeros_like(param)
            self.m2_global[name] += beta3 * param

            # Orthogonalize periodically
            if round_num % slow_chunk == 0:
                self.o2_global[name] = orthogonalize(self.m2_global[name])

        self.W_old = W_global
        return W_global

# Client
class FedM3Client:
    def local_train(self, W_global, local_data):
        optimizer = M3(model.parameters())
        for epoch in range(local_epochs):
            for batch in local_data:
                loss = compute_loss(batch)
                loss.backward()
                optimizer.step()
        return model.state_dict()
```

## 11. Key insights cho luan van

1. **Multi-scale = Multi-level FL**
   - Fast momentum ~ Local adaptation
   - Slow momentum ~ Global knowledge

2. **Orthogonalization = Conflict resolution**
   - Giam xung dot khi aggregate non-IID gradients

3. **Natural fit for FL hierarchy**
   - Client-server architecture map truc tiep voi fast-slow momentum
