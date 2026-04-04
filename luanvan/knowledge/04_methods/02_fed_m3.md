# Fed-M3 Lite: Federated Multi-scale Momentum

> Phien ban hien tai: **Fed-M3 Lite** (khong co Newton-Schulz)
> File implementation: `luanvan/experiments/optimizers/fed_m3.py`

## 1. Tong quan

**Fed-M3 Lite** la phuong phap de xuat dua tren Multi-scale Momentum tu Nested Learning, dieu chinh cho Federated Learning.

### Core Idea
```
Multi-scale optimization:
- Fast momentum (m1): Client-side, adapts quickly to local data
- Slow momentum (m2): Server-side, preserves long-term global direction

Key insight: Multi-scale momentum la core, KHONG PHAI Newton-Schulz.
```

### Tai sao KHONG dung Newton-Schulz?
```
Van de phat hien khi debug:
1. NS output co fixed magnitude (~2-3)
   -> Mat thong tin ve gradient size
   -> Clients contribute equally bat ke gradient lon hay nho

2. Pha vo FedAvg aggregation
   -> Weighted average khong con y nghia

3. Multi-scale momentum du tot
   -> NS la "nice to have", khong phai core
```

## 2. Cong thuc Fed-M3 Lite

### 2.1 Client Update (Fast Scale)

```
Input: grad, m1 (state), m2_global (from server)

# Fast momentum (EMA style, bounded)
m1 = beta1 * m1 + grad

# Combine local + global direction
update = m1 + lam * m2_global

# Apply update
theta = theta - lr * update
```

**Giai thich:**
- `m1`: Fast momentum, theo doi local gradient gan day
- `m2_global`: Slow momentum tu server, long-term global direction
- `lam`: Balance factor (0.3 default), can bang local vs global

### 2.2 Server Aggregation (Slow Scale)

```
Input: client_params, client_buffers

# Step 1: FedAvg aggregation
theta_global = weighted_average(client_params)

# Step 2: Aggregate gradient buffers
buffer_avg = weighted_average(client_buffers)

# Step 3: Update slow momentum (EMA)
m2 = beta3 * m2 + buffer_avg

# Step 4: Normalize to prevent unbounded growth
m2_normalized = m2 / ||m2|| * scale
```

**Giai thich:**
- Server dung FedAvg de aggregate model weights
- Slow momentum `m2` accumulate gradients qua nhieu rounds
- Normalization giu m2 bounded (scale = 0.1)

## 3. Algorithm Chi tiet

### Algorithm 1: Fed-M3 Lite

```
Input:
    - N clients, K selected per round
    - T communication rounds
    - E local epochs
    - beta1: Fast momentum coefficient (0.9)
    - beta3: Slow momentum coefficient (0.9)
    - lam: Balance factor (0.3)
    - lr: Learning rate (0.01)

Server Initialize:
    theta^0 <- random init
    m2 <- 0  # Slow momentum

For round r = 0, 1, ..., T-1:

    # 1. Server broadcasts
    Send theta^r, m2_normalized to selected clients

    # 2. Client local training (parallel)
    For each client k in S_r:
        theta_k <- theta^r
        m1_k <- 0  # RESET moi round
        buffer_k <- 0

        For epoch = 1, ..., E:
            For batch (x, y) in D_k:
                grad = gradient(loss(theta_k; x, y))

                # Fast momentum
                m1_k = beta1 * m1_k + grad

                # Accumulate for server
                buffer_k = buffer_k + grad

                # Combine with slow momentum
                update = m1_k + lam * m2_normalized

                # Apply
                theta_k = theta_k - lr * update

        Send theta_k, buffer_k to server

    # 3. Server aggregation
    # FedAvg
    theta^(r+1) = weighted_avg(theta_k)

    # Update slow momentum
    buffer_avg = weighted_avg(buffer_k)
    m2 = beta3 * m2 + buffer_avg

    # Normalize
    m2_normalized = m2 / ||m2|| * 0.1

Return theta^T
```

### Parameter Table

```
+------------------------------------------------------------------------------+
|                      FED-M3 LITE PARAMETER TABLE                              |
+-------------+-----------+---------+------------------------------------------+
| Parameter   | Location  | Reset?  | Description                              |
+-------------+-----------+---------+------------------------------------------+
| theta       | Both      | -       | Model weights                            |
| m1          | Client    | RESET   | Fast momentum, reset moi round           |
| m2          | Server    | NO      | Slow momentum, accumulate qua rounds     |
| buffer      | Client    | RESET   | Gradient buffer, gui len server          |
+-------------+-----------+---------+------------------------------------------+
| beta1       | Client    | -       | Fast momentum coef (0.9)                 |
| beta3       | Server    | -       | Slow momentum coef (0.9)                 |
| lam         | Client    | -       | Balance local/global (0.3)               |
| lr          | Client    | -       | Learning rate (0.01)                     |
+------------------------------------------------------------------------------+
```

## 4. Implementation

### 4.1 FedM3LiteOptimizer (Client)

```python
class FedM3LiteOptimizer(Optimizer):
    """
    Fed-M3 Lite: Multi-scale Momentum without Newton-Schulz.

    Update rule:
        m1 = beta1 * m1 + grad        # Fast momentum
        update = m1 + lam * m2_global # Combine with slow momentum
        theta = theta - lr * update
    """

    def __init__(self, params, lr=0.01, beta1=0.9, lam=0.3,
                 global_momentum=None):
        self.global_momentum = global_momentum or {}
        self.gradient_buffer = {}  # To send to server

    @torch.no_grad()
    def step(self):
        for p in params:
            grad = p.grad.data

            # Initialize state
            if 'm1' not in state:
                state['m1'] = torch.zeros_like(p.data)

            m1 = state['m1']

            # Fast momentum: m1 = beta1 * m1 + grad
            m1.mul_(beta1).add_(grad)

            # Accumulate gradient for server
            self.gradient_buffer[name].add_(grad)

            # Get slow momentum from server
            m2 = self.global_momentum.get(name, torch.zeros_like(m1))

            # Combine: update = m1 + lam * m2
            update = m1 + lam * m2

            # Apply: theta = theta - lr * update
            p.data.add_(update, alpha=-lr)
```

### 4.2 fed_m3_aggregate (Server)

```python
def fed_m3_aggregate(global_model, client_results, server_state, beta3=0.9):
    """
    Fed-M3 Lite aggregation: FedAvg + slow momentum update.
    """
    total_samples = sum(r['num_samples'] for r in client_results)

    # 1. FedAvg: Aggregate model parameters
    aggregated_params = weighted_average(client_params)
    global_model.load_state_dict(aggregated_params)

    # 2. Aggregate gradient buffers
    aggregated_buffer = weighted_average(client_buffers)

    # 3. Update slow momentum (EMA, NO Newton-Schulz)
    m2 = server_state['m2']
    for key, buffer in aggregated_buffer.items():
        m2[key].mul_(beta3).add_(buffer)

    # 4. Normalize to prevent unbounded growth
    for key, momentum in m2.items():
        norm = torch.norm(momentum)
        if norm > 1e-6:
            global_momentum[key] = momentum / norm * 0.1

    return {'global_momentum': global_momentum, ...}
```

## 5. So sanh voi Fed-M3 Full (da bo)

| Aspect | Fed-M3 Full (docs cu) | Fed-M3 Lite (hien tai) |
|--------|----------------------|------------------------|
| Newton-Schulz | Co (client + server) | **KHONG** |
| Second moment v | Co | **KHONG** |
| Fast momentum | `β1*m1 + (1-β1)*grad` | `β1*m1 + grad` |
| Server update | `(o1 + α*o2) / sqrt(v)` | FedAvg + normalize |
| Complexity | Cao | **Thap** |
| Debug status | That bai (~10% acc) | **76% acc @ round 10** |

### Tai sao Full version that bai?
```
1. Newton-Schulz normalize gradient to fixed magnitude
   -> Clients voi gradient nho duoc "boost" qua muc
   -> Clients voi gradient lon bi "shrink"
   -> Mat thong tin quan trong

2. Second moment v cung bi anh huong
   -> v tinh tren normalized gradients
   -> Khong phan anh true gradient variance

3. Ket qua: Model khong hoc duoc, accuracy ~10% (random)
```

## 6. Hyperparameters

### Recommended Values

| Parameter | Value | Notes |
|-----------|-------|-------|
| beta1 | 0.9 | Fast momentum, standard value |
| beta3 | 0.9 | Slow momentum, standard value |
| lam | 0.3 | Balance factor, tune 0.1-0.5 |
| lr | 0.01 | Learning rate |
| normalize_scale | 0.1 | m2 normalization scale |

### Tuning Guidelines

```
1. Bat dau voi defaults: beta1=0.9, beta3=0.9, lam=0.3

2. Neu model khong stable:
   - Giam lam (0.1, 0.2) -> it anh huong tu global
   - Tang beta3 (0.95, 0.99) -> slow momentum smooth hon

3. Neu convergence cham:
   - Tang lam (0.4, 0.5) -> nhieu global direction
   - Giam beta3 (0.8) -> slow momentum reactive hon

4. Non-IID severe:
   - Giu lam cao (0.3-0.5) -> global direction giup nhieu
```

## 7. So sanh voi cac methods khac

### Fed-M3 Lite vs FedAvg

```
FedAvg:
    theta = theta - lr * grad
    -> Khong co momentum
    -> Khong co global direction

Fed-M3 Lite:
    m1 = beta1 * m1 + grad
    update = m1 + lam * m2_global
    theta = theta - lr * update
    -> Fast momentum smooth local updates
    -> Slow momentum cung cap global direction
```

### Fed-M3 Lite vs FedProx

```
FedProx:
    theta = theta - lr*grad - lr*mu*(theta - theta_global)
    -> Linear penalty keo ve global model
    -> Isotropic (tat ca huong)

Fed-M3 Lite:
    update = m1 + lam * m2_global
    theta = theta - lr * update
    -> Them global DIRECTION (khong phai position)
    -> m2 la gradient direction, khong phai model position
```

### Fed-M3 Lite vs Fed-DGD

```
Fed-DGD:
    k = normalize(theta - theta_global)  # Drift direction
    theta = theta - lr*grad - decay*(k.theta)*k
    -> Decay theo DRIFT direction
    -> Selective forgetting

Fed-M3 Lite:
    update = m1 + lam * m2_global
    -> Them GLOBAL GRADIENT direction
    -> Long-term memory (m2 accumulate qua rounds)
```

| Aspect | FedAvg | FedProx | Fed-DGD | Fed-M3 Lite |
|--------|--------|---------|---------|-------------|
| Momentum | No | No | No | **Yes (multi-scale)** |
| Global info | Position | Position | Drift dir | **Gradient dir** |
| Long-term memory | No | No | No | **Yes (m2)** |
| Complexity | Low | Low | Medium | Medium |

## 8. Ket qua Experiments

### CIFAR-10, Dirichlet alpha=0.5

```
Round 10 results:
- FedAvg:     ~70% (baseline)
- FedProx:    ~72%
- Fed-M3 Lite: 76.14%  <-- Best
```

### Observations

```
1. Multi-scale momentum giup convergence nhanh hon
   - Fast momentum smooth local noise
   - Slow momentum cung cap stable direction

2. Khong can Newton-Schulz de dat ket qua tot
   - Simple normalization du de prevent unbounded growth
   - Giu nguyen gradient magnitude information

3. lam = 0.3 la balance tot
   - Qua nho: Khong tan dung global direction
   - Qua lon: Local adaptation bi giam
```

## 9. Limitations va Future Work

### Limitations

```
1. Extra communication: Gui gradient buffer len server
   - Hien tai: gui ca buffer (same size as model)
   - Co the optimize: chi gui compressed version

2. Server memory: Luu m2 cho moi parameter
   - Same size as model
   - Chap nhan duoc cho small-medium models

3. Hyperparameter sensitivity:
   - lam can tune cho tung dataset
   - normalize_scale (0.1) co the khong optimal
```

### Future Work

```
1. Gradient compression: Giam communication cost
2. Adaptive lam: Tu dong dieu chinh theo non-IID level
3. Federated Adam integration: Ket hop voi adaptive learning rate
4. Theoretical analysis: Convergence guarantees
```

## 10. Usage

### Command

```bash
python run_experiment.py --method fed_m3 --dataset cifar10 \
    --alpha 0.5 --num-rounds 100 --debug
```

### Code Example

```python
from optimizers.fed_m3 import fed_m3_optimizer_fn, fed_m3_aggregate

# Client side
optimizer, extra = fed_m3_optimizer_fn(
    model,
    lr=0.01,
    beta1=0.9,
    lam=0.3,
    extra_state={'global_momentum': server_m2}
)

# Train locally...

# Server side
result = fed_m3_aggregate(
    global_model,
    client_results,
    server_state,
    beta3=0.9
)
```

---

## Appendix: Lich su phat trien

### Version 1: Fed-M3 Full (THAT BAI)
- Newton-Schulz ca client va server
- Second moment v
- Ket qua: ~10% accuracy (random)
- Van de: NS normalize mat gradient magnitude

### Version 2: Fed-M3 Lite (HIEN TAI)
- Bo Newton-Schulz
- Bo second moment
- Simple normalization cho m2
- Ket qua: 76% accuracy @ round 10

---

*Cap nhat: 2026-04-04*
