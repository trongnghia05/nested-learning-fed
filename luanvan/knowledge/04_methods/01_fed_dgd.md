# Fed-DGD: Federated Delta Gradient Descent

> Phien ban hien tai: **Fed-DGD v2 (Drift Direction)**
> File implementation: `luanvan/experiments/optimizers/fed_dgd.py`

---

## 1. Tong quan

### 1.1 Fed-DGD la gi?

Fed-DGD la phuong phap dua tren Delta Gradient Descent tu Nested Learning paper, dieu chinh cho Federated Learning de giam **client drift**.

### 1.2 Core Idea

```
+-------------------------------------------------------------+
|  FED-DGD CORE IDEA:                                          |
|                                                              |
|  Van de: Client drift trong FL                               |
|  - Client train tren local non-IID data                      |
|  - Local model DRIFT xa khoi global model                    |
|  - Khi aggregate: conflict, slow convergence                 |
|                                                              |
|  Giai phap: Decay theo DRIFT direction                       |
|  - k = normalize(theta - theta_global)  # Drift direction    |
|  - Decay component cua theta theo huong k                    |
|  - Ket qua: Giam drift, model gan global hon                 |
+-------------------------------------------------------------+
```

---

## 2. So sanh voi Paper goc

### 2.1 Paper DGD (Nested Learning, Eq. 28-29)

```
W_{t+1} = W_t (I - x_t x_t^T) - eta * grad_W

Trong do:
- x_t: Input vector
- (I - x_t x_t^T): Decay matrix, "quen" info theo huong x_t
- Muc dich: Xu ly data dependencies trong sequential learning
```

### 2.2 Code TITAN (self_modifying.py)

```python
# Preconditioner
kk = torch.einsum("bi,bj->bij", k_t, k_t)  # k ⊗ k
precond = alpha_t * eye - eta_t * kk       # α*I - η*(k⊗k)

# Update
W = W @ precond - eta * g                  # W = W(α*I - η*k⊗k) - η*g
```

### 2.3 Code Fed-DGD (Implementation hien tai)

```python
# k = DRIFT direction (KHONG phai input hay gradient)
drift = p.data - global_params[param_name]
k = drift / torch.norm(drift)  # normalize

# Update (simplified, O(d) thay vi O(d²))
p.data.add_(grad, alpha=-lr)                    # - lr * grad
decay_term = decay_strength * torch.dot(p, k) * k
p.data.add_(decay_term, alpha=-lr)              # - lr * decay * (k·θ)*k
```

### 2.4 Bang so sanh chi tiet

| Aspect | Paper DGD (Eq. 28) | Code TITAN | **Fed-DGD (hien tai)** |
|--------|-------------------|------------|------------------------|
| **k la gi** | Input x_t | Key k_t (projected) | **Drift direction** |
| **Decay matrix** | I - x⊗x | αI - η(k⊗k) | **Khong dung matrix** |
| **Update** | W = W(I-xx^T) - η∇L | W = W@P - η*g | **θ = θ - lr*g - lr*decay*(k·θ)*k** |
| **Memory** | O(d²) | O(d²) | **O(d)** |
| **Muc dich** | Quen input info | Quen key info | **Giam client drift** |
| **Context** | Sequential learning | Neural Memory | **Federated Learning** |

### 2.5 Tai sao khac Paper?

```
1. KHONG CO Neural Memory module
   - Paper dung k = key tu Memory module
   - Fed-DGD dung CNN cho CIFAR-10, khong co Memory
   - Nen can tim k khac

2. MUC DICH KHAC
   - Paper: Quen old input info khi hoc new input
   - Fed-DGD: Giam client drift trong FL

3. EFFICIENCY
   - Paper: Full matrix P, O(d²) memory per layer
   - Fed-DGD: Projection trick, O(d) memory

4. DRIFT la tu nhien trong FL
   - drift = theta_local - theta_global
   - Decay theo drift = keo model ve gan global
   - Giong FedProx nhung theo DIRECTION thay vi position
```

---

## 3. Cong thuc Fed-DGD (Hien tai)

### 3.1 Client Update

```
Input: grad, theta, theta_global

# 1. Tinh DRIFT direction
drift = theta - theta_global
k = drift / ||drift||                    # Unit vector

# 2. Gradient step
theta = theta - lr * grad

# 3. Decay theo DRIFT direction
proj = dot(k, theta)                     # Projection cua theta len k
decay_term = decay_strength * proj * k
theta = theta - lr * decay_term
```

**Cong thuc day du:**
```
theta = theta - lr * grad - lr * decay_strength * (k · theta) * k

Trong do:
- k = normalize(theta - theta_global): DRIFT direction
- (k · theta): Projection cua theta len k
- decay_strength: He so decay (default 0.1)
```

### 3.2 Server Aggregation

```
# 1. FedAvg aggregation (giong FedAvg)
theta_global = weighted_average(theta_i)

# 2. Aggregate drift directions (optional, for logging)
k_global = weighted_average(k_i)
k_global = normalize(k_global)
```

### 3.3 Geometric Interpretation

```
+-------------------------------------------------------------+
|  GEOMETRIC VIEW:                                             |
|                                                              |
|  theta_global ----drift----> theta_local                     |
|                     k = drift direction                      |
|                                                              |
|  Fed-DGD: Decay component cua theta theo huong k             |
|                                                              |
|  Truoc:  theta_local co component lon theo k                 |
|  Sau:    theta_local co component nho hon theo k             |
|          => Gan theta_global hon                             |
|                                                              |
|  Khac FedProx:                                               |
|  - FedProx: Keo POSITION ve theta_global (linear penalty)    |
|  - Fed-DGD: Decay COMPONENT theo drift direction (selective) |
+-------------------------------------------------------------+
```

---

## 4. Algorithm

### 4.1 Pseudocode

```
Algorithm: Fed-DGD (Drift Direction Version)

Input:
    - N clients, K selected per round
    - T communication rounds
    - E local epochs
    - decay_strength: Drift decay coefficient (0.1)
    - lr: Learning rate (0.01)

Server Initialize:
    theta^0 <- random init

For round r = 0, 1, ..., T-1:

    # 1. Server broadcasts
    Send theta^r to selected clients

    # 2. Client local training
    For each client k in S_r:
        theta_k <- theta^r        # Initialize from global
        theta_global <- theta^r   # Save for drift computation

        For epoch = 1, ..., E:
            For batch (x, y) in D_k:
                grad = gradient(loss(theta_k; x, y))

                # Compute DRIFT direction
                drift = theta_k - theta_global
                k = normalize(drift)  # Unit vector

                # Gradient step
                theta_k = theta_k - lr * grad

                # Decay along DRIFT direction
                proj = dot(k, theta_k)
                decay_term = decay_strength * proj * k
                theta_k = theta_k - lr * decay_term

        Send theta_k, k to server

    # 3. Server aggregation (FedAvg)
    theta^(r+1) = weighted_avg(theta_k)

    # Optional: aggregate drift directions for analysis
    k_global = weighted_avg(k_i)

Return theta^T
```

### 4.2 Parameter Table

```
+------------------------------------------------------------------------------+
|                      FED-DGD PARAMETER TABLE                                  |
+-------------+-----------+---------+------------------------------------------+
| Parameter   | Location  | Reset?  | Description                              |
+-------------+-----------+---------+------------------------------------------+
| theta       | Both      | -       | Model weights                            |
| theta_global| Client    | RESET   | Copy of global model, dung de tinh drift |
| k           | Client    | RESET   | Drift direction, tinh moi step           |
+-------------+-----------+---------+------------------------------------------+
| decay_strength| Client  | -       | Drift decay coefficient (0.1)            |
| lr          | Client    | -       | Learning rate (0.01)                     |
| alpha       | Client    | -       | Uniform decay (1.0 = TAT)                |
+------------------------------------------------------------------------------+
```

---

## 5. Implementation

### 5.1 FedDGDOptimizer (Client)

```python
class FedDGDOptimizer(Optimizer):
    """
    Fed-DGD: Simplified Delta Gradient Descent for FL.

    Update rule:
        k = normalize(theta - theta_global)  # Drift direction
        theta = theta - lr * grad - lr * decay_strength * (k·theta) * k
    """

    def __init__(self, params, lr=0.01, alpha=1.0, decay_strength=0.1,
                 global_params=None):
        self.global_params = global_params  # theta_global

    @torch.no_grad()
    def step(self):
        for p in params:
            grad = p.grad.data

            # Compute DRIFT direction
            if param_name in self.global_params:
                global_p = self.global_params[param_name]
                drift = p.data - global_p
                drift_norm = torch.norm(drift)
                if drift_norm > 1e-8:
                    k = drift / drift_norm  # Unit vector
                else:
                    k = torch.zeros_like(p.data)
            else:
                k = torch.zeros_like(p.data)

            # 1. Uniform decay (skip if alpha = 1.0)
            if alpha < 1.0:
                p.data.mul_(alpha)

            # 2. Gradient step
            p.data.add_(grad, alpha=-lr)

            # 3. Decay along DRIFT direction
            if decay_strength > 0 and torch.norm(k) > 1e-8:
                p_flat = p.data.view(-1)
                k_flat = k.view(-1)
                proj = torch.dot(p_flat, k_flat)  # (k · theta)
                decay_term = decay_strength * proj * k
                p.data.add_(decay_term, alpha=-lr)
```

### 5.2 fed_dgd_aggregate (Server)

```python
def fed_dgd_aggregate(global_model, client_results, server_state):
    """Fed-DGD aggregation: FedAvg + optional k aggregation."""

    # 1. FedAvg aggregation
    aggregated_params = weighted_average(client_params)
    global_model.load_state_dict(aggregated_params)

    # 2. Aggregate drift directions (for logging/analysis)
    aggregated_k = weighted_average(client_k)
    global_k = normalize(aggregated_k)

    return {'drift_direction': global_k, ...}
```

---

## 6. So sanh voi cac methods khac

### 6.1 Fed-DGD vs FedProx

```
FedProx:
    theta = theta - lr*grad - lr*mu*(theta - theta_global)
                              |__________________________|
                              LINEAR penalty (keo ve global)
                              Isotropic (tat ca huong)

Fed-DGD:
    k = normalize(theta - theta_global)
    theta = theta - lr*grad - lr*decay*(k · theta)*k
                              |___________________|
                              PROJECTION decay
                              Anisotropic (chi huong drift)
```

| Aspect | FedProx | Fed-DGD |
|--------|---------|---------|
| Penalty type | Linear | Projection |
| Direction | Isotropic (all) | Anisotropic (drift only) |
| Selective | No | Yes |
| Hyperparameter | mu | decay_strength |

### 6.2 Fed-DGD vs Fed-M3 Lite

```
Fed-M3 Lite:
    m1 = beta1 * m1 + grad
    update = m1 + lam * m2_global
    theta = theta - lr * update
    -> Them GRADIENT direction tu server
    -> Long-term memory (m2 accumulate)

Fed-DGD:
    k = normalize(theta - theta_global)
    theta = theta - lr*grad - lr*decay*(k·theta)*k
    -> Decay theo DRIFT direction
    -> Khong co long-term memory
```

| Aspect | Fed-M3 Lite | Fed-DGD |
|--------|-------------|---------|
| Mechanism | Multi-scale momentum | Drift decay |
| Global info | Gradient direction (m2) | Model position (theta_global) |
| Long-term memory | Yes (m2) | No |
| Extra comm | gradient buffer | drift direction k |

---

## 7. Hyperparameters

### 7.1 Default Values

| Parameter | Value | Notes |
|-----------|-------|-------|
| decay_strength | 0.1 | Drift decay, tune 0.05-0.2 |
| alpha | 1.0 | Uniform decay = TAT |
| lr | 0.01 | Learning rate |

### 7.2 Tuning Guidelines

```
1. Bat dau voi defaults: decay_strength=0.1, alpha=1.0

2. Neu model khong stable (drift qua nhieu):
   - Tang decay_strength (0.15, 0.2)
   - Hoac giam alpha (0.99, 0.95) de them uniform decay

3. Neu convergence cham (decay qua manh):
   - Giam decay_strength (0.05, 0.02)
   - Giu alpha = 1.0

4. Non-IID severe:
   - Tang decay_strength (0.15-0.2)
   - Client drift nhieu nen can decay manh hon
```

---

## 8. Lich su phat trien

### Version 1: k = Gradient Direction (DA BO)

```
Idea: k = normalize(accumulated_gradient)
      Decay theo huong gradient

Van de: TU PHA MINH
- Decay theo gradient = giam learning signal
- Model khong hoc duoc
- Ket qua kem
```

### Version 2: k = Drift Direction (HIEN TAI)

```
Idea: k = normalize(theta - theta_global)
      Decay theo huong DRIFT

Tai sao tot hon:
- Drift = su khac biet giua local va global
- Decay theo drift = keo local ve gan global
- KHONG pha gradient, chi giam drift
- Giong FedProx nhung selective (chi theo huong drift)
```

---

## 9. Usage

### Command

```bash
python run_experiment.py --method fed_dgd --dataset cifar10 \
    --alpha 0.5 --dgd-decay-strength 0.1 --num-rounds 100 --debug
```

### Code Example

```python
from optimizers.fed_dgd import fed_dgd_optimizer_fn, fed_dgd_aggregate

# Client side
optimizer, extra = fed_dgd_optimizer_fn(
    model,
    lr=0.01,
    decay_strength=0.1,
    extra_state={'global_params': server_params}
)

# Train locally...

# Server side
result = fed_dgd_aggregate(
    global_model,
    client_results,
    server_state
)
```

---

## 10. Relationship voi Paper goc

### 10.1 Dong gop cua luan van

```
1. ADAPTATION cho FL:
   - Paper dung k = input/key (cho Neural Memory)
   - Luan van dung k = drift direction (cho FL)

2. SIMPLIFIED implementation:
   - Paper: Full matrix P = I - k⊗k, O(d²)
   - Luan van: Projection (k·θ)*k, O(d)

3. NEW interpretation:
   - Paper: "Selective forgetting" of old input info
   - Luan van: "Drift reduction" in federated setting
```

### 10.2 Co the cai tien

```
1. Full matrix P (neu co du memory)
   - Dung P = αI - η(k⊗k) thay vi projection
   - Co the hieu qua hon nhung O(d²)

2. Learned decay_strength
   - Hien tai: fixed hyperparameter
   - Co the: learn tu data (nhu Paper)

3. Multi-scale decay
   - Ket hop voi Fed-M3: decay + multi-scale momentum
```

---

*Cap nhat: 2026-04-04*
