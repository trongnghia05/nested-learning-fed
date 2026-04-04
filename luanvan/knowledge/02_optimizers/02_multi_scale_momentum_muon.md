# Multi-scale Momentum Muon (M3)

---

# MUC LUC

1. [Tong quan](#1-tong-quan)
2. [Muon Optimizer - Nen tang](#2-muon-optimizer---nen-tang)
3. [Multi-scale Momentum](#3-multi-scale-momentum)
4. [Cong thuc M3 chi tiet](#4-cong-thuc-m3-chi-tiet)
5. [Tai sao M3 phu hop voi FL?](#5-tai-sao-m3-phu-hop-voi-fl)
6. [Thiet ke Fed-M3](#6-thiet-ke-fed-m3)
7. [Implementation Notes](#7-implementation-notes)
8. [So sanh voi cac optimizer khac](#8-so-sanh-voi-cac-optimizer-khac)
9. [Cau hoi nghien cuu cho luan van](#9-cau-hoi-nghien-cuu-cho-luan-van)
10. [Pseudo-code M3 cho FL](#10-pseudo-code-m3-cho-fl)
11. [Key insights cho luan van](#11-key-insights-cho-luan-van)
12. **[SO SANH FED-M3 LITE VS M3 PAPER](#12-so-sanh-fed-m3-lite-vs-m3-paper)** ← MOI

---

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

---

## 12. SO SANH FED-M3 LITE VS M3 PAPER

> **Muc dich:** Xac dinh Fed-M3 Lite (implementation hien tai) khac M3 Paper (Algorithm 1) o dau.

### 12.1 Tong quan khac biet

| Component | M3 Paper (Algorithm 1) | Fed-M3 Lite (Implementation) | Khac biet |
|-----------|------------------------|------------------------------|-----------|
| **Momentum style** | ACCUMULATION | EMA | ⚠️ **KHAC** |
| **Newton-Schulz** | CO | KHONG | ⚠️ **KHAC** |
| **Second moment (V)** | CO | KHONG | ⚠️ **KHAC** |
| **Multi-scale** | CO (fast + slow) | CO (fast + slow) | ✅ GIONG |
| **Normalization** | Newton-Schulz | Simple norm (÷ ||m2|| × 0.1) | ⚠️ **KHAC** |

---

### 12.2 Fast Momentum (M1 / m1)

#### M3 Paper:
```
M₁ = M₁ + β₁ · g        (ACCUMULATION - unbounded)
O₁ = Newton-Schulz(M₁)   (Orthogonalize)
```

#### Fed-M3 Lite:
```python
# fed_m3.py, line 120
m1.mul_(beta1).add_(grad)
# => m1 = β1·m1 + g        (EMA - bounded)
# KHONG co Newton-Schulz
```

#### Phan tich:
| | Paper | Fed-M3 Lite |
|-|-------|-------------|
| **Formula** | M = M + β·g | m = β·m + g |
| **Style** | Accumulation | EMA |
| **Bounded?** | KHONG (tang vo han) | CO |
| **Ly do khac** | Paper co NS de normalize | EMA tu bounded, khong can NS |

---

### 12.3 Slow Momentum (M2 / m2)

#### M3 Paper:
```
M₂ = M₂ + β₃ · Σᵢ gᵢ     (ACCUMULATION - unbounded)
O₂ = Newton-Schulz(M₂)    (Orthogonalize, moi f steps)
```

#### Fed-M3 Lite:
```python
# fed_m3.py, line 294
m2[key].mul_(beta3).add_(buffer)
# => m2 = β3·m2 + buffer   (EMA - bounded)

# Line 303: Simple normalization
global_momentum[key] = momentum / norm * 0.1
```

#### Phan tich:
| | Paper | Fed-M3 Lite |
|-|-------|-------------|
| **Formula** | M2 = M2 + β3·Σg | m2 = β3·m2 + buffer |
| **Normalization** | Newton-Schulz | Simple: m2/||m2||×0.1 |
| **Update frequency** | Moi f steps | Moi round |

---

### 12.4 Second Moment (V)

#### M3 Paper:
```
V = V + β₂ · g²
denom = √V + ε
update = (O₁ + α·O₂) / denom     (Adaptive scaling)
```

#### Fed-M3 Lite:
```python
# KHONG CO second moment
update = m1 + lam * m2           (No adaptive scaling)
```

#### Phan tich:
| | Paper | Fed-M3 Lite |
|-|-------|-------------|
| **Second moment** | CO (like Adam) | KHONG |
| **Adaptive scaling** | CO (/ √V) | KHONG |
| **Ly do bo** | Giam complexity cho FL |

---

### 12.5 Newton-Schulz Orthogonalization

#### M3 Paper:
```
O₁ = Newton-Schulz_T(M₁)
O₂ = Newton-Schulz_T(M₂)

# Newton-Schulz iteration (T steps):
X₀ = M / ||M||
X_{k+1} = 0.5 · X_k · (3I - X_k^T · X_k)
```

#### Fed-M3 Lite:
```
KHONG CO Newton-Schulz
```

#### Ly do bo Newton-Schulz trong Fed-M3:
1. **NS output fixed magnitude (~2-3)** → Mat thong tin gradient size
2. **Clients contribute equally** → Khong tot cho weighted FedAvg
3. **Multi-scale momentum la key insight**, khong phai NS
4. **Computational cost** → NS la O(n²) per step

---

### 12.6 Update Rule

#### M3 Paper:
```
Θ = Θ - η · (O₁ + α·O₂) / (√V + ε)
       ↑     ↑      ↑        ↑
      lr    NS(M1) NS(M2)  second moment
```

#### Fed-M3 Lite:
```python
# Client: fed_m3.py, line 134, 137
update = m1 + lam * m2
p.data.add_(update, alpha=-lr)
# => Θ = Θ - lr · (m1 + λ·m2)
```

#### So sanh:
| | Paper | Fed-M3 Lite |
|-|-------|-------------|
| **Fast component** | O₁ = NS(M₁) | m1 (EMA, no NS) |
| **Slow component** | α·O₂ = α·NS(M₂) | λ·m2 (normalized) |
| **Denominator** | √V + ε | 1 (no scaling) |
| **Balance param** | α | λ (lam) |

---

### 12.7 Bang Tong Hop

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    M3 PAPER vs FED-M3 LITE                                  │
├─────────────────────┬───────────────────────┬───────────────────────────────┤
│ Component           │ M3 Paper              │ Fed-M3 Lite                   │
├─────────────────────┼───────────────────────┼───────────────────────────────┤
│ Fast momentum       │ M1 = M1 + β1·g        │ m1 = β1·m1 + g                │
│                     │ (Accumulation)        │ (EMA)                         │
├─────────────────────┼───────────────────────┼───────────────────────────────┤
│ Slow momentum       │ M2 = M2 + β3·Σg       │ m2 = β3·m2 + buffer           │
│                     │ (Accumulation)        │ (EMA)                         │
├─────────────────────┼───────────────────────┼───────────────────────────────┤
│ Second moment       │ V = V + β2·g²         │ KHONG CO                      │
├─────────────────────┼───────────────────────┼───────────────────────────────┤
│ Newton-Schulz       │ CO (T steps)          │ KHONG CO                      │
├─────────────────────┼───────────────────────┼───────────────────────────────┤
│ Normalization       │ NS orthogonalize      │ m2/||m2||×0.1                 │
├─────────────────────┼───────────────────────┼───────────────────────────────┤
│ Update              │ Θ - η(O1+αO2)/√V      │ Θ - lr(m1+λm2)                │
└─────────────────────┴───────────────────────┴───────────────────────────────┘
```

---

### 12.8 Ket luan

#### Fed-M3 Lite GIU LAI tu Paper:
1. ✅ **Multi-scale momentum** (fast + slow)
2. ✅ **Hierarchical structure** (client fast, server slow)
3. ✅ **Long-term memory** (slow momentum qua rounds)

#### Fed-M3 Lite BO/THAY DOI:
1. ❌ **Newton-Schulz** → Bo (fixed magnitude khong tot cho FL)
2. ❌ **Second moment V** → Bo (giam complexity)
3. ⚠️ **Accumulation → EMA** → Thay doi (EMA bounded, on dinh hon)
4. ⚠️ **NS normalization → Simple norm** → Thay doi (m2/||m2||×0.1)

#### Ly do thay doi:
```
1. Newton-Schulz output fixed magnitude (~2-3)
   → Mat thong tin gradient size
   → Khong tot cho weighted FedAvg

2. Accumulation khong bounded
   → M tang vo han neu khong co NS
   → EMA tu bounded, on dinh

3. Multi-scale momentum la CORE INSIGHT
   → Fast (local) + Slow (global) = FL hierarchy
   → Newton-Schulz la "nice to have", khong phai core
```

---

### 12.9 Cau hoi mo rong

1. **Co nen them lai Newton-Schulz?**
   - Thu: Apply NS sau EMA (bounded input)
   - Kiem tra: NS co giup giam client conflict?

2. **Co nen them second moment?**
   - Thu: v = β2·v + g² (Adam style)
   - Kiem tra: Adaptive scaling co giup FL?

3. **EMA vs Accumulation?**
   - Paper dung Accumulation vi co NS normalize
   - Neu bo NS, EMA tot hon (bounded)

---

*Cap nhat: 2026-04-04*
