# Nested Learning Paper - Cong Thuc Chinh Xac

> **Tai lieu goc** cho viec so sanh voi cac thuat toan khac
>
> **Nguon:** Nested Learning: The Illusion of Deep Learning Architectures (Behrouz et al., Google Research)

---

# PHAN 1: CONG THUC CHINH XAC TU PAPER

---

## 1. Gradient Descent (Paper Eq. 3)

$$
W_{t+1} = W_t - \eta_{t+1} \nabla_{W_t} \mathcal{L}(W_t; x_{t+1})
$$

Hoac viet day du:

$$
W_{t+1} = W_t - \eta_{t+1} \nabla_{y_{t+1}} \mathcal{L}(W_t; x_{t+1}) \otimes x_{t+1}
$$

---

## 2. Momentum (Paper Eq. 7-8 va Eq. 16-17)

### 2.1 Phien ban don gian (Eq. 7-8)

$$
\boxed{
\begin{aligned}
W_{t+1} &= W_t - \mathbf{m}_{t+1} \\
\mathbf{m}_{t+1} &= \mathbf{m}_t - \eta_{t+1} \nabla_{W_t} \mathcal{L}(W_t; x_{t+1})
\end{aligned}
}
$$

**Chu y:** Day la tich luy **AM** cua gradient (m = m - η·∇L), W **TRU** m.

### 2.2 Phien ban voi decay (Eq. 16-17)

$$
\boxed{
\begin{aligned}
W_{i+1} &= W_i + \mathbf{m}_{i+1} \\
\mathbf{m}_{i+1} &= \alpha_{i+1} \mathbf{m}_i - \eta_t \nabla \mathcal{L}(W_i; x_i)
\end{aligned}
}
$$

**Trong do:**
- $\alpha_i$ : Momentum decay coefficient
- $\eta_t$ : Learning rate

> Paper: "momentum can indeed be viewed as a meta memory module that learns how to memorize gradients"

---

## 3. Delta Gradient Descent - DGD (Paper Eq. 27-29)

### 3.1 Van de voi GD

GD truyen thong dung dot-product objective:

$$
\min_W \langle W x_t, \nabla_{y_t} \mathcal{L}(W_t; x_t) \rangle
$$

> Paper: "the above formulation cause ignoring the dependencies of data samples"

### 3.2 L2 Regression Objective (Eq. 27)

$$
\min_W \| W x_t - \nabla_{y_t} \mathcal{L}(W_t; x_t) \|_2^2
$$

### 3.3 DGD Update Rule (Eq. 28-29)

$$
\boxed{
W_{t+1} = W_t (I - x_t x_t^\top) - \eta_{t+1} \nabla_{W_t} \mathcal{L}(W_t; x_t)
}
$$

Hoac viet day du:

$$
W_{t+1} = W_t (I - x_t x_t^\top) - \eta_{t+1} \nabla_{y_t} \mathcal{L}(W_t; x_t) \otimes x_t
$$

**Trong do:**
- $(I - x_t x_t^\top)$ : **Adaptive decay matrix**
- $\nabla_{y_t} \mathcal{L}$ : Local Surprise Signal (LSS)

---

## 4. Momentum voi Preconditioning (Paper Eq. 20-21)

$$
\boxed{
\begin{aligned}
W_{i+1} &= W_i + \mathbf{m}_{i+1} \\
\mathbf{m}_{i+1} &= \alpha_{i+1} \mathbf{m}_i - \eta_t \mathbf{P}_i \nabla \mathcal{L}(W_i; x_i)
\end{aligned}
}
$$

**Trong do:** $\mathbf{P}_i$ la preconditioner matrix.

---

## 5. Delta Rule cho Momentum (Paper Eq. 21-22)

Dung L2 regression objective cho momentum:

$$
\min_{\mathbf{m}} \| \mathbf{m} \nabla \mathcal{L}(W_i; x_i)^\top - \mathbf{P}_i \|_2^2
$$

**Update rule:**

$$
\boxed{
\begin{aligned}
W_{i+1} &= W_i + \mathbf{m}_{i+1} \\
\mathbf{m}_{i+1} &= \left( \alpha_{i+1} I - \nabla \mathcal{L}(W_i; x_i)^\top \nabla \mathcal{L}(W_i; x_i) \right) \mathbf{m}_i - \eta_t \mathbf{P}_i \nabla \mathcal{L}(W_i; x_i)
\end{aligned}
}
$$

> Paper: "This update is based on delta-rule and so it allows the memory (momentum) to better manage its limited capacity"

---

## 6. Deep Momentum GD - DMGD (Paper Eq. 23)

$$
\boxed{
W_{i+1} = W_i + \mathbf{m}_{i+1}(\mathbf{u}_i), \quad \mathbf{m}_{i+1} = \alpha_{i+1} \mathbf{m}_i - \eta_t \nabla \mathcal{L}^{(2)}(\mathbf{m}_i; \mathbf{u}_i, I)
}
$$

**Trong do:**
- $\mathbf{u}_i = \nabla \mathcal{L}(W_i; x_i)$
- $\mathcal{L}^{(2)}$ : Internal objective cua momentum
- $\mathbf{m}(\cdot)$ : MLP (thay vi linear)

---

## 7. Muon Optimizer (Paper Eq. 24)

$$
\boxed{
W_{i+1} = W_i + \sigma(\mathbf{m}_{i+1}(\mathbf{u}_i)), \quad \mathbf{m}_{i+1} = \alpha_{i+1} \mathbf{m}_i - \eta_t \nabla \mathcal{L}^{(2)}(\mathbf{m}_i; \mathbf{u}_i, I)
}
$$

**Trong do:** $\sigma(\cdot) = \text{Newton-Schulz}(\cdot)$

> Paper: "we let σ(·) = Newton-Schulz(·)... the resulted optimizer is equivalent to Muon optimizer"

### 7.1 Newton-Schulz Iteration

$$
\begin{aligned}
X_0 &= M / \|M\| \\
X_{k+1} &= \frac{1}{2} X_k (3I - X_k^\top X_k)
\end{aligned}
$$

---

## 8. Continuum Memory System (Paper Eq. 30-31)

$$
\boxed{
\theta_{i+1}^{(f_\ell)} = \theta_i^{(f_\ell)} -
\begin{cases}
\sum_{t=i-C^{(\ell)}}^{i} \eta_t^{(\ell)} f(\theta_t^{(f_\ell)}; x_t) & \text{if } i \equiv 0 \pmod{C^{(\ell)}} \\
0 & \text{otherwise}
\end{cases}
}
$$

**Trong do:**
- $C^{(\ell)}$ : Chunk size (slow update moi $C^{(\ell)}$ steps)
- $f(\cdot)$ : Error signal (e.g., gradient)

---

## 9. Tong hop Cong Thuc Paper

| Algorithm | Equation | Weight Update | Momentum Update |
|-----------|----------|---------------|-----------------|
| GD | Eq. 3 | $W = W - \eta \nabla_W \mathcal{L}$ | - |
| Momentum | Eq. 16-17 | $W = W + m$ | $m = \alpha m - \eta \nabla \mathcal{L}$ |
| DGD | Eq. 28-29 | $W = W(I - xx^\top) - \eta \nabla_W \mathcal{L}$ | - |
| Delta Momentum | Eq. 21-22 | $W = W + m$ | $m = (\alpha I - g^\top g) m - \eta P g$ |
| Muon | Eq. 24 | $W = W + \text{NS}(m)$ | $m = \alpha m - \eta \nabla \mathcal{L}^{(2)}$ |

---

# PHAN 2: CODE IMPLEMENTATION

---

## 1. M3 Code (src/nested_learning/optim/m3.py)

### 1.1 Newton-Schulz trong Code

```python
def _newton_schulz(matrix, steps, eps=1e-6):
    x = matrix
    norm = torch.linalg.norm(x)
    x = x / (norm + eps)                    # Normalize
    eye = torch.eye(n)
    for _ in range(steps):
        x = 0.5 * x @ (3.0 * eye - x.T @ x) # NS iteration
    return x
```

**DUNG voi paper:** $X = \frac{1}{2} X (3I - X^\top X)$

### 1.2 M3 Step trong Code

```python
# Line 105: Fast momentum update
m1.add_(grad, alpha=beta1)
# => m1 = m1 + beta1 * grad

# Line 106: Second moment update
v.addcmul_(grad, grad, value=beta2)
# => v = v + beta2 * grad^2

# Line 107: Accumulate for slow momentum
slow_buffer.add_(grad)
# => slow_buffer = slow_buffer + grad

# Line 109: Orthogonalize fast momentum
o1 = _orthogonalize(m1, steps=ns_steps, eps=eps)

# Line 111-112: Compute update
denom = v.sqrt().add_(eps)
update = (o1 + alpha * o2) / denom

# Line 113: Apply update
p.add_(update, alpha=-lr)
# => p = p - lr * update

# Line 115-120: Slow momentum (every slow_chunk steps)
if step % slow_chunk == 0:
    m2.add_(slow_buffer, alpha=beta3)
    # => m2 = m2 + beta3 * slow_buffer
    slow_buffer.zero_()
    o2 = _orthogonalize(m2)
```

---

## 2. So sanh Paper vs Code

### 2.1 Momentum Update

| Aspect | Paper (Eq. 16-17) | Code (m3.py) |
|--------|-------------------|--------------|
| **Formula** | $m = \alpha m - \eta \nabla \mathcal{L}$ | `m = m + beta * grad` |
| **Decay** | Co ($\alpha$ decay term cu) | **KHONG** (khong decay) |
| **Gradient sign** | TRU ($- \eta \nabla$) | **CONG** ($+ \beta \cdot g$) |
| **Behavior** | EMA-style, bounded | Accumulation, **UNBOUNDED** |

### 2.2 Weight Update

| Aspect | Paper (Eq. 16-17) | Code (m3.py) |
|--------|-------------------|--------------|
| **Formula** | $W = W + m$ | `p = p - lr * update` |
| **Sign** | CONG m | TRU update |
| **Note** | m chua am gradient | update = NS(m)/sqrt(v) |

### 2.3 Second Moment

| Aspect | Paper Muon | Code M3 |
|--------|------------|---------|
| **Existence** | **KHONG CO** | Co (`v = v + beta2 * grad^2`) |
| **Purpose** | - | Adaptive learning rate (Adam-style) |

### 2.4 Multi-scale

| Aspect | Paper CMS (Eq. 31) | Code M3 |
|--------|-------------------|---------|
| **Slow update** | $\sum \eta_t f(\theta; x_t)$ | `m2 = m2 + beta3 * buffer` |
| **Frequency** | Moi $C^{(\ell)}$ steps | Moi `slow_chunk` steps |
| **Match** | **TUONG TU** | Explicit fast/slow |

---

## 3. Khac biet Chinh

### 3.1 KHAC BIET #1: Momentum Formula

**Paper:**
$$
m_{i+1} = \alpha \cdot m_i - \eta \cdot \nabla \mathcal{L}
$$

- Decay term cu voi $\alpha$
- TRU gradient moi voi $\eta$
- m **bounded** (do decay)

**Code:**
```python
m1.add_(grad, alpha=beta1)  # m1 = m1 + beta1 * grad
```

- **KHONG** decay term cu
- **CONG** gradient moi voi beta1
- m **unbounded** (tang vo han)

### 3.2 KHAC BIET #2: Second Moment

**Paper Muon:** Khong co second moment

**Code M3:**
```python
v.addcmul_(grad, grad, value=beta2)  # v = v + beta2 * grad^2
update = (o1 + alpha * o2) / denom   # denom = sqrt(v) + eps
```

Them Adam-style adaptive learning rate.

### 3.3 TUONG DONG: Newton-Schulz

**Paper:** $X = \frac{1}{2} X (3I - X^\top X)$

**Code:**
```python
x = 0.5 * x @ (3.0 * eye - x.T @ x)
```

**CHINH XAC** giong paper.

---

## 4. Giai thich Tai sao Code Khac Paper

### 4.1 Accumulation thay vi EMA

**Gia thuyet 1:** Newton-Schulz normalize output
- NS tra ve orthogonal matrix voi norm ~ 1
- Magnitude cua m khong quan trong
- Chi huong (direction) la quan trong

**Gia thuyet 2:** "Compress all gradients"
- Paper noi momentum la "memory that compress gradients"
- Accumulation giu **TAT CA** gradient history
- EMA "quen" gradient cu

### 4.2 Second Moment

- Paper Muon khong co, nhung Adam co
- Code ket hop Muon + Adam
- Giup adaptive learning rate per-parameter

---

## 5. Cong thuc Code Chinh xac

### M3 Code Formula:

$$
\begin{aligned}
m_1 &= m_1 + \beta_1 \cdot g \\
m_2 &= m_2 + \beta_3 \cdot \text{buffer} \quad (\text{moi } C \text{ steps}) \\
v &= v + \beta_2 \cdot g^2 \\
o_1 &= \text{NS}(m_1) \\
o_2 &= \text{NS}(m_2) \\
\theta &= \theta - \text{lr} \cdot \frac{o_1 + \alpha \cdot o_2}{\sqrt{v} + \epsilon}
\end{aligned}
$$

### So sanh truc quan:

```
PAPER Muon:
    m = α·m - η·∇L          (EMA, decay, bounded)
    W = W + NS(m)

CODE M3:
    m = m + β·g             (Accumulation, no decay, unbounded)
    v = v + β2·g²           (Second moment - THEM MOI)
    W = W - lr·NS(m)/√v     (Adaptive LR - THEM MOI)
```

---

*Cap nhat: 2026-04-04*
