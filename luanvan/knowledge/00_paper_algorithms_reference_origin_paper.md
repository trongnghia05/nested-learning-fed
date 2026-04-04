# Nested Learning Paper - Cong Thuc Chinh Xac

> **Tai lieu goc** cho viec so sanh voi cac thuat toan khac
>
> **Nguon:** Nested Learning: The Illusion of Deep Learning Architectures (Behrouz et al., Google Research)
>
> **arXiv:** https://arxiv.org/abs/2512.24695

---

# MUC LUC

## PHAN 1: CONG THUC CHINH XAC TU PAPER
1. [Gradient Descent (Eq. 3)](#1-gradient-descent-paper-eq-3)
2. [Momentum Truyen Thong (Eq. 16-17)](#2-momentum-truyen-thong-paper-eq-16-17)
3. [Delta Gradient Descent - DGD (Eq. 27-29)](#3-delta-gradient-descent---dgd-paper-eq-27-29)
   - 3.1 Van de voi GD truyen thong
   - 3.2 L2 Regression Objective (Eq. 27)
   - 3.3 DGD Update Rule (Eq. 28-29)
   - 3.4 Giai thich cac thanh phan
   - 3.5 So sanh GD vs DGD
   - 3.6 Code DGD vs Paper DGD
4. [Muon Optimizer (Eq. 24)](#4-muon-optimizer-paper-eq-24)
5. [Multi-scale Momentum Muon - M3 (Algorithm 1)](#5-multi-scale-momentum-muon---m3-paper-algorithm-1)
   - 5.1 Algorithm 1 - Chinh xac tu Paper
   - 5.2 Cong thuc toan hoc
   - 5.3 Dac diem quan trong
6. [Newton-Schulz Iteration](#6-newton-schulz-iteration)
7. [Continuum Memory System (Eq. 30-31)](#7-continuum-memory-system-paper-eq-30-31)
8. [Tong hop Cong Thuc Paper](#8-tong-hop-cong-thuc-paper)

## PHAN 2: CODE IMPLEMENTATION
1. [M3 Code (m3.py)](#1-m3-code-srcnested_learningoptimm3py)
2. [So sanh Paper vs Code](#2-so-sanh-paper-vs-code)
3. [Ket luan](#3-ket-luan)
4. [Hyperparameters Mac Dinh](#4-hyperparameters-mac-dinh)

---

# PHAN 1: CONG THUC CHINH XAC TU PAPER

---

## 1. Gradient Descent (Paper Eq. 3)

$$
W_{t+1} = W_t - \eta_{t+1} \nabla_{W_t} \mathcal{L}(W_t; x_{t+1})
$$

---

## 2. Momentum Truyen Thong (Paper Eq. 16-17)

$$
\boxed{
\begin{aligned}
W_{i+1} &= W_i + \mathbf{m}_{i+1} \\
\mathbf{m}_{i+1} &= \alpha_{i+1} \mathbf{m}_i - \eta_t \nabla \mathcal{L}(W_i; x_i)
\end{aligned}
}
$$

> **Chu y:** Day la momentum TRUYEN THONG (EMA style), **KHAC voi M3!**

### So sanh voi M3:

| | Momentum Truyen Thong (Eq. 16-17) | M3 (Algorithm 1) |
|---|-----------------------------------|------------------|
| **Cong thuc** | $m = \alpha \cdot m - \eta \cdot g$ | $M = M + \beta \cdot g$ |
| **Style** | EMA (Exponential Moving Average) | Accumulation (Cong don) |
| **Decay** | Co ($\alpha < 1$ lam giam m cu) | Khong |
| **Bounded** | Co (m khong tang vo han) | Khong (M tang mai) |

**M3 dung ACCUMULATION** vi sau do co Newton-Schulz normalize, nen magnitude khong quan trong.

---

## 3. Delta Gradient Descent - DGD (Paper Eq. 27-29)

### 3.1 Van de voi Gradient Descent truyen thong

GD truyen thong dung **dot-product objective**:

$$
\min_W \langle W x_t, \nabla_{y_t} \mathcal{L}(W_t; x_t) \rangle
$$

> **Van de:** "the above formulation cause ignoring the dependencies of data samples" - Paper

### 3.2 L2 Regression Objective (Eq. 27)

Thay dot-product bang **L2 regression**:

$$
\boxed{
\min_W \| W x_t - \nabla_{y_t} \mathcal{L}(W_t; x_t) \|_2^2
}
$$

### 3.3 DGD Update Rule (Eq. 28-29)

$$
\boxed{
W_{t+1} = W_t (I - x_t x_t^\top) - \eta_{t+1} \nabla_{W_t} \mathcal{L}(W_t; x_t)
}
$$

Hoac viet day du voi outer product:

$$
W_{t+1} = W_t (I - x_t x_t^\top) - \eta_{t+1} \nabla_{y_t} \mathcal{L}(W_t; x_t) \otimes x_t
$$

### 3.4 Giai thich cac thanh phan

| Thanh phan | Ky hieu | Y nghia |
|------------|---------|---------|
| **Adaptive decay matrix** | $(I - x_t x_t^\top)$ | Projection vuong goc voi $x_t$, "quen" thong tin cu trong huong $x_t$ |
| **Local Surprise Signal** | $\nabla_{y_t} \mathcal{L}$ | Tin hieu loi tai output |
| **Outer product** | $\nabla_{y_t} \mathcal{L} \otimes x_t$ | Gradient cua weight matrix |

### 3.5 So sanh GD vs DGD

| | GD Truyen Thong | DGD |
|---|-----------------|-----|
| **Objective** | Dot-product | L2 Regression |
| **Decay** | Khong | Co $(I - x_t x_t^\top)$ |
| **Dependencies** | Bo qua | Xet den |
| **Update** | $W = W - \eta \nabla_W \mathcal{L}$ | $W = W(I - xx^\top) - \eta \nabla_W \mathcal{L}$ |

---

### 3.6 Code DGD (self_modifying.py) vs Paper DGD

#### Paper DGD (Eq. 28-29):
$$
W_{t+1} = W_t (I - x_t x_t^\top) - \eta \nabla_W \mathcal{L}
$$

#### Code DGD (src/nested_learning/titan/self_modifying.py, line 460-461, 600-603):

```python
# Line 460-461: Compute preconditioner
kk = torch.einsum("bi,bj->bij", k_t, k_t)  # k ⊗ k
precond = alpha_t * eye - eta_t * kk       # α*I - η*(k⊗k)

# Line 600-601: Apply update
if use_rank1_precond:
    W = W @ precond - eta * g              # W = W(α*I - η*k⊗k) - η*g
else:
    W = alpha * W - eta * g                # W = α*W - η*g (standard)
```

#### So sanh chi tiet:

| | Paper DGD (Eq. 28-29) | Code DGD (self_modifying.py) |
|---|----------------------|------------------------------|
| **Decay matrix** | $(I - x_t x_t^\top)$ | $(\alpha I - \eta (k \otimes k))$ |
| **k la gi** | Input $x_t$ | Key $k_t$ (projected input) |
| **Co α** | Khong (α = 1 implicit) | Co (adaptive $\alpha_t$) |
| **η trong decay** | Khong | Co |
| **Cong thuc** | $W(I-xx^\top) - \eta g$ | $W(\alpha I - \eta kk) - \eta g$ |

#### Giai thich khac biet:

**Paper DGD:**
$$
\text{precond} = I - x_t x_t^\top
$$
- Chi co decay (khong co α)
- Decay strength = 1 (fixed)

**Code DGD:**
$$
\text{precond} = \alpha_t \cdot I - \eta_t \cdot (k_t \otimes k_t)
$$
- Co α adaptive (retention/decay control)
- Decay strength = η (learning rate)
- k la key (projected input), khong phai raw input

#### Ket luan:

Code DGD la **BIEN THE MO RONG** cua Paper DGD:
1. Them **α adaptive** de control retention
2. Them **η trong decay** de scale decay strength
3. Dung **key k** thay vi raw input x

---

## 4. Muon Optimizer (Paper Eq. 24)

$$
\boxed{
W_{i+1} = W_i + \sigma(\mathbf{m}_{i+1}(\mathbf{u}_i)), \quad \mathbf{m}_{i+1} = \alpha_{i+1} \mathbf{m}_i - \eta_t \nabla \mathcal{L}^{(2)}(\mathbf{m}_i; \mathbf{u}_i, I)
}
$$

**Trong do:** $\sigma(\cdot) = \text{Newton-Schulz}(\cdot)$

---

## 5. Multi-scale Momentum Muon - M3 (Paper Algorithm 1)

### 5.1 Algorithm 1 - CHINH XAC TU PAPER

```
Algorithm 1: Multi-scale Momentum Muon (M3)

Input:
  - Θ₀: Initial weights
  - L(·): Objective function
  - η > 0: Learning rate
  - T: Newton-Schulz steps
  - β₁, β₂, β₃ ∈ (0,1): Momentum factors
  - α ≥ 0: Balance factor
  - ε > 0: Numerical stability
  - f: Frequency (slow update interval)

Initialize: M₁⁽¹⁾ = 0, M₁⁽²⁾ = 0, V₀ = 0

For lower-frequency iteration k = 0, 1, 2, ...
│
│  // SLOW MOMENTUM (moi f steps)
│  M⁽²⁾_t = M⁽²⁾_{t-1} + β₃ · Σᵢ₌₍ₖ₋₁₎f^{kf} gᵢ
│  O⁽²⁾_t = Newton-Schulz_T(M⁽²⁾_t)
│
│  For t = kf+1, kf+2, ..., (k+1)f
│  │
│  │  // COMPUTE GRADIENT
│  │  g_t = ∇_Θ L(Θ_t)
│  │
│  │  // FAST MOMENTUM (moi step)
│  │  M⁽¹⁾_t = M⁽¹⁾_{t-1} + β₁ · g_t
│  │
│  │  // SECOND MOMENT (moi step)
│  │  V_t = V_{t-1} + β₂ · g_t²
│  │
│  │  // ORTHOGONALIZE
│  │  O⁽¹⁾_t = Newton-Schulz_T(M⁽¹⁾_t)
│  │
│  │  // UPDATE WEIGHTS
│  │  Θ_t = Θ_{t-1} - η · (O⁽¹⁾_t + α·O⁽²⁾_t) / (√V_t + ε)
│  │
│  End For
│
End For
```

### 5.2 Cong Thuc Toan Hoc

$$
\boxed{
\begin{aligned}
M^{(1)}_t &= M^{(1)}_{t-1} + \beta_1 \cdot g_t & \text{(Fast, moi step)} \\
M^{(2)}_t &= M^{(2)}_{t-1} + \beta_3 \cdot \sum_{i=(k-1)f}^{kf} g_i & \text{(Slow, moi } f \text{ steps)} \\
V_t &= V_{t-1} + \beta_2 \cdot g_t^2 & \text{(Second moment)} \\
O^{(1)}_t &= \text{Newton-Schulz}_T(M^{(1)}_t) \\
O^{(2)}_t &= \text{Newton-Schulz}_T(M^{(2)}_t) \\
\Theta_t &= \Theta_{t-1} - \eta \cdot \frac{O^{(1)}_t + \alpha \cdot O^{(2)}_t}{\sqrt{V_t} + \epsilon}
\end{aligned}
}
$$

### 5.3 Dac diem Quan Trong

| Aspect | M3 (Algorithm 1) | Momentum Truyen Thong (Eq. 16-17) |
|--------|------------------|-----------------------------------|
| **Formula** | $M = M + \beta \cdot g$ | $m = \alpha \cdot m - \eta \cdot g$ |
| **Style** | **ACCUMULATION** | EMA (decay) |
| **Decay** | KHONG co decay | Co decay ($\alpha < 1$) |
| **Bounded** | UNBOUNDED (tang vo han) | Bounded |
| **Multi-scale** | Co (fast + slow) | Khong |
| **Second moment** | Co ($V$) | Khong |

> **QUAN TRONG:** M3 dung **ACCUMULATION** (`M = M + β·g`) chu KHONG phai EMA (`m = α·m - η·g`). Day la thiet ke CO CHU DICH cua paper!

---

## 6. Newton-Schulz Iteration

$$
\begin{aligned}
X_0 &= M / \|M\| \\
X_{k+1} &= \frac{1}{2} X_k (3I - X_k^\top X_k)
\end{aligned}
$$

Lap lai T lan. Output: $X^\top X \approx I$ (orthogonal matrix)

---

## 7. Continuum Memory System (Paper Eq. 30-31)

$$
\boxed{
\theta_{i+1}^{(f_\ell)} = \theta_i^{(f_\ell)} -
\begin{cases}
\sum_{t=i-C^{(\ell)}}^{i} \eta_t^{(\ell)} f(\theta_t^{(f_\ell)}; x_t) & \text{if } i \equiv 0 \pmod{C^{(\ell)}} \\
0 & \text{otherwise}
\end{cases}
}
$$

---

## 8. Tong hop Cong Thuc Paper

| Algorithm | Style | Formula |
|-----------|-------|---------|
| GD | - | $W = W - \eta \nabla \mathcal{L}$ |
| Momentum | EMA | $m = \alpha m - \eta \nabla \mathcal{L}$, $W = W + m$ |
| DGD | Decay | $W = W(I - xx^\top) - \eta \nabla \mathcal{L}$ |
| Muon | EMA + NS | $W = W + \text{NS}(m)$ |
| **M3** | **ACCUMULATION** | $M = M + \beta g$, $W = W - \eta \cdot \text{NS}(M)/\sqrt{V}$ |

---

# PHAN 2: CODE IMPLEMENTATION

---

## 1. M3 Code (src/nested_learning/optim/m3.py)

### 1.1 Code Thuc Te

```python
# Line 105: Fast momentum (ACCUMULATION)
m1.add_(grad, alpha=beta1)
# => M1 = M1 + beta1 * grad  ✅ DUNG Algorithm 1

# Line 106: Second moment (ACCUMULATION)
v.addcmul_(grad, grad, value=beta2)
# => V = V + beta2 * grad²  ✅ DUNG Algorithm 1

# Line 107: Buffer for slow momentum
slow_buffer.add_(grad)
# => buffer = buffer + grad  ✅ DUNG Algorithm 1

# Line 109: Orthogonalize fast momentum
o1 = _orthogonalize(m1, steps=ns_steps, eps=eps)
# => O1 = NS(M1)  ✅ DUNG Algorithm 1

# Line 111-113: Update weights
denom = v.sqrt().add_(eps)
update = (o1 + alpha * o2) / denom
p.add_(update, alpha=-lr)
# => Θ = Θ - η * (O1 + α*O2) / √V  ✅ DUNG Algorithm 1

# Line 115-120: Slow momentum (every slow_chunk steps)
if step % slow_chunk == 0:
    m2.add_(slow_buffer, alpha=beta3)
    # => M2 = M2 + beta3 * buffer  ✅ DUNG Algorithm 1
    slow_buffer.zero_()
    o2 = _orthogonalize(m2)
```

---

## 2. So sanh Paper vs Code

| Component | Paper Algorithm 1 | Code m3.py | Ket qua |
|-----------|-------------------|------------|---------|
| Fast momentum | $M^{(1)} = M^{(1)} + \beta_1 g$ | `m1 = m1 + beta1*grad` | ✅ **KHOP** |
| Second moment | $V = V + \beta_2 g^2$ | `v = v + beta2*grad²` | ✅ **KHOP** |
| Slow momentum | $M^{(2)} = M^{(2)} + \beta_3 \sum g$ | `m2 = m2 + beta3*buffer` | ✅ **KHOP** |
| Newton-Schulz | $O = \text{NS}_T(M)$ | `o = _orthogonalize(m)` | ✅ **KHOP** |
| Weight update | $\Theta = \Theta - \eta \frac{O^{(1)} + \alpha O^{(2)}}{\sqrt{V} + \epsilon}$ | `p = p - lr*(o1+α*o2)/√v` | ✅ **KHOP** |
| Frequency | Every $f$ steps | Every `slow_chunk` steps | ✅ **KHOP** |

---

## 3. KET LUAN

### ✅ CODE DUNG 100% THEO PAPER ALGORITHM 1

Truoc do toi da **NHAM** khi so sanh M3 voi Momentum truyen thong (Eq. 16-17).

**M3 (Algorithm 1)** va **Momentum (Eq. 16-17)** la **HAI THUAT TOAN KHAC NHAU:**

| | Momentum (Eq. 16-17) | M3 (Algorithm 1) |
|-|----------------------|------------------|
| Style | EMA: $m = \alpha m - \eta g$ | Accumulation: $M = M + \beta g$ |
| Decay | Co | Khong |
| Multi-scale | Khong | Co |
| Second moment | Khong | Co |
| Newton-Schulz | Khong (trong Eq. 16-17) | Co |

**M3 la thiet ke MOI** - ket hop:
- Accumulation momentum (khong decay)
- Multi-scale (fast + slow)
- Newton-Schulz orthogonalization
- Adam-style second moment

---

## 4. Hyperparameters Mac Dinh

| Parameter | Paper | Code | Description |
|-----------|-------|------|-------------|
| $\eta$ (lr) | - | 1e-3 | Learning rate |
| $\beta_1$ | (0,1) | 0.9 | Fast momentum factor |
| $\beta_2$ | (0,1) | 0.999 | Second moment factor |
| $\beta_3$ | (0,1) | 0.9 | Slow momentum factor |
| $\alpha$ | ≥ 0 | 1.0 | Balance fast/slow |
| $\epsilon$ | > 0 | 1e-8 | Numerical stability |
| $T$ | - | 3 | Newton-Schulz steps |
| $f$ | - | 100 | Slow update frequency |

---

*Cap nhat: 2026-04-04*
