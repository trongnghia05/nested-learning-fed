# Đánh giá Critical: Paper vs Code Implementation

> Tài liệu này SO SÁNH và ĐÁNH GIÁ code implementation với paper.
> Không mặc định code là đúng - cần verify từng concept.

---

## 1. Khái niệm "Frequency" (Tần số)

### Paper định nghĩa (Section 2.2, Definition 2):

> "For any component A, we define its frequency, denoted as f_A, as its **number of updates per unit of time**"
>
> "We let **one update step over one data point** to be the unit of time"

**Ý nghĩa paper:**
- Frequency = Số lần update / 1 đơn vị thời gian
- 1 đơn vị thời gian = 1 step trên 1 data point
- f_A = 1 nghĩa là update MỖI step
- f_A = 0.01 nghĩa là update mỗi 100 steps

### Code implementation (levels.py):

```python
@dataclass(frozen=True)
class LevelSpec:
    name: str
    update_period: int  # <-- Dùng PERIOD, không phải frequency
    warmup_steps: int = 0
    jitter: int = 0
```

### ĐÁNH GIÁ:

```
┌─────────────────────────────────────────────────────────────┐
│  SAI KHÁC VỀ CÁCH BIỂU DIỄN:                                │
│                                                              │
│  Paper: frequency f = số lần update / unit time             │
│         f = 1 → update mỗi step                             │
│         f = 0.1 → update mỗi 10 steps                       │
│                                                              │
│  Code: update_period = số steps giữa các updates            │
│        period = 1 → update mỗi step                         │
│        period = 10 → update mỗi 10 steps                    │
│                                                              │
│  QUAN HỆ: frequency = 1 / period                            │
│                                                              │
│  VERDICT: ✓ ĐÚNG về ý nghĩa, chỉ khác cách biểu diễn       │
│           Period dễ dùng hơn trong code (integer)           │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Khái niệm "Level" và Ordering

### Paper định nghĩa (Section 2.2):

> "A ≻ B (A faster than B) if: (1) f_A > f_B, or (2) f_A = f_B but the computation of B's state at time t requires the computation of A's state at time t"
>
> "**The higher the level is, the lower its frequency**"

**Ý nghĩa paper:**
- Level CAO = Frequency THẤP = Update ÍT
- Level THẤP = Frequency CAO = Update NHIỀU

### Code implementation (levels.py):

```python
def levels_in_frequency_order(self) -> List[LevelSpec]:
    return sorted(self._specs.values(), key=lambda spec: spec.update_period)
```

### ĐÁNH GIÁ:

```
┌─────────────────────────────────────────────────────────────┐
│  PHÂN TÍCH:                                                  │
│                                                              │
│  Code sort theo update_period TĂNG DẦN:                     │
│    period=1 (first) → period=10 → period=100 (last)         │
│                                                              │
│  Nghĩa là:                                                  │
│    - Đầu tiên: period nhỏ = frequency CAO = update NHIỀU    │
│    - Cuối cùng: period lớn = frequency THẤP = update ÍT     │
│                                                              │
│  Theo paper: "higher level = lower frequency"               │
│    - Level thấp = frequency cao (đầu list)                  │
│    - Level cao = frequency thấp (cuối list)                 │
│                                                              │
│  VERDICT: ✓ ĐÚNG - Code sort đúng thứ tự theo paper        │
│                                                              │
│  NHƯNG: Code KHÔNG đánh số level rõ ràng!                   │
│         Chỉ có tên (name), không có level number            │
│         Đây là điểm có thể gây confusion                    │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. M3 Optimizer: So sánh Algorithm 1

### Paper Algorithm 1 (tôi reconstruct từ paper):

```
Input: θ (params), η (lr), β1, β2, β3, α, K (slow_chunk), ns_steps
Initialize: m1=0, m2=0, v=0, buffer=0

For t = 1, 2, ...:
    g_t = ∇L(θ_t)

    # Fast momentum
    m1_t = β1 * m1_{t-1} + g_t          # (1)

    # Second moment
    v_t = β2 * v_{t-1} + g_t²           # (2)

    # Accumulate for slow
    buffer_t = buffer_{t-1} + g_t       # (3)

    # Orthogonalize fast
    o1_t = Newton_Schulz(m1_t)          # (4)

    # Update params
    θ_t = θ_{t-1} - η * (o1_t + α*o2) / √(v_t + ε)  # (5)

    # Every K steps: update slow momentum
    if t mod K == 0:
        m2_t = β3 * m2_{t-1} + buffer_t  # (6)
        buffer_t = 0                      # (7)
        o2 = Newton_Schulz(m2_t)          # (8)
```

### Code implementation (m3.py):

```python
# Line 105-113
m1.add_(grad, alpha=beta1)           # m1 += beta1 * grad
v.addcmul_(grad, grad, value=beta2)  # v += beta2 * grad²
slow_buffer.add_(grad)                # buffer += grad

o1 = _orthogonalize(m1, steps=ns_steps, eps=eps)
o2 = state["o2"]
denom = v.sqrt().add_(eps)
update = (o1 + alpha * o2) / denom
p.add_(update, alpha=-lr)             # p -= lr * update

# Line 115-120
if slow_chunk > 0 and state["step"] % slow_chunk == 0:
    m2.add_(slow_buffer, alpha=beta3)
    slow_buffer.zero_()
    state["o2"] = _orthogonalize(m2, steps=ns_steps, eps=eps)
```

### ĐÁNH GIÁ CHI TIẾT:

```
┌─────────────────────────────────────────────────────────────┐
│  (1) Fast momentum: m1 = β1 * m1 + grad                     │
│                                                              │
│  Paper: m1_t = β1 * m1_{t-1} + g_t                          │
│  Code:  m1.add_(grad, alpha=beta1)  # m1 += beta1 * grad    │
│                                                              │
│  ⚠️ KHÁC BIỆT!                                              │
│  Paper: m1 = β1*m1 + g (momentum DECAY rồi ADD)             │
│  Code:  m1 = m1 + β1*g (ADD với coefficient)                │
│                                                              │
│  Đây là 2 công thức KHÁC NHAU!                              │
│  - Paper: EMA style (exponential moving average)            │
│  - Code: Accumulation style                                 │
│                                                              │
│  VERDICT: ⚠️ CÓ THỂ SAI hoặc là BIẾN THỂ có chủ đích       │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  (2) Second moment: v = β2 * v + grad²                      │
│                                                              │
│  Paper: v_t = β2 * v_{t-1} + g_t²                           │
│  Code:  v.addcmul_(grad, grad, value=beta2)                 │
│         # v += beta2 * grad * grad                          │
│                                                              │
│  ⚠️ CÙNG VẤN ĐỀ!                                            │
│  Paper: v = β2*v + g² (EMA)                                 │
│  Code:  v = v + β2*g² (Accumulation)                        │
│                                                              │
│  VERDICT: ⚠️ KHÁC với paper                                 │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  (5) Update rule                                            │
│                                                              │
│  Paper: θ = θ - η * (o1 + α*o2) / √(v + ε)                  │
│  Code:  update = (o1 + alpha * o2) / denom                  │
│         p.add_(update, alpha=-lr)                           │
│                                                              │
│  VERDICT: ✓ ĐÚNG                                            │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  (6-8) Slow momentum update                                 │
│                                                              │
│  Paper: m2 = β3 * m2 + buffer                               │
│  Code:  m2.add_(slow_buffer, alpha=beta3)                   │
│         # m2 += beta3 * buffer                              │
│                                                              │
│  ⚠️ CÙNG VẤN ĐỀ!                                            │
│  Paper: EMA style                                           │
│  Code:  Accumulation style                                  │
│                                                              │
│  VERDICT: ⚠️ KHÁC với paper                                 │
└─────────────────────────────────────────────────────────────┘
```

### KẾT LUẬN VỀ M3:

```
┌─────────────────────────────────────────────────────────────┐
│  PHÁT HIỆN QUAN TRỌNG:                                      │
│                                                              │
│  Code dùng ACCUMULATION thay vì EMA:                        │
│                                                              │
│  EMA (paper):         m = β*m + g                           │
│  Accumulation (code): m = m + β*g                           │
│                                                              │
│  ĐÂY LÀ 2 DYNAMICS RẤT KHÁC:                                │
│                                                              │
│  EMA:                                                       │
│    - m converge về weighted average của g                   │
│    - Bounded (không explode)                                │
│    - β quyết định "memory length"                           │
│                                                              │
│  Accumulation:                                              │
│    - m TĂNG LIÊN TỤC (unbounded)                            │
│    - Cần normalize hoặc sẽ explode                          │
│    - β chỉ là scaling factor                                │
│                                                              │
│  CÂU HỎI: Đây là bug hay intentional variant?              │
│                                                              │
│  GIẢ THUYẾT: Có thể Newton-Schulz orthogonalization         │
│  đã normalize nên không bị explode?                         │
│  Cần verify thêm với experiments.                           │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Newton-Schulz Orthogonalization

### Paper formula:

```
X = M / ||M||
For i = 1..steps:
    X = 0.5 * X @ (3I - X^T @ X)
```

### Code (m3.py, lines 8-20):

```python
def _newton_schulz(matrix, steps, eps=1e-6):
    x = matrix
    norm = torch.linalg.norm(x)
    x = x / (norm + eps)              # Normalize
    eye = torch.eye(n, device=device, dtype=dtype)
    for _ in range(steps):
        x = 0.5 * x @ (3.0 * eye - x.T @ x)
    return x
```

### ĐÁNH GIÁ:

```
┌─────────────────────────────────────────────────────────────┐
│  VERDICT: ✓ ĐÚNG HOÀN TOÀN với paper                       │
│                                                              │
│  Công thức khớp 100%:                                       │
│  1. Normalize: X = M / ||M||                                │
│  2. Iterate: X = 0.5 * X @ (3I - X^T @ X)                   │
│                                                              │
│  Code thêm eps cho numerical stability - GOOD PRACTICE      │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. TitanMemory: Surprise và Update

### Paper TITAN (Section 3.1):

```
Surprise = ∇ℓ(M; x)  (gradient của loss)

Update rule:
M_t = M_{t-1} - θ_t * ∇ℓ(M_{t-1}; x_t)

Loss:
ℓ(M; x) = ||M(k) - v||²
```

### Code (titan/memory.py):

```python
def surprise(self, residual):
    return residual.norm(dim=-1, keepdim=True)  # ||residual||

def update(self, *, key, value, lr=1e-3):
    prediction = self.forward(key)
    loss = torch.mean((prediction - value) ** 2)  # ||M(k) - v||²

    grads = torch.autograd.grad(loss, self.net.parameters())
    for param, grad in zip(self.net.parameters(), grads):
        param.add_(grad, alpha=-lr)  # param -= lr * grad
```

### ĐÁNH GIÁ:

```
┌─────────────────────────────────────────────────────────────┐
│  (1) Surprise definition:                                   │
│                                                              │
│  Paper: Surprise = ∇ℓ(M; x) = gradient                      │
│  Code:  surprise = ||residual|| = ||prediction - target||   │
│                                                              │
│  ⚠️ KHÁC BIỆT!                                              │
│  Paper: Surprise là GRADIENT vector                         │
│  Code:  Surprise là NORM của residual (scalar)              │
│                                                              │
│  NHƯNG: Cả hai đều đo "model sai bao nhiêu"                │
│         Chỉ khác biểu diễn (vector vs scalar)               │
│                                                              │
│  VERDICT: ⚠️ BIẾN THỂ - đơn giản hóa cho practical use     │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  (2) Loss function:                                         │
│                                                              │
│  Paper: ℓ(M; x) = ||M(k) - v||²                             │
│  Code:  loss = torch.mean((prediction - value) ** 2)        │
│                                                              │
│  VERDICT: ✓ ĐÚNG (squared L2 loss)                          │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  (3) Update rule:                                           │
│                                                              │
│  Paper: M_t = M_{t-1} - θ * ∇ℓ                              │
│  Code:  param.add_(grad, alpha=-lr)  # param -= lr * grad   │
│                                                              │
│  VERDICT: ✓ ĐÚNG (standard gradient descent)                │
└─────────────────────────────────────────────────────────────┘
```

---

## 6. TITAN với Momentum (Past Surprise)

### Paper TITAN (Eq. 9):

```
M_t = M_{t-1} + S_t
S_t = η_t * S_{t-1} - θ_t * ∇ℓ(M_{t-1}; x_t)
      ^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^
      Past Surprise    Momentary Surprise
```

### Code implementation:

```
KHÔNG TÌM THẤY trong titan/memory.py!

TitanMemory class chỉ có simple gradient descent:
    param -= lr * grad

KHÔNG có momentum term S_t!
```

### ĐÁNH GIÁ:

```
┌─────────────────────────────────────────────────────────────┐
│  ⚠️ THIẾU FEATURE!                                          │
│                                                              │
│  Paper TITAN có:                                            │
│  - Momentary surprise: ∇ℓ                                   │
│  - Past surprise: S_{t-1} (momentum)                        │
│  - Data-dependent η_t, θ_t                                  │
│                                                              │
│  Code TitanMemory CHỈ có:                                   │
│  - Simple gradient descent                                  │
│  - Fixed learning rate                                      │
│  - KHÔNG có momentum                                        │
│                                                              │
│  VERDICT: ⚠️ Code là SIMPLIFIED VERSION của paper           │
│           Thiếu past surprise và adaptive rates             │
│                                                              │
│  CÓ THỂ: Full version nằm ở SelfModifyingTitans?           │
└─────────────────────────────────────────────────────────────┘
```

---

## 7. Self-Modifying Titans: DGD Preconditioner

### Paper (Eq. 90):

```
P = α*I - η*(k ⊗ k)
w2 = w2 @ P - η * g2
w1 = α * w1 - η * g1
```

### Code (titan/self_modifying.py, lines 456-475, 589-614):

```python
# Line 460-461: Tính preconditioner
kk = torch.einsum("bi,bj->bij", k_t, k_t)  # k ⊗ k
precond = alpha_t[:, None, None] * eye - eta_t[:, None, None] * kk
# precond = α*I - η*(k⊗k) ✓ ĐÚNG

# Line 600-604: Apply update
if self.config.use_rank1_precond:
    fast.w2 = torch.matmul(fast.w2, precond) - eta_t[:, None, None] * g2
    # w2 = w2 @ P - η*g2 ✓ ĐÚNG
else:
    fast.w2 = alpha_t[:, None, None] * fast.w2 - eta_t[:, None, None] * g2
    # w2 = α*w2 - η*g2 (simplified version)

fast.w1 = alpha_t[:, None, None] * fast.w1 - eta_t[:, None, None] * g1
# w1 = α*w1 - η*g1 ✓ ĐÚNG
```

### ĐÁNH GIÁ:

```
┌─────────────────────────────────────────────────────────────┐
│  DGD PRECONDITIONER:                                        │
│                                                              │
│  Paper Eq. 90:                                              │
│    P = α*I - η*(k ⊗ k)                                      │
│    w2 = w2 @ P - η * g2                                     │
│    w1 = α * w1 - η * g1                                     │
│                                                              │
│  Code (use_rank1_precond=True):                             │
│    kk = einsum("bi,bj->bij", k, k)  # k ⊗ k                │
│    precond = α*I - η*kk             # P                     │
│    w2 = w2 @ precond - η*g2         # w2 update             │
│    w1 = α*w1 - η*g1                 # w1 update             │
│                                                              │
│  VERDICT: ✓ ĐÚNG HOÀN TOÀN với paper khi use_rank1_precond │
│                                                              │
│  LƯU Ý: Code có 2 modes:                                    │
│    - use_rank1_precond=True: Dùng full DGD (paper Eq. 90)   │
│    - use_rank1_precond=False: Simplified (chỉ α*w - η*g)    │
└─────────────────────────────────────────────────────────────┘
```

### Adaptive η và α:

```
┌─────────────────────────────────────────────────────────────┐
│  Paper TITAN: η_t và α_t là DATA-DEPENDENT                  │
│                                                              │
│  Code implementation (lines 296-302):                       │
│                                                              │
│  eta_chunk = self._memory_forward(x_chunk, state.eta)       │
│  eta_chunk = F.softplus(eta_chunk) * self.config.eta_scale  │
│                                                              │
│  alpha_chunk = self._memory_forward(x_chunk, state.alpha)   │
│  alpha_chunk = torch.sigmoid(alpha_chunk)                   │
│                                                              │
│  PHÂN TÍCH:                                                 │
│  - η = softplus(M_eta(x)) * scale                           │
│    → softplus đảm bảo η > 0 (learning rate dương)           │
│    → Learned từ input x (data-dependent) ✓                  │
│                                                              │
│  - α = sigmoid(M_alpha(x))                                  │
│    → sigmoid đảm bảo α ∈ (0, 1) (decay factor hợp lệ)       │
│    → Learned từ input x (data-dependent) ✓                  │
│                                                              │
│  VERDICT: ✓ ĐÚNG - Code implement đầy đủ adaptive rates     │
└─────────────────────────────────────────────────────────────┘
```

### So sánh với TitanMemory (simplified):

```
┌─────────────────────────────────────────────────────────────┐
│  SelfModifyingTitans (FULL):                                │
│  ├── 6 memories: M_k, M_v, M_q, M_η, M_α, M_memory         │
│  ├── DGD preconditioner: P = α*I - η*(k⊗k)                 │
│  ├── Adaptive η từ M_eta network                           │
│  ├── Adaptive α từ M_alpha network                         │
│  ├── Optional momentum trên gradients                       │
│  └── Per-token gradient với vmap                           │
│                                                              │
│  TitanMemory (SIMPLIFIED):                                  │
│  ├── 1 MLP network                                          │
│  ├── Simple gradient descent (không DGD)                    │
│  ├── Fixed learning rate                                    │
│  └── Không có momentum                                      │
│                                                              │
│  => TitanMemory là HEAVILY SIMPLIFIED version               │
│  => SelfModifyingTitans mới là full paper implementation    │
└─────────────────────────────────────────────────────────────┘
```

---

## 8. CMS: Multi-level Implementation

### Paper (Section 3, Eq. 31):

```
y_t = MLP^(f_k)(MLP^(f_{k-1})(...MLP^(f_1)(x_t)))

Trong đó:
- Mỗi MLP^(f_ℓ) update mỗi C^(ℓ) steps
- C^(ℓ) = max_f C / f_ℓ
```

### Code (cms.py):

```python
class CMS(nn.Module):
    def forward(self, x, *, return_intermediates=False):
        current = x
        for spec in self.level_specs:
            block = self.blocks[spec.name]
            current = block(current)  # Chain sequentially
        return current
```

### ĐÁNH GIÁ:

```
┌─────────────────────────────────────────────────────────────┐
│  PHÂN TÍCH:                                                  │
│                                                              │
│  Paper định nghĩa:                                          │
│  - Mỗi level có FREQUENCY RIÊNG                             │
│  - Level update ở các thời điểm KHÁC NHAU                   │
│                                                              │
│  Code forward():                                             │
│  - Chỉ chain MLPs sequentially                              │
│  - KHÔNG thấy logic update theo frequency                   │
│                                                              │
│  ⚠️ THIẾU: Logic multi-frequency update trong CMS.forward()│
│                                                              │
│  TUY NHIÊN: Logic này có thể nằm ở:                         │
│  - LevelClock (levels.py): should_update() method           │
│  - HOPEBlock: Gọi CMS với scheduling                        │
│                                                              │
│  => CMS class chỉ là container                              │
│  => Scheduling logic ở nơi khác (caller)                    │
│                                                              │
│  VERDICT: ⚠️ CMS forward() thiếu frequency logic            │
│           Cần kết hợp với LevelClock khi sử dụng            │
└─────────────────────────────────────────────────────────────┘
```

---

## 9. Vấn đề M3 Accumulation vs EMA - Phân tích sâu

### Đặt vấn đề:

```
Paper công thức:
    m = β * m + g          (EMA)

Code implementation:
    m.add_(g, alpha=beta)  (Accumulation: m = m + β*g)
```

### Hệ quả của sự khác biệt:

```
EMA (paper):
    m_t = β*m_{t-1} + g_t
    m_t = β^t*m_0 + Σ β^(t-i)*g_i

    → Bounded: |m_t| ≤ max|g|/(1-β) khi β<1
    → Converge về weighted average của g
    → Gradients cũ decay exponentially

Accumulation (code):
    m_t = m_{t-1} + β*g_t
    m_t = m_0 + β*Σg_i

    → Unbounded: m_t tăng vô hạn theo t
    → Tổng tích lũy của gradients
    → Gradients cũ KHÔNG decay

⚠️ ĐÂY LÀ 2 DYNAMICS HOÀN TOÀN KHÁC!
```

### Tại sao code có thể vẫn hoạt động?

```
┌─────────────────────────────────────────────────────────────┐
│  GIẢ THUYẾT 1: Newton-Schulz normalize                      │
│                                                              │
│  Code:                                                       │
│    m1.add_(grad, alpha=beta1)     # Accumulation            │
│    o1 = _orthogonalize(m1, ...)   # Newton-Schulz           │
│                                                              │
│  Newton-Schulz đầu tiên normalize: X = M / ||M||            │
│  => Dù m1 unbounded, o1 vẫn bounded (unit norm)             │
│  => Có thể đây là cách họ "fix" accumulation                │
│                                                              │
│  NHƯNG: Vẫn mất thông tin về scale của momentum             │
│         Gradient lớn vs nhỏ đều normalize thành unit        │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  GIẢ THUYẾT 2: Intentional design choice                    │
│                                                              │
│  Accumulation + Newton-Schulz có thể là variant có chủ đích:│
│  - Tích lũy tất cả gradients (không quên)                   │
│  - Orthogonalize để balance directions                      │
│  - Divide by sqrt(v) để adaptive scaling                    │
│                                                              │
│  Có thể họ thử và thấy hoạt động tốt hơn EMA?              │
│  => Cần xem có paper/docs giải thích không                  │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  GIẢ THUYẾT 3: Bug/oversight                                │
│                                                              │
│  Có thể đây là bug:                                         │
│  - add_(x, alpha=β) thêm β*x, không phải β*m + x            │
│  - Developer nhầm API của PyTorch?                          │
│                                                              │
│  Cách fix nếu muốn EMA:                                     │
│    m1.mul_(beta1).add_(grad)   # m = β*m + g                │
│                                                              │
│  => Cần test cả 2 versions để so sánh                       │
└─────────────────────────────────────────────────────────────┘
```

### Kết luận về M3:

```
┌─────────────────────────────────────────────────────────────┐
│  KHÔNG THỂ KẾT LUẬN CHẮC CHẮN:                              │
│                                                              │
│  1. Có thể là intentional variant (Newton-Schulz normalize) │
│  2. Có thể là bug/oversight                                 │
│  3. Có thể có documentation giải thích ở đâu đó             │
│                                                              │
│  KHUYẾN NGHỊ CHO FL IMPLEMENTATION:                         │
│  - Implement ĐÚNG theo paper (EMA)                          │
│  - Sau đó có thể test variant accumulation                  │
│  - So sánh performance                                       │
│  - Không copy code mù quáng                                  │
└─────────────────────────────────────────────────────────────┘
```

---

## 10. Tổng kết Đánh giá

### Điểm ĐÚNG với paper:

| Component | Verdict | Notes |
|-----------|---------|-------|
| Newton-Schulz | ✓ ĐÚNG | Công thức khớp 100% |
| Level ordering | ✓ ĐÚNG | Higher level = lower frequency |
| L2 loss function | ✓ ĐÚNG | ||M(k) - v||² |
| Basic GD update | ✓ ĐÚNG | param -= lr * grad |

### Điểm KHÁC với paper:

| Component | Issue | Severity |
|-----------|-------|----------|
| M3 momentum | Accumulation vs EMA | ⚠️ HIGH |
| M3 second moment | Accumulation vs EMA | ⚠️ HIGH |
| Surprise definition | Norm vs Gradient | ⚠️ MEDIUM |
| TITAN momentum | Không có past surprise | ⚠️ MEDIUM |
| TITAN adaptive lr | Không có η_t, θ_t | ⚠️ MEDIUM |

### Điểm CẦN VERIFY THÊM:

```
1. M3: Accumulation có intentional không?
   - Có thể Newton-Schulz normalize nên OK?
   - Cần test xem có explode không

2. TitanMemory: Simplified version hay bug?
   - Full TITAN có thể ở SelfModifyingTitans?
   - Cần đọc thêm code

3. DGD preconditioner: Cần verify trong self_modifying.py
```

---

## 9. Implications cho FL Implementation

```
┌─────────────────────────────────────────────────────────────┐
│  KHI IMPLEMENT FED-DGD và FED-M3:                           │
│                                                              │
│  1. KHÔNG copy code trực tiếp                               │
│     - Code có thể là simplified/variant version             │
│     - Cần đối chiếu với paper trước                         │
│                                                              │
│  2. Với M3: Quyết định dùng EMA hay Accumulation            │
│     - EMA (paper): Bounded, stable                          │
│     - Accumulation (code): Có thể cần normalize             │
│                                                              │
│  3. Với TITAN: Có thể cần implement full version            │
│     - Thêm past surprise (momentum)                         │
│     - Thêm adaptive learning rates                          │
│                                                              │
│  4. Luôn verify với paper definitions                       │
│     - Không assume code là ground truth                     │
│     - Paper là reference chính                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 11. DeepMomentum Optimizer (deep.py)

### Paper reference:

Nested Learning paper đề cập đến "deep optimizers" như là các component update với frequencies khác nhau.

### Code implementation:

```python
class DeepMomentum(nn.Module):
    def forward(self, grad, *, context=None, param_key=None):
        # ...
        update = grad
        if self.variant in {"preconditioned", "muon"}:
            update = self._precondition(grad, state)
        if self.variant in {"dmgd", "muon"}:
            update = self.nonlinearity(update)  # Tanh

        # EMA momentum
        state.grad_avg.mul_(self.beta).add_(update, alpha=1 - self.beta)
        return state.grad_avg

    def _precondition(self, grad, state):
        # Adam-style second moment
        state.sq_avg.mul_(self.beta2).addcmul_(grad, grad, value=1 - self.beta2)
        denom = state.sq_avg.sqrt().add_(self.eps)
        return grad / denom
```

### ĐÁNH GIÁ:

```
┌─────────────────────────────────────────────────────────────┐
│  PHÁT HIỆN QUAN TRỌNG:                                      │
│                                                              │
│  DeepMomentum DÙNG EMA ĐÚNG:                                │
│                                                              │
│  grad_avg = β * grad_avg + (1-β) * update                   │
│                                                              │
│  Code: state.grad_avg.mul_(self.beta).add_(update, alpha=1-self.beta)
│        grad_avg = β * grad_avg + (1-β) * update ✓ EMA ĐÚNG │
│                                                              │
│  Second moment cũng đúng:                                   │
│  sq_avg = β2 * sq_avg + (1-β2) * grad²  ✓ Adam-style       │
│                                                              │
│  VERDICT: ✓ DeepMomentum implement EMA ĐÚNG                 │
│                                                              │
│  ⚠️ NHƯNG: M3 optimizer KHÔNG dùng EMA!                     │
│            M3 dùng accumulation: m += β*g                   │
│            Hai optimizer trong cùng repo có công thức KHÁC! │
└─────────────────────────────────────────────────────────────┘
```

### Comparison Table:

```
┌─────────────────┬─────────────────────────┬─────────────────────┐
│  Optimizer      │  Momentum Formula       │  Verdict            │
├─────────────────┼─────────────────────────┼─────────────────────┤
│  DeepMomentum   │  m = β*m + (1-β)*g      │  ✓ EMA (Adam-style) │
│  M3             │  m = m + β*g            │  ⚠️ Accumulation    │
│  Paper (SGD+M)  │  m = β*m + g            │  Standard momentum  │
└─────────────────┴─────────────────────────┴─────────────────────┘

Note: 3 cách làm KHÁC NHAU trong cùng 1 repo!
```

---

## 12. HOPE Blocks Implementation

### Paper (Eqs. 94-97):

```
HOPE = Self-Modifying Titans + CMS (multi-level MLPs)

Cụ thể:
y = CMS(SelfModifyingTitans(x))

Với:
- SelfModifyingTitans: Fast memory với DGD preconditioner
- CMS: Slow components với different frequencies
```

### Code (hope/block.py):

```python
# 3 variants trong code:

class HOPEBlock:
    # Attention + TitanMemory (simple) + CMS
    self.attn = SelfAttention(...)
    self.titan_memory = TitanMemory(...)  # Simple version!
    self.cms = CMS(...)

class HOPEAttentionBlock:
    # Attention + CMS (không có Titan)
    self.attn = SelfAttention(...)
    self.cms = CMS(...)

class HOPESelfModBlock:
    # SelfModifyingTitans (full) + CMS
    self.selfmod = SelfModifyingTitans(...)  # Full version!
    self.cms = CMS(...)
```

### ĐÁNH GIÁ:

```
┌─────────────────────────────────────────────────────────────┐
│  HOPE VARIANTS ANALYSIS:                                    │
│                                                              │
│  Paper định nghĩa: HOPE = Self-Modifying Titans + CMS       │
│                                                              │
│  Code có 3 variants:                                        │
│                                                              │
│  1. HOPEBlock:                                              │
│     - Dùng TitanMemory (simplified)                         │
│     - ⚠️ KHÔNG đúng paper vì TitanMemory thiếu features    │
│                                                              │
│  2. HOPEAttentionBlock:                                     │
│     - Không có Titan memory                                 │
│     - ⚠️ Đây là VARIANT, không phải HOPE chính             │
│                                                              │
│  3. HOPESelfModBlock:                                       │
│     - Dùng SelfModifyingTitans (full)                       │
│     - ✓ ĐÚNG với paper                                      │
│                                                              │
│  VERDICT:                                                   │
│  - HOPESelfModBlock = paper-compliant HOPE                  │
│  - HOPEBlock, HOPEAttentionBlock = simplified variants      │
│                                                              │
│  CHÚ Ý: Phải dùng HOPESelfModBlock cho FL implementation   │
│         nếu muốn đúng với paper                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 13. Frequency Scheduling: LevelClock & LevelManager

### Paper định nghĩa:

```
Frequency f_A = số updates / unit time
Component A update khi: t mod (1/f_A) == 0
```

### Code implementation (levels.py, optim/manager.py):

```python
# LevelSpec định nghĩa period
@dataclass(frozen=True)
class LevelSpec:
    name: str
    update_period: int  # = 1/frequency
    warmup_steps: int = 0
    jitter: int = 0

# LevelClock check should_update
class LevelClock:
    def should_update(self, level: str) -> bool:
        spec = self._specs[level]
        period = int(spec.update_period)
        if period <= 0:
            return True
        return (self._global_step - warmup) % period == 0
```

### ĐÁNH GIÁ:

```
┌─────────────────────────────────────────────────────────────┐
│  FREQUENCY SCHEDULING:                                       │
│                                                              │
│  Paper: Update khi t mod (1/f) == 0                         │
│  Code:  Update khi (step - warmup) mod period == 0          │
│                                                              │
│  Với period = 1/frequency, hai công thức TƯƠNG ĐƯƠNG       │
│                                                              │
│  THÊM FEATURES trong code:                                  │
│  - warmup_steps: Không update trong warmup phase            │
│  - jitter: Thêm randomness vào timing (không trong paper)   │
│                                                              │
│  VERDICT: ✓ ĐÚNG logic cơ bản                               │
│           + Có thêm features (warmup, jitter)               │
│                                                              │
│  CHÚ Ý: jitter KHÔNG có trong paper                         │
│         Có thể là engineering choice để training stability  │
└─────────────────────────────────────────────────────────────┘
```

---

## 14. Teach Signal và Loss Function

### Paper (Eq. 95-96):

```
Loss = ||h(x) - δ||²

Với:
- h(x) = output của module
- δ = teach signal (target)
```

### Code (hope/block.py):

```python
def _chunk_loss(prediction, delta_target, mask_f, *, reduction, ...):
    target = prediction.detach() - delta_target
    diff_sq = (prediction - target).pow(2)  # L2 loss
    masked = diff_sq * mask_f
    return masked.sum() / mask_f.sum()
```

### ĐÁNH GIÁ:

```
┌─────────────────────────────────────────────────────────────┐
│  TEACH SIGNAL:                                              │
│                                                              │
│  Paper: Loss = ||h(x) - δ||²                                │
│                                                              │
│  Code: target = prediction.detach() - delta_target          │
│        loss = ||prediction - target||²                      │
│             = ||prediction - (prediction - δ)||²            │
│             = ||δ||²  ← Chờ đã, điều này có vẻ sai!        │
│                                                              │
│  ⚠️ PHÂN TÍCH SÂU:                                          │
│                                                              │
│  Công thức code có vẻ phức tạp:                            │
│    target = pred.detach() - delta_target                   │
│    loss = (pred - target)² = (pred - pred.detach() + δ)²   │
│         = (δ + (pred - pred.detach()))²                    │
│                                                              │
│  Khi pred requires_grad:                                    │
│    ∂loss/∂pred = 2 * (pred - target) = 2 * (pred - pred + δ)
│                ≈ 2 * δ (gradient direction)                │
│                                                              │
│  VERDICT: ⚠️ BIẾN THỂ - gradient-like loss                  │
│           Không đơn giản là ||h(x) - δ||²                   │
│           Cần hiểu rõ hơn intent                            │
└─────────────────────────────────────────────────────────────┘
```

---

## 15. Tổng kết Đầy đủ

### Bảng Summary:

| Component | Paper | Code | Verdict |
|-----------|-------|------|---------|
| Newton-Schulz | X = 0.5X(3I - X^TX) | Khớp 100% | ✓ ĐÚNG |
| Level ordering | Higher = Lower freq | Khớp | ✓ ĐÚNG |
| DGD preconditioner | P = αI - η(k⊗k) | Khớp trong SelfModifyingTitans | ✓ ĐÚNG |
| Adaptive η, α | Data-dependent | Learned networks | ✓ ĐÚNG |
| **M3 momentum** | m = β*m + g | m = m + β*g | ⚠️ **KHÁC** |
| **M3 second moment** | v = β*v + g² | v = v + β*g² | ⚠️ **KHÁC** |
| DeepMomentum | Standard EMA | m = β*m + (1-β)*g | ✓ EMA đúng nhưng khác scale |
| TITAN memory | Full (past surprise) | Simplified | ⚠️ THIẾU |
| HOPE structure | SelfMod + CMS | HOPESelfModBlock | ✓ ĐÚNG (chọn đúng variant) |
| Frequency scheduling | Update per period | LevelClock | ✓ ĐÚNG |
| Loss function | ||h - δ||² | Complex variant | ⚠️ BIẾN THỂ |

### Implications cho FL Implementation:

```
┌─────────────────────────────────────────────────────────────┐
│  KHI IMPLEMENT FED-DGD và FED-M3:                           │
│                                                              │
│  1. M3 MOMENTUM: QUYẾT ĐỊNH QUAN TRỌNG                      │
│     ┌─────────────────────────────────────────────────────┐ │
│     │ Option A: Theo paper (EMA)                          │ │
│     │   m = β*m + g                                       │ │
│     │   → Bounded, stable, standard momentum              │ │
│     │                                                     │ │
│     │ Option B: Theo code (Accumulation)                  │ │
│     │   m = m + β*g                                       │ │
│     │   → Unbounded nhưng Newton-Schulz normalize         │ │
│     │   → Có thể là intentional variant                   │ │
│     │                                                     │ │
│     │ KHUYẾN NGHỊ: Implement cả 2, so sánh performance    │ │
│     └─────────────────────────────────────────────────────┘ │
│                                                              │
│  2. CHỌN ĐÚNG COMPONENT:                                    │
│     - Dùng SelfModifyingTitans (không phải TitanMemory)    │
│     - Dùng HOPESelfModBlock (không phải HOPEBlock)         │
│                                                              │
│  3. DGD PRECONDITIONER: Đã đúng trong code                  │
│     - Có thể reuse logic từ self_modifying.py              │
│                                                              │
│  4. FREQUENCY SCHEDULING:                                   │
│     - Trong FL: Có thể map level → client update frequency │
│     - Fast level = local updates                           │
│     - Slow level = global aggregation                      │
│                                                              │
│  5. LUÔN VERIFY VỚI PAPER:                                  │
│     - Code có thể có bugs hoặc variants                    │
│     - Paper là ground truth                                │
│     - Test implementations carefully                        │
└─────────────────────────────────────────────────────────────┘
```

### Các Câu hỏi Mở:

```
1. M3 accumulation: Bug hay intentional?
   → Cần liên hệ tác giả hoặc test empirically

2. DeepMomentum vs M3: Tại sao khác công thức?
   → Có thể là 2 approaches khác nhau cho 2 use cases

3. Teach signal loss variant: Có paper nào mô tả không?
   → Cần research thêm

4. Jitter trong scheduling: Có lợi ích gì?
   → Engineering trick cho stability?
```

---

*Đánh giá này dựa trên so sánh trực tiếp paper definitions với code implementation.*
*Cập nhật: 2026-03-28*
