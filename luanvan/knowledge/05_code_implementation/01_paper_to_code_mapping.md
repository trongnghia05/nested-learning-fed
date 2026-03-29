# Paper to Code Mapping: Implementation trong Repository

> Tài liệu này map các concepts từ papers sang code implementation trong repo.
> Mục đích: Hiểu code để sau này implement Fed-DGD và Fed-M3 cho FL.

---

## 1. Tổng quan cấu trúc code

```
src/nested_learning/
├── optim/
│   ├── m3.py              # M3 optimizer (Algorithm 1 trong paper)
│   ├── deep.py            # Deep Momentum / DGD variants
│   └── manager.py         # Level-based optimizer manager
├── titan/
│   ├── memory.py          # TitanMemory (Neural Memory Module)
│   └── self_modifying.py  # Self-Modifying Titans (Eq. 83-93)
├── hope/
│   ├── block.py           # HOPE blocks (3 variants)
│   └── self_mod.py        # Self-modifier network
├── cms.py                 # Continuum Memory System
├── levels.py              # LevelSpec for frequency scheduling
├── fast_state.py          # Fast state management
├── functional.py          # Tensor utilities
└── memorize.py            # Surprise-based learning
```

---

## 2. M3 Optimizer (Multi-scale Momentum Muon)

### Paper Reference
- Nested Learning paper, Algorithm 1
- Section 2.3: Optimizers as Learning Modules

### File: `src/nested_learning/optim/m3.py`

### Paper Algorithm vs Code

**Paper Algorithm 1:**
```
Input: params, lr, β1, β2, β3, α, slow_chunk, ns_steps
Initialize: m1=0, m2=0, v=0, slow_buffer=0, o2=0

For each step t:
    1. m1 += β1 * grad
    2. v += β2 * grad²
    3. slow_buffer += grad
    4. o1 = orthogonalize(m1)
    5. update = (o1 + α * o2) / sqrt(v + ε)
    6. param -= lr * update

    If t % slow_chunk == 0:
        7. m2 += β3 * slow_buffer
        8. slow_buffer = 0
        9. o2 = orthogonalize(m2)
```

**Code Implementation (lines 69-121):**
```python
class M3(torch.optim.Optimizer):
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group["params"]:
                grad = p.grad
                state = self.state[p]

                # Initialize state
                if not state:
                    state["step"] = 0
                    state["m1"] = torch.zeros_like(p)      # Fast momentum
                    state["m2"] = torch.zeros_like(p)      # Slow momentum
                    state["v"] = torch.zeros_like(p)       # Second moment
                    state["slow_buffer"] = torch.zeros_like(p)
                    state["o2"] = torch.zeros_like(p)      # Orthogonalized slow

                state["step"] += 1

                # Step 1-3: Update fast momentum, second moment, buffer
                m1.add_(grad, alpha=beta1)
                v.addcmul_(grad, grad, value=beta2)
                slow_buffer.add_(grad)

                # Step 4-6: Orthogonalize and update
                o1 = _orthogonalize(m1, steps=ns_steps, eps=eps)
                denom = v.sqrt().add_(eps)
                update = (o1 + alpha * o2) / denom
                p.add_(update, alpha=-lr)

                # Step 7-9: Slow momentum update every slow_chunk steps
                if slow_chunk > 0 and state["step"] % slow_chunk == 0:
                    m2.add_(slow_buffer, alpha=beta3)
                    slow_buffer.zero_()
                    state["o2"] = _orthogonalize(m2, steps=ns_steps, eps=eps)
```

### Newton-Schulz Orthogonalization (lines 8-20)

**Paper formula:**
```
X = M / ||M||
For i in 1..steps:
    X = 0.5 * X @ (3I - X^T @ X)
```

**Code:**
```python
def _newton_schulz(matrix: torch.Tensor, steps: int, eps: float = 1e-6):
    x = matrix
    norm = torch.linalg.norm(x)
    x = x / (norm + eps)
    eye = torch.eye(n, device=device, dtype=dtype)
    for _ in range(steps):
        x = 0.5 * x @ (3.0 * eye - x.T @ x)
    return x
```

### Key Parameters

| Paper | Code | Default | Ý nghĩa |
|-------|------|---------|---------|
| β1 | beta1 | 0.9 | Fast momentum decay |
| β2 | beta2 | 0.999 | Second moment decay |
| β3 | beta3 | 0.9 | Slow momentum decay |
| α | alpha | 1.0 | Weight của slow momentum |
| K | slow_chunk | 100 | Update slow mỗi K steps |
| ns_steps | ns_steps | 3 | Newton-Schulz iterations |

---

## 3. Deep Momentum / DGD

### Paper Reference
- Section 2.3: Delta Rule và DGD
- Equation 90: Preconditioner P = αI - η(k⊗k)

### File: `src/nested_learning/optim/deep.py`

### Variants

**Code (lines 16-102):**
```python
class DeepMomentum(nn.Module):
    def __init__(self, *, beta=0.9, beta2=0.999, eps=1e-8, variant="preconditioned"):
        self.variant = variant
        # Variants: "preconditioned", "muon", "dmgd", "l2_objective", "nl_l2_precond"
```

### DGD-like Preconditioning (NL variant)

**Paper Eq. 90:**
```
P = α*I - η*(k ⊗ k)
=> Project gradient orthogonal to context direction
```

**Code `_nl_precondition` (lines 46-74):**
```python
def _nl_precondition(self, grad, context):
    if context is None:
        return grad, metrics

    ctx = context
    ctx_norm = torch.norm(ctx)

    if ctx_norm > 0:
        unit = ctx / (ctx_norm + self.eps)
        # Project grad orthogonal to context (rank-1 projector)
        projection = (grad * unit).sum(dim=-1, keepdim=True) * unit
        update = grad - projection  # Grad minus component along context
        return update, metrics
    return grad, metrics
```

### Momentum Update

**Code (lines 76-102):**
```python
def forward(self, grad, *, context=None, param_key=None):
    # Get/create state
    state = self.state.get(key)

    # Apply variant-specific processing
    update = grad
    if self.variant in {"preconditioned", "muon"}:
        update = self._precondition(grad, state)  # RMSprop-style
    if self.variant == "nl_l2_precond":
        update, metrics = self._nl_precondition(grad, context)
    if self.variant in {"dmgd", "muon"}:
        update = self.nonlinearity(update)  # tanh

    # Momentum update
    state.grad_avg.mul_(self.beta).add_(update, alpha=1 - self.beta)
    return state.grad_avg
```

---

## 4. TITAN Neural Memory

### Paper Reference
- TITAN paper Section 3.1: Long-term Memory
- Equation 11: ℓ(M; x) = ||M(k) - v||²

### File: `src/nested_learning/titan/memory.py`

### TitanMemory Class

**Code (lines 31-88):**
```python
class TitanMemory(AssocMemory):
    """Simplified TITAN-style associative memory."""

    def __init__(self, config: TitanMemoryConfig):
        # Build MLP network
        hidden = config.dim * config.hidden_multiplier  # 4x
        blocks = []
        for layer_idx in range(config.layers - 1):
            blocks.extend([nn.Linear(...), activation])
        blocks.append(nn.Linear(hidden, config.dim))
        self.net = nn.Sequential(*blocks)
        self.norm = nn.LayerNorm(config.dim)
        self.grad_clip = 1.0
```

### Forward Pass (Query → Output)

```python
def forward(self, query: torch.Tensor):
    attn = self.net(query)
    # Gradient clipping during training
    if self.training and self.grad_clip > 0:
        with torch.no_grad():
            norm = attn.norm(dim=-1, keepdim=True)
            scale = torch.clamp(norm / self.grad_clip, min=1.0)
        attn = attn / scale
    return self.norm(attn)
```

### Surprise Signal

**Paper:** Surprise = ||∇ℓ(M; x)||

**Code:**
```python
def surprise(self, residual: torch.Tensor):
    return residual.norm(dim=-1, keepdim=True)
```

### Memory Update (Gradient-based)

**Paper Eq. 8:** M_t = M_{t-1} - θ * ∇ℓ(M; x)

**Code:**
```python
@torch.no_grad()
def update(self, *, key, value, error_signal=None, lr=1e-3):
    with torch.enable_grad():
        key_detached = key.detach().requires_grad_(True)
        prediction = self.forward(key_detached)
        target = value.detach()

        # L2 loss (Eq. 11)
        if error_signal is None:
            loss = torch.mean((prediction - target) ** 2)
        else:
            loss = torch.mean(error_signal * prediction)

    # Gradient descent update
    grads = torch.autograd.grad(loss, list(self.net.parameters()))
    for param, grad in zip(self.net.parameters(), grads):
        param.add_(grad, alpha=-lr)
```

---

## 5. Continuum Memory System (CMS)

### Paper Reference
- Section 3: Equation 31
- Multi-frequency MLP updates

### File: `src/nested_learning/cms.py`

### CMSBlock (Single Level MLP)

**Code (lines 11-45):**
```python
class CMSBlock(nn.Module):
    def __init__(self, dim, hidden_multiplier=4, activation="gelu", grad_clip=1.0):
        hidden = dim * hidden_multiplier
        self.net = nn.Sequential(
            nn.LayerNorm(dim),        # Optional normalization
            nn.Linear(dim, hidden),
            activation,
            nn.Linear(hidden, dim),
        )

    def forward(self, x):
        delta = self.net(x)
        # Gradient clipping
        if self.training and self.grad_clip > 0:
            norm = delta.norm(dim=-1, keepdim=True)
            scale = torch.clamp(norm / self.grad_clip, min=1.0)
            delta = delta / scale
        return x + delta  # Residual connection
```

### CMS (Multi-Level System)

**Paper Eq. 31:**
```
y_t = MLP^(f_k)(MLP^(f_{k-1})(...MLP^(f_1)(x_t)))
```

**Code (lines 48-92):**
```python
class CMS(nn.Module):
    """Continuum Memory System with multi-frequency updates."""

    def __init__(self, *, dim, levels: Sequence[LevelSpec], ...):
        # Create one CMSBlock per level
        self.blocks = nn.ModuleDict({
            spec.name: CMSBlock(dim, ...)
            for spec in self.level_specs
        })

    def forward(self, x, *, return_intermediates=False):
        current = x
        for spec in self.level_specs:
            block = self.blocks[spec.name]
            current = block(current)  # Chain through levels
        return current
```

### LevelSpec (Frequency Scheduling)

**File: `src/nested_learning/levels.py`**

```python
@dataclass
class LevelSpec:
    name: str              # Level identifier
    update_period: int     # Update every N tokens
    warmup_steps: int = 0  # Skip first N steps
    jitter: int = 0        # Random jitter for period
```

---

## 6. Self-Modifying Titans

### Paper Reference
- Equations 83-93
- 6 memories: M_k, M_v, M_q, M_η, M_α, M_memory

### File: `src/nested_learning/titan/self_modifying.py`

### Config

```python
@dataclass
class SelfModifyingTitansConfig:
    dim: int
    eta_scale: float = 1e-3        # Learning rate scale
    chunk_size_other: int = 1      # Chunk size for M_k, M_v, etc.
    chunk_size_memory: int = None  # Chunk size for M_memory
    objective: str = "l2"          # "l2" or "dot"
    use_rank1_precond: bool = True # DGD-style preconditioning
    use_alpha: bool = True         # Decay factor
    momentum: float = 0.0          # Gradient momentum
    qk_l2_norm: bool = True        # Normalize Q, K
```

### 6 Memory Modules

**Paper Eq. 83-90:**
```
M_k: Key generation
M_v: Value generation
M_q: Query generation
M_η: Learning rate (eta) generation
M_α: Decay factor (alpha) generation
M_memory: Main associative memory
```

**Code (inside SelfModifyingTitans class):**
```python
# Each memory is a ResidualMLPMemory
self.M_k = ResidualMLPMemory(...)
self.M_v = ResidualMLPMemory(...)
self.M_q = ResidualMLPMemory(...)
self.M_eta = ResidualMLPMemory(...)
self.M_alpha = ResidualMLPMemory(...)
self.M_memory = ResidualMLPMemory(...)
```

### ResidualMLPMemory (Eq. 91)

```python
class ResidualMLPMemory(nn.Module):
    def __init__(self, *, in_dim, out_dim, hidden_dim, activation, use_skip=True):
        self.w2 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.w1 = nn.Linear(hidden_dim, out_dim, bias=False)
        self.w_skip = nn.Linear(in_dim, out_dim) if use_skip else None

    def forward(self, x):
        hidden = self.activation(self.w2(x))
        out = self.w1(hidden)
        if self.w_skip is not None:
            return self.w_skip(x) + out
        return x + out  # Residual
```

### DGD Update Rule (Eq. 90)

**Paper:**
```
P = α*I - η*(k ⊗ k)   # Preconditioner
w2 = w2 @ P - η * g2
w1 = α * w1 - η * g1
```

**Code pattern:**
```python
# Compute per-token gradients via torch.func.grad + vmap
grads = vmap(grad(loss_fn))(...)

# For each token:
eta_t = M_eta(x_t)
alpha_t = M_alpha(x_t)
kk_t = outer(k_t, k_t)

# Preconditioner
P = alpha_t * I - eta_t * kk_t

# Update
w2 = w2 @ P - eta_t * g2
w1 = alpha_t * w1 - eta_t * g1
```

---

## 7. HOPE Architecture

### Paper Reference
- Section 3: HOPE = Self-Modifying Titans + CMS
- Figure 3

### File: `src/nested_learning/hope/block.py`

### 3 Variants

```python
# 1. HOPEAttentionBlock (Eq. 94-97)
#    Attention → CMS (no TITAN memory)

# 2. HOPESelfModBlock (Eq. 83-93)
#    Self-Modifying Titans → CMS

# 3. HOPEBlock (Full Hybrid)
#    Attention → TitanMemory → SelfModifier → CMS
```

### HOPEBlock Components

```python
class HOPEBlock(nn.Module):
    def __init__(self, config):
        self.attention = SelfAttention(...)
        self.titan_memory = TitanMemory(...)
        self.self_modifier = SelfModifier(...)
        self.cms = CMS(levels=...)
```

---

## 8. Surprise-based Learning

### File: `src/nested_learning/memorize.py`

### Surprise Metrics

```python
# L2 Metric: norm(teach_signal)
surprise = torch.norm(teach_signal, dim=-1)

# Gating Logic
def _passes_surprise(surprise_value):
    return surprise >= threshold
```

### Integration với TITAN/CMS

```python
# TITAN memory update conditional on surprise
if self._passes_surprise(surprise):
    self.titan_memory.update(key=k, value=v, lr=lr)

# CMS chunk update conditional on surprise
if surprise >= self.surprise_threshold:
    self._update_cms_chunk(...)
```

---

## 9. Mapping cho Fed-DGD và Fed-M3

### Fed-DGD (dựa trên DeepMomentum)

```
Cần sử dụng/modify:
├── src/nested_learning/optim/deep.py
│   ├── DeepMomentum class
│   ├── _nl_precondition() - rank-1 projector
│   └── Momentum update logic
│
└── Thêm:
    ├── Client-level optimizer
    ├── Server aggregation với decay
    └── Non-IID handling
```

### Fed-M3 (dựa trên M3)

```
Cần sử dụng/modify:
├── src/nested_learning/optim/m3.py
│   ├── M3 class
│   ├── Fast/slow momentum separation
│   ├── Newton-Schulz orthogonalization
│   └── slow_chunk scheduling
│
└── Thêm:
    ├── Client fast momentum
    ├── Server slow momentum
    ├── Cross-client orthogonalization
    └── Hierarchical aggregation
```

---

## 10. Key Design Patterns để học

### 10.1 Chunked Updates
```python
# Process sequence in chunks
for chunk_start in range(0, seq_len, chunk_size):
    chunk = sequence[chunk_start:chunk_start + chunk_size]
    self._process_chunk(chunk)
```

### 10.2 Fast State Management
```python
# Separate meta-learned params from per-context fast state
meta_params = self.net.parameters()  # Learned during training
fast_state = FastState(...)           # Updated during inference
```

### 10.3 Functional Gradients
```python
# Per-sample gradients using vmap
from torch.func import grad, vmap
per_sample_grads = vmap(grad(loss_fn))(batch)
```

### 10.4 Level-based Scheduling
```python
# Different update frequencies per level
if step % level.update_period == 0:
    self._update_level(level)
```

---

## 11. Files quan trọng để đọc khi implement FL

| Priority | File | Lý do |
|----------|------|-------|
| 1 | `optim/m3.py` | M3 optimizer hoàn chỉnh |
| 2 | `optim/deep.py` | DGD-style preconditioning |
| 3 | `titan/memory.py` | Surprise-based update |
| 4 | `cms.py` | Multi-level structure |
| 5 | `levels.py` | Frequency scheduling |
| 6 | `optim/manager.py` | Per-level optimizer |

---

*Tài liệu này map trực tiếp từ papers sang code implementation.*
