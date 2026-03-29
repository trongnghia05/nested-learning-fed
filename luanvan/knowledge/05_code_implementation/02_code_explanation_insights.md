# Giải thích Code Implementation: Insights và Phân tích

> Tài liệu này GIẢI THÍCH code, không chỉ liệt kê.
> Mục đích: Hiểu TẠI SAO code được viết như vậy để sau này implement FL.

---

## 1. Insight tổng quan: Mọi thứ đều là Associative Memory

### Paper nói gì?
> "We refer to any operator M that is capable of learning to map a set of data points (e.g., key) to another set (e.g., value) as an associative memory"

### Code thể hiện như thế nào?

```
Nhìn vào cấu trúc code, MỌI THỨ đều follow pattern giống nhau:

┌─────────────────────────────────────────────────────────────┐
│  TitanMemory:                                               │
│      Input (key) → MLP → Output (value)                     │
│      Update: Gradient descent trên loss = ||M(k) - v||²     │
│                                                              │
│  CMSBlock:                                                  │
│      Input → MLP → Output + Residual                        │
│      Update: Gradient descent (qua outer loop)              │
│                                                              │
│  ResidualMLPMemory (trong Self-Modifying Titans):           │
│      Input → MLP → Output + Skip connection                 │
│      Update: DGD với preconditioner                         │
│                                                              │
│  Momentum trong M3:                                         │
│      Input (gradient) → Accumulate → Output (update)        │
│      "Memory" của các gradients trước                       │
└─────────────────────────────────────────────────────────────┘

=> TẤT CẢ đều là: Input → Transform → Output
=> TẤT CẢ đều "nhớ" thông tin qua parameters/state
=> Đây chính là ý tưởng cốt lõi của Nested Learning!
```

---

## 2. M3 Optimizer: Tại sao cần Fast và Slow Momentum?

### Vấn đề với single momentum

```python
# Standard momentum (SGD với momentum)
m = beta * m + grad
W = W - lr * m

# Vấn đề:
# - Chỉ có 1 "tốc độ" nhớ
# - Hoặc phản ứng nhanh (beta nhỏ) → mất long-term trend
# - Hoặc ổn định (beta lớn) → phản ứng chậm với changes
```

### M3 giải quyết bằng cách tách thành 2 scales

**Ý tưởng từ paper:**
```
Con người có:
- Short-term memory: Nhớ việc vừa xảy ra
- Long-term memory: Nhớ xu hướng, patterns

M3 mô phỏng:
- Fast momentum (m1): Phản ứng nhanh, update MỖI step
- Slow momentum (m2): Ổn định, update MỖI K steps
```

**Code thể hiện:**
```python
# File: optim/m3.py

# Fast momentum - MỖI STEP
m1.add_(grad, alpha=beta1)  # m1 += beta1 * grad

# Slow buffer - TÍCH LŨY gradients
slow_buffer.add_(grad)

# Slow momentum - MỖI K STEPS
if state["step"] % slow_chunk == 0:
    m2.add_(slow_buffer, alpha=beta3)  # m2 += beta3 * slow_buffer
    slow_buffer.zero_()  # Reset buffer
```

### Tại sao cần Newton-Schulz Orthogonalization?

**Vấn đề:**
```
Khi tích lũy gradients, chúng có thể bị CORRELATED:

grad_1 = [1, 2, 0]
grad_2 = [2, 4, 0]  # Cùng hướng với grad_1!

m = grad_1 + grad_2 = [3, 6, 0]
=> Quá tập trung vào 1 hướng
=> Bỏ qua các hướng khác
=> Training không balanced
```

**Giải pháp - Orthogonalization:**
```python
# File: optim/m3.py, lines 8-20

def _newton_schulz(matrix, steps, eps=1e-6):
    """
    Biến matrix thành (gần) orthogonal matrix.

    Sau orthogonalization:
    - Các hàng/cột gần như vuông góc nhau
    - Không có hướng nào bị dominant
    - Training balanced hơn
    """
    x = matrix / (norm + eps)  # Normalize trước

    # Iterate để converge đến orthogonal
    for _ in range(steps):
        # Công thức: X = 0.5 * X @ (3I - X^T @ X)
        # Đây là Newton-Raphson để tìm matrix square root
        x = 0.5 * x @ (3.0 * eye - x.T @ x)

    return x
```

**Ý nghĩa cho Federated Learning:**
```
Trong FL:
- Gradients từ các clients có thể CONFLICTING
- Client A đẩy model theo hướng X
- Client B đẩy model theo hướng -X

Newton-Schulz có thể giúp:
- Giảm conflict khi aggregate
- Balance contributions từ các clients
- ĐÂY LÀ LÝ DO TA CẦN M3 CHO FL!
```

---

## 3. DGD (Delta Gradient Descent): Ý tưởng "Quên trước khi Học"

### Vấn đề của standard GD

```python
# Standard GD
W = W - lr * grad

# Vấn đề: INTERFERENCE
# Khi học sample mới, có thể "ghi đè" lên knowledge cũ
# Đây là catastrophic forgetting!
```

### DGD thêm Adaptive Decay

**Paper Eq. 90:**
```
P = α*I - η*(k ⊗ k)
W_new = W @ P - η * grad

Trong đó:
- I: Identity matrix
- k ⊗ k: Outer product của input với chính nó
- P: Preconditioner/Decay matrix
```

**Ý nghĩa trực quan:**
```
TRƯỚC KHI học từ input x:
    1. Tính decay = I - x @ x^T / ||x||²
    2. W = W @ decay  ← "QUÊN" thông tin liên quan đến x
    3. W = W - lr * grad  ← Học thông tin mới về x

Tại sao?
    - decay là projection VUÔNG GÓC với x
    - W @ decay = Phần của W KHÔNG liên quan đến x
    - Ta xóa thông tin cũ về x trước khi ghi thông tin mới
    - Giảm interference!
```

**Code thể hiện (trong deep.py):**
```python
# File: optim/deep.py, lines 46-74

def _nl_precondition(self, grad, context):
    """
    NL-style preconditioning: Project gradient orthogonal to context.

    Đây là simplified version của DGD decay.
    """
    if context is None:
        return grad

    # Normalize context vector
    ctx_norm = torch.norm(ctx)
    unit = ctx / (ctx_norm + self.eps)

    # Project grad ORTHOGONAL to context
    # Công thức: grad_new = grad - (grad · unit) * unit
    projection = (grad * unit).sum(dim=-1, keepdim=True) * unit
    update = grad - projection

    # update bây giờ VUÔNG GÓC với context
    # Không ảnh hưởng đến thông tin đã học về context
    return update
```

**Ý nghĩa cho Federated Learning:**
```
Trong FL với Non-IID data:
- Client A chỉ có class 0, 1
- Client B chỉ có class 2, 3

Khi aggregate:
- Gradients từ A có thể "xóa" knowledge về class 2, 3
- Đây là CLIENT DRIFT!

DGD có thể giúp:
- Mỗi client "quên" selective thay vì toàn bộ
- Giảm interference giữa clients
- ĐÂY LÀ LÝ DO TA CẦN DGD CHO FL!
```

---

## 4. TITAN Memory: Surprise-based Learning

### Ý tưởng từ Neuroscience

```
Con người nhớ gì?
- Sự kiện BÌNH THƯỜNG → Nhanh quên
- Sự kiện BẤT NGỜ → Nhớ lâu

Ví dụ:
- Đi làm như mọi ngày → Không nhớ gì
- Hôm nay gặp tai nạn → NHỚ SUỐT ĐỜI
```

### Code thể hiện

```python
# File: titan/memory.py

class TitanMemory(AssocMemory):

    def surprise(self, residual):
        """
        Surprise = ||residual||

        residual = prediction - target
        = "Model sai bao nhiêu"

        Sai nhiều → Surprise lớn → Cần nhớ
        Sai ít → Surprise nhỏ → Không cần update
        """
        return residual.norm(dim=-1, keepdim=True)

    def update(self, *, key, value, lr=1e-3):
        """
        Update memory chỉ khi surprise đủ lớn.
        """
        # Tính loss = ||M(key) - value||²
        prediction = self.forward(key)
        loss = torch.mean((prediction - value) ** 2)

        # Gradient descent
        grads = torch.autograd.grad(loss, self.net.parameters())
        for param, grad in zip(self.net.parameters(), grads):
            param.add_(grad, alpha=-lr)
```

### Tích hợp với Gating

```python
# File: memorize.py (concept)

def should_update(surprise_value, threshold):
    """
    Chỉ update khi surprise >= threshold.

    Lợi ích:
    - Tiết kiệm computation
    - Không update với noise
    - Focus vào thông tin quan trọng
    """
    return surprise_value >= threshold
```

**Ý nghĩa cho Federated Learning:**
```
Trong FL:
- Không phải mọi local update đều quan trọng
- Có thể chỉ communicate khi có "surprise" lớn

Ý tưởng:
- Client chỉ gửi update khi local loss thay đổi nhiều
- Tiết kiệm communication cost
- Focus vào meaningful updates
```

---

## 5. CMS: Multi-frequency Updates

### Ý tưởng

```
Traditional view:
    - Short-term memory (attention)
    - Long-term memory (weights)
    = Chỉ 2 levels

Nested Learning view:
    - NHIỀU levels với frequencies khác nhau
    - Không chỉ "short" và "long"
    - Có thể có "medium", "very long", etc.
```

### Code thể hiện

```python
# File: cms.py

class CMS(nn.Module):
    """
    Continuum Memory System:
    - Nhiều MLPs xếp chồng
    - Mỗi MLP có update_period riêng
    """

    def __init__(self, *, dim, levels: Sequence[LevelSpec]):
        # Tạo 1 CMSBlock cho mỗi level
        self.blocks = nn.ModuleDict({
            spec.name: CMSBlock(dim, ...)
            for spec in self.level_specs
        })

    def forward(self, x):
        current = x
        # Chain qua tất cả levels
        for spec in self.level_specs:
            block = self.blocks[spec.name]
            current = block(current)
        return current
```

### LevelSpec cho scheduling

```python
# File: levels.py

@dataclass
class LevelSpec:
    name: str              # Tên level (vd: "fast", "medium", "slow")
    update_period: int     # Update mỗi bao nhiêu tokens

# Ví dụ:
levels = [
    LevelSpec(name="fast", update_period=1),     # Mỗi token
    LevelSpec(name="medium", update_period=10),  # Mỗi 10 tokens
    LevelSpec(name="slow", update_period=100),   # Mỗi 100 tokens
]
```

**Ý nghĩa cho Federated Learning:**
```
FL tự nhiên có multi-level structure!

Level 1 (fast):   Local batch updates    (mỗi batch)
Level 2 (medium): Local epoch updates    (mỗi epoch)
Level 3 (slow):   Global aggregation     (mỗi round)
Level 4 (slower): Global momentum        (mỗi K rounds)

=> CMS concept trực tiếp áp dụng được cho FL!
```

---

## 6. Self-Modifying Titans: Meta-Learning

### Ý tưởng

```
Normal learning:
    - Update rule FIXED (vd: W = W - lr * grad)
    - lr, momentum, etc. là hyperparameters

Self-Modifying:
    - Update rule LEARNED
    - Model học cách update chính nó
    - lr, decay là OUTPUT của neural networks
```

### 6 Memories và vai trò

```python
# File: titan/self_modifying.py

class SelfModifyingTitans:
    """
    6 memories, mỗi cái có vai trò riêng:
    """

    # M_k: Sinh key từ input
    # Input x → M_k → key k
    self.M_k = ResidualMLPMemory(...)

    # M_v: Sinh value từ input
    # Input x → M_v → value v (target)
    self.M_v = ResidualMLPMemory(...)

    # M_q: Sinh query từ input
    # Input x → M_q → query q (để retrieve)
    self.M_q = ResidualMLPMemory(...)

    # M_eta: Sinh learning rate
    # Input x → M_eta → η (learning rate cho x này)
    # ADAPTIVE learning rate per input!
    self.M_eta = ResidualMLPMemory(...)

    # M_alpha: Sinh decay factor
    # Input x → M_alpha → α (decay cho x này)
    # ADAPTIVE decay per input!
    self.M_alpha = ResidualMLPMemory(...)

    # M_memory: Main associative memory
    # Query q → M_memory → retrieved value
    self.M_memory = ResidualMLPMemory(...)
```

### Tại sao cần adaptive η và α?

```
Fixed hyperparameters:
    - lr = 0.001 cho TẤT CẢ inputs
    - Không optimal cho mọi trường hợp

Adaptive (learned):
    - Easy input → η nhỏ (đã biết rồi, không cần học mạnh)
    - Hard input → η lớn (cần học nhiều)
    - Related input → α gần 1 (giữ knowledge)
    - Unrelated input → α gần 0 (có thể quên)

=> Model tự quyết định học như thế nào!
```

**Ý nghĩa cho Federated Learning:**
```
Trong FL:
- Mỗi client có data khác nhau
- Fixed lr/decay không optimal

Ý tưởng:
- Mỗi client có thể học η, α riêng
- Adaptive cho data distribution của client
- Hoặc server có thể học global η, α
```

---

## 7. Điểm tương đồng quan trọng (QUAN TRỌNG!)

### 7.1 Surprise Signal ≈ Gradient

```
TITAN paper:
    Surprise = ∇ℓ(M; x) = Gradient của loss

Nested Learning paper:
    LSS = ∇_y L = Gradient theo output

=> CÙNG Ý TƯỞNG!
=> Gradient = Độ "bất ngờ" = Cần học bao nhiêu
```

### 7.2 Momentum ≈ Memory

```
SGD Momentum:
    m = β * m + grad
    = Tích lũy gradients
    = "Nhớ" hướng đi

TITAN Memory:
    M = M + update
    = Tích lũy knowledge
    = "Nhớ" key-value pairs

=> CÙNG PATTERN!
=> Momentum là memory của gradients
=> TITAN memory là memory của data
```

### 7.3 Forgetting ≈ Decay

```
TITAN:
    M_t = (1 - α) * M_{t-1} + S_t
          ^^^^^^^^
          Forgetting

DGD:
    W = W @ (I - x @ x^T) - lr * grad
        ^^^^^^^^^^^^^^^^^
        Adaptive decay

M3 slow momentum:
    m2 = β3 * m2 + buffer
         ^^^
         Decay old, add new

=> CÙNG Ý TƯỞNG!
=> Cần quên selective để học hiệu quả
```

### 7.4 Multi-scale ≈ Nested Levels

```
M3: fast (m1) + slow (m2)
CMS: multiple MLPs với different frequencies
FL: local updates + global aggregation

=> CÙNG STRUCTURE!
=> Fast/slow, local/global, inner/outer
=> Đây là "nested" trong Nested Learning
```

---

## 8. Bảng tổng hợp cho FL Implementation

### Mapping concepts → FL

| Concept | Trong papers | Áp dụng FL như thế nào |
|---------|--------------|------------------------|
| Fast momentum | M3's m1 | Client-level momentum |
| Slow momentum | M3's m2 | Server-level momentum |
| Orthogonalization | Newton-Schulz | Giảm client gradient conflicts |
| Adaptive decay | DGD's (I - xx^T) | Giảm client drift |
| Surprise gating | TITAN's threshold | Communicate only when important |
| Multi-level | CMS levels | Local/global hierarchy |
| Adaptive lr | M_eta | Per-client learning rate |

### Code files cần study kỹ

```
1. optim/m3.py
   - Học: Fast/slow separation, orthogonalization
   - Dùng cho: Fed-M3 server/client momentum

2. optim/deep.py
   - Học: Rank-1 projector, context-aware update
   - Dùng cho: Fed-DGD adaptive decay

3. titan/memory.py
   - Học: Surprise-based gating
   - Dùng cho: Communication efficiency

4. cms.py + levels.py
   - Học: Multi-level scheduling
   - Dùng cho: FL round/epoch structure
```

---

## 9. Tóm tắt Insights

```
┌─────────────────────────────────────────────────────────────┐
│  KEY INSIGHTS TỪ CODE:                                      │
│                                                              │
│  1. MỌI THỨ là Associative Memory                           │
│     - Weights, momentum, attention đều là memory            │
│     - Learning = Update memory hiệu quả                     │
│                                                              │
│  2. Multi-scale là quan trọng                               │
│     - Không chỉ 1 tốc độ học                                │
│     - Fast cho short-term, slow cho long-term               │
│     - FL tự nhiên có structure này (local/global)           │
│                                                              │
│  3. Forgetting có kiểm soát                                 │
│     - Không xóa random                                      │
│     - Xóa selective (liên quan đến input hiện tại)          │
│     - Giảm interference, catastrophic forgetting            │
│                                                              │
│  4. Orthogonalization giảm conflicts                        │
│     - Gradients có thể correlated/conflicting               │
│     - Newton-Schulz làm balanced                            │
│     - Đặc biệt quan trọng cho FL với non-IID                │
│                                                              │
│  5. Surprise-based learning                                 │
│     - Không phải mọi update đều cần thiết                   │
│     - Focus vào "bất ngờ" = thông tin mới                   │
│     - Tiết kiệm computation và communication                │
└─────────────────────────────────────────────────────────────┘
```

---

*Tài liệu này giải thích TẠI SAO code được viết như vậy, không chỉ WHAT.*
