# TITAN, Nested Learning và HOPE: Giải thích chi tiết

> Tài liệu này giải thích mối quan hệ giữa 3 khái niệm quan trọng dựa trên papers gốc.
>
> **Sources:**
> - TITAN: "Titans: Learning to Memorize at Test Time" (Behrouz et al., 2024)
> - Nested Learning: "Nested Learning" (cùng nhóm tác giả)

---

## 1. TITAN Paper: Ý tưởng và Chi tiết

### 1.1 Vấn đề TITAN giải quyết

**Theo TITAN paper, Abstract:**
> "While recurrent models aim to compress the data into a fixed-size memory (called hidden state), attention allows attending to the entire context window, capturing the direct dependencies of all tokens. This more accurate modeling of dependencies, however, comes with a quadratic cost"

```
┌─────────────────────────────────────────────────────────────┐
│  VẤN ĐỀ CỦA CÁC KIẾN TRÚC HIỆN TẠI:                        │
│                                                              │
│  Transformer (Attention):                                   │
│    ✓ Chính xác (nhìn toàn bộ context)                       │
│    ✗ O(N²) complexity                                       │
│    ✗ Giới hạn context window                                │
│                                                              │
│  RNN / Linear Transformer:                                  │
│    ✓ O(N) complexity                                        │
│    ✗ Compress vào fixed-size memory                         │
│    ✗ Mất thông tin khi sequence dài                         │
│                                                              │
│  => Cần kiến trúc VỪA hiệu quả VỪA nhớ được long-term       │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Memory Perspective (Section 2, TITAN paper)

**Theo paper Section 2:**
> "Memory is a fundamental mental process... memory is a confederation of systems—e.g., short-term, working, and long-term memory—each serving a different function"

```
TITAN nhìn các kiến trúc như HỆ THỐNG MEMORY:

┌─────────────────────────────────────────────────────────────┐
│  Transformer:                                               │
│    - Memory = KV cache (growing)                            │
│    - Write: Append key-value pairs (không compress)         │
│    - Read: Attention (similarity query-key)                 │
│    => Short-term memory (chỉ trong context window)          │
│                                                              │
│  RNN:                                                       │
│    - Memory = Hidden state (fixed-size vector)              │
│    - Write: f(M_{t-1}, x_t) (có compress)                   │
│    - Read: g(M_t, x_t)                                      │
│    => Cố gắng làm long-term nhưng bị giới hạn capacity      │
│                                                              │
│  Linear Transformer:                                        │
│    - Memory = Matrix M (fixed-size)                         │
│    - Write: M_t = M_{t-1} + K_t^T * V_t (additive)          │
│    - Read: y_t = Q_t * M_t                                  │
│    => Memory overflow khi sequence dài                      │
└─────────────────────────────────────────────────────────────┘
```

### 1.3 Neural Memory Module (Section 3.1, TITAN paper)

**Ý tưởng chính:**
> "We present a (deep) neural long-term memory that (as a meta in-context model) learns how to memorize/store the data into its parameters at test time"

#### 1.3.1 Surprise-based Learning

**Theo paper Section 3.1:**
> "Inspired by human long-term memory system, we design this memory module so an event that violates the expectations (being surprising) is more memorable"

```
Con người nhớ gì?
    - Sự kiện BÌNH THƯỜNG -> Dễ quên
    - Sự kiện BẤT NGỜ -> Nhớ lâu

TITAN áp dụng:
    Surprise = ∇ℓ(M; x) = Gradient của loss theo input

    - Gradient LỚN -> Input khác biệt với quá khứ -> BẤT NGỜ -> Nhớ
    - Gradient NHỎ -> Input quen thuộc -> Không cần nhớ thêm
```

#### 1.3.2 Công thức Update (Equation 8-9, TITAN paper)

**Phiên bản đơn giản (Eq. 8):**
```
M_t = M_{t-1} - θ_t * ∇ℓ(M_{t-1}; x_t)
              ^^^^^^^^^^^^^^^^^^^^^^^^
                    Surprise
```

**Vấn đề:** Sau vài bước surprising, gradient có thể rất nhỏ -> bỏ lỡ thông tin quan trọng.

**Phiên bản cải tiến với Past Surprise (Eq. 9):**
```
M_t = M_{t-1} + S_t
S_t = η_t * S_{t-1} - θ_t * ∇ℓ(M_{t-1}; x_t)
      ^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^^
      Past Surprise   Momentary Surprise

Trong đó:
- S_t: Tổng surprise (past + momentary)
- η_t: Surprise decay (data-dependent)
- θ_t: Learning rate cho momentary surprise
```

**Theo paper:**
> "Interestingly, this formulation is similar to gradient descent with momentum, where S_t is the momentum element"

#### 1.3.3 Forgetting Mechanism (Equation 12, TITAN paper)

**Theo paper Section 3.1:**
> "When dealing with very large sequences, it is crucial to manage which past information should be forgotten"

```
M_t = (1 - α_t) * M_{t-1} + S_t
      ^^^^^^^^^^
      Forgetting gate

- α_t → 0: Giữ nguyên memory cũ
- α_t → 1: Xóa toàn bộ memory
- α_t: Data-dependent, học được
```

#### 1.3.4 Objective Function (Equation 11, TITAN paper)

**Theo paper:**
> "We focus on associative memory, in which we aim to store the past data as the pairs of keys and values"

```
Loss function:
    ℓ(M_{t-1}; x_t) = ||M_{t-1}(k_t) - v_t||²

Trong đó:
    k_t = x_t * W_K    (key)
    v_t = x_t * W_V    (value)

=> Memory M học mapping: key -> value
=> Giống associative memory trong Nested Learning!
```

### 1.4 Kiến trúc TITAN (Section 4, TITAN paper)

**Theo paper Section 4:**
> "We present Titans, a family of deep models that consists of three hyper-heads"

```
┌─────────────────────────────────────────────────────────────┐
│                    TITAN ARCHITECTURE                        │
│                                                              │
│  ┌─────────────────┐                                        │
│  │  1. CORE        │  Attention với window nhỏ              │
│  │  (Short-term)   │  Xử lý context gần                     │
│  └────────┬────────┘                                        │
│           │                                                  │
│  ┌────────▼────────┐                                        │
│  │  2. LONG-TERM   │  Neural Memory Module M                │
│  │  MEMORY         │  Học tại test time                     │
│  │                 │  Surprise-based update                 │
│  └────────┬────────┘                                        │
│           │                                                  │
│  ┌────────▼────────┐                                        │
│  │  3. PERSISTENT  │  Learnable parameters                  │
│  │  MEMORY         │  Data-independent                      │
│  │                 │  Encode task knowledge                 │
│  └─────────────────┘                                        │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Nested Learning Paper: Ý tưởng và Chi tiết

### 2.1 Paradigm mới (Section 1-2, NL paper)

**Theo paper Abstract:**
> "We present a new learning paradigm, called Nested Learning (NL), that coherently represents a model with a set of nested, multi-level, and/or parallel optimization problems, each of which with its own context flow"

```
┌─────────────────────────────────────────────────────────────┐
│  DEEP LEARNING truyền thống:                                │
│                                                              │
│    Model = Stack of layers                                  │
│    Training = Backprop, update TẤT CẢ weights CÙNG LÚC      │
│                                                              │
│  NESTED LEARNING:                                           │
│                                                              │
│    Model = Nested optimization problems                     │
│    Training = Mỗi level có TẦN SỐ UPDATE RIÊNG              │
│                                                              │
│    Level 1 (outer): min_{W1} L1(W1; ...)                    │
│                         ↓                                    │
│    Level 2:         min_{W2} L2(W2; W1, ...)                │
│                         ↓                                    │
│    Level 3 (inner): min_{W3} L3(W3; W2, W1, ...)            │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Định nghĩa quan trọng (Section 2.2, NL paper)

#### Definition 2 (Update Frequency):
> "For any component A, we define its frequency f_A as its number of updates per unit of time"

#### Ordering:
> "A ≻ B (A faster than B) if: (1) f_A > f_B, or (2) f_A = f_B but computing B requires computing A first"

#### Level:
> "The higher the level is, the lower its frequency"

```
┌─────────────────────────────────────────────────────────────┐
│  ĐỊNH NGHĨA CHÍNH XÁC TỪ PAPER:                             │
│                                                              │
│  Level CAO  = Frequency THẤP = Update ÍT   = OUTER          │
│  Level THẤP = Frequency CAO  = Update NHIỀU = INNER         │
│                                                              │
│  Ví dụ GD + Momentum:                                       │
│    Level 1 (inner): m = α*m + grad     ← Tính TRƯỚC         │
│    Level 2 (outer): W = W - η*m        ← Tính SAU           │
└─────────────────────────────────────────────────────────────┘
```

### 2.3 Associative Memory (Section 2.1, NL paper)

**Theo paper:**
> "We refer to any operator M that is capable of learning to map a set of data points (e.g., key) to another set (e.g., value) as an associative memory"

```
Associative Memory M:
    Input: Keys K = {k1, k2, ..., kn}
    Output: Values V = {v1, v2, ...}

    M học: M(ki) ≈ vi

Ví dụ trong Neural Network:
    - MLP weights W: M(x) = W*x
    - Attention KV cache
    - Optimizer momentum m
```

### 2.4 Local Surprise Signal (Section 2.1, NL paper)

**Theo paper:**
> "The gradient with respect to the output, ∇_y L, which we call Local Surprise Signal (LSS)"

```
Gradient của loss:
    ∇_W L = ∇_y L * x^T
            ^^^^^
             LSS

LSS = "Model sai bao nhiêu ở output"
    - LSS lớn -> Output sai nhiều -> Cần update mạnh
    - LSS nhỏ -> Output đúng -> Không cần update

=> GIỐNG SURPRISE TRONG TITAN!
```

### 2.5 Ba Contributions của Nested Learning

**Theo paper Abstract:**

#### Contribution 1: Deep Optimizers (Section 2.3)
```
Insight: Optimizers (SGD, Adam, Momentum) là associative memory!

SGD + Momentum:
    m_{t+1} = α * m_t + grad
    W_{t+1} = W_t - η * m_{t+1}

    => 2-level optimization
    => m là memory của gradients

NL đề xuất:
    - DGD (Delta Gradient Descent): Adaptive decay
    - M3 (Multi-scale Momentum): Fast + Slow momentum
```

#### Contribution 2: Self-Modifying Titans (Section 3)
```
Mở rộng TITAN:
    - Model học cách UPDATE CHÍNH NÓ
    - Thay vì fixed update rule, học update rule
    - Dùng Deep Optimizers (DGD, M3) thay vì GD thường
```

#### Contribution 3: Continuum Memory System (Section 3)
```
Mở rộng long-term/short-term memory:

Traditional:
    - Short-term memory (attention)
    - Long-term memory (weights)

CMS:
    - MLP^(f1): Update mỗi token (tần số cao nhất)
    - MLP^(f2): Update mỗi chunk
    - ...
    - MLP^(fk): Update mỗi segment (tần số thấp nhất)

=> NHIỀU MỨC frequency, không chỉ 2
```

---

## 3. Mối quan hệ TITAN và Nested Learning

### 3.1 TITAN dưới góc nhìn Nested Learning

```
TITAN neural memory update:
    M_t = (1 - α_t) * M_{t-1} + S_t
    S_t = η_t * S_{t-1} - θ_t * ∇ℓ(M_{t-1}; x_t)

Nested Learning phân tích:
    - S_t (surprise momentum) = Level 1 (inner, tính trước)
    - M_t (memory weights)    = Level 2 (outer, tính sau)

    => TITAN là 2-LEVEL OPTIMIZATION!
    => Giống GD + Momentum
```

### 3.2 Nested Learning mở rộng TITAN như thế nào?

```
┌─────────────────────────────────────────────────────────────┐
│  TITAN gốc:                                                  │
│    - Neural Memory M (MLP)                                  │
│    - Update bằng GD + Momentum                              │
│    - 2 levels: Surprise momentum + Memory weights           │
│                                                              │
│  Nested Learning MỞ RỘNG:                                   │
│                                                              │
│  1. Deep Optimizers:                                        │
│     - Thay GD bằng DGD (có adaptive decay)                  │
│     - Thay single momentum bằng M3 (fast + slow)            │
│     => Nhiều levels hơn                                     │
│                                                              │
│  2. Self-Modifying:                                         │
│     - Model không chỉ update memory                         │
│     - Model học cách UPDATE CHÍNH UPDATE RULE               │
│     => Meta-learning                                        │
│                                                              │
│  3. Continuum Memory System:                                │
│     - Không chỉ 1 MLP cho memory                            │
│     - Nhiều MLPs với frequencies khác nhau                  │
│     => Multi-scale memory                                   │
└─────────────────────────────────────────────────────────────┘
```

### 3.3 So sánh thuật ngữ

| TITAN paper | Nested Learning paper | Ý nghĩa |
|-------------|----------------------|---------|
| Surprise | Local Surprise Signal (LSS) | Gradient, độ bất ngờ |
| Neural Memory M | Associative Memory | Lưu trữ key-value mapping |
| Momentum S_t | Momentum m | Tích lũy gradients |
| Forgetting α | Adaptive decay | Cơ chế quên |
| Long-term memory | Outer level (low freq) | Nhớ lâu dài |
| Short-term memory | Inner level (high freq) | Nhớ ngắn hạn |

---

## 4. HOPE: Proof of Concept cho Nested Learning

### 4.1 HOPE là gì?

**Theo paper Section 3:**
> "Combining this self-referential sequence model with continuum memory system results in HOPE architecture"

**Theo paper Abstract:**
> "Combining our self-modifying sequence model with the continuum memory system, we present a learning module, called HOPE, showing promising results"

```
┌─────────────────────────────────────────────────────────────┐
│                          HOPE                                │
│                                                              │
│  HOPE = Self-Modifying Titans + CMS + Deep Optimizers       │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Self-Modifying Titans (từ TITAN + NL insights)    │    │
│  │    - Neural Memory học tại test time               │    │
│  │    - Dùng DGD/M3 thay vì GD thường                 │    │
│  └─────────────────────────────────────────────────────┘    │
│                          +                                   │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Continuum Memory System (mới từ NL)               │    │
│  │    - MLP^(f1), MLP^(f2), ..., MLP^(fk)             │    │
│  │    - Mỗi MLP update với frequency khác nhau        │    │
│  └─────────────────────────────────────────────────────┘    │
│                          +                                   │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Deep Optimizers (mới từ NL)                       │    │
│  │    - DGD với adaptive decay                        │    │
│  │    - M3 với multi-scale momentum                   │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Vai trò của HOPE trong paper

```
┌─────────────────────────────────────────────────────────────┐
│  Nested Learning paper structure:                           │
│                                                              │
│  Section 2: LÝ THUYẾT                                       │
│    - Associative Memory                                     │
│    - Nested Optimization                                    │
│    - Deep Optimizers                                        │
│    => "Đây là PARADIGM mới"                                 │
│                                                              │
│  Section 3: HOPE = THỰC NGHIỆM / DEMO                       │
│    - Áp dụng lý thuyết NL vào thực tế                       │
│    - Kết hợp tất cả contributions                           │
│    => "Đây là CHỨNG MINH paradigm hoạt động"                │
│                                                              │
│  Section 4: KẾT QUẢ                                         │
│    - HOPE vs baselines                                      │
│    - HOPE beats TITAN (bản gốc)                             │
│    => "NL thực sự cải thiện được TITAN"                     │
└─────────────────────────────────────────────────────────────┘
```

### 4.3 HOPE KHÔNG phải contribution mới

```
QUAN TRỌNG:

3 contributions của NL paper:
    1. Deep Optimizers         ← LÝ THUYẾT / KỸ THUẬT MỚI
    2. Self-Modifying Titans   ← LÝ THUYẾT / KỸ THUẬT MỚI
    3. Continuum Memory System ← LÝ THUYẾT / KỸ THUẬT MỚI

HOPE:
    = (1) + (2) + (3)
    = FRAMEWORK kết hợp
    = PROOF OF CONCEPT
    = KHÔNG phải contribution riêng

Mục đích HOPE:
    - Chứng minh NL paradigm hiệu quả
    - Cho thấy kết hợp các techniques hoạt động
    - Đạt SOTA trên benchmarks
```

### 4.4 Kết quả của HOPE (Table 1, NL paper)

```
Model               | Avg. Score
--------------------|------------
Transformer++       | 52.25
RetNet              | 52.02
DeltaNet            | 52.14
Samba*              | 54.00
Titans (LMM)        | 56.82      ← TITAN gốc
HOPE (ours)         | 57.23      ← HOPE tốt hơn TITAN

=> HOPE > TITAN > Others
=> Nested Learning insights cải thiện TITAN
```

---

## 5. Tổng kết

### 5.1 Mối quan hệ tổng thể

```
┌─────────────────────────────────────────────────────────────┐
│                                                              │
│  TITAN paper (2024):                                        │
│    "Learning to Memorize at Test Time"                      │
│    - Neural Memory Module                                   │
│    - Surprise-based learning                                │
│    - 3 components: Core, Long-term, Persistent              │
│                                                              │
│         ↓ Cùng nhóm tác giả, phát triển tiếp                │
│                                                              │
│  Nested Learning paper:                                     │
│    "A new learning paradigm"                                │
│    - Nhìn model như nested optimization                     │
│    - Multi-frequency updates                                │
│    - Deep Optimizers (DGD, M3)                              │
│                                                              │
│         ↓ Áp dụng NL vào TITAN                              │
│                                                              │
│  HOPE:                                                      │
│    = TITAN + CMS + Deep Optimizers                          │
│    = Proof of concept cho NL                                │
│    = Kết quả: Tốt hơn TITAN gốc                             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 Tại sao cần hiểu cả hai papers?

```
Để implement Fed-DGD và Fed-M3 cho luận văn:

1. Từ TITAN:
   - Hiểu Neural Memory hoạt động như thế nào
   - Hiểu Surprise-based learning
   - Hiểu Forgetting mechanism

2. Từ Nested Learning:
   - Hiểu multi-level optimization
   - Hiểu DGD và M3 optimizers
   - Hiểu cách áp dụng vào FL

3. Áp dụng:
   - Fed-DGD: DGD + Federated Averaging
   - Fed-M3: M3 + Hierarchical FL
   - Giải quyết: Non-IID, client drift, forgetting
```

---

## 6. Thuật ngữ tổng hợp

| Thuật ngữ | Paper | Định nghĩa |
|-----------|-------|------------|
| Neural Memory (M) | TITAN | MLP học tại test time |
| Surprise | TITAN | ∇ℓ(M; x), gradient của loss |
| Past Surprise (S) | TITAN | Momentum của surprise |
| Forgetting (α) | TITAN | Weight decay, cơ chế quên |
| Associative Memory | NL | Operator map keys → values |
| LSS | NL | ∇_y L, gradient theo output |
| Update Frequency | NL | Số lần update per unit time |
| Level | NL | Higher level = lower frequency |
| CMS | NL | Chain of MLPs với multi-frequency |
| DGD | NL | GD với adaptive decay |
| M3 | NL | Fast + slow momentum + orthogonalization |
| HOPE | NL | Self-Modifying Titans + CMS + Deep Optimizers |

---

*Tài liệu này dựa hoàn toàn trên nội dung từ 2 papers gốc.*
