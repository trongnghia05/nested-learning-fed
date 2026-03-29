# Fed-DGD: Federated Delta Gradient Descent

> Thiết kế chi tiết Fed-DGD cho Federated Learning với non-IID data.

---

## 1. Ý tưởng Chính

### 1.1 DGD trong Nested Learning Paper

```
┌─────────────────────────────────────────────────────────────┐
│  DGD CORE IDEA (Paper Section 3.4, Eq. 90):                 │
│                                                              │
│  Update rule với preconditioner:                            │
│                                                              │
│  P = α*I - η*(k ⊗ k)                                        │
│  W = W @ P - η * ∇L                                         │
│                                                              │
│  Trong đó:                                                  │
│  - α: Decay factor (learned, ∈ (0,1))                       │
│  - η: Learning rate (learned, > 0)                          │
│  - k: Key vector (input representation)                     │
│  - k ⊗ k: Outer product (rank-1 matrix)                    │
│                                                              │
│  Ý nghĩa:                                                   │
│  - P "xóa" thông tin cũ liên quan đến k trước khi ghi mới   │
│  - Adaptive decay: Chỉ xóa phần liên quan, giữ phần khác   │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Áp dụng vào FL

```
┌─────────────────────────────────────────────────────────────┐
│  FED-DGD IDEA:                                              │
│                                                              │
│  Vấn đề non-IID trong FL:                                   │
│  - Client 1 train với data {0,1} → weights bias về {0,1}   │
│  - Client 2 train với data {8,9} → weights bias về {8,9}   │
│  - Khi aggregate: conflicts, client drift                   │
│                                                              │
│  DGD có thể giúp:                                           │
│  - Adaptive decay "quên" local bias trước khi aggregate    │
│  - Preconditioner P điều chỉnh theo data distribution      │
│  - Giảm interference giữa clients                          │
│                                                              │
│  Hai cách áp dụng:                                          │
│  ┌─────────────────────┐  ┌─────────────────────┐           │
│  │  Option A:          │  │  Option B:          │           │
│  │  DGD ở Client       │  │  DGD ở Server       │           │
│  │  (Local decay)      │  │  (Global decay)     │           │
│  └─────────────────────┘  └─────────────────────┘           │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Design Options

### Option A: DGD ở Client (Local Decay)

```
┌─────────────────────────────────────────────────────────────┐
│  Mỗi client sử dụng DGD update locally:                     │
│                                                              │
│  For each local step:                                       │
│      k = input representation                               │
│      P_i = α_i * I - η_i * (k ⊗ k)                         │
│      W_i = W_i @ P_i - η_i * ∇L_i                          │
│                                                              │
│  Lợi ích:                                                   │
│  + Mỗi client decay theo local data                         │
│  + Adaptive với local distribution                          │
│                                                              │
│  Hạn chế:                                                   │
│  - α, η cần learn per-client (complex)                     │
│  - Vẫn có thể drift nếu decay không đủ                     │
└─────────────────────────────────────────────────────────────┘
```

### Option B: DGD ở Server (Global Decay)

```
┌─────────────────────────────────────────────────────────────┐
│  Server apply DGD khi aggregate:                            │
│                                                              │
│  1. Clients train bình thường (SGD)                         │
│  2. Upload Δθ_i và representation k_i                       │
│  3. Server compute:                                         │
│     k_global = aggregate(k_i)                               │
│     P = α * I - η * (k_global ⊗ k_global)                  │
│     θ_new = θ_old @ P + aggregate(Δθ_i)                    │
│                                                              │
│  Lợi ích:                                                   │
│  + Đơn giản hơn (centralized decay)                         │
│  + α, η có thể global                                       │
│                                                              │
│  Hạn chế:                                                   │
│  - Không adaptive với từng client                           │
│  - k_global có thể không representative                     │
└─────────────────────────────────────────────────────────────┘
```

### Option C: Hybrid (Recommended)

```
┌─────────────────────────────────────────────────────────────┐
│  Kết hợp cả hai:                                            │
│                                                              │
│  Client side:                                               │
│  - DGD update với local (α_local, k_local)                 │
│  - Learn α_local từ local data                              │
│                                                              │
│  Server side:                                               │
│  - Aggregate với global decay                               │
│  - Apply P_global trước khi broadcast                       │
│                                                              │
│  Benefit: Multi-level decay (local + global)                │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Proposed Algorithm: Fed-DGD

### 3.1 Notation

| Symbol | Meaning | Notes |
|--------|---------|-------|
| N | Số clients | |
| T | Số local steps per round | |
| R | Số communication rounds | |
| α | Decay factor ∈ (0, 1) | Controls uniform decay |
| η | Learning rate | **Dùng cho CẢ preconditioner VÀ gradient step** (theo paper) |
| k_i | Key representation của client i | |
| P_i | Preconditioner của client i | P = αI - η(k⊗k) |

> **Note về η:** Trong TITAN paper, η được dùng đồng nhất cho cả:
> - Preconditioner: `P = αI - η(k⊗k)`
> - Update: `W = W @ P - η * grad`
>
> Điều này có nghĩa η càng lớn → decay theo hướng k càng mạnh VÀ update càng lớn.
> Đây là thiết kế có chủ đích: η kiểm soát "tốc độ học" ở cả hai mức.

### 3.2 Parameter Summary Table

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         FED-DGD PARAMETER TABLE                              │
├──────────────┬──────────┬─────────┬──────────────────────────────────────────┤
│ Parameter    │ Location │ Reset?  │ Description                              │
├──────────────┼──────────┼─────────┼──────────────────────────────────────────┤
│ θ_global     │ Server   │ NO      │ Global model weights                     │
│ α_global     │ Server   │ NO      │ Global decay factor (optional)           │
│ k_global     │ Server   │ YES*    │ Aggregated key (recomputed each round)   │
├──────────────┼──────────┼─────────┼──────────────────────────────────────────┤
│ θ_i          │ Client   │ RESET   │ Local model ← θ_global mỗi round         │
│ k_i          │ Client   │ RESET   │ Local key ← recomputed from local data   │
│ k_sum_i      │ Client   │ RESET   │ Accumulator for k ← 0 mỗi round          │
│ α_i          │ Client   │ Configurable │ Local decay (có thể fixed hoặc learned) │
├──────────────┼──────────┼─────────┼──────────────────────────────────────────┤
│ η            │ Both     │ NO      │ Learning rate (dùng cho cả P và grad)    │
└──────────────┴──────────┴─────────┴──────────────────────────────────────────┘

*k_global được tính lại từ k_i mỗi round, không "reset" theo nghĩa thông thường

FORMULA REFERENCE:
- Preconditioner: P_i = α_i * I - η * (k_i ⊗ k_i)
- Local update:   W = W @ P_i - η * ∇L
- Aggregation:    θ_global = θ_global + (1/N) * Σ Δθ_i
```

### 3.3 k Representation (Key Decision)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Trong TITAN: k = key vector từ Neural Memory                              │
│  Trong Fed-DGD (CNN, không có Memory): k = GRADIENT DIRECTION              │
│                                                                             │
│  Cách tính k = gradient direction:                                          │
│                                                                             │
│  For each local step t:                                                     │
│      g_t = ∇L(θ; batch)                                                     │
│      k_sum = k_sum + g_t                                                    │
│                                                                             │
│  After T steps:                                                             │
│      k = normalize(k_sum)  # Unit vector theo hướng gradient trung bình    │
│                                                                             │
│  Ý nghĩa:                                                                   │
│  - k đại diện cho "hướng học" của local data                               │
│  - P = αI - η(k⊗k) sẽ decay theo hướng này                                │
│  - Giúp "quên" local bias trước khi aggregate                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.4 Pseudocode

```
Algorithm: Fed-DGD (Hybrid Version)

Server Initialize:
    θ^0 ← random init
    α_global ← 0.9 (or learned)

For round r = 1, 2, ..., R:

    # 1. Server broadcasts
    Broadcast θ^(r-1) to all clients

    # 2. Client local training with DGD
    For each client i in parallel:
        θ_i ← θ^(r-1)
        k_sum_i ← 0  # Accumulated key representations

        For t = 1, 2, ..., T:
            x_batch, y_batch ← sample local data

            # Compute gradient
            g_t ← ∇L_i(θ_i; x_batch, y_batch)

            # Accumulate gradient for k (gradient direction)
            k_sum_i ← k_sum_i + g_t

            # DGD Preconditioner (simplified for vectors)
            # For matrix weights: W = W @ P - η*g
            # For vectors: θ = α*θ - η*(k·θ)*k - η*g

            # Option 1: Full DGD (for matrix layers)
            For each weight matrix W in θ_i:
                k_proj ← project k to match W dimensions
                P ← α * I - η * (k_proj ⊗ k_proj)
                W ← W @ P - η * ∇L_W

            # Option 2: Simplified DGD (for all params)
            # θ_i ← α * θ_i - η * g_t

        # Normalize to get gradient direction
        k_i ← normalize(k_sum_i)  # k_i = k_sum_i / ||k_sum_i||

        # Upload
        Send Δθ_i = θ_i - θ^(r-1), k_i to server

    # 3. Server aggregation with global decay
    # Aggregate keys
    k_global ← (1/N) * Σ k_i

    # Aggregate model updates
    Δθ_avg ← (1/N) * Σ Δθ_i

    # Apply global decay (optional)
    # P_global ← α_global * I - η * (k_global ⊗ k_global)
    # θ^r ← θ^(r-1) @ P_global + Δθ_avg

    # Simplified version:
    θ^r ← α_global * θ^(r-1) + Δθ_avg

Return θ^R
```

---

## 4. Key Components

### 4.1 Key Representation k

```
┌─────────────────────────────────────────────────────────────┐
│  k là gì?                                                   │
│                                                              │
│  Trong TITAN/NL paper:                                      │
│  - k = key vector từ Neural Memory module                   │
│  - k đại diện cho input pattern                             │
│                                                              │
│  Trong Fed-DGD (với CNN, không có Memory module):          │
│  - k = GRADIENT DIRECTION (normalized accumulated gradient)│
│                                                              │
│  ★ CHOSEN: k = gradient direction                          │
│    k = normalize(Σ g_t) over T local steps                 │
│                                                              │
│  Lý do chọn gradient direction:                            │
│  - Không cần thêm encoder network                           │
│  - Gradient tự nhiên capture "hướng học" của local data    │
│  - Consistent với ý tưởng DGD: decay theo hướng đã học     │
│                                                              │
│  Alternatives (có thể test trong ablation):                │
│  - Mean embedding: k = mean(encoder(x))                    │
│  - Class centroid: k = mean of class embeddings            │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Preconditioner P

```
P = α*I - η*(k ⊗ k)

Phân tích:
- α*I: Decay toàn bộ weights với factor α
- η*(k ⊗ k): Decay THÊM theo hướng k

Effect:
- Nếu weight w aligned với k: decay nhiều hơn
- Nếu weight w orthogonal với k: decay ít hơn

Trong FL context:
- k đại diện local bias
- P "xóa" local bias trước khi aggregate
- Giữ lại knowledge không liên quan đến local bias
```

### 4.3 Adaptive α và η

```
┌─────────────────────────────────────────────────────────────┐
│  Trong Self-Modifying Titans:                               │
│  - α = sigmoid(M_α(x)) → ∈ (0, 1)                          │
│  - η = softplus(M_η(x)) * scale → > 0                      │
│                                                              │
│  Cả hai LEARNED từ data!                                    │
│                                                              │
│  Trong Fed-DGD, có thể:                                     │
│                                                              │
│  Option 1: Fixed values                                     │
│    α = 0.9, η = 0.01 (hyperparameters)                     │
│                                                              │
│  Option 2: Scheduled                                        │
│    α_r = α_0 * decay^r (giảm theo round)                   │
│                                                              │
│  Option 3: Learned (complex)                                │
│    Mỗi client có network predict α, η                      │
│    Cần thêm meta-learning component                         │
│                                                              │
│  KHUYẾN NGHỊ: Bắt đầu với Option 1, sau đó thử Option 2    │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. So sánh với Methods khác

| Method | Decay | Adaptive | Communication |
|--------|-------|----------|---------------|
| FedAvg | None | No | θ only |
| FedProx | Proximal term | No | θ only |
| **Fed-DGD** | Preconditioner P | Yes (data-dependent) | θ + k |

---

## 6. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                        SERVER                                │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  θ_global                                            │    │
│  │  k_global = aggregate(k_i)                           │    │
│  │  P_global = α*I - η*(k_global ⊗ k_global)           │    │
│  └─────────────────────────────────────────────────────┘    │
│                          │                                   │
│              Broadcast θ_global                              │
│                          ▼                                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐               │
│  │ CLIENT 1  │  │ CLIENT 2  │  │ CLIENT N  │               │
│  │ Data:{0,1}│  │ Data:{2,3}│  │ Data:{8,9}│               │
│  ├───────────┤  ├───────────┤  ├───────────┤               │
│  │ θ_1       │  │ θ_2       │  │ θ_N       │               │
│  │ k_1       │  │ k_2       │  │ k_N       │               │
│  │           │  │           │  │           │               │
│  │ DGD:      │  │ DGD:      │  │ DGD:      │               │
│  │ P_1=αI-   │  │ P_2=αI-   │  │ P_N=αI-   │               │
│  │  η(k⊗k)  │  │  η(k⊗k)  │  │  η(k⊗k)  │               │
│  └───────────┘  └───────────┘  └───────────┘               │
│       │              │              │                        │
│       └──────────────┼──────────────┘                        │
│                      │                                       │
│              Upload: Δθ_i, k_i                              │
│                      ▼                                       │
│                   SERVER                                     │
└─────────────────────────────────────────────────────────────┘
```

---

## 7. Hyperparameters

### 7.1 Recommended Values

| Param | Value | Notes |
|-------|-------|-------|
| α | 0.9 - 0.99 | Decay factor |
| η | 0.01 - 0.1 | Learning rate |
| T | 5 - 20 | Local steps |
| R | 100 - 500 | Communication rounds |

### 7.2 Ablation Studies

```
1. Ảnh hưởng của α:
   - Test α = {0.5, 0.7, 0.9, 0.95, 0.99}
   - α nhỏ = decay mạnh (quên nhiều)
   - α lớn = decay yếu (giữ nhiều)

2. Full DGD vs Simplified:
   - Full: P = αI - η(k⊗k)
   - Simplified: Just α*θ
   - So sánh accuracy và complexity

3. Key representation k:
   - Mean embedding vs Gradient direction
   - Ảnh hưởng đến decay quality

4. Local vs Global decay:
   - Chỉ local DGD
   - Chỉ global DGD
   - Hybrid
```

---

## 8. Expected Results

### 8.1 Hypothesis

```
┌─────────────────────────────────────────────────────────────┐
│  H1: Fed-DGD > FedAvg trên severe non-IID                  │
│      - DGD decay giảm local bias                            │
│      - Aggregate ít conflict hơn                            │
│                                                              │
│  H2: Fed-DGD ≈ FedProx về accuracy                         │
│      - Cả hai đều address client drift                      │
│      - Nhưng mechanisms khác nhau                           │
│                                                              │
│  H3: Fed-DGD tốt hơn khi data có clear structure           │
│      - k capture data structure                             │
│      - Decay theo structure hiệu quả                        │
│                                                              │
│  H4: α optimal phụ thuộc vào non-IID level                 │
│      - Severe non-IID → α nhỏ (decay nhiều)                │
│      - Mild non-IID → α lớn (giữ nhiều)                    │
└─────────────────────────────────────────────────────────────┘
```

### 8.2 Comparison Table (Expected)

| Scenario | FedAvg | FedProx | Fed-DGD |
|----------|--------|---------|---------|
| IID | ★★★ | ★★★ | ★★★ |
| Mild non-IID | ★★☆ | ★★★ | ★★★ |
| Severe non-IID | ★☆☆ | ★★☆ | ★★★ |

---

## 9. Implementation Challenges

```
⚠️ Challenge 1: Computing k ⊗ k cho high-dim
   - k ∈ R^d → k⊗k ∈ R^(d×d)
   - Memory O(d²) có thể lớn
   Solution:
   - Low-rank approximation
   - Block-wise computation
   - Chỉ apply cho layers nhỏ

⚠️ Challenge 2: Heterogeneous k dimensions
   - Different layers có different dimensions
   - k cần match với mỗi layer
   Solution:
   - Per-layer k representations
   - Hoặc project k về common dimension

⚠️ Challenge 3: Learning α, η
   - Nếu muốn adaptive như paper
   - Cần thêm meta-learning component
   Solution:
   - Bắt đầu với fixed values
   - Sau đó thêm simple scheduling

⚠️ Challenge 4: Communication overhead
   - Upload k tốn bandwidth
   - k có thể large
   Solution:
   - Compress k (quantization)
   - Hoặc chỉ upload periodically
```

---

## 10. Comparison: Fed-DGD vs Fed-M3

| Aspect | Fed-DGD | Fed-M3 |
|--------|---------|--------|
| **Core mechanism** | Adaptive decay P | Multi-scale momentum |
| **Handle non-IID** | "Quên" local bias | Balance local + global |
| **Complexity** | Medium (P computation) | Medium (NS orthogonalization) |
| **Communication** | θ + k | θ + buffer |
| **Memory (client)** | θ, k | θ, m1, v, buffer |
| **Best for** | Clear data structure | Long-term dependencies |

---

## 11. Next Steps

```
[ ] 1. Implement Fed-DGD prototype
[ ] 2. Test key representation (k = gradient direction)
[ ] 3. Compare với FedAvg, FedProx
[ ] 4. Run ablation studies (α values)
[ ] 5. Compare với Fed-M3
```

---

*Cập nhật: 2026-03-29*
