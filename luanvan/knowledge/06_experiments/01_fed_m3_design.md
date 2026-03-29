# Fed-M3: Federated Multi-scale Momentum

> Thiết kế chi tiết Fed-M3 cho Federated Learning với non-IID data.

---

## 1. Ý tưởng Chính

```
┌─────────────────────────────────────────────────────────────┐
│  CORE IDEA:                                                 │
│                                                              │
│  M3 có 2 loại momentum:                                     │
│  - Fast momentum: Update mỗi step (high frequency)          │
│  - Slow momentum: Update mỗi K steps (low frequency)        │
│                                                              │
│  Áp dụng vào FL:                                            │
│  - Fast momentum → LOCAL (mỗi client giữ riêng)            │
│  - Slow momentum → GLOBAL (server aggregate)                │
│                                                              │
│  Lợi ích:                                                   │
│  - Fast: Adapt nhanh với local non-IID distribution        │
│  - Slow: Giữ global direction, chống client drift          │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        SERVER                                │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Global Slow Momentum: m2                            │    │
│  │  Global Orthogonalized: o2 = Newton_Schulz(m2)       │    │
│  └─────────────────────────────────────────────────────┘    │
│                          │                                   │
│              Broadcast o2 (global direction)                │
│                          ▼                                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐               │
│  │ CLIENT 1  │  │ CLIENT 2  │  │ CLIENT N  │               │
│  ├───────────┤  ├───────────┤  ├───────────┤               │
│  │ θ_1       │  │ θ_2       │  │ θ_N       │  Model        │
│  │ m1_1      │  │ m1_2      │  │ m1_N      │  Fast mom     │
│  │ v_1       │  │ v_2       │  │ v_N       │  2nd moment   │
│  │ buffer_1  │  │ buffer_2  │  │ buffer_N  │  Grad buffer  │
│  └───────────┘  └───────────┘  └───────────┘               │
│       │              │              │                        │
│       └──────────────┼──────────────┘                        │
│                      │                                       │
│              Upload: buffers, model updates                  │
│                      ▼                                       │
│                   SERVER                                     │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Algorithm

### 3.1 Notation

| Symbol | Meaning |
|--------|---------|
| N | Số clients |
| K | Slow momentum period |
| T | Số local steps per round |
| R | Số communication rounds |
| β1 | Fast momentum coefficient |
| β2 | Second moment coefficient |
| β3 | Slow momentum coefficient |
| λ | Balance factor (local vs global), λ ∈ [0, 1] |
| η | Learning rate |

### 3.2 Bảng Tổng hợp Parameters (QUAN TRỌNG)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                     TỔNG HỢP TẤT CẢ PARAMETERS TRONG FED-M3                     │
└─────────────────────────────────────────────────────────────────────────────────┘
```

| Parameter | Vị trí | Reset mỗi round? | Công thức | Ý nghĩa |
|-----------|--------|------------------|-----------|---------|
| **θ_i** | Client | YES (← θ_global) | - | Model weights của client i |
| **θ_global** | Server | NO (update liên tục) | θ = θ + avg(Δθ_i) | Global model |
| **m1_i** | Client | YES (← 0) | m1 = β1*m1 + g | Fast momentum, nhớ gradients gần đây |
| **m2** | Server | NO (giữ qua rounds) | m2 = β3*m2 + buffer | Slow momentum, nhớ long-term direction |
| **v_i** | Client | YES (← 0) | v = β2*v + g² | Second moment, normalize gradient scale |
| **buffer_i** | Client | YES (← 0) | buffer = Σg | Tích lũy gradients để gửi server |
| **o1_i** | Client | Tính mỗi step | NS(m1_i) | Local direction (orthogonalized) |
| **o2** | Server | Tính mỗi round | NS(m2) | Global direction (orthogonalized) |

#### Chi tiết giải thích:

```
┌─────────────────────────────────────────────────────────────┐
│  θ_i (Client Model)                                         │
│  ─────────────────────────────────────────────────────────  │
│  Vị trí: CLIENT                                             │
│  Reset: YES - nhận θ_global từ server mỗi round đầu        │
│  Lý do: Đảm bảo tất cả clients bắt đầu từ cùng 1 điểm     │
├─────────────────────────────────────────────────────────────┤
│  m1_i (Fast Momentum)                                       │
│  ─────────────────────────────────────────────────────────  │
│  Vị trí: CLIENT (mỗi client giữ riêng)                     │
│  Reset: YES - về 0 mỗi round                               │
│  Lý do:                                                     │
│    - m1 tính trên model cũ, không phù hợp model mới        │
│    - Theo convention FedAvg: optimizer state reset          │
│    - Mỗi client có data khác → m1 khác → không nên share   │
├─────────────────────────────────────────────────────────────┤
│  m2 (Slow Momentum) ⭐ KHÁC BIỆT CHÍNH                      │
│  ─────────────────────────────────────────────────────────  │
│  Vị trí: SERVER (global, duy nhất)                         │
│  Reset: NO - giữ qua tất cả rounds                         │
│  Lý do:                                                     │
│    - m2 capture "long-term global direction"               │
│    - Cần tích lũy từ TẤT CẢ clients qua NHIỀU rounds       │
│    - Dùng để chống client drift                            │
│    - Nếu reset → mất long-term information                 │
├─────────────────────────────────────────────────────────────┤
│  v_i (Second Moment)                                        │
│  ─────────────────────────────────────────────────────────  │
│  Vị trí: CLIENT (mỗi client giữ riêng)                     │
│  Reset: YES - về 0 mỗi round                               │
│  Lý do KHÔNG ở server:                                      │
│    - v đo gradient MAGNITUDE (scale)                       │
│    - Mỗi client có data khác → gradient magnitude khác     │
│    - v dùng để normalize gradient của CHÍNH client đó      │
│    - Nếu dùng v_global → normalize sai scale               │
│  Lý do reset:                                               │
│    - Theo convention FedAvg                                │
│    - Model mới → gradient statistics mới                   │
├─────────────────────────────────────────────────────────────┤
│  buffer_i (Gradient Buffer)                                 │
│  ─────────────────────────────────────────────────────────  │
│  Vị trí: CLIENT (tạm thời, gửi lên server)                 │
│  Reset: YES - về 0 sau khi gửi                             │
│  Lý do:                                                     │
│    - Chỉ là container để tích lũy gradients trong round    │
│    - Gửi lên server để update m2                           │
│    - Sau khi gửi xong → reset để round sau dùng lại        │
├─────────────────────────────────────────────────────────────┤
│  o1_i, o2 (Orthogonalized Directions)                       │
│  ─────────────────────────────────────────────────────────  │
│  o1_i: CLIENT - tính từ m1_i mỗi step                      │
│  o2:   SERVER - tính từ m2 mỗi round, broadcast cho clients│
│  Không cần lưu trữ lâu dài, tính khi cần                   │
└─────────────────────────────────────────────────────────────┘
```

#### Sơ đồ tóm tắt:

```
                    SERVER
    ┌─────────────────────────────────────┐
    │  θ_global  (NO reset - update)      │
    │  m2        (NO reset - accumulate)  │
    │  o2 = NS(m2)                        │
    └─────────────────────────────────────┘
                      │
                      │ Broadcast: θ_global, o2
                      ▼
    ┌─────────────────────────────────────┐
    │            CLIENTS                   │
    │                                      │
    │  θ_i      (RESET ← θ_global)        │
    │  m1_i     (RESET ← 0)               │
    │  v_i      (RESET ← 0)               │
    │  buffer_i (RESET ← 0)               │
    │  o1_i = NS(m1_i)                    │
    └─────────────────────────────────────┘
                      │
                      │ Upload: Δθ_i, buffer_i
                      ▼
                    SERVER
```

#### Hyperparameters (coefficients):

| Param | Giá trị | Ở đâu | Ý nghĩa |
|-------|---------|-------|---------|
| β1 | 0.9 | Client | Decay rate của fast momentum |
| β2 | 0.999 | Client | Decay rate của second moment |
| β3 | 0.9 | Server | Decay rate của slow momentum |
| λ | 0.1-0.5 | Client | Balance local vs global direction |
| η | 0.01 | Client | Learning rate |

### 3.3 Pseudocode

```
Algorithm: Fed-M3

════════════════════════════════════════════════════════════════
SERVER INITIALIZE:
════════════════════════════════════════════════════════════════
    θ_global ← random init
    m2 ← 0                    # Global slow momentum (KHÔNG reset)
    o2 ← 0                    # Global direction

════════════════════════════════════════════════════════════════
FOR round r = 1, 2, ..., R:
════════════════════════════════════════════════════════════════

    ┌─────────────────────────────────────────────────────────┐
    │ STEP 1: SERVER BROADCASTS                               │
    └─────────────────────────────────────────────────────────┘
    Broadcast θ_global, o2 to all clients

    ┌─────────────────────────────────────────────────────────┐
    │ STEP 2: CLIENT LOCAL TRAINING (parallel)                │
    └─────────────────────────────────────────────────────────┘
    For each client i in parallel:

        # === RESET all client states (mỗi round) ===
        θ_i ← θ_global        # Reset model về global
        m1_i ← 0              # Reset fast momentum
        v_i ← 0               # Reset second moment
        buffer_i ← 0          # Reset gradient buffer

        # === Local training loop ===
        For t = 1, 2, ..., T:
            g_t ← ∇L_i(θ_i; batch)       # Local gradient

            # Fast momentum (M3 paper formula)
            m1_i ← β1 * m1_i + g_t

            # Second moment (M3 paper formula, KHÔNG có 1-β2)
            v_i ← β2 * v_i + g_t²

            # Accumulate gradients for server
            buffer_i ← buffer_i + g_t

            # Orthogonalize fast momentum (per-layer, giữ shape)
            o1_i ← Newton_Schulz(m1_i)

            # Compute update: combine local + global direction
            update ← (o1_i + λ * o2) / √(v_i + ε)

            # Apply update
            θ_i ← θ_i - η * update

        # === Upload to server ===
        Δθ_i ← θ_i - θ_global
        Send (Δθ_i, buffer_i) to server

    ┌─────────────────────────────────────────────────────────┐
    │ STEP 3: SERVER AGGREGATION                              │
    └─────────────────────────────────────────────────────────┘

    # Aggregate model updates (FedAvg style)
    θ_global ← θ_global + (1/N) * Σ Δθ_i

    # Aggregate gradient buffers
    buffer_global ← (1/N) * Σ buffer_i

    # Update global slow momentum (KHÔNG reset, tích lũy qua rounds)
    m2 ← β3 * m2 + buffer_global

    # Orthogonalize for next round
    o2 ← Newton_Schulz(m2)

════════════════════════════════════════════════════════════════
RETURN θ_global
════════════════════════════════════════════════════════════════
```

### 3.4 Newton-Schulz Orthogonalization

```python
def Newton_Schulz(M, steps=5, eps=1e-6):
    """
    Orthogonalize matrix M using Newton-Schulz iteration.

    Paper: X_{k+1} = 0.5 * X_k @ (3I - X_k^T @ X_k)
    """
    X = M / (||M|| + eps)  # Normalize
    I = identity matrix

    for _ in range(steps):
        X = 0.5 * X @ (3*I - X.T @ X)

    return X
```

---

## 4. Key Design Decisions

### 4.1 Tại sao Fast momentum ở Local?

```
┌─────────────────────────────────────────────────────────────┐
│  REASONING:                                                 │
│                                                              │
│  - Mỗi client có data distribution khác nhau               │
│  - Fast momentum cần adapt nhanh với local patterns         │
│  - Nếu share fast momentum → bị dominated bởi majority     │
│                                                              │
│  VÍ DỤ:                                                     │
│  Client 1: labels {0,1} → fast momentum hướng về {0,1}     │
│  Client 2: labels {8,9} → fast momentum hướng về {8,9}     │
│                                                              │
│  Nếu share: momentum bị trung bình hóa → không optimal     │
│  Nếu local: mỗi client có momentum phù hợp                 │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Tại sao Slow momentum ở Global?

```
┌─────────────────────────────────────────────────────────────┐
│  REASONING:                                                 │
│                                                              │
│  - Slow momentum capture "long-term trends"                 │
│  - Cần aggregrate từ ALL clients để có global view          │
│  - Chống client drift: kéo clients về hướng chung          │
│                                                              │
│  VÍ DỤ:                                                     │
│  - Round 1: Client 1 thấy {0,1}, drift về hướng đó         │
│  - Round 2: Client 2 thấy {8,9}, drift về hướng khác       │
│                                                              │
│  Global slow momentum: Tích lũy cả hai → hướng balanced    │
│  → Broadcast lại cho clients → chống drift                  │
└─────────────────────────────────────────────────────────────┘
```

### 4.3 Role của λ (Balance Factor)

```
λ nhỏ (→ 0): Ưu tiên local direction (o1)
    - Tốt khi data rất non-IID
    - Mỗi client cần personalization cao

λ lớn (→ 1): Ưu tiên global direction (o2)
    - Tốt khi data gần IID
    - Cần convergence nhanh đến global optimum

λ optimal: Cần tune qua experiments
    - Suggestion: λ = 0.1 cho severe non-IID
    - Suggestion: λ = 0.5 cho moderate non-IID

NOTE: λ KHÁC với α trong Fed-DGD
      - λ (Fed-M3): balance local vs global
      - α (Fed-DGD): decay factor trong P = αI - η(k⊗k)
```

---

## 5. So sánh với M3 Gốc

| Aspect | M3 (Original) | Fed-M3 (Proposed) |
|--------|---------------|-------------------|
| Fast momentum | Single model | Per-client |
| Slow momentum | Single model | Global (server) |
| Newton-Schulz | After each step | Local + Global |
| Update rule | o1 + α*o2 | o1_local + λ*o2_global |
| Communication | N/A | Model + Buffer upload |

---

## 6. Hyperparameters

### 6.1 Recommended Values

| Param | Value | Notes |
|-------|-------|-------|
| β1 | 0.9 | Standard momentum |
| β2 | 0.999 | Adam-style |
| β3 | 0.9 | Slow momentum |
| λ | 0.1 - 0.5 | Balance factor, tune based on non-IID level |
| η | 0.01 | Learning rate |
| K | 10 - 100 | Slow update period |
| T | 5 - 20 | Local steps per round |
| NS_steps | 5 | Newton-Schulz iterations |

### 6.2 Ablation Studies cần làm

```
1. Ảnh hưởng của λ (balance factor):
   - Test λ = {0.0, 0.1, 0.3, 0.5, 0.7, 1.0}
   - Trên các mức non-IID khác nhau

2. Ảnh hưởng của β3 (slow momentum):
   - Test β3 = {0.5, 0.9, 0.99, 0.999}
   - So sánh convergence speed

3. Local steps T:
   - Test T = {1, 5, 10, 20}
   - Trade-off: computation vs communication

4. Newton-Schulz steps:
   - Test steps = {1, 3, 5, 10}
   - Ảnh hưởng đến orthogonalization quality
```

---

## 7. Expected Results

### 7.1 Hypothesis

```
┌─────────────────────────────────────────────────────────────┐
│  H1: Fed-M3 > FedAvg trên non-IID data                     │
│      - Slow momentum giữ global direction                   │
│      - Fast momentum adapt với local                        │
│                                                              │
│  H2: Fed-M3 ≈ FedAvg trên IID data                         │
│      - Khi IID, không cần local adaptation                  │
│      - Overhead của M3 không đáng kể                        │
│                                                              │
│  H3: λ tối ưu phụ thuộc vào mức non-IID                    │
│      - Severe non-IID → λ nhỏ (ưu tiên local)              │
│      - Mild non-IID → λ lớn (ưu tiên global)               │
│                                                              │
│  H4: Newton-Schulz giảm gradient conflicts                  │
│      - Orthogonalization balance các directions             │
│      - Convergence ổn định hơn                              │
└─────────────────────────────────────────────────────────────┘
```

### 7.2 Metrics to Track

| Metric | Description |
|--------|-------------|
| Test Accuracy | Global model accuracy |
| Convergence Speed | Rounds to reach target accuracy |
| Client Variance | Variance of client accuracies |
| Gradient Norm | Track gradient magnitudes |
| Communication Cost | Total bytes transferred |

---

## 8. Implementation Notes

### 8.1 Câu hỏi cần quyết định khi implement

```
Q1: EMA hay Accumulation cho momentum?
    - Paper: m = β*m + g (EMA)
    - Code repo: m = m + β*g (Accumulation)
    - KHUYẾN NGHỊ: Implement cả 2, so sánh

Q2: Newton-Schulz cho vectors hay matrices?
    - M3 gốc: cho matrices (weight matrices)
    - ĐÃ QUYẾT ĐỊNH: Per-layer, KEEP SHAPE
    - Mỗi layer giữ nguyên shape, apply NS riêng
    - Không flatten toàn bộ model thành 1 vector

Q3: Buffer gửi gì?
    - Option A: Raw gradients Σg
    - Option B: Normalized Σg / T
    - KHUYẾN NGHỊ: Normalized (ít bias do T khác nhau)

Q4: Khi nào update slow momentum?
    - Option A: Mỗi round
    - Option B: Mỗi K rounds
    - KHUYẾN NGHỊ: Mỗi round (K đã implicit trong T*R)
```

### 8.2 Potential Issues

```
⚠️ Issue 1: Momentum explosion
   - Nếu dùng accumulation, m có thể unbounded
   - Solution: Newton-Schulz normalize, hoặc dùng EMA

⚠️ Issue 2: Stale global direction
   - o2 từ round trước có thể outdated
   - Solution: Giảm λ, hoặc update o2 more frequently

⚠️ Issue 3: Memory overhead
   - Mỗi client phải giữ m1, v, buffer
   - Solution: Có thể share v across clients (approximate)

⚠️ Issue 4: Newton-Schulz cho high-dim
   - NS yêu cầu matrix operations
   - Solution: Block-wise NS, hoặc skip cho layers nhỏ
```

---

## 9. Variants to Explore

```
Variant 1: Fed-M3-Lite
    - Bỏ Newton-Schulz (giảm computation)
    - Chỉ dùng momentum scaling

Variant 2: Fed-M3-Adaptive
    - λ adaptive theo round/client
    - λ_i = f(divergence of client i)

Variant 3: Fed-M3-Personalized
    - Slow momentum per-client (không global)
    - Cho extreme personalization scenarios

Variant 4: Fed-M3-Async
    - Asynchronous aggregation
    - Clients không cần đợi nhau
```

---

## 10. Next Steps

```
[ ] 1. Implement Fed-M3 prototype
[ ] 2. Test trên FMNIST với pathological non-IID
[ ] 3. Compare với FedAvg baseline
[ ] 4. Tune hyperparameters
[ ] 5. Run full experiments
[ ] 6. Analyze results
```

---

*Cập nhật: 2026-03-29*
