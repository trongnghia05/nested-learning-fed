# Fed-M3: Tong hop cac huong tiep can

> Document nay tong hop cac y tuong thiet ke Fed-M3 de thu nghiem.
> Cap nhat: 2026-03-30
> **Trang thai hien tai: Dang dung Approach 6 (Lite) - 76.14% accuracy @ Round 10**

---

## Nested Learning - Y tuong cot loi

### Van de ma Nested Learning giai quyet

**Deep Learning truyen thong:** Tat ca components update cung toc do

```
Input → Layer1 → Layer2 → ... → Output
         ↓         ↓              ↓
       Update   Update         Update
       (cung luc, cung toc do trong backprop)
```

**Van de:**
- Mot so kien thuc can hoc **CHAM** (long-term patterns)
- Mot so can hoc **NHANH** (adapt to current input)
- Cung learning rate cho tat ca → khong toi uu

### Giai phap: Multi-scale / Multi-frequency Learning

```
Level 1 (CHAM):  Global knowledge, long-term patterns
    │            → Update it, giu lau
    │
    └── Level 2 (NHANH): Local adaptation, recent changes
                         → Update nhieu, thay doi nhanh
```

### Vi du trong M3 Optimizer

```
┌─────────────────────────────────────────────────────────────┐
│  Slow Momentum (m2)                                         │
│  - Update moi 100 steps                                     │
│  - Giu "huong di tong the" cua optimization                │
│  - Nhu "bo nho dai han"                                     │
├─────────────────────────────────────────────────────────────┤
│  Fast Momentum (m1)                                         │
│  - Update moi step                                          │
│  - Theo doi gradient gan day                                │
│  - Nhu "bo nho ngan han"                                    │
├─────────────────────────────────────────────────────────────┤
│  Current Gradient                                           │
│  - Thong tin ngay lap tuc                                   │
└─────────────────────────────────────────────────────────────┘

Update = (o1 + α * o2) / sqrt(v)
         ↑       ↑
      fast    slow
```

### Loi ich cua Multi-scale

| Van de | Giai phap |
|--------|-----------|
| Gradient noisy | Slow momentum loc nhieu |
| Forgetting | Slow momentum giu kien thuc cu |
| Zigzag convergence | Ket hop fast + slow cho huong di on dinh |

---

## Ap dung vao Federated Learning

### Mapping tu nhien

```
Nested Learning          →    Federated Learning
─────────────────────────────────────────────────
Slow (long-term)         →    Server (global knowledge)
Fast (short-term)        →    Client (local adaptation)
```

### Ky vong cua Fed + Nested Learning

| Van de FL | Nested Learning giai quyet |
|-----------|---------------------------|
| **Client drift** | Slow momentum o server giu "huong di chung" |
| **Non-IID** | Multi-scale giup balance local vs global |
| **Forgetting** | Long-term memory giu kien thuc tu rounds truoc |
| **Conflicting gradients** | Orthogonalization giam xung dot |

### Cau hoi thiet ke quan trong

**Server co nen chi FedAvg (average params) khong?**

Theo tinh than Nested Learning, **Server nen LEARN, khong chi AGGREGATE:**

```python
# Hien tai (FedAvg style):
theta_global = average(theta_i)  # Chi aggregate, khong "learn"

# Nested Learning style:
theta_global = theta_old - lr_slow * slow_update  # Server cung "learn"!
```

Day la ly do tai sao Approach 3 (Server-Side) va Approach 5 (Meta-Learning)
co the dung tinh than Nested Learning hon Approach 1 (Client-Side).

---

## Baseline: FedAvg

```python
# Client
theta_local = theta_global
for epoch in range(E):
    for batch in data:
        grad = compute_gradient(theta_local, batch)
        theta_local = theta_local - lr * grad  # SGD don gian

# Server
theta_global = weighted_average([theta_1, theta_2, ..., theta_N])
```

**Ket qua tham khao:** ~37% accuracy Round 1 (CIFAR-10, alpha=0.5)

---

## Approach 1: Fed-M3 Client-Side + NS (DA THAT BAI ❌)

### Y tuong
- Client dung M3 optimizer (Newton-Schulz + multi-scale momentum)
- Server dung FedAvg + tich luy slow momentum

### Code
```python
# Client
m1 = 0, v = v_init  # Reset moi round
for step in training:
    m1 = m1 + beta1 * grad           # Fast momentum
    v  = v  + beta2 * grad^2         # Second moment
    o1 = Newton_Schulz(m1)           # Orthogonalize
    o2 = server.global_direction     # Tu server
    update = (o1 + lam * o2) / sqrt(v)
    theta = theta - lr * update

# Server
theta_global = weighted_average(theta_i)           # FedAvg
m2 = m2 + beta3 * aggregated_gradient_buffer       # Slow momentum
o2 = Newton_Schulz(m2)                             # Global direction
```

### Van de
1. Newton-Schulz lam mat magnitude → clients dong gop bang nhau (sai!)
2. v reset moi round → denom qua nho → update qua lon
3. Local models hoc tot nhung global model khong hoc

### Cac bien the da thu

| Variant | Thay doi | Ket qua |
|---------|----------|---------|
| v_init=1.0 | Default | ~10% (that bai) |
| v_init=100 | Tang v_init | ~20% (van kem FedAvg) |
| v_init=1000 | Tang v_init nhieu | Khong cai thien |

**Ket luan:** Van de khong phai v_init, ma la Newton-Schulz lam mat magnitude.

---

## Approach 2: Fed-M3 voi Scaled Newton-Schulz (DA THAT BAI ❌)

### Y tuong
Newton-Schulz chi thay doi HUONG, giu nguyen MAGNITUDE.

### Code thay doi
```python
def newton_schulz_scaled(matrix):
    norm = ||matrix||              # Luu magnitude goc
    x = matrix / norm              # Normalize
    x = newton_schulz_iterations(x)
    return x * norm                # Scale lai!

# Client update
o1 = newton_schulz_scaled(m1)      # Giu magnitude
update = (o1 + lam * o2) / sqrt(v)
```

### Van de thuc te
- m1 dung accumulation: `m1 = m1 + beta1 * grad`
- ||m1|| tang lien tuc theo so steps
- Scale bang ||m1|| → update KHONG LON → model explode

### Ket qua
- **Te hon Approach 1**
- Update magnitude qua lon do ||m1|| accumulate

### Trang thai: THAT BAI ❌

---

## Approach 3: Fed-M3 Server-Side Only

### Y tuong
- Client dung SGD/Adam binh thuong (nhu FedAvg)
- Server ap dung M3-style update SAU khi aggregate

### Code
```python
# Client (giong FedAvg)
theta_local = theta_global
for epoch in range(E):
    grad = compute_gradient(theta_local, batch)
    theta_local = theta_local - lr * grad  # SGD binh thuong

# Server (M3-style)
theta_avg = weighted_average(theta_i)
pseudo_grad = theta_global_old - theta_avg     # "Gradient" tu thay doi

m1 = m1 + beta1 * pseudo_grad                  # Fast momentum
m2 = m2 + beta3 * pseudo_grad (moi K rounds)   # Slow momentum
v  = v  + beta2 * pseudo_grad^2

o1 = Newton_Schulz(m1)
o2 = Newton_Schulz(m2)
update = (o1 + lam * o2) / sqrt(v)

theta_global = theta_global_old - lr_server * update
```

### Uu diem
- Client don gian, khong can thay doi
- M3 chay o server, co du thoi gian tich luy v
- Khong co van de "magnitude mat khi aggregate"

### Nhuoc diem
- Server phai tinh toan nhieu hon
- Nested Learning chi o server, khong tan dung duoc o client

### Trang thai: CHUA THU NGHIEM

---

## Approach 4: Aggregate Gradients (khong phai Params)

### Y tuong
- Client tinh gradient, gui gradient (KHONG update local)
- Server aggregate gradients va ap dung M3 update

### Code
```python
# Client
accumulated_grad = 0
for epoch in range(E):
    for batch in data:
        grad = compute_gradient(theta_global, batch)  # Dung theta_global!
        accumulated_grad += grad
return accumulated_grad  # Gui gradient, khong gui params

# Server
agg_grad = weighted_average(accumulated_grad_i)

m1 = m1 + beta1 * agg_grad
m2 = m2 + beta3 * agg_grad (moi K rounds)
v  = v  + beta2 * agg_grad^2

update = (Newton_Schulz(m1) + lam * Newton_Schulz(m2)) / sqrt(v)
theta_global = theta_global - lr * update
```

### Uu diem
- Giong distributed training, M3 chay centralized
- v tich luy dung cach qua nhieu rounds
- Khong co van de client drift

### Nhuoc diem
- Communication cost cao hon (gui gradients thay vi params)
- Client khong hoc gi, chi tinh gradient
- Mat loi ich cua "local adaptation"

### Trang thai: CHUA THU NGHIEM

---

## Approach 5: Nested Thuc Su (Meta-Learning Style)

### Y tuong
- Outer loop (Server): Update slow, hoc "global knowledge"
- Inner loop (Client): Update fast, adapt to local data

### Code
```python
# Outer loop - Server (moi K rounds)
if round % K == 0:
    slow_grad = aggregate(client_slow_updates)
    m2 = m2 + beta3 * slow_grad
    theta_global = theta_global - lr_outer * Newton_Schulz(m2)

# Inner loop - Client (moi round)
theta_local = theta_global  # Start from global
for epoch in range(E):
    grad = compute_gradient(theta_local, batch)
    m1 = m1 + beta1 * grad
    theta_local = theta_local - lr_inner * Newton_Schulz(m1)

# Client gui ca theta_local va accumulated_grad
return theta_local, accumulated_grad
```

### Uu diem
- Nested Learning dung nghia (2 levels)
- Giong MAML: global model la "initialization tot"
- Local models adapt nhanh, global model stable

### Nhuoc diem
- Phuc tap hon cac approach khac
- Can tune nhieu hyperparameters (K, lr_outer, lr_inner, beta1, beta3)

### Trang thai: CHUA THU NGHIEM

---

## Approach 6: Fed-M3 Lite (DANG SU DUNG)

### Y tuong
- Bo Newton-Schulz hoan toan
- Chi dung multi-scale momentum (khong orthogonalize)
- **Core insight: Multi-scale momentum la y tuong chinh cua Nested Learning, khong phai NS**

### Implementation hien tai

```python
# ====== CLIENT ======
# Reset m1 moi round (fast, local)
m1 = 0

for step in training:
    grad = compute_gradient()

    # Fast momentum (EMA style - BOUNDED)
    # QUAN TRONG: m1 = beta1*m1 + grad (EMA)
    # KHONG PHAI: m1 = m1 + beta1*grad (accumulation - UNBOUNDED!)
    m1 = beta1 * m1 + grad

    # Ket hop fast (local) + slow (global) momentum
    update = m1 + lam * m2_global

    # Apply update
    theta = theta - lr * update

    # Tich luy gradient de gui server
    buffer = buffer + grad

# Gui ve server: theta_local, buffer

# ====== SERVER ======
# 1. FedAvg: aggregate params
theta_global = weighted_average(theta_i)

# 2. Aggregate gradient buffers
agg_buffer = weighted_average(buffer_i)

# 3. Update slow momentum (EMA style - BOUNDED)
# QUAN TRONG: m2 = beta3*m2 + buffer (EMA)
m2 = beta3 * m2 + agg_buffer

# 4. Normalize m2 de tranh gia tri qua lon
# Scale ve magnitude hop ly (~0.1)
m2_normalized = m2 / ||m2|| * 0.1

# 5. Gui m2_normalized cho clients vong sau
```

### Tai sao EMA thay vi Accumulation?

```
Accumulation (SAI):  m = m + beta * grad
  → m tang vo han theo thoi gian
  → Update = m qua lon → model explode

EMA (DUNG):  m = beta * m + grad
  → m bi bounded (beta < 1 nen phan cu decay di)
  → Update on dinh, khong explode
```

### Nested Learning o dau?

```
┌─────────────────────────────────────────────────────────────┐
│  SLOW (Server-side, m2)                                     │
│  - Update moi round (sau khi aggregate)                     │
│  - Tich luy "huong di chung" tu tat ca clients             │
│  - Khong reset qua cac rounds → long-term memory           │
├─────────────────────────────────────────────────────────────┤
│  FAST (Client-side, m1)                                     │
│  - Update moi step trong training                          │
│  - Adapt nhanh theo local data                              │
│  - Reset moi round → short-term, local                     │
└─────────────────────────────────────────────────────────────┘

Update = m1 + λ * m2
         ↑       ↑
      fast    slow
      local   global
```

### Uu diem
- Don gian, de hieu, de debug
- Giu magnitude (khong co van de NS)
- Van co multi-scale momentum (core cua Nested Learning)
- **BOUNDED**: EMA dam bao m1, m2 khong explode

### Nhuoc diem
- Mat loi ich cua orthogonalization (neu co)
- Can chon lambda (lam) phu hop

### Ket qua thu nghiem

| Round | Fed-M3 Lite | FedAvg | Notes |
|-------|-------------|--------|-------|
| 1     | ~37%        | ~37%   | Tuong duong |
| 10    | **76.14%**  | TBD    | Kha quan! |
| 50    | TBD         | TBD    | Can chay them |
| 100   | TBD         | TBD    | Can chay them |

### Trang thai: DANG THU NGHIEM ✓

---

## Bang tong hop

| Approach | Nested o dau | NS o dau | Do phuc tap | Trang thai |
|----------|--------------|----------|-------------|------------|
| 1. Client-Side + NS | Client | Client | Trung binh | ❌ That bai (~10%) |
| 2. Scaled NS | Client | Client (scaled) | Trung binh | ❌ That bai (te hon) |
| 3. Server-Side Only | Server | Server | Thap | Chua thu |
| 4. Aggregate Gradients | Server | Server | Trung binh | Chua thu |
| 5. Meta-Learning Style | Ca hai | Ca hai | Cao | Chua thu |
| **6. Lite (no NS)** | **Ca hai** | **Khong co** | **Thap** | **✓ Dang dung (76%)** |

---

## Thu tu thu nghiem (CAP NHAT)

### Da thu nghiem ❌
1. **Approach 1 (Client-Side + NS)** - ~10% accuracy, that bai
   - Van de: NS output fixed magnitude, v reset moi round
2. **Approach 2 (Scaled NS)** - Te hon Approach 1
   - Van de: m1 accumulate nen ||m1|| rat lon, scale lam update explode
3. **v_init tuning** - v_init=100, 1000 khong cai thien dang ke

### Dang thu nghiem ✓
4. **Approach 6 (Lite)** - 76.14% at Round 10
   - Bo NS, chi dung multi-scale momentum
   - EMA momentum (bounded) thay vi accumulation

### Chua thu (neu can)
5. **Approach 3 (Server-Side)** - NS chi o server
6. **Approach 4 (Aggregate Gradients)** - Gui gradient thay vi params
7. **Approach 5 (Meta-Learning)** - Phuc tap, can thoi gian

---

## Metrics can theo doi

1. **Global Test Accuracy** - Metric chinh
2. **Per-Client Accuracy Variance** - Do fairness
3. **Convergence Speed** - So rounds de dat X% accuracy
4. **Communication Cost** - Bytes gui/nhan
5. **Update Magnitude** - Debug, kiem tra stability

---

## Notes

- Moi approach can so sanh voi FedAvg baseline (37% Round 1)
- Chay it nhat 50-100 rounds de thay convergence
- Dung cung seed de fair comparison
- Log day du de debug

---

*Cap nhat: 2026-03-30*
