# Mathematical Foundations of Nested Learning

## 1. Tu Gradient Descent den Associative Memory

### 1.1 Standard Gradient Descent

Cho 1-layer MLP voi weight W, input x, output y = Wx:

```
Objective: min_W L(W; D_train)

Update rule:
    W_{t+1} = W_t - eta * nabla_W L(W_t; x_t)
```

### 1.2 Phan tich Gradient

Gradient cua loss theo W:
```
nabla_W L(W; x) = nabla_y L(W; x) @ x^T
                  ^^^^^^^^^^^^^^^^
                  Local Surprise Signal (LSS)
```

Trong do:
- `nabla_y L`: Gradient theo output (do "sai" cua output)
- `x^T`: Input (transpose)
- `@`: Outer product

### 1.3 Viet lai nhu Optimization Problem

Dat u_t = nabla_y L(W_t; x_t), ta co:

```
W_{t+1} = argmin_W [ <W*x_t, u_t> + (1/2*eta)||W - W_t||^2 ]
```

**Chung minh:**
```
Lay dao ham theo W va dat = 0:
    x_t @ u_t^T + (1/eta)(W - W_t) = 0
    W = W_t - eta * u_t @ x_t^T
    W = W_t - eta * nabla_W L(W_t; x_t)  ✓
```

### 1.4 Nhin nhu Associative Memory

```
Objective: <W*x_t, u_t>
         = <Memory(key), value>
         = Dot-product similarity

Memory M = W
Key k = x_t
Value v = u_t = nabla_y L

=> GD dang hoc: "Anh xa input x_t sang LSS u_t"
```

---

## 2. Momentum nhu 2-Level Optimization

### 2.1 GD + Momentum

```
m_{t+1} = alpha * m_t + nabla_W L(W_t; x_t)
W_{t+1} = W_t - eta * m_{t+1}
```

### 2.2 Phan tich

**Theo paper: "Higher level = lower frequency"**
**"A > B if computing B requires computing A first" => A is inner (lower level)**

**Level 1 (inner/lower - momentum update, tinh truoc):**
```
m_{t+1} = argmin_m [ -<m, nabla L> + (1/alpha)||m - m_t||^2 ]

Chung minh:
    Dao ham: -nabla L + (2/alpha)(m - m_t) = 0
    m = m_t + (alpha/2) * nabla L
    (Voi alpha' = alpha/2, ta duoc update tuong tu)
```

**Level 2 (outer/higher - weight update, tinh sau, phu thuoc m):**
```
W_{t+1} = W_t - eta * m_{t+1}
```

### 2.3 Y nghia

```
Momentum m la "memory" luu tru gradients:
    - Input (key): gradient hien tai
    - Output (value): weighted sum of past gradients
    - Objective: balance giua past va present

=> Momentum giup "nho" xu huong dai han cua gradients
```

---

## 3. Delta Rule va Delta Gradient Descent

### 3.1 Van de cua Hebbian (dot-product)

Dot-product objective:
```
min_W <W*x_t, u_t>
```

**Van de:** Khong xu ly duoc **dependencies** giua cac data points.

### 3.2 L2 Regression Objective

Thay doi objective:
```
min_W ||W*x_t - u_t||^2
```

**Giai:**
```
Dao ham: 2(W*x_t - u_t) @ x_t^T = 0
         W @ (x_t @ x_t^T) = u_t @ x_t^T
         W = u_t @ x_t^T @ (x_t @ x_t^T)^{-1}
```

### 3.3 Iterative Update (DGD)

Voi regularization:
```
min_W ||W*x_t - u_t||^2 + lambda||W - W_t||^2
```

**Giai (mot buoc GD):**
```
W_{t+1} = W_t - eta * nabla[ ||W_t*x_t - u_t||^2 ]
        = W_t - eta * 2(W_t*x_t - u_t) @ x_t^T
        = W_t - 2*eta*(W_t @ x_t @ x_t^T - u_t @ x_t^T)
        = W_t @ (I - 2*eta*x_t @ x_t^T) - 2*eta*u_t @ x_t^T
```

**Voi eta' = 2*eta va normalize x:**
```
W_{t+1} = W_t @ (I - x_t @ x_t^T / ||x_t||^2) - eta' * grad
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
          ADAPTIVE DECAY TERM
```

### 3.4 Y nghia cua Adaptive Decay

```
Decay matrix: D = I - x_t @ x_t^T / ||x_t||^2

Tinh chat:
1. D la projection orthogonal len khong gian vuong goc voi x_t
2. D "xoa" thong tin trong huong x_t
3. D giu nguyen thong tin vuong goc voi x_t

=> Truoc khi hoc tu x_t, "quen" di thong tin cu ve x_t
=> Giam interference giua cac data points
```

---

## 4. Multi-scale Momentum (M3)

### 4.1 Y tuong

Thay vi 1 momentum, dung 2 momentum voi **tan so khac nhau**:

```
Fast momentum M1: Cap nhat moi step
Slow momentum M2: Cap nhat moi K steps
```

### 4.2 Algorithm

```python
# Moi step t:
grad = compute_gradient()

# Fast momentum (moi step)
m1 = beta1 * m1 + grad
o1 = orthogonalize(m1)  # Newton-Schulz

# Accumulate cho slow
slow_buffer += grad

# Second moment (nhu Adam)
v = beta2 * v + grad^2

# Update
update = (o1 + alpha * o2) / sqrt(v)
W = W - lr * update

# Slow momentum (moi K steps)
if step % K == 0:
    m2 = beta3 * m2 + slow_buffer
    o2 = orthogonalize(m2)
    slow_buffer = 0
```

### 4.3 Newton-Schulz Orthogonalization

**Muc dich:** Bien matrix M thanh orthogonal matrix.

**Algorithm:**
```python
def newton_schulz(M, steps=3):
    X = M / ||M||  # Normalize
    for _ in range(steps):
        X = 0.5 * X @ (3*I - X^T @ X)
    return X
```

**Tinh chat:**
- Output X thoa man: X^T @ X ≈ I
- Giam correlation giua cac rows/columns
- On dinh numerically

### 4.4 Tai sao Orthogonalize Momentum?

```
Van de: Gradients co the bi correlated
    grad_1 = [1, 2, 0]
    grad_2 = [2, 4, 0]  # Collinear voi grad_1

Momentum accumulate: m = grad_1 + grad_2 = [3, 6, 0]
    => Qua tap trung vao mot huong
    => Co the bo qua thong tin tu huong khac

Sau orthogonalize:
    o = orthogonalize(m)
    => Spread ra cac huong
    => Balance hon
```

---

## 5. Lien he cac Concepts

### 5.1 Hierarchy cua Optimizers

```
SGD:           1-level  (W update)
SGD+Momentum:  2-level  (W, m)
Adam:          3-level  (W, m, v)
M3:            4-level  (W, m1, m2, v)
```

### 5.2 Frequency Ordering

```
Optimizer    | Component | Frequency | Level
-------------|-----------|-----------|------
SGD          | W         | 1/step    | 1
SGD+Mom      | m         | 1/step    | 2
             | W         | 1/step    | 1
Adam         | v         | 1/step    | 3
             | m         | 1/step    | 2
             | W         | 1/step    | 1
M3           | m1        | 1/step    | 4
             | v         | 1/step    | 3
             | m2        | 1/K       | 2
             | W         | 1/step    | 1
```

### 5.3 Trong FL Context

```
Theo paper: "Higher level = lower frequency"

FedAvg:
    Level 2 (cao, cham): Global W (update moi round)
    Level 1 (thap, nhanh): Local W (update moi batch)

Fed-M3:
    Level 4 (cao nhat, cham nhat): Server slow momentum (moi K rounds)
    Level 3: Global W (moi round)
    Level 2: Client fast momentum (moi batch)
    Level 1 (thap nhat, nhanh nhat): Client gradients (moi batch)
```

---

## 6. Bai tap tu kiem tra

### Bai 1: Chung minh
Chung minh rang GD update co the viet nhu:
```
W_{t+1} = argmin_W [ <W*x_t, nabla_y L> + (1/2*eta)||W - W_t||^2 ]
```

### Bai 2: Phan tich
Cho Adam optimizer:
```
m = beta1 * m + (1-beta1) * grad
v = beta2 * v + (1-beta2) * grad^2
W = W - lr * m / sqrt(v)
```
Xac dinh cac "levels" va "objectives" tuong ung.

### Bai 3: Thiet ke
De xuat cach ap dung DGD cho FL:
- Client update rule?
- Server aggregation rule?
- Communication cost?

### Bai 4: So sanh
So sanh M3 voi Adam:
- So levels?
- Uu diem cua multi-scale momentum?
- Uu diem cua orthogonalization?

---

## 7. Key Formulas Summary

| Concept | Formula |
|---------|---------|
| GD | W = W - eta * grad |
| GD as memory | W = argmin <W*x, u> + reg |
| Momentum | m = alpha*m + grad |
| DGD | W = W*(I - xx^T) - eta*grad |
| Newton-Schulz | X = 0.5*X*(3I - X^T*X) |
| M3 fast | m1 = beta1*m1 + grad |
| M3 slow | m2 = beta3*m2 + slow_buffer |
