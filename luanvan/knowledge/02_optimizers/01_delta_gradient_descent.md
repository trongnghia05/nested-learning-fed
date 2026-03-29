# Delta Gradient Descent (DGD)

## 1. Van de voi Gradient Descent truyen thong

### Standard Gradient Descent
```
W_{t+1} = W_t - eta * nabla_W L(W_t; x_t)
```

### Van de: Bo qua data dependencies
GD truyen thong xem cac data points nhu **doc lap**, nhung trong thuc te:
- Tokens trong sequence co dependencies
- Gradients tu cac samples co tuong quan
- Khong phu hop voi **non-IID data**

### Cong thuc nhu Associative Memory
GD co the viet lai nhu:
```
min_W <W*x_t, nabla_y L(W_t; x_t)>
```
-> Chi dung dot-product similarity, khong xu ly duoc dependencies

## 2. Delta Gradient Descent - Y tuong

### Tu dot-product sang L2 regression
Thay vi dot-product, DGD dung **L2 loss** de do chat luong anh xa:

```
min_W ||W*x_t - nabla_y L(W_t; x_t)||_2^2
```

### Loi ich
- Xu ly duoc **tuong quan** giua cac data points
- Memory (weights) hoc cach "xoa" thong tin cu truoc khi ghi moi
- Tuong tu **Delta Rule** trong neuroscience

## 3. Cong thuc DGD

### Update Rule
```
W_{t+1} = W_t * (I - x_t * x_t^T) - eta * nabla_y L(W_t; x_t) @ x_t^T
```

Trong do:
- `(I - x_t * x_t^T)`: **Adaptive decay term** - xoa thong tin cu lien quan den x_t
- `nabla_y L(W_t; x_t) @ x_t^T`: Gradient update binh thuong

### So sanh voi GD thuong
```
GD:  W_{t+1} = W_t - eta * grad
DGD: W_{t+1} = W_t * decay_matrix - eta * grad
```

### Dac diem Adaptive Decay
- Decay phu thuoc vao **input hien tai** x_t
- Xoa thong tin cu chi trong **huong lien quan** den x_t
- Giu nguyen thong tin khong lien quan

## 4. Phan tich Toan hoc

### Bai toan toi uu
```
W_{t+1} = argmin_W [ ||W*x_t - u_t||_2^2 + lambda||W - W_t||_F^2 ]
```
Voi u_t = nabla_y L(W_t; x_t)

### Giai
Lay dao ham va dat bang 0:
```
2(W*x_t - u_t)*x_t^T + 2*lambda*(W - W_t) = 0
W*x_t*x_t^T + lambda*W = u_t*x_t^T + lambda*W_t
W*(x_t*x_t^T + lambda*I) = u_t*x_t^T + lambda*W_t
```

Voi lambda -> 0 va mot so gan dung:
```
W_{t+1} ≈ W_t*(I - x_t*x_t^T) - eta * u_t @ x_t^T
```

## 5. DGD voi Momentum

### Ket hop DGD va Momentum
```
m_{t+1} = alpha * m_t + (I - x_t*x_t^T) * nabla L
W_{t+1} = W_t - eta * m_{t+1}
```

### Hoac:
```
W_{t+1} = W_t * decay(x_t) - eta * m_{t+1}
```

## 6. Tai sao DGD phu hop voi Federated Learning?

### Van de trong FL: Non-IID Data
- Moi client co data distribution khac nhau
- Local updates tao ra **client drift**
- Gradients tu cac clients **conflict** nhau

### DGD giai quyet nhu the nao
1. **Adaptive decay**: Weights "quen" thong tin cu truoc khi hoc moi
   - Giam accumulation cua conflicting gradients

2. **Data-dependent decay**: Decay phu thuoc vao data hien tai
   - Moi client co decay rieng phu hop voi local data

3. **Better handling of sequential updates**:
   - Xu ly tot cac local epochs lien tiep
   - Giam overfitting vao local data

### So sanh
```
FedAvg + SGD:
    Client i: W_local = W_global - eta * sum(grads)
    -> Accumulate tat ca gradients, de bi drift

FedAvg + DGD:
    Client i: W_local = W_global * prod(decays) - eta * weighted_sum(grads)
    -> Decay thong tin cu, giam drift
```

## 7. Thach thuc khi ap dung DGD cho FL

### 1. Aggregation cua decay terms
- Moi client co decay matrix khac nhau
- Lam sao tong hop decay tai server?

**Phuong an A: Explicit aggregation**
```
decay_global = average(decay_1, decay_2, ..., decay_K)
```

**Phuong an B: Implicit (chi tong hop weights cuoi)**
```
W_global = average(W_1, W_2, ..., W_K)
# Decay da duoc ap dung trong local training
```

### 2. Communication overhead
- Can truyen them decay information?
- Trade-off giua accuracy va communication cost

### 3. Hyperparameter tuning
- Learning rate eta
- Decay strength (neu co scaling factor)
- Number of local epochs

## 8. Pseudo-code DGD

```python
def dgd_step(W, x, y, eta):
    # Forward pass
    y_pred = W @ x

    # Compute loss and gradient
    loss = loss_fn(y_pred, y)
    grad_y = grad_output(y_pred, y)  # LSS
    grad_W = outer_product(grad_y, x)

    # Compute adaptive decay
    decay_matrix = I - outer_product(x, x)

    # DGD update
    W_new = W @ decay_matrix - eta * grad_W

    return W_new
```

## 9. Lien he voi cac phuong phap khac

### Delta Rule (Widrow-Hoff, 1960)
```
W_{t+1} = W_t + eta * (target - W_t*x) * x^T
```
-> DGD la generalization cua Delta Rule

### DeltaNet (Yang et al., 2024)
- Modern linear attention voi delta update
- Parallelizable training
- DGD trong NL xay dung tren y tuong nay

## 10. Cau hoi nghien cuu cho luan van

1. **Fed-DGD design**:
   - Nen dung explicit hay implicit decay aggregation?

2. **Convergence analysis**:
   - Fed-DGD co convergence guarantee khong?

3. **Communication efficiency**:
   - So sanh communication cost voi FedAvg, FedProx

4. **Ablation studies**:
   - Dong gop cua adaptive decay trong moi truong non-IID?
