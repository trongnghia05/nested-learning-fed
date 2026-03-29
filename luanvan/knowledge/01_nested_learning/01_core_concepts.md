# Nested Learning: Core Concepts

## 1. Tong quan

**Nested Learning (NL)** la mot paradigm moi trong machine learning, duoc gioi thieu boi Behrouz et al. (2025) tai Google Research.

### Y tuong chinh
Thay vi nhin neural network nhu "stack cac layers", NL nhin no nhu **he thong cac bai toan toi uu long nhau**, moi bai toan co:
- Context flow rieng
- Tan so cap nhat rieng
- Objective function rieng

```
Deep Learning truyen thong:
    Input -> Layer1 -> Layer2 -> ... -> Output
    (Tat ca update cung luc trong backprop)

Nested Learning:
    Level 1 (cham nhat): Pre-training weights
        |
        +-- Level 2: Optimizer state (momentum)
            |
            +-- Level 3: Attention memory
                |
                +-- Level 4 (nhanh nhat): Per-token updates
```

## 2. Associative Memory - Nen tang cua NL

### Dinh nghia
**Associative Memory** la operator anh xa tap keys sang tap values:

```
M: K -> V

M* = argmin_M  L(M(K); V)
```

Trong do:
- K: tap keys (co the la tokens, gradients, sub-sequences,...)
- V: tap values tuong ung
- L: objective do chat luong anh xa

### Insight quan trong
**Tat ca thanh phan trong deep learning deu la associative memory:**
- Neural network: Memory luu tru anh xa input -> output
- Optimizer momentum: Memory luu tru gradients
- Attention: Memory luu tru key-value pairs

## 3. Update Frequency - Phan cap tan so cap nhat

### Dinh nghia
Voi moi component A, **update frequency** f_A la so lan cap nhat tren mot don vi thoi gian.

### Ordering
- A > B neu f_A > f_B (A cap nhat nhanh hon B)
- Components co cung frequency nam cung "level"
- Level cao hon -> frequency thap hon

### Vi du trong Transformer
```
Level 1 (f = 1/training_steps): MLP weights
Level 2 (f = 1/batch): Optimizer momentum
Level 3 (f = 1/token): Attention scores
Level 4 (f = real-time): Activation values
```

## 4. Neural Learning Module

### Dinh nghia
**Neural Learning Module** la cach bieu dien model nhu he thong lien ket cac components, moi component co gradient flow rieng.

### Dac diem
1. Multi-level: Nhieu muc do abstraction
2. Multi-timescale: Nhieu toc do cap nhat
3. Nested: Cac bai toan toi uu long nhau

### Hinh thuc hoa
Mot model co the bieu dien nhu:
```
min_{theta^(1)} L^(1)(theta^(1); ...)
    subject to: theta^(2) = argmin L^(2)(theta^(2); theta^(1), ...)
        subject to: theta^(3) = argmin L^(3)(theta^(3); theta^(2), ...)
            ...
```

## 5. Local Surprise Signal (LSS)

### Dinh nghia
**LSS** la gradient cua output doi voi input, do luong "su bat ngo" cua model:

```
LSS = nabla_y L(W; x)
```

### Y nghia
- LSS cao: Input "bat ngo" voi model -> can nho nhieu hon
- LSS thap: Input "binh thuong" -> khong can cap nhat manh

### Ung dung
- Trong TITAN: Dung LSS de quyet dinh co cap nhat memory khong
- Trong training: LSS chinh la gradient signal

## 6. Tai sao NL quan trong cho luan van?

### Lien he voi Federated Learning
```
FL truyen thong:
    Server: Aggregate weights (1 level)
    Client: Local training (1 level)
    -> Total: 2-level optimization

FL + Nested Learning:
    Server: Multi-scale aggregation (nhieu levels)
        - Slow momentum cho long-term knowledge
        - Fast aggregation cho recent updates
    Client: Multi-level local training
        - DGD voi adaptive decay
        - M3 voi multi-scale momentum
    -> Total: 4+ level optimization
```

### Loi ich
1. **Xu ly non-IID**: Multi-scale momentum giu thong tin dai han
2. **Giam client drift**: Adaptive decay trong DGD
3. **Preserve knowledge**: CMS-style multi-frequency updates

## 7. Cac cong thuc quan trong

### Gradient Descent nhu Associative Memory
```
W_{t+1} = argmin_W <W*x_t, nabla_y L(W_t; x_t)> + (1/2*eta)||W - W_t||^2
```

### Momentum nhu 2-level Optimization
```
Level 1: W_{t+1} = W_t - m_{t+1}
Level 2: m_{t+1} = argmin_m -<m, nabla L> + eta||m - m_t||^2
```

### Delta Rule (L2 objective)
```
W_{t+1} = argmin_W ||W*x_t - nabla_y L(W_t; x_t)||^2 + regularization
       = W_t(I - x_t*x_t^T) - eta * nabla L
```

## 8. Key Papers

1. **Behrouz et al. (2025)**: "Nested Learning: The Illusion of Deep Learning Architectures"
   - Paper chinh ve Nested Learning
   - Gioi thieu DGD, M3, CMS, HOPE

2. **Behrouz et al. (2024)**: "Titans: Learning to Memorize at Test Time"
   - Neural long-term memory
   - Surprise-based learning

## 9. Cau hoi tu kiem tra

1. Tai sao gradient descent voi momentum la 2-level optimization?
2. Local Surprise Signal duoc tinh nhu the nao?
3. Update frequency anh huong den thiet ke optimizer ra sao?
4. Lam sao ap dung y tuong multi-level optimization vao FL?
