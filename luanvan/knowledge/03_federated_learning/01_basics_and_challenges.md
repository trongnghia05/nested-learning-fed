# Federated Learning: Basics and Challenges

## 1. Federated Learning la gi?

### Dinh nghia
**Federated Learning (FL)** la phuong phap huan luyen model phan tan, cho phep nhieu clients cung huan luyen mot model ma **khong chia se du lieu**.

### Kien truc co ban
```
                    Server
                      |
        +-------------+-------------+
        |             |             |
    Client 1      Client 2      Client K
    (Data D1)     (Data D2)     (Data DK)
```

### So sanh voi Centralized Learning
```
Centralized:
    Tat ca data -> Server -> Train model

Federated:
    Data o lai client -> Train local -> Gui updates -> Aggregate
```

## 2. FedAvg - Thuat toan co ban

### Algorithm
```
Server:
    Initialize W_0
    For round t = 0, 1, ..., T-1:
        Select clients S_t
        For each client k in S_t (parallel):
            W_k^{t+1} = LocalTrain(W^t, D_k, epochs=E)
        W^{t+1} = sum(n_k * W_k^{t+1}) / sum(n_k)

Client LocalTrain(W, D, epochs):
    For epoch = 1, ..., E:
        For batch in D:
            W = W - eta * grad(batch)
    Return W
```

### Uu diem
- Don gian, de implement
- Bao mat du lieu (data khong roi khoi client)
- Giam bandwidth (chi gui model weights)

## 3. Van de Non-IID Data

### Non-IID la gi?
**Non-IID (Non-Independent and Identically Distributed)**: Du lieu tai cac clients co phan phoi **khac nhau**.

### Cac loai Non-IID

#### 1. Label Skew (Skew nhan)
```
Client 1: Chi co class 0, 1, 2
Client 2: Chi co class 3, 4, 5
Client 3: Chi co class 6, 7, 8, 9
```

#### 2. Feature Skew (Skew dac trung)
```
Client 1: Anh chup ban ngay
Client 2: Anh chup ban dem
-> Cung labels nhung features khac
```

#### 3. Quantity Skew (Skew so luong)
```
Client 1: 10,000 samples
Client 2: 100 samples
Client 3: 50 samples
```

#### 4. Dirichlet Distribution (Pho bien trong nghien cuu)
```
p ~ Dir(alpha)
- alpha -> 0: Extreme non-IID (moi client 1 class)
- alpha -> inf: IID (uniform distribution)
- alpha = 0.5: Moderate non-IID
```

### Anh huong cua Non-IID

1. **Client Drift**
   - Moi client toi uu theo huong khac
   - Aggregate bi "cancel out"

2. **Slow Convergence**
   - Can nhieu rounds hon
   - Accuracy thap hon

3. **Accuracy Drop**
   - Zhao et al. (2018): Giam den 55% accuracy

## 4. Cac thach thuc chinh trong FL

### 4.1 Client Drift

**Nguyen nhan:**
```
Local optimum cua client k: W*_k
Global optimum: W*

Non-IID -> W*_k != W* -> Drift
```

**Hau qua:**
- Local updates huong ve W*_k
- Aggregate bi sai lech

### 4.2 Catastrophic Forgetting

**Nguyen nhan:**
- Round t: Learn tu client 1, 2
- Round t+1: Learn tu client 3, 4
- -> Quen kien thuc tu round t

**Trong FL:**
- Server aggregate ghi de kien thuc cu
- Khong co mechanism giu long-term memory

### 4.3 Communication Cost

**Van de:**
- Model co hang trieu parameters
- Moi round gui/nhan full model
- Bandwidth han che (mobile, IoT)

**Metrics:**
- Communication rounds
- Bytes per round
- Total communication cost

### 4.4 System Heterogeneity

- Clients co hardware khac nhau
- Training speed khac nhau
- Dropouts va failures

## 5. Cac phuong phap hien tai

### 5.1 FedProx (Li et al., 2020)

**Y tuong:** Them proximal term de giu local model gan global model

```
L_local = L_task + (mu/2) * ||W - W_global||^2
```

**Uu diem:**
- Giam client drift

**Han che:**
- Hyperparameter mu kho tune
- Van dung single-level optimization

### 5.2 SCAFFOLD (Karimireddy et al., 2020)

**Y tuong:** Dung control variates de correct drift

```
Local update: W = W - eta * (grad - c_k + c)
    c_k: Client control variate
    c: Server control variate
```

**Uu diem:**
- Giam variance
- Convergence tot hon

**Han che:**
- Tang communication (2x: gui W va c)
- Them memory tai client

### 5.3 FedAdam/FedYogi (Reddi et al., 2021)

**Y tuong:** Dung adaptive optimizer tai server

```
Server aggregate:
    delta = W_new - W_old
    m = beta1 * m + (1-beta1) * delta
    v = beta2 * v + (1-beta2) * delta^2
    W = W - lr * m / (sqrt(v) + eps)
```

**Uu diem:**
- Adaptive learning rate
- On dinh hon FedAvg

**Han che:**
- Van la single-level
- Khong giai quyet conflict truc tiep

### 5.4 SlowMo (Wang et al., 2020)

**Y tuong:** Slow momentum tai server

```
Server:
    delta = aggregate(client_deltas)
    m = beta * m + delta
    W = W - lr * m
```

**Uu diem:**
- Smoothing updates
- Giam noise

**Lien he voi Fed-M3:**
- SlowMo chi co 1 level momentum
- Fed-M3 co multi-scale + orthogonalization

## 6. Gap nghien cuu

### Cac phuong phap hien tai thieu:

1. **Multi-level optimization**
   - Chi dung 1-2 levels
   - Khong tan dung nested structure

2. **Conflict resolution**
   - Simple averaging
   - Khong xu ly gradient conflicts

3. **Long-term memory**
   - Chi co momentum ngan han
   - Khong giu knowledge dai han

### De xuat cua luan van:

| Gap | Fed-DGD | Fed-M3 |
|-----|---------|--------|
| Multi-level | Adaptive decay | Fast + Slow momentum |
| Conflict | Data-dependent decay | Newton-Schulz |
| Memory | Implicit | Slow momentum |

## 7. Metrics danh gia

### 7.1 Accuracy
- **Global accuracy**: Acc tren test set chung
- **Personalized accuracy**: Avg acc tren test set cua tung client

### 7.2 Convergence
- **Rounds to target**: So rounds de dat accuracy X%
- **Final accuracy**: Accuracy sau T rounds

### 7.3 Communication
- **Rounds**: So lan giao tiep
- **Bytes**: Tong data truyen
- **Cost**: Rounds x Bytes per round

### 7.4 Stability
- **Variance**: Do dao dong acc qua cac rounds
- **Drift distance**: ||W_local - W_global||

## 8. Experimental Setup tieu chuan

### Datasets
- **FMNIST**: 10 classes, 28x28 grayscale
- **CIFAR-10**: 10 classes, 32x32 RGB
- **CIFAR-100**: 100 classes, 32x32 RGB

### Non-IID Splits
```python
# Dirichlet split
def dirichlet_split(dataset, num_clients, alpha):
    label_distribution = np.random.dirichlet(
        [alpha] * num_clients,
        num_classes
    )
    # Assign samples to clients based on distribution
    ...
```

### Models
- **Simple CNN**: 2 conv + 2 fc (~1M params)
- **ResNet-18**: Standard architecture (~11M params)

### Baselines
1. FedAvg (McMahan et al., 2017)
2. FedProx (Li et al., 2020)
3. SCAFFOLD (Karimireddy et al., 2020)
4. FedAdam (Reddi et al., 2021)
5. FedYogi (Reddi et al., 2021)

## 9. Lien he Nested Learning va FL

### Mapping
```
Nested Learning:
    Level 1: Slow parameters (pre-training)
    Level 2: Fast parameters (in-context)
    Level 3: Per-token updates

Federated Learning:
    Level 1: Global model (server)
    Level 2: Local model (client)
    Level 3: Per-batch updates

Combined (Fed-NL):
    Level 1: Server slow momentum
    Level 2: Server fast aggregation
    Level 3: Client momentum
    Level 4: Client per-batch updates
```

### Key Insight
FL tu nhien la **nested optimization**:
- Outer: Server toi uu global model
- Inner: Client toi uu local model

Nested Learning cho cong cu de thiet ke tot hon cho moi level.
