# FedProx: Federated Optimization with Proximal Term

> Paper: "Federated Optimization in Heterogeneous Networks" (Li et al., MLSys 2020)
> arXiv: https://arxiv.org/abs/1812.06127

## 1. Van de voi FedAvg

### FedAvg Standard
```
Client k: W_local = W_global - eta * sum(local_grads)
Server:   W_global = (1/K) * sum(W_local_k)
```

### Van de: Client Drift
- Moi client train tren **local data** (non-IID)
- Local model **drift xa** khoi global model
- Khi aggregate, cac local models **conflict** nhau
- Convergence cham hoac diverge

### Nguyen nhan
```
FedAvg objective (implicit):
    min_W  F_k(W)   # Chi toi uu local loss

-> Khong co gi ngan can W_local drift xa W_global
-> Cang nhieu local epochs, cang drift nhieu
```

## 2. FedProx - Y tuong

### Tu FedAvg sang FedProx
Them **proximal term** de phat (penalty) khi local model drift xa global model:

```
FedAvg:   min_W  F_k(W)
                 |_____|
                 local loss

FedProx:  min_W  F_k(W) + (mu/2) * ||W - W_global||^2
                 |_____|   |________________________|
                 local loss    PROXIMAL TERM
```

### Loi ich cua Proximal Term
1. **Giu local model gan global**: Penalty khi drift
2. **Stability**: Cho phep nhieu local epochs hon ma khong diverge
3. **Handles heterogeneity**: Hoat dong tot voi non-IID data
4. **Simple**: Chi them 1 hyperparameter (mu)

## 3. Cong thuc FedProx

### Local Objective
```
h_k(W; W^t) = F_k(W) + (mu/2) * ||W - W^t||^2

Trong do:
- F_k(W):       Local loss (cross-entropy, MSE, etc.)
- W^t:          Global model tai round t
- mu:           Proximal coefficient (hyperparameter)
- ||.||^2:      Squared L2 norm (Frobenius norm)
```

### Gradient cua Proximal Term
```
d/dW [ (mu/2) * ||W - W_global||^2 ]

= (mu/2) * d/dW [ sum_i (W_i - W_global_i)^2 ]

= (mu/2) * 2 * (W - W_global)

= mu * (W - W_global)
```

### Total Gradient
```
grad_total = grad_F_k(W) + mu * (W - W_global)
             |__________|   |__________________|
             gradient cua   gradient cua
             local loss     proximal term
```

### Update Rule
```
W = W - eta * grad_total
  = W - eta * (grad_F_k + mu * (W - W_global))
  = W - eta * grad_F_k - eta * mu * (W - W_global)
    |________________|   |_______________________|
    Gradient step        Proximal regularization
```

## 4. Phan tich Toan hoc

### Proximal Term la gi?
Proximal term `(mu/2) * ||W - W_global||^2` la mot dang **L2 regularization** voi reference point la `W_global` thay vi `0`.

```
L2 regularization:       (lambda/2) * ||W||^2        -> Keo W ve 0
Proximal regularization: (mu/2) * ||W - W_global||^2 -> Keo W ve W_global
```

### Tac dung
```
Khi W drift xa W_global:
    ||W - W_global||^2 LON
    -> Penalty LON
    -> Gradient mu*(W - W_global) LON
    -> Keo W ve gan W_global MANH

Khi W gan W_global:
    ||W - W_global||^2 NHO
    -> Penalty NHO
    -> Cho phep W di chuyen theo gradient cua local loss
```

### So sanh voi Weight Decay
```
Weight Decay:
    W = W - eta * grad - eta * lambda * W
    -> Keo W ve 0

FedProx:
    W = W - eta * grad - eta * mu * (W - W_global)
    -> Keo W ve W_global (khong phai 0)
```

## 5. Hyperparameter mu

### Vai tro cua mu
```
mu = 0:
    -> Proximal term = 0
    -> Giong het FedAvg
    -> Khong co regularization

mu nho (0.001 - 0.01):
    -> Regularization nhe
    -> Cho phep drift vua phai
    -> Tot cho mild non-IID

mu vua (0.01 - 0.1):
    -> Regularization vua
    -> Balance local learning va global consistency
    -> KHOI DAU VOI GIA TRI NAY

mu lon (0.1 - 1.0):
    -> Regularization manh
    -> Local model rat gan global
    -> Co the under-fit local data
    -> Tot cho severe non-IID
```

### Tuning Guidelines
```
1. Bat dau voi mu = 0.01
2. Neu accuracy thap, divergence -> tang mu
3. Neu accuracy khong tang, loss cao -> giam mu
4. So sanh voi FedAvg (mu=0) de validate
```

### Recommended Values
| Non-IID Level | Alpha (Dirichlet) | Recommended mu |
|---------------|-------------------|----------------|
| IID           | -                 | 0 (FedAvg)     |
| Mild          | 1.0               | 0.001 - 0.01   |
| Moderate      | 0.5               | 0.01 - 0.1     |
| Severe        | 0.1               | 0.1 - 1.0      |

## 6. Algorithm

### Pseudocode (Algorithm 2 from paper)
```
Algorithm: FedProx

Input:
    - K: So clients duoc chon moi round
    - T: So local epochs
    - R: So communication rounds
    - mu: Proximal coefficient
    - eta: Learning rate
    - W_0: Initial global model

For round r = 0, 1, ..., R-1:

    # 1. Server broadcasts
    Server gui W^r cho tat ca clients duoc chon

    # 2. Client local training
    For each client k in parallel:
        W_k <- W^r  # Initialize tu global

        For t = 1, 2, ..., T (local epochs):
            For each batch (x, y) in local data:
                # Compute loss gradient
                g = gradient(F_k(W_k; x, y))

                # FedProx update (2 cach tuong duong)
                # Cach 1: Them vao loss
                loss = F_k + (mu/2) * ||W_k - W^r||^2
                g_total = gradient(loss)
                W_k = W_k - eta * g_total

                # Cach 2: Them vao update (EQUIVALENT)
                W_k = W_k - eta * g - eta * mu * (W_k - W^r)

        Send W_k to server

    # 3. Server aggregation (SAME AS FedAvg)
    W^(r+1) = (1/K) * sum(W_k)

Return W^R
```

### Key Points
1. **Local training khac FedAvg**: Them proximal term
2. **Aggregation giong FedAvg**: Weighted average
3. **W^r duoc giu nguyen**: Khong update trong qua trinh local training

## 7. Implementation

### Cach 1: Them vao Loss (Autograd)
```python
def train_fedprox_loss(model, global_model, dataloader, mu, lr):
    optimizer = SGD(model.parameters(), lr=lr)

    for x, y in dataloader:
        optimizer.zero_grad()

        # Forward
        y_pred = model(x)
        loss = F.cross_entropy(y_pred, y)

        # Add proximal term to loss
        proximal_term = 0.0
        for w, w_global in zip(model.parameters(), global_model.parameters()):
            proximal_term += torch.sum((w - w_global) ** 2)

        loss = loss + (mu / 2) * proximal_term

        # Backward (autograd tinh gradient cua ca proximal term)
        loss.backward()
        optimizer.step()
```

### Cach 2: Them vao Optimizer Step (Manual)
```python
class FedProxOptimizer(Optimizer):
    def __init__(self, params, lr, mu, global_params):
        self.mu = mu
        self.global_params = global_params
        super().__init__(params, dict(lr=lr))

    def step(self):
        for group in self.param_groups:
            lr = group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                global_p = self.global_params[name]

                # QUAN TRONG: Tinh proximal term TRUOC gradient step
                prox_term = p.data - global_p  # W - W_global

                # Update: W = W - lr*grad - lr*mu*(W - W_global)
                p.data.add_(grad, alpha=-lr)
                p.data.add_(prox_term, alpha=-lr * self.mu)
```

### Cach 1 vs Cach 2
| Aspect | Cach 1 (Loss) | Cach 2 (Optimizer) |
|--------|---------------|-------------------|
| Implementation | Them vao loss function | Them vao optimizer.step() |
| Gradient computation | Autograd | Manual |
| Khi nao tinh prox | Trong forward pass | Trong step() |
| Flexibility | Can sua train loop | Chi sua optimizer |
| Ket qua | **GIONG NHAU** | **GIONG NHAU** |

## 8. So sanh voi cac phuong phap khac

### FedProx vs FedAvg
```
FedAvg:
    W = W - eta * grad
    -> Khong co regularization
    -> De bi drift

FedProx:
    W = W - eta * grad - eta * mu * (W - W_global)
    -> Penalty khi drift
    -> Stable hon
```

### FedProx vs Fed-DGD (Drift Direction)
```
FedProx:
    Update: W = W - eta*grad - eta*mu*(W - W_global)
                              |___________________|
                              LINEAR penalty (tat ca huong)

Fed-DGD:
    k = normalize(W - W_global)  # Drift direction
    Update: W = W - eta*grad - eta*decay*(k . W)*k
                              |_______________|
                              PROJECTION decay (chi huong k)
```

| Aspect | FedProx | Fed-DGD |
|--------|---------|---------|
| Penalty type | Linear | Projection |
| Direction | Isotropic (tat ca) | Anisotropic (chi k) |
| Selective | Khong | Co |
| Memory | Chi W_global | W_global + k |
| Complexity | Don gian | Phuc tap hon |

### Geometric Interpretation
```
FedProx: KEO W ve phia W_global
    W_global <-------- W
              mu*(W - W_global)

    -> Keo TAT CA components cua W

Fed-DGD: DECAY W theo huong drift
    W --------k-------> (drift direction)
                |
                v
         chi decay thanh phan theo k

    -> Chi decay component SONG SONG voi k
    -> Giu nguyen component VUONG GOC voi k
```

## 9. Advantages va Disadvantages

### Advantages
```
+ Don gian: Chi them 1 term vao update
+ Hieu qua: Giam client drift ro rang
+ It hyperparameters: Chi co mu
+ Khong can extra memory: Chi luu W_global
+ Well-studied: Paper duoc cite nhieu (4000+)
+ Theoretical guarantees: Co convergence proof
```

### Disadvantages
```
- Isotropic: Penalty moi huong nhu nhau
  -> Khong selective nhu DGD

- Tuning mu: Can chon mu phu hop cho moi dataset

- Over-regularization: mu qua lon -> under-fit

- Khong co long-term memory: Nhu Fed-M3 (slow momentum)

- Communication cost: Giong FedAvg (khong giam)
```

## 10. Khi nao nen dung FedProx?

### Nen dung khi
```
1. Non-IID data moderate den severe
2. Can stability (khong bi diverge)
3. Muon simple baseline truoc khi thu advanced methods
4. Chua biet nen dung method nao
```

### Khong nen dung khi
```
1. IID data (FedAvg du tot)
2. Can selective forgetting (dung Fed-DGD)
3. Can long-term memory (dung Fed-M3)
4. Communication cost la bottleneck (FedProx khong help)
```

## 11. Experiment Setup

### Hyperparameters
```
mu:          0.01 (default), tune in {0.001, 0.01, 0.1, 1.0}
lr:          0.01
batch_size:  32
local_epochs: 5
```

### Command
```bash
python run_experiment.py --method fedprox --dataset cifar10 \
    --alpha 0.5 --fedprox-mu 0.01 --num-rounds 100
```

### Expected Results
| Method | IID | alpha=1.0 | alpha=0.5 | alpha=0.1 |
|--------|-----|-----------|-----------|-----------|
| FedAvg | +++ | ++ | + | - |
| FedProx | +++ | +++ | ++ | + |

## 12. References

```
[1] Li, T., Sahu, A. K., Zaheer, M., Sanjabi, M., Talwalkar, A., & Smith, V. (2020).
    Federated Optimization in Heterogeneous Networks.
    MLSys 2020.
    https://arxiv.org/abs/1812.06127

[2] McMahan, B., Moore, E., Ramage, D., Hampson, S., & y Arcas, B. A. (2017).
    Communication-Efficient Learning of Deep Networks from Decentralized Data.
    AISTATS 2017. (FedAvg paper)

[3] Official implementation:
    https://github.com/litian96/FedProx
```

---

*Cap nhat: 2026-04-04*