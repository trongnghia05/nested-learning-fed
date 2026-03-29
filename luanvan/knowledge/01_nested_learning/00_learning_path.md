# Learning Path: Nested Learning

## Muc tieu
Sau khi hoan thanh, ban se hieu:
1. Nested Learning la gi va tai sao quan trong
2. Associative Memory va vai tro cua no
3. Update Frequency va multi-level structure
4. Cach ap dung vao Federated Learning

---

## Buoc 1: Hieu van de cua Deep Learning hien tai

### Cau hoi can tra loi:
- Tai sao stack nhieu layers khong phai luc nao cung tot?
- LLM hien tai co han che gi?

### Doc:
- Paper Section 1: Introduction (google_papers/Nested_Learning/Nested_Learning.md)

### Key insight:
```
Van de cua LLM:
1. Static sau pre-training (khong hoc them duoc)
2. Chi co in-context learning trong context window
3. Tuong tu "anterograde amnesia" - chi nho qua khu xa va hien tai gan

=> Can paradigm moi cho phep CONTINUAL LEARNING
```

---

## Buoc 2: Associative Memory - Nen tang

### Cau hoi can tra loi:
- Associative Memory la gi?
- Tai sao neural network la associative memory?
- Learning vs Memorization khac nhau the nao?

---

### 2.1 Associative Memory la gi?

**Dinh nghia don gian:**
Associative Memory la mot **bo nho lien ket** - no nho moi quan he giua cac cap (key, value).

**Vi du thuc te:**
```
Nao nguoi:
    Key: Mui banh mi      ->  Value: Nho ve tiem banh gan nha
    Key: Giai dieu bai hat ->  Value: Nho ve nguoi yeu cu
    Key: Khuon mat ban    ->  Value: Nho ten ban do

=> Nao "lien ket" mot input voi mot output tuong ung
```

**Dinh nghia toan hoc:**
```
Associative Memory M la operator: Keys -> Values

M* = argmin_M  L(M(K); V)

Trong do:
- K: Tap keys (inputs)
- V: Tap values (outputs mong muon)
- L: Ham do do sai khac
- M: Bo nho (parameters)
```

**Vi du cu the:**
```
Cho K = {x1, x2, x3} va V = {y1, y2, y3}

Memory M hoc de:
    M(x1) ~ y1
    M(x2) ~ y2
    M(x3) ~ y3

=> M "nho" cac cap (xi, yi)
```

---

### 2.2 Tai sao Neural Network la Associative Memory?

**Xet 1-layer MLP don gian:**
```
Input x -> [W] -> Output y = W·x
```

**Khi training voi Gradient Descent:**
```
W_{t+1} = W_t - eta * nabla_W L(W_t; x_t)
        = W_t - eta * (nabla_y L) * x_t^T
```

**Viet lai nhu bai toan toi uu:**
```
W_{t+1} = argmin_W [ <W·x_t, nabla_y L> + regularization ]
```

**Day chinh la Associative Memory voi:**
```
Key   = x_t           (input hien tai)
Value = nabla_y L     (Local Surprise Signal - gradient theo output)
Memory = W            (weights cua network)

=> Neural network dang hoc: "Khi thay input x, output sai bao nhieu?"
=> Weights W LUU TRU moi lien ket nay
```

**Tai sao goi la "memory"?**
```
Truoc training:
    W = random -> M(x) = random output

Sau khi thay (x1, y1):
    W update -> M(x1) ~ y1

Sau khi thay (x2, y2):
    W update -> M(x1) ~ y1, M(x2) ~ y2

=> W "NHO" cac patterns da thay
=> Do la ly do goi la MEMORY
```

**Cac thanh phan khac cung la memory:**

| Component | Key | Value | Memory |
|-----------|-----|-------|--------|
| MLP weights | input x | surprise signal | W |
| Attention | query q | value v | KV cache |
| Momentum | gradient | weighted sum | m |
| Adam's v | gradient | squared gradient | v |

---

### 2.3 Learning vs Memorization khac nhau the nao?

**Dinh nghia (tu Neuropsychology):**
```
MEMORIZATION (Ghi nho):
    = Neural update gay ra boi input
    = Don gian la "luu thong tin vao memory"
    = CO THE huu ich hoac VO ICH

LEARNING (Hoc):
    = Qua trinh thu duoc memory HIEU QUA va HUU ICH
    = Khong chi nho, ma nho DUNG CACH
    = Generalize duoc sang data moi
```

**Vi du minh hoa:**
```
MEMORIZATION (xau):
    Hoc sinh nho: "2+3=5, 2+4=6, 2+5=7, ..."
    -> Nho tung phep tinh rieng le
    -> Gap 2+100 = khong biet!

LEARNING (tot):
    Hoc sinh hieu: "Cong nghia la dem them"
    -> Nam duoc QUY LUAT
    -> Gap 2+100 = 102 ✓
```

**Trong Machine Learning:**
```
MEMORIZATION:
    - Overfitting training data
    - Nho chinh xac tung sample
    - Test accuracy thap

LEARNING:
    - Generalization tot
    - Hoc duoc patterns
    - Test accuracy cao
```

**Trong Nested Learning context:**
```
Memory M dang MEMORIZE:
    M ghi nho cac cap (key, value)

Qua trinh LEARNING:
    Tim M* sao cho:
    1. M* nho duoc training data
    2. M* generalize duoc sang test data
    3. M* hieu qua (it parameters, nhanh)

=> Learning = Tim cach memorize THONG MINH
```

**Cong thuc the hien su khac biet:**
```
MEMORIZATION:
    M(k) = v  cho moi (k,v) trong training set
    -> Co the can M rat lon
    -> Co the khong generalize

LEARNING:
    M* = argmin_M [ L(M(K); V) + lambda·Complexity(M) ]
                                 ^^^^^^^^^^^^^^^^^^^^^
                                 Regularization!
    -> Tim M don gian nhung van chinh xac
    -> Generalize tot hon
```

---

### Bai tap tu kiem tra Buoc 2:
1. Viet lai cong thuc GD nhu bai toan optimization
2. Xac dinh key, value, memory trong linear attention
3. Cho vi du ve memorization vs learning trong thuc te

---

## Buoc 3: Local Surprise Signal (LSS)

### Cau hoi can tra loi:
- LSS la gi?
- Tai sao goi la "surprise"?
- LSS duoc dung nhu the nao trong NL?

### Dinh nghia don gian:

**Surprise Signal = Do "bat ngo" cua model khi thay mot input.**

```
Neu model du doan DUNG  -> It bat ngo  -> LSS nho
Neu model du doan SAI   -> Rat bat ngo -> LSS lon
```

### Dinh nghia toan hoc:

```
LSS = nabla_y L(W; x)
    = Gradient cua Loss theo OUTPUT (khong phai theo weights!)
```

**Vi du cu the:**
```
Input x = [1, 2, 3]
Model output y = W·x = [0.8]
Target (ground truth) = [1.0]

Loss L = (y - target)^2 = (0.8 - 1.0)^2 = 0.04

LSS = nabla_y L = 2(y - target) = 2(0.8 - 1.0) = -0.4
                                  ^^^^^^^^^^^^
                                  Sai so!
```

### Tai sao goi la "Surprise"?

**Lien he voi nao nguoi:**
```
Nao nguoi:
    - Thong tin BAT NGO -> Duoc nho lau hon
    - Thong tin BINH THUONG -> De quen

Vi du:
    - Ban di lam nhu moi ngay -> Khong nho gi dac biet
    - Hom nay co tai nan tren duong -> NHO RAT RO!
```

**Trong Neural Network:**
```
Model da biet pattern:
    Input quen thuoc -> Output dung -> Loss nho -> LSS nho
    -> Khong can update nhieu

Model gap pattern moi:
    Input la -> Output sai -> Loss lon -> LSS LON
    -> Can update manh de "nho" pattern moi nay
```

### LSS trong Gradient Descent

**Cong thuc day du:**
```
nabla_W L = nabla_y L  ·  x^T
            ^^^^^^^^^     ^^^
               LSS       Input

=> Gradient = (Do bat ngo) × (Input gay ra su bat ngo)
```

**Y nghia:**
```
Neu LSS = 0 (khong bat ngo):
    nabla_W L = 0 · x^T = 0
    -> Khong update weights
    -> Model da biet roi, khong can hoc them

Neu LSS lon (rat bat ngo):
    nabla_W L = (lon) · x^T = lon
    -> Update weights manh
    -> Model can hoc pattern moi nay
```

### Vi du truc quan - Classification voi 3 classes:

```
Input: Anh con meo
Target: [1, 0, 0] (class 0 = meo)

Case 1 - Model gioi:
    Output: [0.95, 0.03, 0.02]
    Loss: rat nho
    LSS = nabla_y L ~ [0.05, 0.03, 0.02]  <- Gan nhu khong bat ngo
    -> Update nhe

Case 2 - Model te:
    Output: [0.1, 0.8, 0.1]  (nghi la cho!)
    Loss: rat lon
    LSS = nabla_y L ~ [-0.9, 0.8, 0.1]  <- RAT BAT NGO!
    -> Update manh de sua sai
```

### LSS trong TITAN va Nested Learning

**Trong TITAN (paper ve neural memory):**
```
Surprise-based gating:
    if ||LSS|| > threshold:
        Update memory  (thong tin quan trong, can nho)
    else:
        Skip update    (thong tin binh thuong, bo qua)
```

**Trong Nested Learning:**
```
LSS dong vai tro "VALUE" trong Associative Memory:

Memory M hoc mapping:
    Key: input x
    Value: LSS (do bat ngo)

=> Model hoc: "Voi input x, toi da sai bao nhieu?"
=> Lan sau gap x tuong tu, model biet cach dieu chinh
```

### So sanh cac loai Gradient

| Gradient | Cong thuc | Y nghia |
|----------|-----------|---------|
| nabla_W L | Gradient theo weights | Cach update model |
| nabla_y L (LSS) | Gradient theo output | Do bat ngo |
| nabla_x L | Gradient theo input | Dung cho adversarial attacks |

### Tom tat LSS

```
LSS = nabla_y L = "Model sai bao nhieu o output?"

LSS nho -> Du doan dung -> Khong can hoc them
LSS lon -> Du doan sai  -> Can hoc manh

=> LSS quyet dinh CUONG DO hoc tap
=> Thong tin bat ngo duoc uu tien ghi nho
```

---

## Buoc 4: Update Frequency va Multi-level

### Cau hoi can tra loi:
- Update frequency la gi?
- Cac components duoc sap xep nhu the nao?
- Nested structure tao ra loi ich gi?

### Dinh nghia:

```
UPDATE FREQUENCY f_A:
    So lan component A duoc update tren 1 don vi thoi gian

ORDERING:
    A > B neu:
    1. f_A > f_B, HOAC
    2. f_A = f_B nhung B phu thuoc vao A
```

### Vi du trong Transformer:

```
Theo paper: "The higher the level is, the lower its frequency"

Level 1 (f CAO nhat):  Activations      - thay doi moi token (inner)
Level 2:               Attention scores - tinh moi token
Level 3:               Optimizer state  - update moi batch
Level 4 (f THAP nhat): MLP weights     - update moi step (outer)

=> Level THAP = frequency CAO = inner optimization
=> Level CAO = frequency THAP = outer optimization
=> Moi level co "context flow" rieng
=> Moi level co "objective" rieng
```

### Key insight:

```
DEEP LEARNING TRUYEN THONG:
    - Nhin nhu "stack of layers"
    - Tat ca update cung luc (backprop)

NESTED LEARNING:
    - Nhin nhu "nested optimization problems"
    - Moi level update voi tan so khac nhau
    - Mo ra kha nang thiet ke NHIEU LEVELS hon
```

---

## Buoc 5: Optimizer nhu Associative Memory

### Cau hoi can tra loi:
- Tai sao momentum la memory?
- Adam la gi trong NL framework?
- Lam sao thiet ke optimizer tot hon?

### GD + Momentum = 2-level optimization:

```
Theo paper: "Higher level = lower frequency"
            "A > B if computing B requires computing A first"

Level 1 (inner, tinh truoc): m_{t+1} = alpha*m_t + grad
Level 2 (outer, tinh sau):   W_{t+1} = W_t - m_{t+1}

=> m la "fast network" (level thap, inner)
=> W la "slow network" (level cao, outer, phu thuoc m)
=> Momentum la MEMORY cua gradients
=> No "nho" cac gradients truoc do
```

### Cong thuc nhu optimization:

```
m_{t+1} = argmin_m  -<m, grad> + eta||m - m_t||^2

=> Momentum dang toi uu bai toan:
   "Tim m gan voi m_t nhung cung gan voi grad moi"
```

### Adam trong NL:

```
Adam co 2 memory:
1. m (first moment): Nho gradients
2. v (second moment): Nho gradient magnitudes

=> Adam la 3-level optimization!
```

---

## Buoc 6: Ket noi voi Federated Learning

### Mapping:

```
Theo paper: "Higher level = lower frequency"

NESTED LEARNING:
    Level 3 (cao, cham): Pre-training weights
    Level 2:             Optimizer state
    Level 1 (thap, nhanh): In-context updates

FEDERATED LEARNING:
    Level 3 (cao, cham): Global model (server) - update moi round
    Level 2:             Aggregation
    Level 1 (thap, nhanh): Local training (client) - update moi batch

=> FL TU NHIEN la nested structure!
=> Co the ap dung NL techniques
```

### Y tuong cho luan van:

```
1. DGD: Ap dung adaptive decay cho local training
   - Giam client drift
   - Xu ly non-IID

2. M3: Ap dung multi-scale momentum
   - Fast momentum: Client level
   - Slow momentum: Server level
   - Newton-Schulz: Giam gradient conflict
```

---

## Checklist tu kiem tra

### Khai niem co ban:
- [ ] Giai thich duoc Associative Memory
- [ ] Giai thich duoc Local Surprise Signal
- [ ] Giai thich duoc Update Frequency

### Optimizer:
- [ ] Chung minh GD+Momentum la 2-level optimization
- [ ] Giai thich tai sao Adam co the coi la 3-level
- [ ] Hieu duoc Delta Rule va DGD

### Ket noi:
- [ ] Map duoc NL concepts sang FL
- [ ] Giai thich duoc y tuong Fed-DGD
- [ ] Giai thich duoc y tuong Fed-M3

---

## Tai lieu doc them

### Bat buoc:
1. `01_core_concepts.md` - File nay
2. `02_optimizers/01_delta_gradient_descent.md`
3. `02_optimizers/02_multi_scale_momentum_muon.md`

### Paper goc:
1. `google_papers/Nested_Learning/Nested_Learning.md` - Sections 1, 2
2. `google_papers/TITANs/TITANs.md` - Section 3 (Neural Memory)

### Code:
1. `src/nested_learning/optim/m3.py` - M3 implementation
