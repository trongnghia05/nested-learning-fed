# Knowledge Base - Luan Van Thac Si

## De tai: Nghien cuu phuong phap Federated Learning dua tren Nested Learning

**Hoc vien:** Mai Trong Nghia
**Huong dan:** TS. Tran Trong Hieu
**Co so:** Truong Dai hoc Khoa Hoc Tu Nhien, DHQG Ha Noi

---

## Muc luc Knowledge Base

### 1. Nested Learning Fundamentals
- `01_nested_learning/00_learning_path.md` - **BAT DAU TU DAY** - Lo trinh hoc
- `01_nested_learning/01_core_concepts.md` - Khai niem co ban ve Nested Learning
- `01_nested_learning/02_mathematical_foundations.md` - Co so toan hoc chi tiet
- `01_nested_learning/03_deep_learning_vs_nested_learning.md` - **So sanh DL vs NL** - Khac biet thuc su
- `01_nested_learning/04_titan_nested_learning_hope.md` - **TITAN, NL, HOPE** - Moi quan he chi tiet

### 2. Optimizers
- `02_optimizers/01_delta_gradient_descent.md` - Delta Gradient Descent (DGD)
- `02_optimizers/02_multi_scale_momentum_muon.md` - Multi-scale Momentum Muon (M3)

### 3. Federated Learning (chua hoan thanh)
- `03_federated_learning/01_basics.md` - FL co ban va FedAvg
- `03_federated_learning/02_non_iid_challenges.md` - Van de non-IID
- `03_federated_learning/03_existing_methods.md` - FedProx, SCAFFOLD, etc.

### 4. Proposed Methods
- `04_proposed_methods/01_fed_dgd.md` - Federated Delta Gradient Descent
- `04_proposed_methods/02_fed_m3.md` - Federated Multi-scale Momentum Muon

### 5. Code Implementation (DA CO)
- `05_code_implementation/01_paper_to_code_mapping.md` - Map paper -> code (chi tiet)
- `05_code_implementation/02_code_explanation_insights.md` - Giai thich va insights
- `05_code_implementation/03_critical_analysis_paper_vs_code.md` - **QUAN TRONG** - So sanh Paper vs Code, tim ra bugs/variants

### 6. Experiments (DA HOAN THANH)
- `06_experiments/00_experiment_overview.md` - **TONG QUAN** - Overview tat ca experiments
- `06_experiments/01_fed_m3_design.md` - **FED-M3** - Thiet ke chi tiet Fed-M3 algorithm
- `06_experiments/02_fed_dgd_design.md` - **FED-DGD** - Thiet ke chi tiet Fed-DGD algorithm
- `06_experiments/03_experiment_setup.md` - **SETUP** - Hardware, model, hyperparameters
- `06_experiments/04_non_iid_scenarios.md` - **NON-IID** - Cac kich ban: Pathological, Dirichlet, Combined

---

## Roadmap nghien cuu

### Phase 1: Literature Review (01-02/2026)
- [ ] Doc va hieu Nested Learning paper
- [ ] Doc va hieu TITAN paper
- [ ] Tong hop cac phuong phap FL hien co
- [ ] Phan tich DGD va M3 chi tiet

### Phase 2: Method Design (02-03/2026)
- [ ] Thiet ke Fed-DGD algorithm
- [ ] Thiet ke Fed-M3 algorithm
- [ ] Viet pseudo-code va convergence analysis sketch

### Phase 3: Implementation (03-04/2026)
- [ ] Implement Fed-DGD trong PyTorch
- [ ] Implement Fed-M3 trong PyTorch
- [ ] Setup FL simulation framework
- [ ] Tao non-IID data splits

### Phase 4: Experiments (04-05/2026)
- [ ] Chay thi nghiem tren FMNIST
- [ ] Chay thi nghiem tren CIFAR-10
- [ ] Ablation studies
- [ ] So sanh voi baselines

### Phase 5: Writing (05-06/2026)
- [ ] Viet chuong 1-2: Co so ly thuyet
- [ ] Viet chuong 3: Phuong phap de xuat
- [ ] Viet chuong 4: Thi nghiem
- [ ] Hoan thien va bao ve

---

## Key Research Questions

1. **DGD trong FL:**
   - Adaptive decay co giam client drift khong?
   - Nen aggregate decay explicitly hay implicitly?

2. **M3 trong FL:**
   - Multi-scale momentum co giu long-term information khong?
   - Newton-Schulz orthogonalization co giam gradient conflicts khong?

3. **So sanh:**
   - Fed-DGD vs Fed-M3: Phuong phap nao tot hon?
   - Trade-off giua accuracy va computation cost?

---

## Cac file quan trong trong repo

### Papers
- `google_papers/Nested_Learning/Nested_Learning.md` - Paper chinh
- `google_papers/TITANs/TITANs.md` - TITAN paper

### Code reference
- `src/nested_learning/optim/m3.py` - M3 implementation
- `src/nested_learning/optim/deep.py` - Deep optimizer variants

### Documentation
- `docs/PAPER_COMPLIANCE.md` - Fidelity notes
- `README.md` - Huong dan su dung repo

---

## Lien he nhanh

| Concept | File | Key formula |
|---------|------|-------------|
| Nested Learning | `01_core_concepts.md` | Multi-level optimization |
| DGD | `01_delta_gradient_descent.md` | W = W*(I-xx^T) - eta*grad |
| M3 | `02_multi_scale_momentum_muon.md` | Fast + Slow momentum |
| Fed-DGD | `01_fed_dgd.md` | DGD + FedAvg |
| Fed-M3 | `02_fed_m3.md` | M3 + Hierarchical FL |

---

## Notes & Updates

### 2026-03-28: Critical Analysis Complete

**PHAT HIEN QUAN TRONG:**

1. **M3 Optimizer dung Accumulation thay vi EMA**
   - Paper: `m = β*m + g` (EMA)
   - Code: `m = m + β*g` (Accumulation)
   - Chua ro day la bug hay intentional variant
   - Newton-Schulz normalize nen co the van work

2. **DeepMomentum vs M3 khac nhau**
   - DeepMomentum: `m = β*m + (1-β)*g` (Adam-style EMA)
   - M3: `m = m + β*g` (Accumulation)
   - 2 optimizer trong cung repo co cong thuc KHAC

3. **HOPE variants**
   - HOPESelfModBlock = paper-compliant (dung SelfModifyingTitans)
   - HOPEBlock = simplified (dung TitanMemory)
   - Chon dung variant khi implement FL

4. **TitanMemory la simplified version**
   - Thieu past surprise (momentum)
   - Thieu adaptive learning rates
   - Full version nam o SelfModifyingTitans

**Xem chi tiet:** `05_code_implementation/03_critical_analysis_paper_vs_code.md`
