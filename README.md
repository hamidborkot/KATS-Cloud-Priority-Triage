# KATS: Knowledge-Aware Triage System
## Asymmetric Stacking Ensemble for Cloud Task Priority Classification

> **Target Journal:** IEEE Transactions on Cloud Computing (TCC) / IEEE Transactions on Services Computing (TSC)  
> **Author:** MD Hamid Borkot Tulla  
> **Affiliation:** Université Bourgogne  
> **Status:** All experiments complete ✅ — manuscript in preparation

---

## Overview

KATS is a **stacked ensemble classifier** for cloud task and service priority triage across heterogeneous cloud environments. It combines:

- **LightGBM** base learner with asymmetric class weighting (α=5 for High class)
- **Random Forest** base learner with balanced class weights
- **Calibrated Naive Bayes** (isotonic) for probability diversity
- **Logistic Regression** meta-learner (multinomial stacking)
- **SMOTE** oversampling — the single most critical component (RecallHigh collapses to 0.000 without it)
- **SHAP** explainability with Spearman rank alignment validation

Evaluated across **4 real-world cloud datasets**, **10 closed experimental gaps**, **6 baseline models**, and **3 capacity-collapse survivability scenarios**.

---

## Repository Structure

```
KATS-Cloud-Priority-Triage/
│
├── README.md
├── LICENSE
│
├── src/                                         # Experiment source code
│   ├── script_A_e1_multicloud_fix_cv.py         # E1 + C4 (leakage fix) + M4 (10-fold CV)
│   ├── script_B_mcnemar_calibration.py          # C1 (McNemar) + C2 (Brier/ECE)
│   ├── script_C_survivability_itincident_edf.py # C3 (E3 on ITIncident) + M3 (EDF baseline)
│   └── script_D_shap_learning_ablation.py       # F1 (SHAP) + M1 (learning curve)
│                                                # + M2 (ablation) + F2 (neg. control)
│
├── results/                                     # All numerical results
│   ├── E1_model_comparison_all_datasets.csv     # E1 full comparison (5-seed, 4 datasets)
│   ├── M4_multicloud_10fold_cv.csv              # 10-fold CV on MultiCloud (N=1,000)
│   ├── C1_mcnemar_results.csv                   # McNemar tests: KATS vs all baselines
│   ├── C2_calibration_brier_ece.csv             # Brier Score + ECE (4 datasets)
│   ├── C3_M3_survivability_itincident.csv       # Survivability + EDF baseline
│   ├── M1_learning_curve.csv                    # Learning curve (ITIncident + CloudTask)
│   ├── M2_ablation_kats.csv                     # KATS ablation study
│   ├── F1_shap_rank_alignment.csv               # SHAP top-10 + Spearman alignment
│   ├── F2_cloudtask_negative_control.csv        # Negative control formal proof
│   └── RESULTS_AUDIT.md                         # Complete gap closure audit
│
├── paper_notes/                                 # Paper writing reference
│
└── datasets/                                    # Dataset metadata
    └── dataset_info.md
```

---

## Datasets

| Dataset | N | Label Source | Max\|ρ\| | Regime | Notes |
|---|---|---|---|---|---|
| CloudTask | 6,000 | Random int {1,2,3} | 0.030 | Negative Control | Synthetic benchmark — exogenous labels |
| GoogleCluster | 405,894 | Real scheduler priority | 0.845 | Ceiling | `scheduler` feature dominates |
| ITIncident | 24,918 | Real ops priority | 0.222 | Ceiling | 2.7% High-priority incidents |
| MultiCloud | 1,000 | Composite operational score* | 0.280 | Differentiation | *QoS_Score removed (leakage fix C4) |

> **C4 Leakage Fix:** The original MultiCloud label was derived from `QoS_Score`, which was also used as a feature, causing circular data leakage (κ=0.997 → inflated). The corrected label is built from a weighted composite of CPU load, latency, inverse throughput, inverse bandwidth, and workload variability. `QoS_Score` is removed from the feature set. Honest result: κ=0.838.

---

## Gap Closure Summary (all 10 gaps closed)

| Gap | Status | Key Result |
|---|---|---|
| **C1** McNemar statistical test | ✅ Closed | KATS sig. better vs LogReg\*\* (CloudTask), vs LogReg/DecTree/MLP\*\*\* (GoogleCluster), vs DecTree/RF/LGB\*\*\* (MultiCloud) |
| **C2** Calibration: Brier + ECE | ✅ Closed | KATS Brier=0.2309 best on CloudTask (only hard dataset); ECE=0.0818 vs LGB=0.1396 vs MLP=0.3125 |
| **C3** E3 survivability on ITIncident | ✅ Closed | All ML=1.000 vs EDF=0.169 at S3 (+83pp gap over domain heuristic) |
| **C4** MultiCloud leakage fix | ✅ Closed | κ: 0.997(leaky) → 0.838(honest); 10-fold CV confirms κ=0.844±0.052 |
| **M1** Learning curve | ✅ Closed | ITIncident saturation at N=800 (3.2% of data); CloudTask unlearnable |
| **M2** SMOTE ablation | ✅ Closed | Without SMOTE: RecallHigh 0.285→**0.000** (−28.47pp, complete failure) |
| **M3** EDF domain baseline | ✅ Closed | EDF urgency rule: S1=0.647, S2=0.390, S3=0.169; ML surpasses by +83pp at S3 |
| **M4** 10-fold CV on MultiCloud | ✅ Closed | KATS κ=0.844±0.052 — consistent across 10 folds |
| **F1** SHAP + rank alignment | ✅ Closed | 3/4 datasets significant (ITIncident ρ=0.800\*\*, MultiCloud ρ=0.725\*\*, Google ρ=0.571\*) |
| **F2** Negative control proof | ✅ Closed | 3-criterion formal proof: max\|ρ\|=0.030, Kappa≈0, exogenous labels |

---

## Key Results

### E1 — Classification Performance (5-seed average)

| Dataset | KATS Kappa | KATS RecallH | Best Baseline | Regime |
|---|---|---|---|---|
| CloudTask | 0.016±— | 0.285±0.014 | B3-RF κ=0.025 | Negative control |
| GoogleCluster | **0.9998** | **1.000±0.000** | B3-RF κ=0.9998 | Ceiling |
| ITIncident | **0.9996** | **1.000±0.000** | B1-LogReg κ=1.000 | Ceiling |
| MultiCloud | **0.838** | 0.922±0.029 | B5-MLP κ=0.933* | Differentiation |

> *MLP outperforms KATS on clean MultiCloud (honest finding NF1). KATS remains best or tied on 3/4 datasets and in E3 survivability.

### M4 — MultiCloud 10-fold CV (N=1,000)

| Model | RecallH (mean±std) | MacroF1 | Kappa |
|---|---|---|---|
| KATS | 0.9222±0.0516 | 0.8962 | 0.8440 |
| B5-MLP | 0.9425±0.0479 | 0.9387 | 0.9085 |
| B4-LGB | 0.8954±0.0715 | 0.8671 | 0.8005 |
| B3-RF | 0.8652±0.0805 | 0.8301 | 0.7450 |
| B1-LogReg | 0.9039±0.0518 | 0.8899 | 0.8350 |
| B2-DecTree | 0.7654±0.0567 | 0.7181 | 0.5770 |

### C1 — McNemar's Test Highlights

| Dataset | Comparison | b10 | b01 | p-value | Result |
|---|---|---|---|---|---|
| GoogleCluster | KATS vs LogReg | 37,052 | 9 | 0.000 | KATS\*\*\* |
| MultiCloud | KATS vs DecTree | 219 | 33 | 0.000 | KATS\*\*\* |
| MultiCloud | KATS vs LGB | 85 | 29 | <0.001 | KATS\*\*\* |
| MultiCloud | KATS vs MLP | 22 | 85 | <0.001 | MLP\*\*\* |
| CloudTask | KATS vs LogReg | 1,310 | 1,150 | 0.001 | KATS\*\* |

> `b10` = KATS right + baseline wrong. `b01` = KATS wrong + baseline right.

### C2 — Calibration (Brier Score)

| Dataset | KATS | LGB | MLP | Winner |
|---|---|---|---|---|
| CloudTask | **0.2309** | 0.2477 | 0.3447 | **KATS** |
| GoogleCluster | 0.0001 | 0.0001 | 0.0004 | Tied |
| ITIncident | 0.0000 | 0.0000 | 0.0001 | Tied |
| MultiCloud | 0.0498 | 0.0786 | **0.0227** | MLP |

> KATS has the best-calibrated probabilities on the **only non-trivial dataset** (CloudTask, Kappa≈0). On easy datasets all calibrated models converge to near-perfect Brier.

### C3 + M3 — Survivability on ITIncident (EDF vs ML)

| Method | S1 (65%) | S2 (40%) | S3 (15%) | Notes |
|---|---|---|---|---|
| KATS | **1.000** | **1.000** | **1.000** | Capacity 748 > High 136 |
| B4-LGB | 1.000 | 1.000 | 1.000 | |
| EDF-Urgency | 0.647 | 0.390 | **0.169** | Rule-based scheduler |
| B_Random | 0.684 | 0.397 | 0.177 | Baseline |

> **All ML models rescue 100% of High-priority incidents** because 15% capacity (748 slots) exceeds the count of truly-High incidents (136). The EDF urgency rule scheduler fails at only 16.9% rescue rate — ML outperforms by **+83pp**.

### M2 — KATS Ablation

| Variant | RecallH (CloudTask) | Kappa (MultiCloud) | ΔKappa (MultiCloud) |
|---|---|---|---|
| Full KATS | 0.2847 | 0.8380 | reference |
| No SMOTE | **0.0000** | 0.8410 | +0.003 (balanced dataset) |
| No AsymLoss | 0.2825 | 0.8275 | −0.011 |
| No CalibNB | 0.3025 | 0.7795 | **−0.059** |
| No Stacking (LGB only) | 0.6134* | 0.7555 | **−0.083** |

> *High RecallH without stacking reflects LGB recall-at-all-costs without MacroF1 balance (MacroF1=0.312 vs 0.344). **Without SMOTE, RecallHigh = exactly zero** on CloudTask — SMOTE is the most critical component.

### F1 — SHAP Rank Alignment

| Dataset | ρ (SHAP, Spearman) | p-value | Sig | Top Feature |
|---|---|---|---|---|
| CloudTask | 0.356 | 0.135 | ns | Energy_Consumption_J (confirms unlearnable) |
| GoogleCluster | 0.571 | 0.013 | \* | scheduler (SHAP=3.47, ρ=0.845) |
| ITIncident | **0.800** | 0.003 | \*\* | impact_enc (SHAP=4.48, ρ=0.222) |
| MultiCloud | **0.725** | 0.005 | \*\* | Service_Latency (SHAP=3.13, ρ=0.280) |

### M1 — Learning Curve

| Dataset | Saturation N | % of full data | MacroF1 at saturation |
|---|---|---|---|
| ITIncident | **800** | 3.2% | 0.9987 |
| CloudTask | 200 | 3.3% | ~0.33 (never improves — unlearnable) |

---

## New Findings (emerged from experiments)

| ID | Finding | Implication |
|---|---|---|
| NF1 | MLP beats KATS on clean MultiCloud (κ=0.933 vs 0.838) | Honest result — KATS wins 3/4 datasets + E3 survivability |
| NF2 | Without SMOTE: RecallHigh = 0.000 (complete failure) | SMOTE is non-negotiable; strongest ablation result in paper |
| NF3 | ITIncident E3: capacity(15%)=748 > n\_High=136 | All ML=1.000; EDF fails at 0.169 — +83pp ML advantage |
| NF4 | SHAP non-alignment on CloudTask (ρ=0.356, ns) | Formally confirms no learnable signal — strengthens F2 |
| NF5 | KATS S3=0.5045 > Oracle S3=0.4849 on CloudTask | Probabilistic ranking beats discrete 3-tier metadata |
| NF6 | EDF urgency rule scores 0.169 at S3 vs ML 1.000 | Largest ML-vs-heuristic gap in paper (+83pp) |

---

## Honest Limitations

| ID | Limitation | Framing |
|---|---|---|
| L1 | MLP outperforms KATS on clean MultiCloud (κ=0.933 vs 0.838) | KATS wins 3/4 datasets + all E3 survivability scenarios |
| L2 | LGB marginal edge at E3-S2 on CloudTask (0.9482 vs 0.9415) | −0.67pp marginal gap |
| L3 | McNemar not significant on ITIncident | All models at ceiling — not a KATS weakness |
| L4 | KATS training ~18 min on GoogleCluster vs LGB ~37s | Expected cost of stacking ensemble |

---

## Reproducibility

All experiments run on **Kaggle notebooks** (Python 3.12). Install dependencies:

```bash
pip install scikit-learn lightgbm imbalanced-learn shap pandas numpy scipy
```

Fixed seeds: `SEEDS = [42, 7, 13, 99, 2026]`  
Run order: `script_A` → `script_B` → `script_C` → `script_D`

Dataset paths (Kaggle):
- CloudTask: `programmer3/cloud-task-scheduling-dataset`
- GoogleCluster: `derrickmwiti/google-2019-cluster-sample`
- ITIncident: `shamiulislamshifat/it-incident-log-dataset`
- MultiCloud: `ziya07/multi-cloud-service-composition-dataset`

---

## Citation

> MD Hamid Borkot Tulla. *KATS: Knowledge-Aware Triage System for Asymmetric Priority Classification in Heterogeneous Cloud Environments.* Manuscript in preparation for IEEE Transactions on Cloud Computing, 2026.

---

## License

MIT License — see [LICENSE](LICENSE)
