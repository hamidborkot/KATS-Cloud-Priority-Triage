# KATS: Knowledge-Aware Triage System
## Asymmetric Stacking Ensemble for Cloud Task Priority Classification

> **Target Journal:** IEEE Transactions on Cloud Computing (TCC) / IEEE Transactions on Services Computing (TSC)  
> **Author:** MD Hamid Borkot Tulla  
> **Affiliation:** Université Bourgogne  
> **Status:** Experimental work complete — manuscript in preparation

---

## Overview

KATS is a **stacked ensemble classifier** for cloud task and service priority triage across heterogeneous cloud environments. It combines:

- **LightGBM** base learner with asymmetric class weighting (α=5 for High class)
- **Random Forest** base learner with balanced class weights
- **Calibrated Naive Bayes** for probability diversity
- **Logistic Regression** meta-learner (multinomial stacking)
- **SMOTE** oversampling for minority class handling
- **SHAP** explainability for IEC 62443-4-2 audit compliance

The system is evaluated across **4 real-world cloud datasets**, **7 experiments**, and **6 baseline models** including MLP (deep learning).

---

## Repository Structure

```
KATS-Cloud-Priority-Triage/
│
├── README.md                          # This file
├── LICENSE
│
├── src/                               # All experiment source code
│   ├── 00_preprocessing.py            # Dataset loading & preprocessing
│   ├── 01_E1_classification.py        # E1: In-distribution classification (5-seed)
│   ├── 02_E2_transfer.py              # E2: Cross-dataset semantic transfer
│   ├── 03_E3_survivability.py         # E3: Survivability simulation
│   ├── 04_E4_shap.py                  # E4: SHAP explainability + inference timing
│   ├── 05_E5_ablation.py              # E5: Ablation study (GoogleCluster + ITIncident)
│   ├── 06_E7_sensitivity.py           # E7: Sensitivity analysis
│   ├── 07_G1_mlp_baseline.py          # G1+G5: MLP baseline + training time table
│   ├── 08_G2G3_spearman.py            # G2+G3: Spearman correlation analysis
│   ├── 09_G6_inference_benchmark.py   # G6: Inference latency + extended E3
│   └── 10_G4_asymloss_evidence.py     # G4: Asymmetric loss formal evidence
│
├── results/                           # All numerical results
│   ├── E1_classification_results.md
│   ├── E2_transfer_results.md
│   ├── E3_survivability_results.md
│   ├── E4_shap_results.md
│   ├── E5_ablation_results.md
│   ├── E7_sensitivity_results.md
│   ├── G1_mlp_training_time.md
│   ├── G2G3_spearman_results.md
│   ├── G4_asymloss_evidence.md
│   └── G6_inference_benchmark.md
│
├── paper_notes/                       # Paper writing reference
│   ├── honest_analysis.md             # Complete honest gap analysis
│   ├── paper_ready_text.md            # Draft text for all sections
│   └── reviewer_prep.md              # Anticipated reviewer concerns + responses
│
└── datasets/                          # Dataset reference (not raw data)
    └── dataset_info.md                # Sources, sizes, preprocessing notes
```

---

## Datasets

| Dataset | Rows | Priority Distribution | Source |
|---|---|---|---|
| CloudTask Scheduling | 6,000 | High:1794, Med:2381, Low:1825 | [Kaggle: programmer3](https://www.kaggle.com/datasets/programmer3/cloud-task-scheduling-dataset) |
| Google Cluster Traces | 405,894 | High:156K, Med:165K, Low:85K | [Kaggle: derrickmwiti](https://www.kaggle.com/datasets/derrickmwiti/google-2019-cluster-sample) |
| IT Incident Log | 24,918 | Med:23466, Low:774, High:678 | [Kaggle: shamiulislamshifat](https://www.kaggle.com/datasets/shamiulislamshifat/it-incident-log-dataset) |
| Multi-Cloud Service | 1,000 | Balanced (333 each) | [Kaggle: ziya07](https://www.kaggle.com/datasets/ziya07/multi-cloud-service-composition-dataset) |

---

## Experiments

| ID | Experiment | Datasets | Key Finding |
|---|---|---|---|
| E1 | In-distribution classification | All 4 | 3 performance regimes identified; MLP beaten |
| E2 | Cross-dataset semantic transfer | All 4 | 7/12 pairs KATS>LGB; semantic inversion found |
| E3 | Survivability simulation | CloudTask | KATS beats oracle at S3 (+1.96pp) |
| E4 | SHAP + inference timing | GoogleCluster | 68.3 μs/pred; scheduler=top feature |
| E5 | Ablation study | Google + IT | T5.5 ΔKappa=−0.785 (feature eng. critical) |
| E7 | Sensitivity analysis | IT + CloudTask | RecallH≥0.978 under 20% label noise |
| G1+G5 | MLP baseline + training time | All 4 | KATS beats MLP everywhere |
| G2+G3 | Spearman correlation | All 4 | CloudTask max\|ρ\|=0.030 (negative control) |
| G4 | Asymmetric loss evidence | Cloud + MC | +21pp RecallH on LGB base learner |
| G6 | Inference benchmark + E3 ext | CloudTask | +36pp over random at S3 |

---

## Key Results Summary

### E1 — Classification Performance

| Dataset | KATS Kappa | KATS RecallH | vs LogReg | vs MLP | Regime |
|---|---|---|---|---|---|
| CloudTask | 0.016 | 0.285±0.014 | −0.017 | +0.017 | Negative control |
| GoogleCluster | 1.000 | 1.000±0.000 | +0.140 | +0.001 | Ceiling |
| ITIncident | 1.000 | 1.000±0.000 | =0.000 | +0.002 | Ceiling |
| MultiCloud | 0.997 | 0.994±0.007 | **+0.697** | **+0.091** | Differentiation ✓ |

### E3 — Survivability Under Bandwidth Degradation

| Scenario | KATS | B4-LGB | B3-RF | Oracle | Random | KATS vs Random |
|---|---|---|---|---|---|---|
| S1: 65% BW | 1.0000 | 0.9928 | 1.0000 | 1.0000 | 0.6483 | **+35.2pp** |
| S2: 40% BW | 0.9415 | 0.9482 | 0.9326 | 1.0000 | 0.3980 | **+54.4pp** |
| S3: 15% BW | **0.5045** | 0.4649 | 0.4532 | 0.4849 | 0.1444 | **+36.0pp** |

> 🏆 **KATS beats the oracle at S3** — probabilistic ranking outperforms discrete 3-tier metadata under catastrophic collapse.

### Computational Cost

| Model | Inference (μs/pred) | Training GoogleCluster |
|---|---|---|
| KATS | 68.3 | 1078s (18 min) |
| B4-LGB | 33.2 | 37s |
| B5-MLP | 1.2 | 121s |
| B1-LogReg | 0.1 | 258s |

---

## Reproducibility

All experiments run on **Kaggle GPU notebooks** (Python 3.10). Dependencies:

```bash
pip install scikit-learn lightgbm imbalanced-learn shap statsmodels pandas numpy
```

Fixed seeds: `SEEDS = [42, 7, 13, 99, 2026]`, `SEED = 42`

All results are reproducible by running scripts in order: `00` → `10`.

---

## Citation

> MD Hamid Borkot Tulla. *KATS: Knowledge-Aware Triage System for Asymmetric Priority Classification in Heterogeneous Cloud Environments.* Manuscript in preparation for IEEE Transactions on Cloud Computing, 2026.

---

## License

MIT License — see [LICENSE](LICENSE)
