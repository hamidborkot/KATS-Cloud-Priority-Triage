# KATS — Final Verified Results (All Scripts Complete)

> Last updated: May 2026 | Scripts A–H all complete | Ready for IEEE TCC writing

---

## Dataset Summary

| Dataset | Rows | Domain | Label Source | Regime |
|---|---|---|---|---|
| CloudTask | 6,000 | Cloud Scheduling | Simulated random {1,2,3} | Negative Control |
| GoogleCluster | 405,894 | Production Cluster | Borg priority bins | Ceiling |
| ITIncident | 24,918 | IT Operations | Incident severity mapping | Ceiling |
| MultiCloud | 1,000 | Multi-Cloud QoS | Composite score (QoS_Score removed) | Differentiation |

---

## E1 — Classification Results (7 models × 4 datasets, 5 seeds each)

### CloudTask (Negative Control — all Kappa ≈ 0 by design)

| Model | RecallH | PrecH | MacroF1 | Kappa | AUC-ROC |
|---|---|---|---|---|---|
| KATS | 0.2847±0.014 | 0.3214 | 0.3442 | 0.0159 | 0.5171 |
| B1-LogReg | 0.3599±0.017 | 0.3071 | 0.3278 | -0.0025 | 0.5008 |
| B2-DecTree | 0.3326±0.017 | 0.3123 | 0.3403 | 0.0103 | 0.5054 |
| B3-RF | 0.2585±0.014 | 0.3334 | 0.3441 | 0.0253 | 0.5183 |
| B4-LGB | 0.2836±0.029 | 0.3213 | 0.3454 | 0.0180 | 0.5166 |
| B5-MLP | 0.2841±0.053 | 0.2921 | 0.3314 | -0.0011 | 0.5063 |
| B6-XGBoost | 0.6981±0.020 | 0.3116 | 0.2976 | 0.0177 | 0.5221 |

> XGBoost highest RecallH (0.698) but lowest MacroF1 (0.298) = prediction bias, not learning. Validates negative control.

### GoogleCluster (Ceiling)

| Model | RecallH | MacroF1 | Kappa | AUC-ROC |
|---|---|---|---|---|
| KATS | 1.0000±0.000 | 0.9999 | 0.9998 | 1.0000 |
| B1-LogReg | 0.8849±0.024 | 0.9007 | 0.8597 | 0.9711 |
| B2-DecTree | 0.9994±0.000 | 0.9996 | 0.9993 | 0.9997 |
| B3-RF | 0.9999±0.000 | 0.9999 | 0.9998 | 1.0000 |
| B4-LGB | 0.9999±0.000 | 0.9999 | 0.9998 | 1.0000 |
| B5-MLP | 0.9987±0.000 | 0.9991 | 0.9987 | 1.0000 |
| B6-XGBoost | 0.9999±0.000 | 0.9999 | 0.9998 | 1.0000 |

### ITIncident (Ceiling)

| Model | RecallH | MacroF1 | Kappa | AUC-ROC |
|---|---|---|---|---|
| KATS | 1.0000±0.000 | 0.9997 | 0.9996 | 1.0000 |
| B1-LogReg | 1.0000±0.000 | 1.0000 | 1.0000 | 1.0000 |
| B2-DecTree | 1.0000±0.000 | 1.0000 | 1.0000 | 1.0000 |
| B3-RF | 1.0000±0.000 | 0.9991 | 0.9986 | 1.0000 |
| B4-LGB | 1.0000±0.000 | 1.0000 | 1.0000 | 1.0000 |
| B5-MLP | 0.9985±0.003 | 0.9990 | 0.9986 | 0.9993 |
| B6-XGBoost | 1.0000±0.000 | 0.9997 | 0.9996 | 1.0000 |

### MultiCloud (Differentiation — key dataset)

| Model | RecallH | MacroF1 | Kappa | AUC-ROC |
|---|---|---|---|---|
| KATS | 0.9218±0.029 | 0.8922 | 0.8380 | 0.9806 |
| B1-LogReg | 0.9067±0.032 | 0.8853 | 0.8290 | 0.9734 |
| B2-DecTree | 0.7532±0.044 | 0.7070 | 0.5591 | 0.7795 |
| B3-RF | 0.8463±0.018 | 0.8017 | 0.7030 | 0.9349 |
| B4-LGB | 0.8794±0.032 | 0.8351 | 0.7540 | 0.9604 |
| **B5-MLP** | **0.9578±0.011** | **0.9549** | **0.9325** | **0.9960** |
| B6-XGBoost | 0.8975±0.026 | 0.8348 | 0.7540 | 0.9567 |

> MLP beats KATS on MultiCloud (honest limitation — documented in Discussion).

---

## McNemar's Test Summary (KATS vs each baseline)

| Dataset | Model | p-value | Sig | Direction |
|---|---|---|---|---|
| CloudTask | B1-LogReg | 1.35e-03 | ** | KATS_BETTER |
| CloudTask | B3-RF | 1.61e-03 | ** | BASE_BETTER |
| CloudTask | B6-XGBoost | 4.03e-03 | ** | KATS_BETTER |
| GoogleCluster | B1-LogReg | ~0 | *** | KATS_BETTER |
| GoogleCluster | B2-DecTree | ~0 | *** | KATS_BETTER |
| GoogleCluster | B5-MLP | ~0 | *** | KATS_BETTER |
| GoogleCluster | B6-XGBoost | 1.64e-02 | * | KATS_BETTER |
| ITIncident | All | ns | — | All at ceiling |
| MultiCloud | B2-DecTree | ~0 | *** | KATS_BETTER |
| MultiCloud | B3-RF | 9.44e-15 | *** | KATS_BETTER |
| MultiCloud | B4-LGB | 2.59e-07 | *** | KATS_BETTER |
| MultiCloud | B5-MLP | 2.05e-09 | *** | BASE_BETTER |
| MultiCloud | B6-XGBoost | 4.12e-07 | *** | KATS_BETTER |

---

## Calibration (Brier + ECE + AUC-ROC)

### Key corrections vs. earlier reports:
- CloudTask ECE: RF=0.038 BEST (not KATS=0.069 — previous claim was wrong)
- MultiCloud: MLP ECE=0.027 < KATS ECE=0.032
- KATS Brier best on CloudTask (0.228 ≈ random-guess level, confirms NC)

---

## E3 Survivability

### CloudTask (bandwidth collapse, 5 seeds)
| Scenario | KATS | B4-LGB | B0-Oracle | B_Random |
|---|---|---|---|---|
| S1 (65%) | 1.000 | 1.000 | 1.000 | — |
| S2 (40%) | 0.941 | 0.948 | — | — |
| S3 (15%) | 0.505 | — | 0.485 | ~0.14 |

> KATS S3=0.505 > Oracle S3=0.485 (+1.96pp): probabilistic ranking beats oracle discrete metadata.

### ITIncident (tight thresholds)
| Scenario | KATS | All ML | EDF-Urgency |
|---|---|---|---|
| S1 (5%) | 1.000 | 1.000 | 0.000 |
| S2 (3%) | 1.000 | 1.000 | 0.000 |
| S3 (2%) | 0.735 | 0.735 | 0.000 |

> All ML models equally rescue 73.5% of High incidents at crisis level vs EDF=0% (+73.5pp).

---

## Ablation (Wilcoxon signed-rank, one-sided, 5 seeds)

### CloudTask
| Variant | ΔRecallH | Wilcoxon p | Sig |
|---|---|---|---|
| T_NoSMOTE | -0.302 | 0.031 | * |
| T_NoAsymLoss | +0.001 | ns | — |
| T_NoCalibNB | +0.001 | ns | — |
| T_NoStacking | +0.126 | 0.031 | * (bias) |

### MultiCloud
| Variant | ΔKappa | Wilcoxon p | Sig |
|---|---|---|---|
| T_NoSMOTE | -0.012 | ns | — |
| T_NoAsymLoss | +0.005 | ns | — |
| T_NoCalibNB | -0.039 | 0.031 | * |
| T_NoStacking | -0.081 | 0.031 | * |

> Pattern: SMOTE critical for imbalanced (CloudTask), Stacking+CalibNB critical for complex boundaries (MultiCloud).

---

## Hyperparameter Grid Search (Script E)

| Parameter | Search Space | Selected | Criterion |
|---|---|---|---|
| α (High weight) | {2,3,5,7,10} | α=3 (MultiCloud), α=5 (CloudTask) | Max RecallH on 3-fold val-CV |
| LGB lr | {0.01, 0.05, 0.10} | 0.05 | Stability (lr=0.10 gives Δ=0.0013) |
| LGB n_estimators | {100, 300, 500} | 500 | Max MacroF1 on val-CV |
| RF n_estimators | {100, 200, 300} | 300 | Stable saturation |
| MLP hidden layers | {(64,32),(128,64,32)} | (128,64,32) | Max MacroF1 on val-CV |
| Stack CV folds | {3, 5} | 5 | Standard practice |

---

## SHAP Feature Analysis (Script D)

| Dataset | SHAP-Spearman ρ | p-value | Interpretation |
|---|---|---|---|
| CloudTask | 0.3561 | 0.135 (ns) | Non-linear signal only; confirms negative control |
| GoogleCluster | 0.5707 | 0.013 (*) | High alignment — SHAP confirms Spearman |
| ITIncident | 0.8000 | 0.003 (**) | Very high alignment — strong XAI evidence |
| MultiCloud | 0.7253 | 0.005 (**) | High alignment — SHAP confirms Spearman |

---

## Paper Limitations (to state openly in Discussion)

1. MLP beats KATS on MultiCloud (κ=0.933 vs 0.838) — small balanced data favors MLP
2. RF beats KATS on CloudTask (McNemar p=0.0016) — but both κ≈0.03, noise-level
3. ITIncident E3 cannot differentiate ML models (all recall=1.000)
4. KATS training cost: ~18 min on GoogleCluster vs 37 sec for single LGB
