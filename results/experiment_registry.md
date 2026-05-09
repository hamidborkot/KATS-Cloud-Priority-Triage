# KATS Experiment Registry — Final Locked State

All experiments are complete and results are frozen for paper writing.

## Experiment Summary

| Script | Purpose | Status | Key Output |
|--------|---------|--------|------------|
| A–D | E1 core classification, SHAP, E3 original | ✅ Complete | Baseline F1/RecallH |
| E | Hyperparameter grid (α, LGB lr/n_est) | ✅ Complete | α=5 selected |
| F | AUC-ROC + XGBoost + 5-seed full rerun | ✅ Complete | Main Table (7 models × 4 datasets) |
| G | McNemar (24 pairs) + Calibration (Brier, ECE) | ✅ Complete | p-values, reliability diagrams |
| H | ITIncident E3 + Wilcoxon ablation | ✅ Complete | EDF gap = 73.5pp |
| I | Computational cost (train time, latency, memory) | ✅ Complete | Cost Table |
| J | SHAP stability (Spearman ρ) + Scope of applicability | ✅ Complete | ρ table, boundary rule |
| K | SLA breach rate + MTTF + real SLA field (K2) | ✅ Complete | 100% SLA catch on ITIncident |

**Total: 11 distinct experiments across 4 datasets × 5 seeds × 7 models**

---

## Key Results

### E1 Classification (Script F)
- 7 models, 4 datasets, 5 seeds, metrics: RecallH, MacroF1, κ, AUC-ROC
- KATS achieves highest RecallH on ITIncident (IR=34.6): RecallH = 0.9997
- CloudTask confirmed as negative control: AUC-ROC ≈ 0.517 (random labels)

### Statistical Tests (Script G + H)
- McNemar: 24 pairwise comparisons, Bonferroni-corrected p < 0.05
- Wilcoxon ablation: SMOTE+weighting contribution confirmed significant
- EDF survivability gap on ITIncident: KATS = +73.5pp over EDF baseline

### Calibration (Script G)
- Brier score, Expected Calibration Error (ECE) for all 7 models
- KATS isotonic calibration layer improves ECE by ~12% over uncalibrated LGB

### Computational Cost (Script I)

| Model | CloudTask | GoogleCluster | ITIncident | MultiCloud |
|-------|-----------|---------------|------------|------------|
| KATS | 61.24s | 912.10s | 65.67s | 23.26s |
| B1-LogReg | 2.86s | 250.94s | 19.69s | 1.04s |
| B4-LGB | 1.82s | 32.74s | 2.45s | 0.48s |
| B5-MLP | 4.83s | 185.58s | 8.28s | 0.50s |

KATS inference latency: max 0.29 ms/sample — acceptable for batch scheduling workflows.

### SHAP Stability (Script J)

| Dataset | Mean ρ | ±Std | Top-5 Agree | Stability |
|---------|--------|------|-------------|----------|
| CloudTask | 0.6525 | 0.098 | 48% | LOW (negative control) |
| GoogleCluster | 0.9554 | 0.013 | 80% | HIGH ✅ |
| ITIncident | 0.9373 | 0.044 | 76% | HIGH ✅ |
| MultiCloud | 0.9736 | 0.016 | 100% | HIGH ✅ |

**ITIncident top features (identical across all 5 seeds):** impact → urgency → made_sla  
**MultiCloud top features (identical across all 5 seeds):** Service_Latency → CPU_Utilization → Throughput

### Scope of Applicability (Script J)

| Dataset | n | IR | KATS_F1 | MLP_F1 | ΔF1 | Winner |
|---------|---|-----|---------|--------|-----|--------|
| MultiCloud | 1,000 | 1.0 | 0.8922 | 0.9549 | −0.063 | MLP |
| CloudTask | 6,000 | 1.3 | 0.3442 | 0.3314 | +0.013 | KATS |
| GoogleCluster | 405,894 | 1.9 | 0.9999 | 0.9991 | +0.001 | TIE |
| ITIncident | 24,918 | 34.6 | 0.9997 | 0.9990 | +0.001 | TIE |

**Deployment boundary rule:**
- IR > 10:1 → KATS recommended (RecallH + SLA advantage)
- n > 25,000 and balanced → LGB sufficient
- n < 5,000 and IR < 3:1 → MLP or LGB preferred

### SLA Breach / Dependability (Script K)

- **K1:** ITIncident — KATS SLA breach rate = 0.00%, MTTF = 4,984 (perfect)
- **K2 (GOLD result):** 671 of 678 High-priority incidents historically breached SLA (99%)  
  KATS catches **100%** of these before breach in test set  
  Operational claim: *zero High-priority tasks misrouted to standard queues*
- CloudTask SLA numbers not used (random labels = meaningless breach rates)

---

## Paper Target

**IEEE Transactions on Cloud Computing (TCC)**  
Impact Factor: 5.9 | Scope match: ✅ Exact  
Estimated length: 12–14 pages (double-column IEEE format)

## Writing Order

1. §5 Experimental Setup
2. §6 E1 Results (Main Table)
3. §4 Methodology (KATS architecture)
4. §9 Survivability + §11 Computational Cost
5. §10 Explainability (SHAP stability)
6. §7–§8 Statistical Tests + Calibration
7. §12 Discussion (scope boundary + negative control)
8. §3 Problem Statement
9. §2 Related Work
10. §1 Introduction + §13 Conclusion
