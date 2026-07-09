# KATS: Knowledge-Aware Tiered Stacking for Priority Triage Under Class Imbalance

Reproducible pipeline and results for the KATS manuscript (submitted to FGCS,
under major revision). This repository contains the full experiment code,
per-dataset results, and ablation studies referenced in the paper and the
response-to-reviewers letter.

## Datasets (5 total, 2 high-imbalance)

| Dataset | Rows (capped) | Features | IR | Domain |
|---|---|---|---|---|
| CloudTask | 6,000 | 14 | 1.00 | Cloud task scheduling (balanced negative control) |
| GoogleCluster | 60,000 | 18 | 1.95 | Cluster job scheduling |
| ITIncident | 24,918 | 12 | 34.61 | IT incident priority (impact/urgency excluded — leakage audit) |
| MultiCloud | 1,000 | 8 | 1.00 | Multi-cloud service placement (balanced negative control) |
| CICIDS2017 | 60,000 | 52 | 20.99 | Network intrusion detection |

All datasets pass a feature-leakage audit (balanced-accuracy threshold 0.75);
no single feature exceeds this threshold on any dataset — see
`results/leakage_audit_*.csv`.

## Key Findings

1. **SMOTE is a precision-recall balancing mechanism, not a pure recall
   booster.** On ITIncident (IR=34.6), removing SMOTE *increases* RecallHigh
   by ~0.14 but *decreases* MacroF1 by ~0.07 — without synthetic minority
   oversampling, the model over-predicts "High" indiscriminately.
2. **Asymmetric class weighting, not stacking, is KATS's dominant component.**
   Removing it costs 0.37 RecallHigh on ITIncident, the largest single
   ablation effect observed.
3. **KATS does not universally maximize RecallHigh.** On ITIncident and
   CICIDS2017 — the two real high-imbalance datasets — simpler baselines
   (LogReg, XGBoost) achieve higher raw RecallHigh; KATS instead offers a
   calibrated, tunable precision-recall trade-off. This is reported
   transparently in `results/cost_benefit_analysis_FINAL.csv`, including
   negative net-benefit cases.
4. Family-wise Holm-Bonferroni correction across all 35 McNemar tests confirms
   17 statistically significant KATS-vs-baseline differences, concentrated on
   ITIncident and CICIDS2017; GoogleCluster's near-ceiling performance yields
   no significant differences after correction.

## Repository Structure

```
/results
  E1_full_results_5datasets_FINAL.csv       # KATS + 7 baselines x 5 datasets, 5-seed averages
  M2_ablation_5datasets_FINAL.csv           # Component ablation (SMOTE, AsymLoss, CalibNB, Stacking)
  McNemar_Holm_corrected_5datasets.csv      # Pairwise significance, family-wise corrected
  cost_benefit_analysis_FINAL.csv           # SLA-penalty-based net benefit, 5-seed averaged
  temporal_split_robustness_FINAL.csv       # ITIncident: chronological vs random split
  scheduler_circularity_check_FINAL.csv     # GoogleCluster: with/without `scheduler` feature
  leakage_audit_*.csv                       # Per-dataset single-feature leakage audit
  dataset_summary.csv
/src
  KATS_master_pipeline_v9_SUBMISSION.py     # Full pipeline: loading, audit, E1, ablation, robustness checks
```

## Methodology Notes

- All models evaluated across 5 seeds (42, 7, 13, 99, 2026); reported metrics
  are 5-seed averages.
- SMOTE applied only when IR exceeds 3.0 (`IR_THRESHOLD`); asymmetric class
  weighting similarly gated by the same threshold to prevent artificial
  imbalance pressure on balanced datasets.
- KATS's High-class decision threshold is calibrated on an internal
  validation split carved from training data only (test set never touched),
  optimizing a precision-recall balance objective with a base-rate-scaled
  precision floor.
- `KATS_raw` (default 0.5 threshold) is reported alongside `KATS`
  (threshold-calibrated) in all results tables to isolate the contribution
  of threshold calibration from the base architecture.

## Reproducing Results

```bash
python src/KATS_master_pipeline_v9_SUBMISSION.py
```
Requires: scikit-learn, imbalanced-learn, lightgbm, xgboost, statsmodels, pandas, numpy.
Expected runtime: ~4-5 hours on Kaggle CPU (dominated by CICIDS2017 stacking fits).

## Citation

Manuscript under review at *Future Generation Computer Systems*. Citation
details will be added upon acceptance.
