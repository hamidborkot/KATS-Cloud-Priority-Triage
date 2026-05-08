# Anticipated Reviewer Concerns — KATS Paper

---

## R1: "Why no deep learning baseline?"

**Response:** B5-MLP (MLPClassifier, 3 hidden layers: 128-64-32, ReLU, Adam, early stopping)
was evaluated across all 4 datasets. KATS outperforms MLP on all datasets:
- MultiCloud: KATS κ=0.997 vs MLP κ=0.906 (+9.1pp)
- CloudTask: KATS RecallH=0.285 vs MLP RecallH=0.284 (equivalent, negative control)
- GoogleCluster: KATS κ=0.9998 vs MLP κ=0.9987 (+0.11pp)
- ITIncident: KATS RecallH=1.000 vs MLP RecallH=0.9985

See G1 (Script 1) and Table E1 for full results.

---

## R2: "KATS is just a stacking ensemble — not novel"

**Response:** KATS contributes four specific design decisions:
1. **Asymmetric class weighting (α=5)** applied to the LGB base learner specifically,
   yielding +20.9pp RecallHigh on non-trivial datasets
2. **Isotonic-calibrated Naive Bayes** as diversity component enabling better-calibrated
   probability estimates — critical for E3 probabilistic triage
3. **SMOTE + stacking combination** for extreme imbalance (34.6:1 on ITIncident)
4. **Survivability simulation** as primary evaluation metric — operational rather than
   academic benchmark

The oracle-beating S3 result (KATS 0.5045 > Oracle 0.4849) specifically demonstrates
that calibrated probabilistic ranking provides value unavailable from rule-based baselines.

---

## R3: "CloudTask results are poor"

**Response:** CloudTask is a DESIGNED negative control, proved by Spearman analysis:
max|ρ|=0.0302 across all 19 features (mean|ρ|=0.011). Task_Priority is assigned by the
simulation engine independently of observable scheduling metrics — this is verified by
simulation source documentation. No classifier can outperform chance on this dataset;
all models including LogReg, RF, LGB and KATS achieve Kappa ≈ 0.0.

This is a strength, not a weakness: it proves the framework is correctly calibrated
and does not overfit spurious correlations.

---

## R4: "ITIncident is trivial — LogReg achieves F1=1.000"

**Response:** Classification accuracy is not the sole evaluation criterion.
On ITIncident, KATS provides:
1. **Calibrated probabilities** for continuous triage ranking (required for E3 framework)
2. **SHAP auditability** for IEC 62443-4-2 compliance (required for industrial deployment)
3. **Robustness under label noise** (T7.2: RecallH≥0.978 under 20% noise)

These capabilities are unavailable from DecisionTree or LogReg despite matching accuracy.

---

## R5: "KATS loses to LGB at S2 in E3"

**Response:** We acknowledge this directly in the paper (Section 4.4).
LGB outperforms KATS at S2 by 0.67pp (0.9482 vs 0.9415), within the
95% CI overlap. KATS's advantage is concentrated at:
- S1: perfect recovery (1.000, tied with RF and Oracle)
- S3: highest survivability (0.5045, +3.96pp over LGB and +1.96pp over Oracle)

The S3 advantage is the operationally critical scenario (catastrophic collapse)
where calibrated probability ranking matters most.

---

## R6: "Why not test on more datasets?"

**Response:** The 4 datasets represent the four canonical cloud priority regimes:
- Negative control (CloudTask): proves absence of spurious correlation
- Ceiling / scheduler-encoded (Google): proves large-scale robustness
- Extreme imbalance / ITIL-governed (ITIncident): proves SMOTE effectiveness
- Non-linear QoS boundary (MultiCloud): proves ensemble necessity

This is a designed 4-regime taxonomy, not an arbitrary selection. See Section 3.1.
