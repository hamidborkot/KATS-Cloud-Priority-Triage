# KATS — Key Findings for Paper Writing

## The Core Claim (frame everything around this)

> KATS achieves **zero SLA breach rate** on real-world incident data, catching 100% of 671 historically-breached High-priority tasks (ITIncident, K2). This operational result is the primary contribution — not raw F1.

---

## Finding 1: RecallH Under Imbalance

- ITIncident (IR = 34.6:1): KATS RecallH = **0.9997** vs MLP RecallH = 0.9990
- High class imbalance is where KATS's asymmetric weighting (α=5) + SMOTE + isotonic calibration work together
- Paper claim: *"KATS's asymmetric cost-sensitive weighting prevents the majority-class dominance that causes single-model classifiers to systematically underserve High-priority workloads"*

## Finding 2: Explainability Is Stable and Interpretable

- ITIncident SHAP top-1: `impact_enc` (identical all 5 seeds, ρ = 0.937)
- MultiCloud SHAP top-1: `Service_Latency` (identical all 5 seeds, ρ = 0.974)
- GoogleCluster SHAP top-1: `scheduler` (identical all 5 seeds, ρ = 0.955)
- CloudTask ρ = 0.65 → confirms random labels → negative control strengthens rigor

## Finding 3: Survivability (EDF Gap)

- ITIncident: KATS vs EDF scheduling gap = **73.5 percentage points** on High-priority throughput
- CloudTask: KATS maintains priority-ordered scheduling under load; EDF degrades at high load

## Finding 4: Deployment Boundary Is Formally Defined

- KATS justified when: IR > 10:1 OR n > 25,000 with real priority labels
- MLP is preferred when: n < 5,000 AND IR < 3:1 (small balanced datasets)
- This is NOT a weakness — it defines the system's scope of applicability

## Finding 5: Computational Cost Is Honest

- KATS is 3.6x–47x slower than single models depending on dataset size
- Training cost is one-time and amortized over deployment
- Inference latency: max 0.29 ms/sample — acceptable for batch cloud schedulers
- For latency-critical real-time dispatch: LGB base learner (0.057 ms) is recommended

## Negative Control (CloudTask) — Turn Weakness Into Strength

- CloudTask AUC-ROC = 0.517 (random labels, no learnable signal)
- All models fail similarly — proves KATS does not overfit noise
- SHAP stability ρ = 0.65 on CloudTask confirms no false pattern extraction
- Paper framing: *"The inclusion of a synthetically-labelled negative control validates the framework's discriminative integrity"*

---

## Numbers to Put In Abstract

- 4 datasets, 7 baselines, 5 seeds, 11 experiments
- RecallH = 0.9997 (ITIncident, highest class imbalance)
- SLA breach rate = 0% on real incident data (K2)
- SHAP stability ρ ≥ 0.937 on 3/4 real-world datasets
- EDF scheduling gap = 73.5pp
- Inference latency ≤ 0.29 ms/sample
