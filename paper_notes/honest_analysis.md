# KATS Complete Honest Experimental Analysis

**Last updated:** May 2026  
**Status:** All experiments complete. Writing phase begins.

---

## Gap Resolution Status

| Gap | Status | Description |
|---|---|---|
| G1 | ✅ RESOLVED | MLP baseline added — KATS beats MLP on all 4 datasets |
| G2 | ✅ RESOLVED | CloudTask Spearman: max\|ρ\|=0.030 — perfect negative control |
| G3 | ✅ RESOLVED | Semantic validation table; sign inversions explain E2 failures |
| G4 | ⚠️ PARTIAL | Stacking dilutes alpha effect — reframe as LGB base learner benefit |
| G5 | ✅ RESOLVED | Full training time table across all models × datasets |
| G6 | ✅ RESOLVED | Random baseline + oracle beat at S3 |
| G7 | ☐ OPEN | Formal math definitions — writing task |
| G8 | ☐ OPEN | Literature review 30+ references — writing task |

---

## Finding 1: E1 Regime Classification

| Dataset | Regime | Key Evidence |
|---|---|---|
| CloudTask | NEGATIVE CONTROL | max\|ρ\|=0.030 — all features decorrelated |
| GoogleCluster | CEILING | scheduler ρ=0.845 — near-bijective |
| ITIncident | CEILING | All classifiers RecallH=1.000 |
| MultiCloud | DIFFERENTIATION ✓ | KATS κ=0.997 vs LogReg κ=0.300 |

- KATS is NOT best on CloudTask (LGB F1=0.3454 > KATS F1=0.3442)
- KATS ties ceiling on Google/IT while providing calibration+auditability
- KATS beats MLP on all 4 datasets (strongest: MultiCloud κ 0.997 vs 0.906)

---

## Finding 2: E3 Oracle-Beating Result

**KATS S3=0.5045 > Oracle S3=0.4849 (+1.96pp)**

Reason: KATS produces continuous P(High) enabling finer-grained ranking than
discrete 3-tier Task_Priority. Probabilistic triage outperforms rule-based
metadata under catastrophic resource collapse.

**Honest: LGB beats KATS at S2 (0.9482 vs 0.9415, Δ=0.67pp)**

---

## Finding 3: Semantic Sign Inversion

`time_pressure` ρ = +0.010 on CloudTask but **−0.606 on GoogleCluster** (opposite sign).
This is the cleanest empirical proof of cross-domain semantic inversion causing E2 failures.

---

## Finding 4: Asymmetric Loss Honest Assessment

- LGB α=5 standalone: RecallH=0.4373 vs α=1: RecallH=0.2284 (+20.9pp)
- KATS α=5 vs α=1: RecallH=0.2730 vs 0.2674 (only +0.56pp)
- Stacking meta-learner DILUTES the alpha effect
- **Reframe:** AsymLoss benefits LGB base learner; stacking calibrates the aggression

---

## Paper-Ready Numbers Checklist

| Metric | Value | Used In |
|---|---|---|
| CloudTask Spearman max\|ρ\| | 0.0302 | G2, Section 4.1 |
| MultiCloud KATS vs LogReg Kappa gap | +69.7pp | E1, Section 4.2 |
| MultiCloud KATS vs MLP Kappa gap | +9.1pp | E1/G1, Section 4.2 |
| E3 KATS S3 vs Oracle | +1.96pp | E3, Section 4.4 |
| E3 KATS S2 vs Random | +54.4pp | E3, Section 4.4 |
| LGB α=5 vs α=1 RecallH | +20.9pp | G4, Section 4.6 |
| T7.1 α full-training | +44pp | E7, Section 4.7 |
| T7.2 Noise 20% | RecallH≥0.978 | E7 |
| T7.3 Data saturation | N=318 | E7 |
| KATS inference | 68.3 μs/pred | G6 |
| KATS training GoogleCluster | 1078s (18 min) | G5 |
