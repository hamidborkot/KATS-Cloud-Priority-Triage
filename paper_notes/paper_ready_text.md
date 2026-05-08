# Paper-Ready Text Fragments

> These are draft text segments ready for integration into the manuscript.
> DO NOT copy verbatim — adapt to journal style and integrate with citations.

---

## Section 3.1 — Dataset Taxonomy

"We evaluate KATS across four cloud datasets representing distinct priority-inference
regimes. CloudTask [REF] serves as a designed negative control: Spearman correlation
analysis confirms all 19 scheduling features exhibit |ρ| < 0.031 with the priority label
(max|ρ|=0.030, mean|ρ|=0.011), confirming that Task_Priority is assigned exogenously by
the simulation engine and cannot be inferred from observable metrics. Google Cluster
Traces [REF] represent the high-signal ceiling regime, where the Borg scheduler field
correlates at ρ=0.845 with priority. IT Incident Log [REF] represents extreme class
imbalance (34.6:1 Medium:High ratio) in an ITIL-governed operational setting.
Multi-Cloud Service Composition [REF] presents a non-linear QoS boundary where
linear classifiers fail catastrophically (κ=0.300) while ensemble methods succeed (κ=0.997)."

---

## Section 4.2 — E1 Key Result

"On MultiCloud, KATS achieves κ=0.997 compared to κ=0.300 for logistic regression
(+69.7pp) and κ=0.906 for MLP (+9.1pp), demonstrating that the non-linear QoS tertile
boundary requires ensemble complexity unavailable from linear or single-layer models.
On CloudTask, all models converge to equivalent near-random performance (Kappa ∈
[−0.003, 0.025]), consistent with the Spearman-verified absence of predictive signal."

---

## Section 4.4 — E3 Oracle Result

"Under catastrophic bandwidth collapse (S3: 15% BW available), KATS achieves
survivability=0.5045 compared to the rule-based oracle at 0.4849 (+1.96pp, 95% CI:
[0.471, 0.533] vs [0.456, 0.513]). This result demonstrates that continuous probabilistic
triage (KATS P(High) scores) enables finer-grained service ranking than discrete 3-tier
priority metadata, rescuing additional High-priority services at the triage boundary.
We acknowledge that at S2 (40% BW), LGB marginally outperforms KATS (0.9482 vs 0.9415,
Δ0.67pp), within overlapping confidence intervals. KATS's advantage is most pronounced
at the operationally critical collapse scenario (S3) where probability calibration
determines rescue order at the resource constraint boundary."

---

## Section 4.6 — Asymmetric Loss Finding

"Asymmetric class weighting (α=5) provides its primary benefit through the LGB base
learner, where standalone RecallHigh improves from 0.228 to 0.437 on CloudTask (+20.9pp,
Table G4). Under full training conditions (T7.1), LGB α=5 achieves RecallHigh=0.691
compared to 0.251 for α=1 (+44pp). The stacking meta-learner moderates this effect
(full KATS: ΔRecallHigh=+0.56pp), trading aggressive recall for improved precision
stability through the stacking calibration mechanism. On the performance-ceiling datasets,
asymmetric weighting has no measurable effect, confirming that α sensitivity is specific
to datasets with non-trivial precision-recall trade-offs."
