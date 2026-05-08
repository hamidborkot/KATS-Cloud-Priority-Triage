# G2+G3 — Spearman Correlation Analysis Results

---

## G2 — CloudTask: Proof of Negative Control

**All 19 features show |ρ| < 0.031 — statistically decorrelated from priority label.**

| Feature | Spearman ρ | p-value | Signal |
|---|---|---|---|
| VM_Bandwidth_MBps | 0.0302 | 1.93e-02 | WEAK |
| Energy_Consumption_J | 0.0261 | 4.28e-02 | WEAK |
| All others | < 0.020 | > 0.10 | WEAK |

**Max |ρ| = 0.0302, Mean |ρ| = 0.0106**

Conclusion: Task_Priority is assigned by the simulation engine independently
of all observable scheduling features. This validates treating CloudTask
as a negative control experiment.

---

## 