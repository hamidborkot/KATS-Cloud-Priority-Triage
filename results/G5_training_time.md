# G5 — Training Time Table

**Mean training time in seconds over 5 seeds**  
**Hardware:** Kaggle GPU P100

| Model | CloudTask | GoogleCluster | ITIncident | MultiCloud |
|---|---|---|---|---|
| KATS | 46.55s | **1078.32s** | 55.45s | 5.08s |
| B1-LogReg | 2.30s | 258.38s | 22.73s | 0.57s |
| B2-DecTree | 0.20s | 3.80s | 0.12s | 0.00s |
| B3-RF | 7.79s | 181.72s | 7.33s | 0.62s |
| B4-LGB | 1.73s | 36.64s | 2.56s | 0.12s |
| B5-MLP | 2.79s | 120.60s | 5.98s | 0.28s |

## Key Observations

- KATS training on GoogleCluster = 18 minutes (acceptable for offline deployment)
- KATS is **29× slower** than LGB at training on GoogleCluster (1078s vs 37s)
- KATS is **2× slower** than LGB at inference (68.3 μs vs 33.2 μs)
- **Paper framing:** Offline training cost is amortized over deployment lifetime;
  KATS is trained once and deployed continuously for triage decisions
