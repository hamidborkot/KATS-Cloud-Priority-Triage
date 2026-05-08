# G6 — Inference Latency Benchmark

**Benchmark:** N=10,000 predictions, 5 warm-up runs, mean of 10 timed runs  
**Hardware:** Kaggle CPU (inference benchmark, no GPU)

## Batch Throughput

| Model | N=100 | N=1,000 | N=5,000 | N=10,000 | Throughput (N=10K) |
|---|---|---|---|---|---|
| KATS | 48.43ms | 159.30ms | 563.65ms | 683.31ms | 14,635 pred/s |
| B4-LGB | 9.35ms | 60.78ms | 278.96ms | 331.70ms | 30,148 pred/s |
| B3-RF | 32.89ms | 83.16ms | 231.11ms | 261.18ms | 38,288 pred/s |
| B1-LogReg | 0.10ms | 0.15ms | 0.40ms | 0.56ms | 17,723,269 pred/s |
| B5-MLP | 0.46ms | 2.69ms | 10.64ms | 11.90ms | 840,061 pred/s |

## Per-Item Latency at N=10,000

| Model | μs/pred | Real-time suitable (<1ms)? |
|---|---|---|
| KATS | 68.3 | YES ✓ |
| B4-LGB | 33.2 | YES ✓ |
| B3-RF | 26.1 | YES ✓ |
| B1-LogReg | 0.1 | YES ✓ |
| B5-MLP | 1.2 | YES ✓ |

**All models suitable for real-time triage (<100ms SLA).**  
KATS 68.3 μs/pred is well within operational requirements.
