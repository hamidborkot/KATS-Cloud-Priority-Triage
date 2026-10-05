#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
roc_auc_from_clusterA_predictions.py
Computes the missing macro one-vs-rest ROC-AUC (and re-derives PR-AUC, precision, FP/FN as a cross-check) from
clusterA_predictions.csv (1,167,360 rows = (12000+12000+4984+200 test rows) x 8 models x 5 seeds).
Usage:  python roc_auc_from_clusterA_predictions.py /kaggle/working/KATS_CLUSTER_A_FINAL/clusterA_predictions.csv [clusterA_main_metrics_per_seed.csv]
Prints the detected columns first; if detection fails, edit the three CANDIDATES lists.
"""
import sys, re, os
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score
P = sys.argv[1]; REF = sys.argv[2] if len(sys.argv) > 2 else None
df = pd.read_csv(P); print("columns:", list(df.columns)); print(df.head(3).to_string())
low = {c.lower(): c for c in df.columns}
def pick(cands):
    for c in cands:
        if c in low: return low[c]
YT = pick(["y_true", "true_class", "true_label", "true", "label", "y"]); YP = pick(["y_pred", "pred_class", "predicted_class", "pred", "prediction"])
DS, MD, SD = pick(["dataset"]), pick(["model"]), pick(["seed"])
PC = [c for c in df.columns if re.match(r"(p_|proba_|prob_|probability_)", c, re.I)]
if not all([YT, DS, MD, SD]) or len(PC) != 3: sys.exit(f"could not detect: y_true={YT} dataset={DS} model={MD} seed={SD} prob cols={PC}")
classes = sorted(df[YT].astype(str).unique())                      # e.g. High, Low, Medium
PC = sorted(PC, key=lambda c: [k.lower() for k in classes].index(re.sub(r"^(p_|proba_|prob_|probability_)", "", c, flags=re.I).lower()) if re.sub(r"^(p_|proba_|prob_|probability_)", "", c, flags=re.I).lower() in [k.lower() for k in classes] else 99)
print("classes:", classes, "prob columns (same order):", PC)
hi = classes.index("High") if "High" in classes else 0
rows = []
for (d, m, s), g in df.groupby([DS, MD, SD]):
    y = g[YT].astype(str).values; Y = np.stack([(y == c).astype(int) for c in classes], 1); Pm = g[PC].values
    r = dict(dataset=d, model=m, seed=s, n=len(g), AUC_macro_ovr=roc_auc_score(Y, Pm, average="macro"), AUC_High_ovr=roc_auc_score(Y[:, hi], Pm[:, hi]),
             PRAUC_High_recomputed=average_precision_score(Y[:, hi], Pm[:, hi]))
    if YP:
        yp = g[YP].astype(str).values; tp = ((yp == "High") & (y == "High")).sum(); fp = ((yp == "High") & (y != "High")).sum(); fn = ((yp != "High") & (y == "High")).sum()
        r.update(RecallH_recomputed=tp / max(1, tp + fn), PrecH_recomputed=tp / max(1, tp + fp), FP_High_recomputed=int(fp), FN_High_recomputed=int(fn))
    rows.append(r)
R = pd.DataFrame(rows); R.to_csv("roc_auc_per_seed.csv", index=False)
def tci(x): x = np.asarray(x, float); return stats.t.ppf(.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan
S = R.groupby(["dataset", "model"]).AUC_macro_ovr.agg(["mean", "std", tci]).reset_index().rename(columns={"tci": "ci95"}); S.to_csv("roc_auc_mean_sd_ci95.csv", index=False)
print(S.round(4).to_string())
if REF and os.path.exists(REF):        # cross-check against the reported per-seed metrics (should agree to ~1e-6 for the default-rule models)
    ref = pd.read_csv(REF, sep=None, engine="python"); j = R.merge(ref, on=["dataset", "model", "seed"], suffixes=("", "_ref"))
    for a, b in [("PRAUC_High_recomputed", "PRAUC_High"), ("FP_High_recomputed", "FP_High"), ("FN_High_recomputed", "FN_High")]:
        if a in j and b in j: print(f"max |{a} - reported {b}| = {np.abs(j[a] - j[b]).max():.6g}")
