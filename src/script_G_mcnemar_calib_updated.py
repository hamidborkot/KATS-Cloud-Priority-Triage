# ============================================================
# SCRIPT G — McNemar (7 models) + Calibration Updated
# ============================================================
# PURPOSE:
#   Adds B6-XGBoost to all McNemar comparisons
#   Adds AUC-ROC to calibration table (Brier + ECE + AUC)
#   Requires: f_all_preds.pkl, f_all_probs.pkl (from Script F)
#
# KEY McNemar RESULTS:
#   CloudTask:     KATS>LogReg**, KATS>XGBoost**, RF>KATS** (all κ≈0)
#   GoogleCluster: KATS>LogReg***, KATS>DecTree***, KATS>MLP***, KATS>XGB*
#   ITIncident:    None significant (all models at ceiling)
#   MultiCloud:    KATS>DecTree***, KATS>RF***, KATS>LGB***, KATS>XGB***
#                  MLP>KATS*** (honest limitation documented)
#
# KEY CALIBRATION CORRECTION:
#   CloudTask ECE: RF=0.038 < KATS=0.069 (previous claim was WRONG)
#   MultiCloud:    MLP ECE=0.027 < KATS ECE=0.032
# ============================================================

import pandas as pd
import numpy as np
import pickle, warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, brier_score_loss
from scipy.stats import chi2 as chi2_dist

with open("/kaggle/working/f_all_preds.pkl", "rb") as f:
    all_preds = pickle.load(f)
with open("/kaggle/working/f_all_probs.pkl", "rb") as f:
    all_probs = pickle.load(f)
print("Loaded predictions from Script F")

def mcnemar_test(y_true, y_pred_a, y_pred_b):
    y_true   = np.array(y_true)
    y_pred_a = np.array(y_pred_a)
    y_pred_b = np.array(y_pred_b)
    ca = (y_pred_a == y_true)
    cb = (y_pred_b == y_true)
    b01 = np.sum(~ca &  cb)
    b10 = np.sum( ca & ~cb)
    if (b01 + b10) == 0:
        return 1.0, b01, b10, "TIED"
    chi2 = (abs(b01 - b10) - 1)**2 / (b01 + b10)
    pval = 1 - chi2_dist.cdf(chi2, df=1)
    direction = "KATS_BETTER" if b10 > b01 else "BASE_BETTER"
    return pval, b01, b10, direction

def compute_brier_mc(y_true, y_prob, n_cls):
    y_bin = label_binarize(y_true, classes=list(range(n_cls)))
    return np.mean([brier_score_loss(y_bin[:, c], y_prob[:, c])
                    for c in range(n_cls)])

def compute_ece_mc(y_true, y_prob, n_cls, n_bins=10):
    y_bin = label_binarize(y_true, classes=list(range(n_cls)))
    eces  = []
    bins  = np.linspace(0, 1, n_bins + 1)
    for c in range(n_cls):
        ece = 0.0
        for i in range(n_bins):
            lo, hi = bins[i], bins[i+1]
            mask = (y_prob[:, c] >= lo) & (y_prob[:, c] < hi)
            if mask.sum() == 0:
                continue
            ece += (mask.sum()/len(y_true)) * abs(
                y_bin[mask, c].mean() - y_prob[mask, c].mean())
        eces.append(ece)
    return np.mean(eces)

def compute_auc_mc(y_true, y_prob, n_cls):
    try:
        if n_cls == 2:
            return roc_auc_score(y_true, y_prob[:, 1])
        y_bin = label_binarize(y_true, classes=list(range(n_cls)))
        return roc_auc_score(y_bin, y_prob, average="macro", multi_class="ovr")
    except:
        return np.nan

BASELINES      = ["B1-LogReg","B2-DecTree","B3-RF","B4-LGB","B5-MLP","B6-XGBoost"]
ALL_MODELS     = ["KATS"] + BASELINES
DATASETS_ORDER = ["CloudTask","GoogleCluster","ITIncident","MultiCloud"]

# ── McNemar table ─────────────────────────────────────────────
print("\n" + "=" * 72)
print("  C1 UPDATED — McNEMAR'S TEST (7 models including B6-XGBoost)")
print("=" * 72)
print(f"\n  {'Dataset':<18} {'Baseline':<14} {'b01':>5} {'b10':>5} "
      f"{'p-value':>12} {'Sig':>5} Direction")
print("  " + "-" * 74)

mcnemar_results = {}
for ds in DATASETS_ORDER:
    y_true_kats = all_preds[ds]["KATS"][0]
    y_pred_kats = all_preds[ds]["KATS"][1]
    mcnemar_results[ds] = {}
    for base in BASELINES:
        y_pred_b = all_preds[ds][base][1]
        pval, b01, b10, direction = mcnemar_test(
            y_true_kats, y_pred_kats, y_pred_b)
        sig = "***" if pval < 0.001 else ("**" if pval < 0.01 else
              ("*" if pval < 0.05 else "ns"))
        mcnemar_results[ds][base] = {
            "pval": pval, "b01": b01, "b10": b10, "direction": direction}
        print(f"  {ds:<18} {base:<14} {b01:>5} {b10:>5} "
              f"{pval:>12.4e} {sig:>5}  {direction}")
    print()

# ── Calibration table ─────────────────────────────────────────
print("\n" + "=" * 72)
print("  C2 UPDATED — CALIBRATION: BRIER + ECE + AUC-ROC (7 models)")
print("=" * 72)
print(f"\n  {'Dataset':<18} {'Model':<14} {'Brier↓':>8} {'ECE↓':>8} "
      f"{'AUC-ROC↑':>10}")
print("  " + "-" * 64)

calib_results = {}
for ds in DATASETS_ORDER:
    n_cls = len(set(all_preds[ds]["KATS"][0]))
    calib_results[ds] = {}
    for m in ALL_MODELS:
        y_true = np.array(all_preds[ds][m][0])
        y_prob = np.array(all_probs[ds][m]).reshape(-1, n_cls)
        brier  = compute_brier_mc(y_true, y_prob, n_cls)
        ece    = compute_ece_mc(y_true, y_prob, n_cls)
        auc    = compute_auc_mc(y_true, y_prob, n_cls)
        calib_results[ds][m] = {"Brier": brier, "ECE": ece, "AUC": auc}
        marker = " ←" if m == "KATS" else ""
        print(f"  {ds:<18} {m:<14} {brier:>8.4f} {ece:>8.4f} "
              f"{auc:>10.4f}{marker}")
    print()

# ── AUC-ROC summary ───────────────────────────────────────────
print("  AUC-ROC RANKING TABLE (macro OvR):")
print(f"\n  {'Model':<18}", end="")
for ds in DATASETS_ORDER:
    print(f"  {ds[:10]:>10}", end="")
print(f"  {'Mean':>8}")
print("  " + "-" * 60)
for m in ALL_MODELS:
    vals = [calib_results[ds].get(m, {}).get("AUC", np.nan)
            for ds in DATASETS_ORDER]
    mean_v = np.nanmean(vals)
    print(f"  {m:<18}", end="")
    for v in vals:
        print(f"  {v:>10.4f}", end="")
    print(f"  {mean_v:>8.4f}")

with open("/kaggle/working/g_mcnemar.pkl", "wb") as f:
    pickle.dump(mcnemar_results, f)
with open("/kaggle/working/g_calib.pkl", "wb") as f:
    pickle.dump(calib_results, f)
print("\n  Script G COMPLETE ✓")
