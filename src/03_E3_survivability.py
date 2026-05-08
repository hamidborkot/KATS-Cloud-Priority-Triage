"""
KATS Framework — E3: Survivability Simulation
=============================================
Simulates bandwidth-constrained service migration triage.
Ranks services by P(High) and greedily migrates until BW cap.
Bootstrap 95% CI for all survivability estimates.

Scenarios:
  S1: 65% BW remaining (mild degradation)
  S2: 40% BW remaining (gulf-strike level)
  S3: 15% BW remaining (catastrophic collapse)
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.utils import resample
import lightgbm as lgb

from src.preprocessing import (
    load_cloud_task, encode_labels_fixed, make_class_weights,
    apply_smote_safe, get_kats_ensemble
)

SEED = 42
SCENARIOS = {"S1_Mild(65%BW)": 0.65, "S2_Gulf(40%BW)": 0.40, "S3_Collapse(15%BW)": 0.15}


def survivability_prob(prob_high, bw_series, true_high_mask, bw_cap_frac, n_boot=500):
    """Probability-ranked survivability with bootstrap 95% CI."""
    bw   = bw_series.values
    mask = true_high_mask.values
    cap  = bw.sum() * bw_cap_frac
    order = np.argsort(-prob_high)
    cum   = np.cumsum(bw[order])
    mig   = order[cum <= cap]
    rescued = mask[mig].sum()
    surv    = rescued / max(mask.sum(), 1)

    boots = []
    for i in range(n_boot):
        idx   = resample(np.arange(len(prob_high)), random_state=i)
        bw_b  = bw[idx]; mask_b = mask[idx]; ph_b = prob_high[idx]
        cap_b = bw_b.sum() * bw_cap_frac
        ord_b = np.argsort(-ph_b)
        cum_b = np.cumsum(bw_b[ord_b])
        mig_b = ord_b[cum_b <= cap_b]
        boots.append(mask_b[mig_b].sum() / max(mask_b.sum(), 1))
    lo, hi_ci = np.percentile(boots, [2.5, 97.5])
    return surv, lo, hi_ci, int(rescued)


def run_e3():
    df_cloud, CLOUD_FEATURES = load_cloud_task()
    X_all = df_cloud[CLOUD_FEATURES].fillna(0).astype(float).values
    y_all, le, hi = encode_labels_fixed(df_cloud["priority_label"])
    cw_all = make_class_weights(y_all, hi, alpha=5)

    X_tr, _, y_tr, _ = train_test_split(
        X_all, y_all, test_size=0.2, random_state=SEED, stratify=y_all)
    X_tr_s, y_tr_s = apply_smote_safe(X_tr, y_tr, seed=SEED)
    cw_s = make_class_weights(y_tr_s, hi, alpha=5)

    scaler  = StandardScaler().fit(X_tr_s)
    Xs_tr_s = scaler.transform(X_tr_s)
    Xs_all  = scaler.transform(X_all)

    kats  = get_kats_ensemble(cw_s, SEED);  kats.fit(X_tr_s,  y_tr_s)
    lgb_m = lgb.LGBMClassifier(n_estimators=500, class_weight="balanced",
                                random_state=SEED, verbose=-1); lgb_m.fit(X_tr_s, y_tr_s)
    rf_m  = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                   random_state=SEED); rf_m.fit(X_tr_s, y_tr_s)
    lr_m  = LogisticRegression(max_iter=2000, class_weight="balanced",
                               random_state=SEED); lr_m.fit(X_tr_s, y_tr_s)
    mlp_m = MLPClassifier(hidden_layer_sizes=(128,64,32), max_iter=500,
                          random_state=SEED, early_stopping=True)
    mlp_m.fit(Xs_tr_s, y_tr_s)

    bw_series      = df_cloud["VM_Bandwidth_MBps"]
    true_high_mask = (df_cloud["priority_label"] == "High")
    rule_scores    = df_cloud["Task_Priority"].values
    prob_rule      = (rule_scores - 1) / 2.0
    prob_random    = np.random.RandomState(SEED).uniform(0, 1, len(df_cloud))

    MODELS_E3 = {
        "KATS":           kats.predict_proba(X_all)[:, hi],
        "B4-LGB":         lgb_m.predict_proba(X_all)[:, hi],
        "B3-RF":          rf_m.predict_proba(X_all)[:, hi],
        "B1-LogReg":      lr_m.predict_proba(X_all)[:, hi],
        "B5-MLP":         mlp_m.predict_proba(Xs_all)[:, hi],
        "B0-Rule(Oracle)": prob_rule,
        "B_Random":       prob_random,
    }

    e3_results = {}
    print(f"\n  Fleet={len(df_cloud):,} | TrueHigh={true_high_mask.sum():,}")
    print(f"  {'Method':<22}", end="")
    for sc in SCENARIOS: print(f"  {sc:>28}", end="")
    print()
    print("  " + "-" * 100)

    for mname, prob_h in MODELS_E3.items():
        e3_results[mname] = {}
        row = f"  {mname:<22}"
        for sc_name, bw_frac in SCENARIOS.items():
            s, lo, hi_ci, rescued = survivability_prob(
                prob_h, bw_series, true_high_mask, bw_frac)
            e3_results[mname][sc_name] = (s, lo, hi_ci, rescued)
            row += f"  {s:.4f} [{lo:.3f},{hi_ci:.3f}]"
        print(row)

    return e3_results


if __name__ == "__main__":
    run_e3()
