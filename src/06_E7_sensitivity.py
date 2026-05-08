"""
KATS Framework — E7: Sensitivity Analysis
==========================================
T7.1 — Alpha sweep on CloudTask (non-trivial precision/recall trade-off)
T7.2 — Label noise robustness on ITIncident
T7.3 — Learning curve on MultiCloud
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
import lightgbm as lgb

from src.preprocessing import (
    load_cloud_task, load_it_incident, load_multi_cloud,
    encode_labels_fixed, make_class_weights, apply_smote_safe,
    compute_metrics, get_kats_ensemble
)

SEED = 42
np.random.seed(SEED)


def run_e7():
    # ── T7.1: Alpha sweep on CloudTask ────────────────────────────────────
    print("  T7.1 — Alpha Sensitivity on CloudTask")
    print(f"  {'Alpha':>6} {'RecallH':>12} {'PrecH':>10} {'MacroF1':>10} {'Kappa':>10}")
    print("  " + "-" * 52)

    df_ct, CT_FEATS = load_cloud_task()
    X_ct = df_ct[CT_FEATS].fillna(0).astype(float).values
    y_ct, le_ct, hi_ct = encode_labels_fixed(df_ct["priority_label"])
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_ct, y_ct, test_size=0.2, random_state=SEED, stratify=y_ct)
    X_tr_s, y_tr_s = apply_smote_safe(X_tr, y_tr, seed=SEED)

    for alpha in [1, 2, 3, 5, 7, 9, 12]:
        cw = make_class_weights(y_tr_s, hi_ct, alpha=alpha)
        m  = lgb.LGBMClassifier(n_estimators=300, class_weight=cw,
                                  learning_rate=0.05, random_state=SEED, verbose=-1)
        m.fit(X_tr_s, y_tr_s)
        met = compute_metrics(y_te, m.predict(X_te), le_ct)
        print(f"  {alpha:>6}  {met['RecallHigh']:>12.4f} {met['PrecHigh']:>10.4f} "
              f"{met['MacroF1']:>10.4f} {met['Kappa']:>10.4f}")

    # ── T7.2: Label noise on ITIncident ────────────────────────────────────
    print("\n  T7.2 — Label Noise Robustness on ITIncident")
    print(f"  {'Noise%':>7} {'RecallH':>12} {'MacroF1':>10} {'Kappa':>8}")

    df_it, IT_FEATS = load_it_incident()
    X_it = df_it[IT_FEATS].fillna(0).astype(float).values
    y_it, le_it, hi_it = encode_labels_fixed(df_it["priority_label"])
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_it, y_it, test_size=0.2, random_state=SEED, stratify=y_it)

    for noise_pct in [0, 5, 10, 15, 20]:
        y_noisy = y_tr.copy()
        n_flip  = int(len(y_noisy) * noise_pct / 100)
        if n_flip > 0:
            rng = np.random.RandomState(SEED)
            flip_idx = rng.choice(len(y_noisy), n_flip, replace=False)
            classes = np.unique(y_noisy)
            for idx in flip_idx:
                others = [c for c in classes if c != y_noisy[idx]]
                y_noisy[idx] = rng.choice(others)
        X_tr_n, y_tr_n = apply_smote_safe(X_tr, y_noisy, seed=SEED)
        cw_n = make_class_weights(y_tr_n, hi_it, alpha=5)
        m = lgb.LGBMClassifier(n_estimators=300, class_weight=cw_n,
                                random_state=SEED, verbose=-1)
        m.fit(X_tr_n, y_tr_n)
        met = compute_metrics(y_te, m.predict(X_te), le_it)
        print(f"  {noise_pct:>6}%  {met['RecallHigh']:>12.4f} "
              f"{met['MacroF1']:>10.4f} {met['Kappa']:>8.4f}")

    # ── T7.3: Learning curve on MultiCloud ────────────────────────────────
    print("\n  T7.3 — Learning Curve on MultiCloud")
    print(f"  {'Frac%':>6} {'N_train':>9} {'RecallH':>12} {'MacroF1':>10} {'Kappa':>10}")

    df_mc, MC_FEATS = load_multi_cloud()
    X_mc = df_mc[MC_FEATS].fillna(0).astype(float).values
    y_mc, le_mc, hi_mc = encode_labels_fixed(df_mc["priority_label"])
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_mc, y_mc, test_size=0.2, random_state=SEED, stratify=y_mc)
    X_tr_s, y_tr_s = apply_smote_safe(X_tr, y_tr, seed=SEED)

    for frac in [0.10, 0.20, 0.40, 0.60, 0.80, 1.00]:
        idx_all = []
        for cls in np.unique(y_tr_s):
            cls_idx = np.where(y_tr_s == cls)[0]
            n_take  = max(2, int(len(cls_idx) * frac))
            chosen  = np.random.RandomState(SEED).choice(cls_idx, min(n_take, len(cls_idx)), replace=False)
            idx_all.extend(chosen.tolist())
        idx_all = np.array(idx_all)
        X_f = X_tr_s[idx_all]; y_f = y_tr_s[idx_all]
        cw_f = make_class_weights(y_f, hi_mc, alpha=5)
        min_cls = np.bincount(y_f).min()
        if min_cls >= 10:
            model = get_kats_ensemble(cw_f, seed=SEED)
        else:
            model = lgb.LGBMClassifier(n_estimators=300, class_weight=cw_f,
                                        random_state=SEED, verbose=-1)
        model.fit(X_f, y_f)
        met = compute_metrics(y_te, model.predict(X_te), le_mc)
        print(f"  {int(frac*100):>5}%  {len(idx_all):>9}  "
              f"{met['RecallHigh']:>12.4f} {met['MacroF1']:>10.4f} {met['Kappa']:>10.4f}")


if __name__ == "__main__":
    run_e7()
