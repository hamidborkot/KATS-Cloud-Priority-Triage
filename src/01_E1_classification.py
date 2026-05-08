"""
KATS Framework — E1: In-Distribution Classification
====================================================
5-seed stratified evaluation across all 4 datasets.
Includes KATS + 5 baselines (LogReg, DecTree, RF, LGB, MLP).
McNemar test for KATS vs B4-LGB on each dataset.
"""

import pandas as pd
import numpy as np
import warnings, time
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, cohen_kappa_score
from statsmodels.stats.contingency_tables import mcnemar as mc_test

# Import preprocessing utilities
from src.preprocessing import (
    load_cloud_task, load_google_cluster, load_it_incident, load_multi_cloud,
    encode_labels_fixed, make_class_weights, apply_smote_safe,
    compute_metrics, get_kats_ensemble, get_baselines
)

SEEDS = [42, 7, 13, 99, 2026]
SEED  = 42


def mcnemar_test(y_true, pred_a, pred_b):
    a_right = (pred_a == y_true)
    b_right = (pred_b == y_true)
    n01 = int(np.sum(~a_right &  b_right))
    n10 = int(np.sum( a_right & ~b_right))
    table = np.array([[0, n01], [n10, 0]])
    result = mc_test(table, exact=False, correction=True)
    return result.pvalue, n10, n01


def run_e1():
    df_cloud,  CLOUD_FEATURES  = load_cloud_task()
    df_google, GOOGLE_FEATURES = load_google_cluster()
    df_it,     IT_FEATURES     = load_it_incident()
    df_mc,     MC_FEATURES     = load_multi_cloud()

    DATASETS = {
        "CloudTask":     (df_cloud,  CLOUD_FEATURES),
        "GoogleCluster": (df_google, GOOGLE_FEATURES),
        "ITIncident":    (df_it,     IT_FEATURES),
        "MultiCloud":    (df_mc,     MC_FEATURES),
    }

    e1_results = {}
    train_times = {}

    for ds_name, (df, feats) in DATASETS.items():
        print(f"\n  >> Dataset: {ds_name}")
        X = df[feats].fillna(0).astype(float)
        y_enc, le, high_idx = encode_labels_fixed(df["priority_label"])
        cw = make_class_weights(y_enc, high_idx, alpha=5)

        model_names = ["KATS", "B1-LogReg", "B2-DecTree", "B3-RF", "B4-LGB", "B5-MLP"]
        e1_results[ds_name] = {m: [] for m in model_names}
        train_times[ds_name] = {m: [] for m in model_names}
        last_seed_preds = {}

        scaler   = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        for seed in SEEDS:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X.values, y_enc, test_size=0.20, random_state=seed, stratify=y_enc)
            Xs_tr, Xs_te, _, _ = train_test_split(
                X_scaled, y_enc, test_size=0.20, random_state=seed, stratify=y_enc)

            X_tr_s, y_tr_s = apply_smote_safe(X_tr, y_tr, seed=seed)
            Xs_tr_s, _      = apply_smote_safe(Xs_tr, y_tr, seed=seed)
            cw_s = make_class_weights(y_tr_s, high_idx, alpha=5)

            # KATS
            t0 = time.time()
            kats = get_kats_ensemble(cw_s, seed=seed)
            kats.fit(X_tr_s, y_tr_s)
            train_times[ds_name]["KATS"].append(time.time() - t0)
            pred_kats = kats.predict(X_te)
            e1_results[ds_name]["KATS"].append(compute_metrics(y_te, pred_kats, le))
            last_seed_preds["KATS"] = (y_te, pred_kats)

            # Baselines
            for bname, bmodel in get_baselines(seed).items():
                if bname == "B5-MLP":
                    t0 = time.time()
                    bmodel.fit(Xs_tr_s, y_tr_s)
                    train_times[ds_name][bname].append(time.time() - t0)
                    pred_b = bmodel.predict(Xs_te)
                else:
                    t0 = time.time()
                    bmodel.fit(X_tr_s, y_tr_s)
                    train_times[ds_name][bname].append(time.time() - t0)
                    pred_b = bmodel.predict(X_te)
                e1_results[ds_name][bname].append(compute_metrics(y_te, pred_b, le))
                last_seed_preds[bname] = (y_te, pred_b)

        # Print results
        print(f"  {'Model':<18} {'RecallH':>12}  {'MacroF1':>8} {'Kappa':>8}")
        print("  " + "-" * 52)
        for m in model_names:
            rh  = np.mean([x["RecallHigh"] for x in e1_results[ds_name][m]])
            rhs = np.std( [x["RecallHigh"] for x in e1_results[ds_name][m]])
            f1  = np.mean([x["MacroF1"]    for x in e1_results[ds_name][m]])
            kap = np.mean([x["Kappa"]      for x in e1_results[ds_name][m]])
            tt  = np.mean(train_times[ds_name][m])
            print(f"  {m:<18} {rh:.4f}±{rhs:.4f} {f1:>8.4f} {kap:>8.4f}  [{tt:.1f}s]")

        # McNemar: KATS vs B4-LGB
        y_mc, pred_kats_mc = last_seed_preds["KATS"]
        _,    pred_lgb_mc  = last_seed_preds["B4-LGB"]
        pval, n10, n01 = mcnemar_test(y_mc, pred_kats_mc, pred_lgb_mc)
        print(f"\n  McNemar KATS vs B4-LGB: p={pval:.4f} "
              f"(KATS_better={n10}, LGB_better={n01})")

    return e1_results, train_times


if __name__ == "__main__":
    run_e1()
