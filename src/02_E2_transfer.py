"""
KATS Framework — E2: Cross-Dataset Semantic Transfer
=====================================================
Builds a 6-dimensional semantic feature space and tests
cross-domain transferability of KATS vs LGB baseline.

Semantic dimensions:
  workload_intensity, time_pressure, resource_demand,
  failure_risk, complexity, qos_score
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score
import lightgbm as lgb

from src.preprocessing import (
    load_cloud_task, load_google_cluster, load_it_incident, load_multi_cloud,
    encode_labels_fixed, make_class_weights, apply_smote_safe, get_kats_ensemble
)

SEED = 42
SEMANTIC_FEATS = ["workload_intensity", "time_pressure", "resource_demand",
                  "failure_risk", "complexity", "qos_score"]


def build_semantic_features(df, dataset_name):
    """Map dataset-specific features to universal 6-dim semantic space."""
    out = pd.DataFrame()
    out["priority_label"] = df["priority_label"].astype(str)

    if dataset_name == "CloudTask":
        out["workload_intensity"] = df["Task_Length_MIPS"] / df["Task_Length_MIPS"].max()
        out["time_pressure"]      = 1 - (df["Task_Deadline"] / df["Task_Deadline"].max())
        out["resource_demand"]    = (df["VM_MIPS"]/df["VM_MIPS"].max() +
                                     df["VM_Memory_GB"]/df["VM_Memory_GB"].max() +
                                     df["VM_Bandwidth_MBps"]/df["VM_Bandwidth_MBps"].max()) / 3
        out["failure_risk"]       = df["Degree_of_Imbalance"] / df["Degree_of_Imbalance"].max()
        out["complexity"]         = df["Path_Load"]
        out["qos_score"]          = 1 - (df["Execution_Cost_$"] / df["Execution_Cost_$"].max())

    elif dataset_name == "GoogleCluster":
        out["workload_intensity"] = df["req_cpus"] / (df["req_cpus"].max() + 1e-9)
        out["time_pressure"]      = df["priority"] / df["priority"].max()
        out["resource_demand"]    = (df["req_cpus"] + df["req_memory"]) / 2
        out["resource_demand"]    /= (out["resource_demand"].max() + 1e-9)
        out["failure_risk"]       = df["failed"].astype(float)
        out["complexity"]         = df["instance_index"] / (df["instance_index"].max() + 1e-9)
        out["qos_score"]          = df["avg_cpus"] / (df["avg_cpus"].max() + 1e-9)

    elif dataset_name == "ITIncident":
        out["workload_intensity"] = df["sys_mod_count"] / (df["sys_mod_count"].max() + 1)
        out["time_pressure"]      = 1 - df["made_sla_enc"].astype(float)
        out["resource_demand"]    = df["urgency_enc"] / (df["urgency_enc"].max() + 1)
        out["failure_risk"]       = df["reopen_flag"].astype(float)
        out["complexity"]         = df["reassignment_count"] / (df["reassignment_count"].max() + 1)
        out["qos_score"]          = 1 - (df["impact_enc"] / (df["impact_enc"].max() + 1))

    elif dataset_name == "MultiCloud":
        out["workload_intensity"] = df["CPU_Utilization (%)"] / 100
        out["time_pressure"]      = df["Service_Latency (ms)"] / (df["Service_Latency (ms)"].max() + 1)
        out["resource_demand"]    = (df["CPU_Utilization (%)"]/100 +
                                     df["Memory_Usage (MB)"]/df["Memory_Usage (MB)"].max()) / 2
        out["failure_risk"]       = 1 - df["Optimal_Service_Placement"].astype(float)
        out["complexity"]         = df["Workload_Variability"] / (df["Workload_Variability"].max() + 1)
        out["qos_score"]          = df["QoS_Score"] / (df["QoS_Score"].max() + 1)

    out[SEMANTIC_FEATS] = out[SEMANTIC_FEATS].fillna(0).clip(0, 1)
    return out


def run_e2():
    df_cloud,  _ = load_cloud_task()
    df_google, _ = load_google_cluster()
    df_it,     _ = load_it_incident()
    df_mc,     _ = load_multi_cloud()

    raw_data = {
        "CloudTask": df_cloud, "GoogleCluster": df_google,
        "ITIncident": df_it,   "MultiCloud": df_mc
    }

    sem_data = {ds: build_semantic_features(df, ds) for ds, df in raw_data.items()}
    DS_NAMES = list(sem_data.keys())
    e2_results = {}

    print(f"\n  {'Source -> Target':<32} {'KATS F1':>10} {'LGB F1':>10} {'Gap':>8}")
    print("  " + "-" * 62)

    for src in DS_NAMES:
        src_df = sem_data[src]
        X_src = src_df[SEMANTIC_FEATS].values
        y_src, le_src, hi_src = encode_labels_fixed(src_df["priority_label"])
        cw_src = make_class_weights(y_src, hi_src, alpha=5)

        X_tr, _, y_tr, _ = train_test_split(
            X_src, y_src, test_size=0.2, random_state=SEED, stratify=y_src)
        X_tr_s, y_tr_s = apply_smote_safe(X_tr, y_tr, seed=SEED)

        kats = get_kats_ensemble(cw_src, seed=SEED)
        lgb_m = lgb.LGBMClassifier(n_estimators=300, class_weight="balanced",
                                    random_state=SEED, verbose=-1)
        kats.fit(X_tr_s, y_tr_s)
        lgb_m.fit(X_tr_s, y_tr_s)

        e2_results[src] = {}
        for tgt in DS_NAMES:
            if tgt == src:
                continue
            tgt_df = sem_data[tgt]
            y_tgt_raw = tgt_df["priority_label"]
            valid = y_tgt_raw.isin(le_src.classes_)
            X_tgt_v = tgt_df[SEMANTIC_FEATS].values[valid]
            y_tgt_v = le_src.transform(y_tgt_raw[valid])

            if len(np.unique(y_tgt_v)) < 2:
                e2_results[src][tgt] = None
                continue

            kf1 = f1_score(y_tgt_v, kats.predict(X_tgt_v),  average="macro", zero_division=0)
            lf1 = f1_score(y_tgt_v, lgb_m.predict(X_tgt_v), average="macro", zero_division=0)
            e2_results[src][tgt] = {"KATS_F1": kf1, "LGB_F1": lf1}
            print(f"  {src+' -> '+tgt:<32} {kf1:>10.4f} {lf1:>10.4f} {kf1-lf1:>+8.4f}")

    return e2_results


if __name__ == "__main__":
    run_e2()
