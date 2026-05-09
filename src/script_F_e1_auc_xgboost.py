# ============================================================
# SCRIPT F — FULL E1 RERUN: AUC-ROC + B6-XGBoost
# ============================================================
# PURPOSE:
#   R1: Adds macro OvR AUC-ROC to ALL results tables
#   R3: Adds B6-XGBoost as 6th baseline
#   Runs 5 seeds × 4 datasets × 7 models (KATS + 6 baselines)
#
# KEY RESULTS:
#   CloudTask:     All AUC ≈ 0.51 (confirms random labels)
#   GoogleCluster: All AUC ≥ 0.9997 (confirms ceiling)
#   ITIncident:    All AUC ≥ 0.9993 (confirms ceiling)
#   MultiCloud:    KATS=0.9806, MLP=0.9960 (MLP wins, documented)
#
# NOTABLE FINDING:
#   XGBoost RecallH=0.698 on CloudTask (highest!) but MacroF1=0.298 (lowest)
#   = prediction bias, not learning. Validates negative control.
#
# SAVES: f_e1_results.pkl | f_all_preds.pkl | f_all_probs.pkl
# ============================================================

import pandas as pd
import numpy as np
import ast, warnings, pickle
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    classification_report, cohen_kappa_score, roc_auc_score, f1_score)
import lightgbm as lgb
import xgboost as xgb
from imblearn.over_sampling import SMOTE

SEEDS = [42, 7, 13, 99, 2026]
np.random.seed(42)

# ── Utilities ─────────────────────────────────────────────────
def encode_labels(y_series):
    le = LabelEncoder()
    y_enc = le.fit_transform(y_series.astype(str))
    high_idx = int(np.where(le.classes_ == "High")[0][0])
    return y_enc, le, high_idx

def make_class_weights(y_enc, high_idx, alpha=5):
    classes, counts = np.unique(y_enc, return_counts=True)
    total = len(y_enc)
    cw = {int(c): total/(len(classes)*cnt) for c, cnt in zip(classes, counts)}
    cw[high_idx] *= alpha
    return cw

def apply_smote(X, y, seed=42):
    counts = np.bincount(y)
    k = max(1, counts.min()-1) if counts.min() < 6 else 5
    try:
        return SMOTE(random_state=seed, k_neighbors=k).fit_resample(X, y)
    except:
        return X, y

def compute_metrics(y_true, y_pred, y_prob, le):
    rep = classification_report(y_true, y_pred,
        target_names=le.classes_.tolist(), output_dict=True, zero_division=0)
    n_cls = len(le.classes_)
    try:
        if n_cls == 2:
            auc = roc_auc_score(y_true, y_prob[:, 1])
        else:
            y_bin = label_binarize(y_true, classes=list(range(n_cls)))
            auc   = roc_auc_score(y_bin, y_prob, average="macro", multi_class="ovr")
    except:
        auc = np.nan
    return {
        "RecallHigh": rep.get("High", {}).get("recall",    0.0),
        "PrecHigh":   rep.get("High", {}).get("precision", 0.0),
        "F1High":     rep.get("High", {}).get("f1-score",  0.0),
        "MacroF1":    rep["macro avg"]["f1-score"],
        "Kappa":      cohen_kappa_score(y_true, y_pred),
        "AUC_ROC":    auc,
    }

def get_kats(cw, seed=42):
    """KATS: Stacking ensemble with asymmetric loss LGB + RF + CalibratedNB."""
    return StackingClassifier(
        estimators=[
            ("lgb", lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05,
                max_depth=6, num_leaves=31, class_weight=cw,
                random_state=seed, verbose=-1)),
            ("rf",  RandomForestClassifier(n_estimators=300,
                class_weight="balanced", random_state=seed)),
            ("nb",  CalibratedClassifierCV(GaussianNB(), cv=5, method="isotonic")),
        ],
        final_estimator=LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs",
            multi_class="multinomial", random_state=seed),
        cv=5, passthrough=False)

def get_baselines(cw, seed=42):
    """Six baselines: LogReg, DecTree, RF, LGB, MLP, XGBoost."""
    return {
        "B1-LogReg":  LogisticRegression(max_iter=2000, random_state=seed,
                          class_weight="balanced"),
        "B2-DecTree": DecisionTreeClassifier(random_state=seed,
                          class_weight="balanced"),
        "B3-RF":      RandomForestClassifier(n_estimators=300, random_state=seed,
                          class_weight="balanced"),
        "B4-LGB":     lgb.LGBMClassifier(n_estimators=500, random_state=seed,
                          class_weight="balanced", verbose=-1),
        "B5-MLP":     MLPClassifier(hidden_layer_sizes=(128, 64, 32),
                          activation="relu", solver="adam", max_iter=500,
                          early_stopping=True, validation_fraction=0.1,
                          random_state=seed, learning_rate_init=0.001),
        "B6-XGBoost": xgb.XGBClassifier(n_estimators=500, learning_rate=0.05,
                          max_depth=6, use_label_encoder=False,
                          eval_metric="mlogloss", random_state=seed, verbosity=0),
    }

# ── Load datasets (update paths for local runs) ───────────────
print("=" * 72)
print("  SCRIPT F: FULL E1 RERUN — AUC-ROC + XGBoost")
print("=" * 72)

# CloudTask
df_cloud = pd.read_csv(
    "/kaggle/input/datasets/programmer3/cloud-task-scheduling-dataset/Distributed_Task_Scheduling.csv")
df_cloud["priority_label"]    = df_cloud["Task_Priority"].map({1:"Low",2:"Medium",3:"High"})
df_cloud["resource_type_enc"] = LabelEncoder().fit_transform(df_cloud["Resource_Type"])
df_cloud["sched_algo_enc"]    = LabelEncoder().fit_transform(df_cloud["Scheduling_Algorithm"])
CLOUD_FEATURES = [
    "Task_Length_MIPS","Task_Deadline","Data_Upload_Size_MB","Data_Download_Size_MB",
    "VM_MIPS","VM_Memory_GB","VM_Bandwidth_MBps","Execution_Time_S","Waiting_Time_S",
    "Completion_Time_S","Energy_Consumption_J","Makespan_S","Response_Time_S",
    "Execution_Cost_$","Degree_of_Imbalance","Storage_Utilization","Path_Load",
    "resource_type_enc","sched_algo_enc"]

# GoogleCluster
df_google = pd.read_csv(
    "/kaggle/input/datasets/derrickmwiti/google-2019-cluster-sample/borg_traces_data.csv",
    low_memory=False)
def parse_dict_col(series, key):
    def _p(val):
        try:
            d = ast.literal_eval(str(val))
            return d.get(key, np.nan) if isinstance(d, dict) else np.nan
        except:
            return np.nan
    return series.apply(_p)
for k in ["cpus", "memory"]:
    df_google[f"req_{k}"] = parse_dict_col(df_google["resource_request"], k)
    df_google[f"avg_{k}"] = parse_dict_col(df_google["average_usage"], k)
    df_google[f"max_{k}"] = parse_dict_col(df_google["maximum_usage"], k)
df_google["priority_label"] = df_google["priority"].apply(
    lambda p: "Low" if p < 100 else ("Medium" if p < 200 else "High"))
df_google["event_enc"] = LabelEncoder().fit_transform(df_google["event"].astype(str))
for col in ["cycles_per_instruction","memory_accesses_per_instruction"]:
    df_google[col].fillna(df_google[col].median(), inplace=True)
df_google["scheduler"].fillna(0, inplace=True)
df_google["vertical_scaling"].fillna(1, inplace=True)
for col in ["req_cpus","req_memory","avg_cpus","avg_memory","max_cpus","max_memory"]:
    df_google[col].fillna(df_google[col].median(), inplace=True)
GOOGLE_FEATURES = [
    "scheduling_class","collection_type","instance_index","assigned_memory",
    "page_cache_memory","cycles_per_instruction","memory_accesses_per_instruction",
    "sample_rate","scheduler","vertical_scaling","req_cpus","req_memory",
    "avg_cpus","avg_memory","max_cpus","max_memory","failed","event_enc"]

# ITIncident
df_it = pd.read_csv(
    "/kaggle/input/datasets/shamiulislamshifat/it-incident-log-dataset/incident_event_log.csv",
    low_memory=False)
df_it = df_it.sort_values("sys_mod_count").groupby("number").last().reset_index()
df_it["priority_label"] = df_it["priority"].map({
    "1 - Critical":"High","2 - High":"High",
    "3 - Moderate":"Medium","4 - Low":"Low"})
df_it.dropna(subset=["priority_label"], inplace=True)
for cr, ce in [("impact","impact_enc"),("urgency","urgency_enc"),("category","category_enc"),
               ("location","location_enc"),("contact_type","contact_enc")]:
    df_it[ce] = LabelEncoder().fit_transform(df_it[cr].astype(str))
df_it["made_sla_enc"]  = df_it["made_sla"].astype(int)
df_it["knowledge_enc"] = df_it["knowledge"].astype(int)
df_it["reopen_flag"]   = (df_it["reopen_count"] > 0).astype(int)
IT_FEATURES = ["reassignment_count","reopen_count","sys_mod_count","impact_enc",
               "urgency_enc","category_enc","location_enc","contact_enc",
               "made_sla_enc","knowledge_enc","reopen_flag"]

# MultiCloud (clean)
df_mc = pd.read_csv(
    "/kaggle/input/datasets/ziya07/multi-cloud-service-composition-dataset/multi_cloud_service_dataset.csv")
df_mc["service_type_enc"]   = LabelEncoder().fit_transform(df_mc["Service_Type"])
df_mc["cloud_provider_enc"] = LabelEncoder().fit_transform(df_mc["Cloud_Provider"])
df_mc["edge_node_enc"]      = LabelEncoder().fit_transform(df_mc["Edge_Node_ID"])
cpu_n = df_mc["CPU_Utilization (%)"] / 100.0
lat_n = df_mc["Service_Latency (ms)"] / df_mc["Service_Latency (ms)"].max()
thr_n = 1 - (df_mc["Throughput (Requests/sec)"] / df_mc["Throughput (Requests/sec)"].max())
bw_n  = 1 - (df_mc["Network_Bandwidth (Mbps)"] / df_mc["Network_Bandwidth (Mbps)"].max())
wv_n  = df_mc["Workload_Variability"] / df_mc["Workload_Variability"].max()
composite = 0.30*cpu_n + 0.25*lat_n + 0.20*thr_n + 0.15*bw_n + 0.10*wv_n
df_mc["priority_label"] = pd.qcut(composite, q=3,
    labels=["Low","Medium","High"]).astype(str)
MC_FEATURES = [
    "CPU_Utilization (%)","Memory_Usage (MB)","Storage_Usage (GB)",
    "Network_Bandwidth (Mbps)","Service_Latency (ms)","Response_Time (ms)",
    "Throughput (Requests/sec)","Load_Balancing (%)","Workload_Variability",
    "Optimal_Service_Placement","service_type_enc","cloud_provider_enc","edge_node_enc"]

DATASETS = {
    "CloudTask":     (df_cloud,  CLOUD_FEATURES),
    "GoogleCluster": (df_google, GOOGLE_FEATURES),
    "ITIncident":    (df_it,     IT_FEATURES),
    "MultiCloud":    (df_mc,     MC_FEATURES),
}
MODEL_NAMES = ["KATS","B1-LogReg","B2-DecTree","B3-RF","B4-LGB","B5-MLP","B6-XGBoost"]

# ── E1 MAIN LOOP ──────────────────────────────────────────────
e1_results = {}
all_preds  = {}
all_probs  = {}

for ds_name, (df, feats) in DATASETS.items():
    print(f"\n{'='*72}")
    print(f"  Dataset: {ds_name}  ({df.shape[0]:,} rows)")
    print(f"{'='*72}")
    X = df[feats].fillna(0).astype(float)
    y, le, hi = encode_labels(df["priority_label"])
    n_cls  = len(le.classes_)
    scaler = StandardScaler()
    X_s    = scaler.fit_transform(X)
    e1_results[ds_name] = {m: [] for m in MODEL_NAMES}
    all_preds[ds_name]  = {m: ([], []) for m in MODEL_NAMES}
    all_probs[ds_name]  = {m: [] for m in MODEL_NAMES}

    for seed_i, seed in enumerate(SEEDS):
        X_tr, X_te, y_tr, y_te = train_test_split(
            X.values, y, test_size=0.20, random_state=seed, stratify=y)
        Xs_tr, Xs_te, _, _ = train_test_split(
            X_s, y, test_size=0.20, random_state=seed, stratify=y)
        X_tr_s, y_tr_s  = apply_smote(X_tr, y_tr, seed=seed)
        Xs_tr_s, _       = apply_smote(Xs_tr, y_tr, seed=seed)
        cw_s = make_class_weights(y_tr_s, hi, alpha=5)

        # KATS
        kats = get_kats(cw_s, seed)
        kats.fit(X_tr_s, y_tr_s)
        y_pred_k = kats.predict(X_te)
        y_prob_k = kats.predict_proba(X_te)
        e1_results[ds_name]["KATS"].append(
            compute_metrics(y_te, y_pred_k, y_prob_k, le))
        all_preds[ds_name]["KATS"][0].extend(y_te.tolist())
        all_preds[ds_name]["KATS"][1].extend(y_pred_k.tolist())
        all_probs[ds_name]["KATS"].extend(y_prob_k.tolist())

        # Baselines
        for bname, model in get_baselines(cw_s, seed).items():
            use_scaled = (bname == "B5-MLP")
            Xtr_use   = Xs_tr_s if use_scaled else X_tr_s
            Xte_use   = Xs_te   if use_scaled else X_te
            if bname == "B6-XGBoost":
                model.fit(X_tr_s, y_tr_s,
                    sample_weight=np.array([cw_s[c] for c in y_tr_s]))
            else:
                model.fit(Xtr_use, y_tr_s)
            y_pred_b = model.predict(Xte_use)
            try:
                y_prob_b = model.predict_proba(Xte_use)
            except:
                y_prob_b = np.eye(n_cls)[y_pred_b]
            e1_results[ds_name][bname].append(
                compute_metrics(y_te, y_pred_b, y_prob_b, le))
            all_preds[ds_name][bname][0].extend(y_te.tolist())
            all_preds[ds_name][bname][1].extend(y_pred_b.tolist())
            all_probs[ds_name][bname].extend(y_prob_b.tolist())

        print(f"  Seed {seed} done ({seed_i+1}/5)", end="\r")

    # Print results table
    print(f"\n\n  Results — {ds_name}")
    print(f"  {'Model':<18} {'RecallH':>10} {'PrecH':>8} {'MacroF1':>9} "
          f"{'Kappa':>8} {'AUC-ROC':>9}")
    print("  " + "-" * 66)
    for m in MODEL_NAMES:
        rh  = np.mean([x["RecallHigh"] for x in e1_results[ds_name][m]])
        rhs = np.std( [x["RecallHigh"] for x in e1_results[ds_name][m]])
        ph  = np.mean([x["PrecHigh"]   for x in e1_results[ds_name][m]])
        f1  = np.mean([x["MacroF1"]    for x in e1_results[ds_name][m]])
        k   = np.mean([x["Kappa"]      for x in e1_results[ds_name][m]])
        auc = np.mean([x["AUC_ROC"]    for x in e1_results[ds_name][m]])
        print(f"  {m:<18} {rh:.4f}±{rhs:.4f} {ph:>8.4f} {f1:>9.4f} "
              f"{k:>8.4f} {auc:>9.4f}")

with open("/kaggle/working/f_e1_results.pkl", "wb") as f:
    pickle.dump(e1_results, f)
with open("/kaggle/working/f_all_preds.pkl", "wb") as f:
    pickle.dump(all_preds, f)
with open("/kaggle/working/f_all_probs.pkl", "wb") as f:
    pickle.dump(all_probs, f)
print("\n  Script F COMPLETE ✓")
print("  Saved: f_e1_results.pkl | f_all_preds.pkl | f_all_probs.pkl")
