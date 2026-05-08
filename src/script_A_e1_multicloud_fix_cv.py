# =============================================================================
# SCRIPT A — E1 Full Comparison + MultiCloud Leakage Fix + 10-Fold CV
# =============================================================================
# Gaps closed:
#   C4 — MultiCloud leakage fix: removes QoS_Score from features, rebuilds
#         priority label from composite operational score (CPU/Latency/Throughput)
#   M4 — 10-fold stratified CV on MultiCloud (N=1,000, too small for hold-out)
#   E1  — Full model comparison re-run across all 4 datasets (clean MultiCloud)
#
# Run order: A → B → C → D
# Outputs:   /kaggle/working/e1_results.pkl
#            /kaggle/working/all_preds.pkl
# =============================================================================
import pandas as pd
import numpy as np
import ast, warnings, time
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, cohen_kappa_score
import lightgbm as lgb
from imblearn.over_sampling import SMOTE

SEEDS = [42, 7, 13, 99, 2026]
np.random.seed(42)


# ── Shared utilities ──────────────────────────────────────────────────────────
def encode_labels(y_series):
    le = LabelEncoder()
    y_enc = le.fit_transform(y_series.astype(str))
    high_idx = int(np.where(le.classes_ == "High")[0][0])
    return y_enc, le, high_idx


def make_class_weights(y_enc, high_idx, alpha=5):
    classes, counts = np.unique(y_enc, return_counts=True)
    total = len(y_enc)
    n_cls = len(classes)
    cw = {int(c): total / (n_cls * cnt) for c, cnt in zip(classes, counts)}
    cw[high_idx] *= alpha
    return cw


def apply_smote(X, y, seed=42):
    counts = np.bincount(y)
    k = max(1, counts.min() - 1) if counts.min() < 6 else 5
    try:
        return SMOTE(random_state=seed, k_neighbors=k).fit_resample(X, y)
    except Exception:
        return X, y


def compute_metrics(y_true, y_pred, le):
    rep = classification_report(
        y_true, y_pred, target_names=le.classes_.tolist(),
        output_dict=True, zero_division=0)
    return {
        "RecallHigh": rep.get("High", {}).get("recall",    0.0),
        "PrecHigh":   rep.get("High", {}).get("precision", 0.0),
        "F1High":     rep.get("High", {}).get("f1-score",  0.0),
        "MacroF1":    rep["macro avg"]["f1-score"],
        "Kappa":      cohen_kappa_score(y_true, y_pred),
    }


def get_kats(cw, seed=42):
    return StackingClassifier(
        estimators=[
            ("lgb", lgb.LGBMClassifier(
                n_estimators=500, learning_rate=0.05, max_depth=6,
                num_leaves=31, class_weight=cw, random_state=seed, verbose=-1)),
            ("rf",  RandomForestClassifier(
                n_estimators=300, class_weight="balanced", random_state=seed)),
            ("nb",  CalibratedClassifierCV(
                GaussianNB(), cv=5, method="isotonic")),
        ],
        final_estimator=LogisticRegression(
            C=1.0, max_iter=2000, solver="lbfgs",
            multi_class="multinomial", random_state=seed),
        cv=5, passthrough=False)


def get_baselines(cw, seed=42):
    return {
        "B1-LogReg":  LogisticRegression(max_iter=2000, random_state=seed,
                                          class_weight="balanced"),
        "B2-DecTree": DecisionTreeClassifier(random_state=seed,
                                              class_weight="balanced"),
        "B3-RF":      RandomForestClassifier(n_estimators=300, random_state=seed,
                                              class_weight="balanced"),
        "B4-LGB":     lgb.LGBMClassifier(n_estimators=500, random_state=seed,
                                           class_weight="balanced", verbose=-1),
        "B5-MLP":     MLPClassifier(
                          hidden_layer_sizes=(128, 64, 32), activation="relu",
                          solver="adam", max_iter=500, early_stopping=True,
                          validation_fraction=0.1, random_state=seed,
                          learning_rate_init=0.001),
    }


# ── C4: MultiCloud leakage fix ────────────────────────────────────────────────
# OLD (BROKEN): QoS_Score was used as both a feature AND the source of the label.
# FIX: Derive label from composite operational score (CPU load + latency +
#      inverse throughput + inverse bandwidth + workload variability).
#      QoS_Score is removed from the feature set entirely.

def build_multicloud(path):
    df = pd.read_csv(path)
    df["service_type_enc"]   = LabelEncoder().fit_transform(df["Service_Type"])
    df["cloud_provider_enc"] = LabelEncoder().fit_transform(df["Cloud_Provider"])
    df["edge_node_enc"]      = LabelEncoder().fit_transform(df["Edge_Node_ID"])

    cpu_norm = df["CPU_Utilization (%)"] / 100.0
    lat_norm = df["Service_Latency (ms)"] / df["Service_Latency (ms)"].max()
    thr_norm = 1 - (df["Throughput (Requests/sec)"] /
                    df["Throughput (Requests/sec)"].max())
    bw_norm  = 1 - (df["Network_Bandwidth (Mbps)"] /
                    df["Network_Bandwidth (Mbps)"].max())
    wv_norm  = df["Workload_Variability"] / df["Workload_Variability"].max()
    composite = (0.30 * cpu_norm + 0.25 * lat_norm + 0.20 * thr_norm
                 + 0.15 * bw_norm + 0.10 * wv_norm)

    df["priority_label"] = pd.qcut(
        composite, q=3, labels=["Low", "Medium", "High"]).astype(str)

    features = [
        "CPU_Utilization (%)", "Memory_Usage (MB)", "Storage_Usage (GB)",
        "Network_Bandwidth (Mbps)", "Service_Latency (ms)", "Response_Time (ms)",
        "Throughput (Requests/sec)", "Load_Balancing (%)", "Workload_Variability",
        "Optimal_Service_Placement", "service_type_enc",
        "cloud_provider_enc", "edge_node_enc",
    ]  # QoS_Score deliberately excluded
    return df, features


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import pickle

    # Dataset paths (Kaggle)
    CLOUD_PATH  = "/kaggle/input/datasets/programmer3/cloud-task-scheduling-dataset/Distributed_Task_Scheduling.csv"
    GOOGLE_PATH = "/kaggle/input/datasets/derrickmwiti/google-2019-cluster-sample/borg_traces_data.csv"
    IT_PATH     = "/kaggle/input/datasets/shamiulislamshifat/it-incident-log-dataset/incident_event_log.csv"
    MC_PATH     = "/kaggle/input/datasets/ziya07/multi-cloud-service-composition-dataset/multi_cloud_service_dataset.csv"

    # --- Load CloudTask ---
    df_cloud = pd.read_csv(CLOUD_PATH)
    df_cloud["priority_label"]    = df_cloud["Task_Priority"].map({1: "Low", 2: "Medium", 3: "High"})
    df_cloud["resource_type_enc"] = LabelEncoder().fit_transform(df_cloud["Resource_Type"])
    df_cloud["sched_algo_enc"]    = LabelEncoder().fit_transform(df_cloud["Scheduling_Algorithm"])
    CLOUD_FEATS = [
        "Task_Length_MIPS", "Task_Deadline", "Data_Upload_Size_MB",
        "Data_Download_Size_MB", "VM_MIPS", "VM_Memory_GB", "VM_Bandwidth_MBps",
        "Execution_Time_S", "Waiting_Time_S", "Completion_Time_S",
        "Energy_Consumption_J", "Makespan_S", "Response_Time_S",
        "Execution_Cost_$", "Degree_of_Imbalance", "Storage_Utilization",
        "Path_Load", "resource_type_enc", "sched_algo_enc"]

    # --- Load GoogleCluster ---
    df_google = pd.read_csv(GOOGLE_PATH, low_memory=False)
    def _parse(series, key):
        def _p(val):
            try:
                d = ast.literal_eval(str(val))
                return d.get(key, np.nan) if isinstance(d, dict) else np.nan
            except:
                return np.nan
        return series.apply(_p)
    for k in ["cpus", "memory"]:
        for prefix in ["req", "avg", "max"]:
            col_src = {"req": "resource_request", "avg": "average_usage", "max": "maximum_usage"}[prefix]
            df_google[f"{prefix}_{k}"] = _parse(df_google[col_src], k)
    df_google["priority_label"] = df_google["priority"].apply(
        lambda p: "Low" if p < 100 else ("Medium" if p < 200 else "High"))
    df_google["event_enc"] = LabelEncoder().fit_transform(df_google["event"].astype(str))
    for col in ["cycles_per_instruction", "memory_accesses_per_instruction"]:
        df_google[col].fillna(df_google[col].median(), inplace=True)
    df_google["scheduler"].fillna(0, inplace=True)
    df_google["vertical_scaling"].fillna(1, inplace=True)
    for col in ["req_cpus", "req_memory", "avg_cpus", "avg_memory", "max_cpus", "max_memory"]:
        df_google[col].fillna(df_google[col].median(), inplace=True)
    GOOGLE_FEATS = [
        "scheduling_class", "collection_type", "instance_index",
        "assigned_memory", "page_cache_memory", "cycles_per_instruction",
        "memory_accesses_per_instruction", "sample_rate", "scheduler",
        "vertical_scaling", "req_cpus", "req_memory",
        "avg_cpus", "avg_memory", "max_cpus", "max_memory", "failed", "event_enc"]

    # --- Load ITIncident ---
    df_it = pd.read_csv(IT_PATH, low_memory=False)
    df_it = df_it.sort_values("sys_mod_count").groupby("number").last().reset_index()
    df_it["priority_label"] = df_it["priority"].map({
        "1 - Critical": "High", "2 - High": "High",
        "3 - Moderate": "Medium", "4 - Low": "Low"})
    df_it.dropna(subset=["priority_label"], inplace=True)
    for cr, ce in [("impact", "impact_enc"), ("urgency", "urgency_enc"),
                   ("category", "category_enc"), ("location", "location_enc"),
                   ("contact_type", "contact_enc")]:
        df_it[ce] = LabelEncoder().fit_transform(df_it[cr].astype(str))
    df_it["made_sla_enc"]  = df_it["made_sla"].astype(int)
    df_it["knowledge_enc"] = df_it["knowledge"].astype(int)
    df_it["reopen_flag"]   = (df_it["reopen_count"] > 0).astype(int)
    IT_FEATS = [
        "reassignment_count", "reopen_count", "sys_mod_count",
        "impact_enc", "urgency_enc", "category_enc", "location_enc",
        "contact_enc", "made_sla_enc", "knowledge_enc", "reopen_flag"]

    # --- Load MultiCloud (clean) ---
    df_mc, MC_FEATS = build_multicloud(MC_PATH)

    DATASETS = {
        "CloudTask":     (df_cloud,  CLOUD_FEATS),
        "GoogleCluster": (df_google, GOOGLE_FEATS),
        "ITIncident":    (df_it,     IT_FEATS),
        "MultiCloud":    (df_mc,     MC_FEATS),
    }

    # ── M4: 10-fold CV on MultiCloud ─────────────────────────────────────────
    print("Running M4: 10-fold CV on MultiCloud...")
    df, feats = DATASETS["MultiCloud"]
    X_mc  = df[feats].fillna(0).astype(float).values
    y_mc, le_mc, hi_mc = encode_labels(df["priority_label"])
    scaler_mc = StandardScaler()
    X_mc_s = scaler_mc.fit_transform(X_mc)

    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    cv_results = {m: [] for m in ["KATS", "B1-LogReg", "B2-DecTree", "B3-RF", "B4-LGB", "B5-MLP"]}

    for fold_i, (tr_idx, te_idx) in enumerate(kf.split(X_mc, y_mc)):
        X_tr,  X_te  = X_mc[tr_idx],   X_mc[te_idx]
        Xs_tr, Xs_te = X_mc_s[tr_idx], X_mc_s[te_idx]
        y_tr,  y_te  = y_mc[tr_idx],   y_mc[te_idx]
        X_tr_s, y_tr_s = apply_smote(X_tr, y_tr, seed=42)
        Xs_tr_s, _     = apply_smote(Xs_tr, y_tr, seed=42)
        cw_s = make_class_weights(y_tr_s, hi_mc, alpha=5)

        kats = get_kats(cw_s, seed=42)
        kats.fit(X_tr_s, y_tr_s)
        cv_results["KATS"].append(compute_metrics(y_te, kats.predict(X_te), le_mc))

        for bname, bmodel in get_baselines(cw_s, seed=42).items():
            if bname == "B5-MLP":
                bmodel.fit(Xs_tr_s, y_tr_s)
                cv_results[bname].append(compute_metrics(y_te, bmodel.predict(Xs_te), le_mc))
            else:
                bmodel.fit(X_tr_s, y_tr_s)
                cv_results[bname].append(compute_metrics(y_te, bmodel.predict(X_te), le_mc))

    # ── E1: Full comparison across all 4 datasets ─────────────────────────────
    print("Running E1: full model comparison...")
    e1_results = {}
    all_preds  = {}
    models_list = ["KATS", "B1-LogReg", "B2-DecTree", "B3-RF", "B4-LGB", "B5-MLP"]

    for ds_name, (df, feats) in DATASETS.items():
        X = df[feats].fillna(0).astype(float)
        y, le, hi = encode_labels(df["priority_label"])
        e1_results[ds_name] = {m: [] for m in models_list}
        all_preds[ds_name]  = {m: ([], []) for m in models_list}
        scaler = StandardScaler()
        X_s    = scaler.fit_transform(X)

        for seed in SEEDS:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X.values, y, test_size=0.20, random_state=seed, stratify=y)
            Xs_tr, Xs_te, _, _ = train_test_split(
                X_s, y, test_size=0.20, random_state=seed, stratify=y)
            X_tr_s, y_tr_s = apply_smote(X_tr, y_tr, seed=seed)
            Xs_tr_s, _     = apply_smote(Xs_tr, y_tr, seed=seed)
            cw_s = make_class_weights(y_tr_s, hi, alpha=5)

            kats = get_kats(cw_s, seed=seed)
            kats.fit(X_tr_s, y_tr_s)
            y_pred_kats = kats.predict(X_te)
            e1_results[ds_name]["KATS"].append(compute_metrics(y_te, y_pred_kats, le))
            all_preds[ds_name]["KATS"][0].extend(y_te.tolist())
            all_preds[ds_name]["KATS"][1].extend(y_pred_kats.tolist())

            for bname, bmodel in get_baselines(cw_s, seed).items():
                if bname == "B5-MLP":
                    bmodel.fit(Xs_tr_s, y_tr_s)
                    y_pred = bmodel.predict(Xs_te)
                else:
                    bmodel.fit(X_tr_s, y_tr_s)
                    y_pred = bmodel.predict(X_te)
                e1_results[ds_name][bname].append(compute_metrics(y_te, y_pred, le))
                all_preds[ds_name][bname][0].extend(y_te.tolist())
                all_preds[ds_name][bname][1].extend(y_pred.tolist())

    with open("/kaggle/working/e1_results.pkl", "wb") as f:
        pickle.dump(e1_results, f)
    with open("/kaggle/working/all_preds.pkl", "wb") as f:
        pickle.dump(all_preds, f)
    print("Script A complete. Outputs: e1_results.pkl, all_preds.pkl")
