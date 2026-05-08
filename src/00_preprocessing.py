"""
KATS Framework — Dataset Loading & Preprocessing
================================================
Loads and preprocesses all 4 cloud datasets.
Run this first to verify dataset access before running experiments.

Datasets required (Kaggle input paths):
  /kaggle/input/datasets/programmer3/cloud-task-scheduling-dataset/
  /kaggle/input/datasets/derrickmwiti/google-2019-cluster-sample/
  /kaggle/input/datasets/shamiulislamshifat/it-incident-log-dataset/
  /kaggle/input/datasets/ziya07/multi-cloud-service-composition-dataset/
"""

import pandas as pd
import numpy as np
import ast
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import LabelEncoder

SEED = 42
np.random.seed(SEED)

# =============================================================================
# CORE UTILITIES
# =============================================================================

def encode_labels_fixed(y_series):
    """Encode labels alphabetically: High=0, Low=1, Medium=2.
    Returns encoded array, fitted LabelEncoder, and index of 'High' class."""
    le = LabelEncoder()
    y_enc = le.fit_transform(y_series.astype(str))
    high_idx = int(np.where(le.classes_ == "High")[0][0]) if "High" in le.classes_ else 0
    return y_enc, le, high_idx


def make_class_weights(y_enc, high_idx, alpha=5):
    """
    Build asymmetric class weight dictionary.
    Alpha multiplies the weight of the High class.
    Uses actual integer keys from encoded labels.
    """
    classes, counts = np.unique(y_enc, return_counts=True)
    total = len(y_enc)
    n_classes = len(classes)
    cw = {int(c): total / (n_classes * cnt) for c, cnt in zip(classes, counts)}
    cw[high_idx] = cw[high_idx] * alpha
    return cw


def apply_smote_safe(X_tr, y_tr, seed=42):
    """Apply SMOTE with adaptive k_neighbors for small minority classes."""
    from imblearn.over_sampling import SMOTE
    counts = np.bincount(y_tr)
    min_count = counts.min()
    k = max(1, min_count - 1) if min_count < 6 else 5
    try:
        X_res, y_res = SMOTE(random_state=seed, k_neighbors=k).fit_resample(X_tr, y_tr)
        return X_res, y_res
    except Exception:
        return X_tr, y_tr


def compute_metrics(y_true, y_pred, le):
    """Compute all evaluation metrics including High-class specific metrics."""
    from sklearn.metrics import classification_report, cohen_kappa_score
    labels = le.classes_.tolist()
    report = classification_report(y_true, y_pred, target_names=labels,
                                   output_dict=True, zero_division=0)
    return {
        "RecallHigh":  report.get("High", {}).get("recall",    0.0),
        "PrecHigh":    report.get("High", {}).get("precision", 0.0),
        "F1High":      report.get("High", {}).get("f1-score",  0.0),
        "MacroF1":     report["macro avg"]["f1-score"],
        "MacroRecall": report["macro avg"]["recall"],
        "Kappa":       cohen_kappa_score(y_true, y_pred),
    }


# =============================================================================
# 1A. CLOUD TASK SCHEDULING
# =============================================================================

def load_cloud_task(path="/kaggle/input/datasets/programmer3/cloud-task-scheduling-dataset/Distributed_Task_Scheduling.csv"):
    df = pd.read_csv(path)
    df["priority_label"]    = df["Task_Priority"].map({1: "Low", 2: "Medium", 3: "High"})
    df["resource_type_enc"] = LabelEncoder().fit_transform(df["Resource_Type"])
    df["sched_algo_enc"]    = LabelEncoder().fit_transform(df["Scheduling_Algorithm"])
    features = [
        "Task_Length_MIPS", "Task_Deadline", "Data_Upload_Size_MB",
        "Data_Download_Size_MB", "VM_MIPS", "VM_Memory_GB", "VM_Bandwidth_MBps",
        "Execution_Time_S", "Waiting_Time_S", "Completion_Time_S",
        "Energy_Consumption_J", "Makespan_S", "Response_Time_S",
        "Execution_Cost_$", "Degree_of_Imbalance", "Storage_Utilization",
        "Path_Load", "resource_type_enc", "sched_algo_enc"
    ]
    print(f"[CloudTask] {df.shape[0]:,} rows | {df['priority_label'].value_counts().to_dict()}")
    return df, features


# =============================================================================
# 1B. GOOGLE CLUSTER TRACES
# =============================================================================

def load_google_cluster(path="/kaggle/input/datasets/derrickmwiti/google-2019-cluster-sample/borg_traces_data.csv"):
    df = pd.read_csv(path, low_memory=False)

    def parse_dict_col(series, key):
        def _parse(val):
            try:
                d = ast.literal_eval(str(val))
                return d.get(key, np.nan) if isinstance(d, dict) else np.nan
            except:
                return np.nan
        return series.apply(_parse)

    for k in ["cpus", "memory"]:
        df[f"req_{k}"]  = parse_dict_col(df["resource_request"], k)
        df[f"avg_{k}"]  = parse_dict_col(df["average_usage"],    k)
        df[f"max_{k}"]  = parse_dict_col(df["maximum_usage"],    k)

    df["priority_label"] = df["priority"].apply(
        lambda p: "Low" if p < 100 else ("Medium" if p < 200 else "High"))
    df["event_enc"] = LabelEncoder().fit_transform(df["event"].astype(str))

    for col in ["cycles_per_instruction", "memory_accesses_per_instruction"]:
        df[col].fillna(df[col].median(), inplace=True)
    df["scheduler"].fillna(0, inplace=True)
    df["vertical_scaling"].fillna(1, inplace=True)
    for col in ["req_cpus","req_memory","avg_cpus","avg_memory","max_cpus","max_memory"]:
        df[col].fillna(df[col].median(), inplace=True)

    features = [
        "scheduling_class", "collection_type", "instance_index",
        "assigned_memory", "page_cache_memory", "cycles_per_instruction",
        "memory_accesses_per_instruction", "sample_rate", "scheduler",
        "vertical_scaling", "req_cpus", "req_memory",
        "avg_cpus", "avg_memory", "max_cpus", "max_memory", "failed", "event_enc"
    ]
    print(f"[GoogleCluster] {df.shape[0]:,} rows | {df['priority_label'].value_counts().to_dict()}")
    return df, features


# =============================================================================
# 1C. IT INCIDENT LOG
# =============================================================================

def load_it_incident(path="/kaggle/input/datasets/shamiulislamshifat/it-incident-log-dataset/incident_event_log.csv"):
    df = pd.read_csv(path, low_memory=False)
    df = df.sort_values("sys_mod_count").groupby("number").last().reset_index()

    PRIORITY_MAP = {
        "1 - Critical": "High", "2 - High": "High",
        "3 - Moderate": "Medium", "4 - Low": "Low"
    }
    df["priority_label"] = df["priority"].map(PRIORITY_MAP)
    df.dropna(subset=["priority_label"], inplace=True)

    for col_raw, col_enc in [
        ("impact", "impact_enc"), ("urgency", "urgency_enc"),
        ("category", "category_enc"), ("location", "location_enc"),
        ("contact_type", "contact_enc")
    ]:
        df[col_enc] = LabelEncoder().fit_transform(df[col_raw].astype(str))

    df["made_sla_enc"]  = df["made_sla"].astype(int)
    df["knowledge_enc"] = df["knowledge"].astype(int)
    df["reopen_flag"]   = (df["reopen_count"] > 0).astype(int)

    features = [
        "reassignment_count", "reopen_count", "sys_mod_count",
        "impact_enc", "urgency_enc", "category_enc", "location_enc",
        "contact_enc", "made_sla_enc", "knowledge_enc", "reopen_flag"
    ]
    print(f"[ITIncident] {df.shape[0]:,} rows | {df['priority_label'].value_counts().to_dict()}")
    return df, features


# =============================================================================
# 1D. MULTI-CLOUD SERVICE
# =============================================================================

def load_multi_cloud(path="/kaggle/input/datasets/ziya07/multi-cloud-service-composition-dataset/multi_cloud_service_dataset.csv"):
    df = pd.read_csv(path)
    df["priority_label"]      = pd.qcut(df["QoS_Score"], q=3, labels=["Low","Medium","High"]).astype(str)
    df["service_type_enc"]    = LabelEncoder().fit_transform(df["Service_Type"])
    df["cloud_provider_enc"]  = LabelEncoder().fit_transform(df["Cloud_Provider"])
    df["edge_node_enc"]       = LabelEncoder().fit_transform(df["Edge_Node_ID"])

    features = [
        "CPU_Utilization (%)", "Memory_Usage (MB)", "Storage_Usage (GB)",
        "Network_Bandwidth (Mbps)", "Service_Latency (ms)", "Response_Time (ms)",
        "Throughput (Requests/sec)", "Load_Balancing (%)", "QoS_Score",
        "Workload_Variability", "Optimal_Service_Placement",
        "service_type_enc", "cloud_provider_enc", "edge_node_enc"
    ]
    print(f"[MultiCloud] {df.shape[0]:,} rows | {df['priority_label'].value_counts().to_dict()}")
    return df, features


# =============================================================================
# MODEL FACTORIES
# =============================================================================

def get_kats_ensemble(cw_dict, seed=42):
    """Build KATS stacked ensemble with asymmetric class weights."""
    from sklearn.ensemble import RandomForestClassifier, StackingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.calibration import CalibratedClassifierCV
    import lightgbm as lgb

    return StackingClassifier(
        estimators=[
            ("lgb", lgb.LGBMClassifier(
                n_estimators=500, learning_rate=0.05, max_depth=6,
                num_leaves=31, class_weight=cw_dict,
                random_state=seed, verbose=-1)),
            ("rf",  RandomForestClassifier(
                n_estimators=300, max_depth=None, min_samples_leaf=2,
                class_weight="balanced", random_state=seed)),
            ("nb",  CalibratedClassifierCV(GaussianNB(), cv=5, method="isotonic")),
        ],
        final_estimator=LogisticRegression(
            C=1.0, max_iter=2000, solver="lbfgs",
            multi_class="multinomial", random_state=seed),
        cv=5, passthrough=False
    )


def get_baselines(seed=42):
    """Return all 5 baseline classifiers."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    import lightgbm as lgb
    return {
        "B1-LogReg":  LogisticRegression(max_iter=2000, random_state=seed, class_weight="balanced"),
        "B2-DecTree": DecisionTreeClassifier(random_state=seed, class_weight="balanced"),
        "B3-RF":      RandomForestClassifier(n_estimators=300, random_state=seed, class_weight="balanced"),
        "B4-LGB":     lgb.LGBMClassifier(n_estimators=500, random_state=seed, class_weight="balanced", verbose=-1),
        "B5-MLP":     MLPClassifier(hidden_layer_sizes=(128,64,32), activation="relu",
                                    solver="adam", max_iter=500, early_stopping=True,
                                    validation_fraction=0.1, random_state=seed),
    }


if __name__ == "__main__":
    print("Loading all datasets...")
    df_cloud,  CLOUD_FEATURES  = load_cloud_task()
    df_google, GOOGLE_FEATURES = load_google_cluster()
    df_it,     IT_FEATURES     = load_it_incident()
    df_mc,     MC_FEATURES     = load_multi_cloud()
    print("\nAll datasets loaded successfully.")
