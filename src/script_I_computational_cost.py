# ================================================================
# SCRIPT I — COMPUTATIONAL COST ANALYSIS
# Measures: Training time (s), Inference latency (ms/sample),
#           Memory footprint (MB)
# Why: Required by IEEE TCC for systems papers
# Runs ALL 7 models × 4 datasets — single seed (42) for timing
# Self-contained. Saves i_cost_results.pkl + prints paper table
# Runtime: ~30 min
# ================================================================
import pandas as pd
import numpy as np
import ast, warnings, pickle, time, os, tracemalloc
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score
import lightgbm as lgb
import xgboost as xgb
from imblearn.over_sampling import SMOTE

SEED = 42
np.random.seed(SEED)

# ── Utilities ─────────────────────────────────────────────────
def encode_labels(y_series):
    le = LabelEncoder()
    y_enc = le.fit_transform(y_series.astype(str))
    high_idx = int(np.where(le.classes_ == "High")[0][0])
    return y_enc, le, high_idx

def make_class_weights(y_enc, high_idx, alpha=5):
    classes, counts = np.unique(y_enc, return_counts=True)
    total = len(y_enc)
    cw = {int(c): total/(len(classes)*cnt) for c,cnt in zip(classes,counts)}
    cw[high_idx] *= alpha
    return cw

def apply_smote(X, y, seed=42):
    counts = np.bincount(y)
    k = max(1, counts.min()-1) if counts.min() < 6 else 5
    try:
        return SMOTE(random_state=seed, k_neighbors=k).fit_resample(X, y)
    except:
        return X, y

def get_kats(cw, seed=42):
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

def get_all_models(cw, seed=42):
    return {
        "KATS":       get_kats(cw, seed),
        "B1-LogReg":  LogisticRegression(max_iter=2000, random_state=seed,
                          class_weight="balanced"),
        "B2-DecTree": DecisionTreeClassifier(random_state=seed,
                          class_weight="balanced"),
        "B3-RF":      RandomForestClassifier(n_estimators=300, random_state=seed,
                          class_weight="balanced"),
        "B4-LGB":     lgb.LGBMClassifier(n_estimators=500, random_state=seed,
                          class_weight="balanced", verbose=-1),
        "B5-MLP":     MLPClassifier(hidden_layer_sizes=(128,64,32), activation="relu",
                          solver="adam", max_iter=500, early_stopping=True,
                          validation_fraction=0.1, random_state=seed,
                          learning_rate_init=0.001),
        "B6-XGBoost": xgb.XGBClassifier(n_estimators=500, learning_rate=0.05,
                          max_depth=6, eval_metric="mlogloss",
                          random_state=seed, verbosity=0),
    }

# ── Dataset loading ────────────────────────────────────────────
df_cloud = pd.read_csv(
    "/kaggle/input/datasets/programmer3/cloud-task-scheduling-dataset/Distributed_Task_Scheduling.csv")
df_cloud["priority_label"]    = df_cloud["Task_Priority"].map({1:"Low",2:"Medium",3:"High"})
df_cloud["resource_type_enc"] = LabelEncoder().fit_transform(df_cloud["Resource_Type"])
df_cloud["sched_algo_enc"]    = LabelEncoder().fit_transform(df_cloud["Scheduling_Algorithm"])
CLOUD_F = ["Task_Length_MIPS","Task_Deadline","Data_Upload_Size_MB","Data_Download_Size_MB",
    "VM_MIPS","VM_Memory_GB","VM_Bandwidth_MBps","Execution_Time_S","Waiting_Time_S",
    "Completion_Time_S","Energy_Consumption_J","Makespan_S","Response_Time_S",
    "Execution_Cost_$","Degree_of_Imbalance","Storage_Utilization","Path_Load",
    "resource_type_enc","sched_algo_enc"]

df_google = pd.read_csv(
    "/kaggle/input/datasets/derrickmwiti/google-2019-cluster-sample/borg_traces_data.csv",
    low_memory=False)
def parse_dict_col(series, key):
    def _p(val):
        try:
            d = ast.literal_eval(str(val))
            return d.get(key, np.nan) if isinstance(d, dict) else np.nan
        except: return np.nan
    return series.apply(_p)
for k in ["cpus","memory"]:
    df_google[f"req_{k}"] = parse_dict_col(df_google["resource_request"], k)
    df_google[f"avg_{k}"] = parse_dict_col(df_google["average_usage"], k)
    df_google[f"max_{k}"] = parse_dict_col(df_google["maximum_usage"], k)
df_google["priority_label"] = df_google["priority"].apply(
    lambda p: "Low" if p<100 else ("Medium" if p<200 else "High"))
df_google["event_enc"] = LabelEncoder().fit_transform(df_google["event"].astype(str))
for col in ["cycles_per_instruction","memory_accesses_per_instruction"]:
    df_google[col].fillna(df_google[col].median(), inplace=True)
df_google["scheduler"].fillna(0, inplace=True)
df_google["vertical_scaling"].fillna(1, inplace=True)
for col in ["req_cpus","req_memory","avg_cpus","avg_memory","max_cpus","max_memory"]:
    df_google[col].fillna(df_google[col].median(), inplace=True)
GOOGLE_F = ["scheduling_class","collection_type","instance_index","assigned_memory",
    "page_cache_memory","cycles_per_instruction","memory_accesses_per_instruction",
    "sample_rate","scheduler","vertical_scaling","req_cpus","req_memory",
    "avg_cpus","avg_memory","max_cpus","max_memory","failed","event_enc"]

df_it = pd.read_csv(
    "/kaggle/input/datasets/shamiulislamshifat/it-incident-log-dataset/incident_event_log.csv",
    low_memory=False)
df_it = df_it.sort_values("sys_mod_count").groupby("number").last().reset_index()
df_it["priority_label"] = df_it["priority"].map({
    "1 - Critical":"High","2 - High":"High","3 - Moderate":"Medium","4 - Low":"Low"})
df_it.dropna(subset=["priority_label"], inplace=True)
for cr, ce in [("impact","impact_enc"),("urgency","urgency_enc"),("category","category_enc"),
               ("location","location_enc"),("contact_type","contact_enc")]:
    df_it[ce] = LabelEncoder().fit_transform(df_it[cr].astype(str))
df_it["made_sla_enc"]  = df_it["made_sla"].astype(int)
df_it["knowledge_enc"] = df_it["knowledge"].astype(int)
df_it["reopen_flag"]   = (df_it["reopen_count"] > 0).astype(int)
IT_F = ["reassignment_count","reopen_count","sys_mod_count","impact_enc","urgency_enc",
    "category_enc","location_enc","contact_enc","made_sla_enc","knowledge_enc","reopen_flag"]

df_mc = pd.read_csv(
    "/kaggle/input/datasets/ziya07/multi-cloud-service-composition-dataset/multi_cloud_service_dataset.csv")
df_mc["service_type_enc"]   = LabelEncoder().fit_transform(df_mc["Service_Type"])
df_mc["cloud_provider_enc"] = LabelEncoder().fit_transform(df_mc["Cloud_Provider"])
df_mc["edge_node_enc"]      = LabelEncoder().fit_transform(df_mc["Edge_Node_ID"])
cpu_n = df_mc["CPU_Utilization (%)"]/100.0
lat_n = df_mc["Service_Latency (ms)"]/df_mc["Service_Latency (ms)"].max()
thr_n = 1-(df_mc["Throughput (Requests/sec)"]/df_mc["Throughput (Requests/sec)"].max())
bw_n  = 1-(df_mc["Network_Bandwidth (Mbps)"]/df_mc["Network_Bandwidth (Mbps)"].max())
wv_n  = df_mc["Workload_Variability"]/df_mc["Workload_Variability"].max()
composite = 0.30*cpu_n + 0.25*lat_n + 0.20*thr_n + 0.15*bw_n + 0.10*wv_n
df_mc["priority_label"] = pd.qcut(composite,q=3,labels=["Low","Medium","High"]).astype(str)
MC_F = ["CPU_Utilization (%)","Memory_Usage (MB)","Storage_Usage (GB)",
    "Network_Bandwidth (Mbps)","Service_Latency (ms)","Response_Time (ms)",
    "Throughput (Requests/sec)","Load_Balancing (%)","Workload_Variability",
    "Optimal_Service_Placement","service_type_enc","cloud_provider_enc","edge_node_enc"]

DATASETS = {
    "CloudTask":     (df_cloud,  CLOUD_F),
    "GoogleCluster": (df_google, GOOGLE_F),
    "ITIncident":    (df_it,     IT_F),
    "MultiCloud":    (df_mc,     MC_F),
}

MODEL_NAMES = ["KATS","B1-LogReg","B2-DecTree","B3-RF","B4-LGB","B5-MLP","B6-XGBoost"]
TIMING_REPS = 3
cost_results = {}

for ds_name, (df, feats) in DATASETS.items():
    print(f"\n{'='*72}\n  Dataset: {ds_name}  ({df.shape[0]:,} rows × {len(feats)} features)\n{'='*72}")
    X = df[feats].fillna(0).astype(float).values
    y, le, hi = encode_labels(df["priority_label"])
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=SEED, stratify=y)
    Xs_tr, Xs_te, _, _ = train_test_split(X_s, y, test_size=0.20, random_state=SEED, stratify=y)
    X_tr_s, y_tr_s = apply_smote(X_tr, y_tr, seed=SEED)
    Xs_tr_s, _     = apply_smote(Xs_tr, y_tr, seed=SEED)
    cw_s = make_class_weights(y_tr_s, hi, alpha=5)
    cost_results[ds_name] = {}
    n_test = X_te.shape[0]
    print(f"  Train (post-SMOTE): {X_tr_s.shape[0]:,} | Test: {n_test:,}")
    print(f"  {'Model':<18} {'TrainTime(s)':>14} {'Infer(ms/samp)':>16} {'PeakMem(MB)':>13} {'MacroF1':>9}")
    print("  " + "-" * 74)
    for mname in MODEL_NAMES:
        models = get_all_models(cw_s, SEED)
        model  = models[mname]
        use_scaled = (mname == "B5-MLP")
        Xtr_use = Xs_tr_s if use_scaled else X_tr_s
        Xte_use = Xs_te   if use_scaled else X_te
        tracemalloc.start()
        t0 = time.perf_counter()
        if mname == "B6-XGBoost":
            model.fit(X_tr_s, y_tr_s, sample_weight=np.array([cw_s[c] for c in y_tr_s]))
        else:
            model.fit(Xtr_use, y_tr_s)
        train_time = time.perf_counter() - t0
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_mb = peak_mem / (1024 * 1024)
        infer_times = []
        for _ in range(TIMING_REPS):
            t1 = time.perf_counter()
            _ = model.predict(Xte_use)
            infer_times.append(time.perf_counter() - t1)
        infer_ms = (np.mean(infer_times) / n_test) * 1000
        y_pred = model.predict(Xte_use)
        f1 = f1_score(y_te, y_pred, average="macro", zero_division=0)
        cost_results[ds_name][mname] = {
            "train_s": round(train_time, 3), "infer_ms": round(infer_ms, 4),
            "mem_mb":  round(peak_mb, 2),    "macro_f1": round(f1, 4),
        }
        print(f"  {mname:<18} {train_time:>14.3f} {infer_ms:>16.4f} {peak_mb:>13.2f} {f1:>9.4f}")

with open("/kaggle/working/i_cost_results.pkl","wb") as f:
    pickle.dump(cost_results, f)
print("\n  Script I COMPLETE ✓  (saved i_cost_results.pkl)")
