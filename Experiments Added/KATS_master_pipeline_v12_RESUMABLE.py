"""
KATS MASTER PIPELINE v12 — CRASH-HARDENED + CHECKPOINT/RESUMABLE
====================================================================
Fixes vs. v11 (which crashed on your Kaggle image):

  CRASH FIX #1: Removed `use_label_encoder=False` from every
  XGBClassifier call. This parameter was REMOVED from XGBoost in
  v1.6+ and raises a hard error on Kaggle's current default image.
  This was already broken in your original v9_SUBMISSION.py too -
  it just hadn't been re-run since Kaggle updated XGBoost.

  CRASH FIX #2: roc_auc_score() was being called with a pre-binarized
  y_true AND multi_class='ovr' at the same time - these two are
  mutually exclusive in recent sklearn (multi_class expects the
  ORIGINAL 1D label array, sklearn does the binarization internally).
  Fixed to pass y_true directly.

  CRASH FIX #3: every per-seed model fit is now wrapped in try/except.
  If ONE model on ONE seed fails for any reason, it's logged and
  skipped (recorded as NaN) instead of killing the entire multi-hour
  run.

  OPERATIONAL FIX (the actual cause of "nothing left after
  disconnect"): this script now CHECKPOINTS after every single
  (dataset, seed) unit of work to /kaggle/working/results_v12/_ckpt_*.pkl.
  On every run, it first checks for existing checkpoints and SKIPS
  anything already completed. This means:
    1. If Kaggle disconnects mid-run, just re-run the same cell -
       it resumes from the last completed (dataset, seed) pair,
       not from zero.
    2. You should ALSO use Kaggle's "Save Version -> Save & Run All
       (Commit)" instead of interactive Run. This runs in the
       background on Kaggle's servers and survives you closing the
       browser entirely; the Output tab of that Version keeps every
       file in RESULTS_DIR permanently, independent of session state.

Everything else (feature lists, hyperparameters, leakage-audit
method, SMOTE-in-pipeline fix, NetBenefit-from-same-run fix, IR-sweep,
SHAP comparison) is IDENTICAL in logic to v11 - only the crash points
and resumability are new.
"""

import os
os.environ['PYTHONWARNINGS'] = 'ignore'
import warnings, time, ast, json, pickle, sys
warnings.filterwarnings('ignore')
import datetime
def log(msg):
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, label_binarize
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (classification_report, cohen_kappa_score, brier_score_loss,
                              roc_auc_score, make_scorer, balanced_accuracy_score,
                              average_precision_score, confusion_matrix, f1_score,
                              recall_score)
import lightgbm as lgb
import xgboost as xgb
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests
from scipy.stats import wilcoxon, spearmanr

log(f"xgboost version: {xgb.__version__} | lightgbm version: {lgb.__version__}")

SEED, SEEDS = 42, [42, 7, 13, 99, 2026]
np.random.seed(SEED)
IR_THRESHOLD = 3.0
MAX_TRAIN_ROWS = 60000
LEAK_THRESH = 0.75
N_JOBS = -1
SLA_PENALTY_PER_BREACH_USD = 50.0
COMPUTE_HOURLY_RATE_USD = 0.50
RESULTS_DIR = '/kaggle/working/results_v12'
CKPT_DIR = f"{RESULTS_DIR}/checkpoints"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

# ════════════════════════════════════════════════════════════════
# CHECKPOINT INFRASTRUCTURE — THIS IS WHAT SURVIVES A DISCONNECT
# ════════════════════════════════════════════════════════════════

def ckpt_path(name):
    safe = name.replace('/', '_').replace(' ', '_')
    return f"{CKPT_DIR}/{safe}.pkl"

def ckpt_exists(name):
    return os.path.exists(ckpt_path(name))

def ckpt_save(name, obj):
    with open(ckpt_path(name), 'wb') as f:
        pickle.dump(obj, f)

def ckpt_load(name):
    with open(ckpt_path(name), 'rb') as f:
        return pickle.load(f)

def run_or_resume(name, compute_fn):
    """The core resumability primitive. If a checkpoint for `name`
    already exists, load and return it instantly (no recomputation).
    Otherwise, run compute_fn(), save the result, and return it.
    Call this around every expensive unit of work."""
    if ckpt_exists(name):
        log(f"  [RESUME] {name} already completed - loading from checkpoint.")
        return ckpt_load(name)
    result = compute_fn()
    ckpt_save(name, result)
    return result

# ════════════════════════════════════════════════════════════════
# SECTION 0 — UTILITIES
# ════════════════════════════════════════════════════════════════

def encode_labels(y_series):
    le = LabelEncoder()
    y = le.fit_transform(y_series.astype(str))
    high_idx = int(np.where(le.classes_ == 'High')[0][0])
    return y, le, high_idx

def compute_ir(y):
    counts = np.bincount(y)
    return counts.max() / counts.min()

def make_class_weights(y, high_idx, alpha=5):
    classes, counts = np.unique(y, return_counts=True)
    total = len(y)
    cw = {int(c): total / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
    ir = counts.max() / counts.min()
    if ir > IR_THRESHOLD:
        cw[high_idx] *= alpha
    return cw

def cap_dataset_size(df, label_col, max_rows=MAX_TRAIN_ROWS, seed=SEED):
    if len(df) <= max_rows:
        return df
    frac = max_rows / len(df)
    parts = [grp.sample(frac=frac, random_state=seed) for _, grp in df.groupby(label_col)]
    return pd.concat(parts).reset_index(drop=True)

def compute_metrics(ytrue, ypred, yproba, le, high_idx=None, save_cm=False):
    rep = classification_report(ytrue, ypred, target_names=le.classes_.tolist(),
                                 output_dict=True, zero_division=0)
    nc = len(le.classes_)
    # CRASH FIX #2: pass y_true directly (1D labels), do NOT pre-binarize
    # when using the multi_class parameter - sklearn does that internally.
    try:
        auc = roc_auc_score(ytrue, yproba, multi_class='ovr', average='macro',
                             labels=np.arange(nc))
    except Exception:
        auc = np.nan
    try:
        brier = np.mean([brier_score_loss((ytrue == c).astype(int), yproba[:, c]) for c in range(nc)])
    except Exception:
        brier = np.nan
    pr_auc = np.nan
    if high_idx is not None:
        try:
            pr_auc = average_precision_score((ytrue == high_idx).astype(int), yproba[:, high_idx])
        except Exception:
            pass
    out = dict(RecallHigh=rep.get('High', {}).get('recall', 0.0),
               PrecHigh=rep.get('High', {}).get('precision', 0.0),
               F1High=rep.get('High', {}).get('f1-score', 0.0),
               MacroF1=rep['macro avg']['f1-score'],
               Kappa=cohen_kappa_score(ytrue, ypred), AUC=auc, Brier=brier,
               PR_AUC_High=pr_auc)
    if save_cm:
        out['ConfusionMatrix'] = confusion_matrix(ytrue, ypred).tolist()
    return out

def get_kats(cw, seed=42, ir=None, smote_k=5):
    use_smote = (ir is None) or (ir > IR_THRESHOLD)
    def wrap(est):
        if not use_smote:
            return est
        return ImbPipeline([('smote', SMOTE(random_state=seed, k_neighbors=smote_k)),
                             ('est', est)])
    return StackingClassifier(
        estimators=[
            ('lgb', wrap(lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                                             num_leaves=31, class_weight=cw, random_state=seed,
                                             verbose=-1, n_jobs=N_JOBS))),
            ('rf', wrap(RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                                random_state=seed, n_jobs=N_JOBS))),
            ('nb', wrap(CalibratedClassifierCV(GaussianNB(), cv=3, method='isotonic'))),
        ],
        final_estimator=LogisticRegression(C=1.0, max_iter=2000, random_state=seed,
                                            class_weight=cw),
        stack_method='predict_proba', passthrough=True, cv=3, n_jobs=N_JOBS)

def optimize_high_threshold(model, X_train, y_train, high_idx, seed=42, val_frac=0.15):
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=val_frac, random_state=seed, stratify=y_train)
    model.fit(X_fit, y_fit)
    proba_val = model.predict_proba(X_val)[:, high_idx]
    true_high_all = (y_val == high_idx).astype(int)
    base_rate = true_high_all.mean()
    precision_floor = max(0.30, 1.5 * base_rate)
    best_thresh, best_score = 0.5, -1
    for t in np.arange(0.15, 0.86, 0.05):
        pred_high = (proba_val >= t).astype(int)
        tp = np.sum((pred_high == 1) & (true_high_all == 1))
        fp = np.sum((pred_high == 1) & (true_high_all == 0))
        fn = np.sum((pred_high == 0) & (true_high_all == 1))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        if precision < precision_floor:
            continue
        bal_score = 0.5 * recall + 0.5 * precision
        if bal_score > best_score:
            best_score, best_thresh = bal_score, t
    model.fit(X_train, y_train)
    return model, best_thresh

def predict_with_threshold(model, X, high_idx, threshold, n_classes):
    proba = model.predict_proba(X)
    pred = np.argmax(proba, axis=1)
    high_trigger = proba[:, high_idx] >= threshold
    pred[high_trigger] = high_idx
    return pred, proba

def get_baselines(cw, seed=42):
    """CRASH FIX #1: use_label_encoder=False REMOVED from XGBClassifier."""
    return {
        'LightGBM': lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                                        class_weight=cw, random_state=seed, verbose=-1, n_jobs=N_JOBS),
        'XGBoost': xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                                      eval_metric='mlogloss', random_state=seed,
                                      verbosity=0, n_jobs=N_JOBS),
        'RandomForest': RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                                random_state=seed, n_jobs=N_JOBS),
        'BalancedRF': BalancedRandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=N_JOBS),
        'MLP': MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=300,
                              early_stopping=True, random_state=seed, learning_rate_init=0.001),
        'LogReg': LogisticRegression(max_iter=2000, random_state=seed, class_weight='balanced'),
        'NaiveBayes': CalibratedClassifierCV(GaussianNB(), cv=3, method='isotonic'),
    }

def leakage_audit(df, candidate_features, label_col, dataset_name, threshold=LEAK_THRESH):
    y_enc, le, _ = encode_labels(df[label_col])
    n_classes = len(le.classes_)
    chance = 1.0 / n_classes
    bal_scorer = make_scorer(balanced_accuracy_score)
    rows = []
    for feat in candidate_features:
        X_single = df[[feat]].fillna(0).astype(float).values
        try:
            stump = DecisionTreeClassifier(max_depth=1, random_state=SEED, class_weight='balanced')
            scores = cross_val_score(stump, X_single, y_enc, cv=5, scoring=bal_scorer)
            bal_acc = scores.mean()
        except Exception:
            bal_acc = np.nan
        rows.append((feat, bal_acc))
    rdf = pd.DataFrame(rows, columns=['feature', 'balanced_stump_accuracy']).sort_values(
        'balanced_stump_accuracy', ascending=False).reset_index(drop=True)
    suspects = rdf[rdf['balanced_stump_accuracy'] > threshold]['feature'].tolist()
    clean = [f for f in candidate_features if f not in suspects]
    log(f"--- LEAKAGE AUDIT: {dataset_name} (chance={chance:.4f}, thresh={threshold}) ---")
    for _, r in rdf.iterrows():
        flag = 'REMOVED' if r['feature'] in suspects else ''
        log(f"    {r['feature']:<40} {r['balanced_stump_accuracy']:.4f}  {flag}")
    rdf.to_csv(f"{RESULTS_DIR}/leakage_audit_{dataset_name}.csv", index=False)
    if suspects:
        log(f"  >>> {len(suspects)} feature(s) removed: {suspects}")
    else:
        log(f"  >>> No leakage suspects found. All {len(candidate_features)} features retained.")
    return clean, rdf

def mcnemar_pvalue_directional(y_true, pred_a, pred_b):
    a_correct = (pred_a == y_true)
    b_correct = (pred_b == y_true)
    b10 = int(np.sum(a_correct & ~b_correct))
    b01 = int(np.sum(~a_correct & b_correct))
    table = [[0, b10], [b01, 0]]
    try:
        res = mcnemar(table, exact=(b10 + b01 < 25), correction=True)
        pval = res.pvalue
    except Exception:
        pval = np.nan
    direction = "TIED" if b10 == b01 else ("KATS_BETTER" if b10 > b01 else "BASELINE_BETTER")
    return pval, b10, b01, direction

# ════════════════════════════════════════════════════════════════
# SECTION 1 — LOAD ALL 5 DATASETS (checkpointed as one unit; dataset
# loading is fast, so no finer granularity needed here)
# ════════════════════════════════════════════════════════════════

def load_all_datasets():
    log("="*70); log("SECTION 1: LOADING ALL 5 DATASETS"); log("="*70)

    raw3 = pd.read_csv('/kaggle/input/datasets/programmer3/cloud-task-scheduling-dataset/'
                        'Distributed_Task_Scheduling.csv')
    raw3.columns = [c.lower().strip().replace(' ', '_') for c in raw3.columns]
    algo_complexity = {'SA-ACO': 4, 'G_SOS': 3, 'HMFO': 2}
    raw3['algo_complexity_num'] = raw3['scheduling_algorithm'].map(algo_complexity).fillna(3)
    ds3 = pd.DataFrame()
    ds3['service_criticality'] = np.round(MinMaxScaler((1,10)).fit_transform(
        raw3[['task_priority']])).clip(1,10).astype(int).flatten()
    ds3['data_volume_gb'] = ((raw3['data_upload_size_mb'].fillna(0) +
                               raw3['data_download_size_mb'].fillna(0)) / 1024).clip(0.01, 800)
    ds3['rto_minutes'] = (raw3['execution_time_s'] / 60).clip(2, 480)
    ds3['rpo_minutes'] = (raw3['waiting_time_s'] / 60).clip(0.5, ds3['rto_minutes'])
    ratio3 = raw3['vm_mips'] / raw3['task_length_mips'].clip(lower=1)
    ds3['dependency_count'] = np.round(MinMaxScaler((0,30)).fit_transform(
        ratio3.values.reshape(-1,1))).flatten().astype(int)
    ds3['downstream_critical'] = (ds3['dependency_count'] > 15).astype(int)
    ds3['redundancy_level'] = np.round(MinMaxScaler((0,3)).fit_transform(
        1 - MinMaxScaler().fit_transform(raw3[['path_load']]))).astype(int).flatten()
    ds3['regulatory_flag'] = (raw3['storage_utilization'] > 0.75).astype(int)
    ds3['active_sessions'] = np.round(MinMaxScaler((10,50000)).fit_transform(
        raw3[['vm_memory_gb']])).astype(int).flatten()
    ds3['bandwidth_required_mbps'] = raw3['vm_bandwidth_mbps'].clip(0.1,10000)
    median_rt3 = raw3['response_time_s'].median()
    ds3['latency_sensitivity'] = (raw3['response_time_s'] < median_rt3).astype(int)
    energy_norm3 = MinMaxScaler().fit_transform(raw3[['energy_consumption_j']]).flatten()
    imbal_norm3 = MinMaxScaler().fit_transform(raw3[['degree_of_imbalance']]).flatten()
    ds3['az_risk_score'] = (0.5*energy_norm3 + 0.5*imbal_norm3).clip(0,1)
    ds3['multi_region_deployed'] = (raw3['algo_complexity_num'] >=
                                     raw3['algo_complexity_num'].median()).astype(int)
    ds3['migration_complexity'] = raw3['algo_complexity_num'].astype(int)
    rng_ct = np.random.default_rng(SEED)
    n3 = len(ds3)
    _labels_ct = np.array(['Low']*(n3//3) + ['Medium']*(n3//3) + ['High']*(n3 - 2*(n3//3)))
    rng_ct.shuffle(_labels_ct)
    ds3['priority_label'] = _labels_ct
    dfcloud = ds3.copy()
    CLOUD_CANDIDATES = ['service_criticality','data_volume_gb','rto_minutes','rpo_minutes',
        'dependency_count','downstream_critical','redundancy_level','regulatory_flag',
        'active_sessions','bandwidth_required_mbps','latency_sensitivity','az_risk_score',
        'multi_region_deployed','migration_complexity']
    log(f"CloudTask       {len(dfcloud):>9,} rows | {len(CLOUD_CANDIDATES)} candidate features")

    dfgoogle = pd.read_csv('/kaggle/input/datasets/derrickmwiti/google-2019-cluster-sample/'
                            'borg_traces_data.csv', low_memory=False)
    google_n_full = len(dfgoogle)
    def parse_dict_col(series, key):
        def pval(val):
            try:
                d = ast.literal_eval(str(val))
                return d.get(key, np.nan) if isinstance(d, dict) else np.nan
            except Exception:
                return np.nan
        return series.apply(pval)
    for k in ['cpus','memory']:
        dfgoogle[f'req{k}'] = parse_dict_col(dfgoogle['resource_request'], k)
        dfgoogle[f'avg{k}'] = parse_dict_col(dfgoogle['average_usage'], k)
        dfgoogle[f'max{k}'] = parse_dict_col(dfgoogle['maximum_usage'], k)
    dfgoogle['priority_label'] = dfgoogle['priority'].apply(
        lambda p: 'Low' if p < 100 else ('Medium' if p < 200 else 'High'))
    dfgoogle['eventenc'] = LabelEncoder().fit_transform(dfgoogle['event'].astype(str))
    for col in ['cycles_per_instruction','memory_accesses_per_instruction']:
        dfgoogle[col].fillna(dfgoogle[col].median(), inplace=True)
    dfgoogle['scheduler'].fillna(0, inplace=True)
    dfgoogle['vertical_scaling'].fillna(1, inplace=True)
    for col in ['reqcpus','reqmemory','avgcpus','avgmemory','maxcpus','maxmemory']:
        dfgoogle[col].fillna(dfgoogle[col].median(), inplace=True)
    GOOGLE_CANDIDATES = ['scheduling_class','collection_type','instance_index','assigned_memory',
        'page_cache_memory','cycles_per_instruction','memory_accesses_per_instruction','sample_rate',
        'scheduler','vertical_scaling','reqcpus','reqmemory','avgcpus','avgmemory','maxcpus',
        'maxmemory','failed','eventenc']
    log(f"GoogleCluster   {google_n_full:>9,} rows (full trace) | {len(GOOGLE_CANDIDATES)} candidate features")

    dfit_raw = pd.read_csv('/kaggle/input/datasets/shamiulislamshifat/it-incident-log-dataset/'
                            'incident_event_log.csv', low_memory=False)
    dfit = dfit_raw.sort_values('sys_mod_count').groupby('number').last().reset_index()
    dfit['priority_label'] = dfit['priority'].map({'1 - Critical':'High','2 - High':'High',
        '3 - Moderate':'Medium','4 - Low':'Low'})
    dfit.dropna(subset=['priority_label'], inplace=True)
    for colraw, colenc in [('category','categoryenc'),('location','locationenc'),
                            ('contact_type','contactenc'),('assignment_group','assignenc'),
                            ('cmdb_ci','cmdbenc'),('subcategory','subcatenc')]:
        if colraw in dfit.columns:
            dfit[colenc] = LabelEncoder().fit_transform(dfit[colraw].astype(str))
    dfit['knowledgeenc'] = dfit['knowledge'].astype(int)
    dfit['reopenflag'] = (dfit['reopen_count'] > 0).astype(int)
    IT_CANDIDATES = ['reassignment_count','reopen_count','sys_mod_count','categoryenc',
        'locationenc','contactenc','knowledgeenc','reopenflag']
    IT_CANDIDATES = [c for c in IT_CANDIDATES if c in dfit.columns]
    for extra in ['assignenc','cmdbenc','subcatenc']:
        if extra in dfit.columns:
            IT_CANDIDATES.append(extra)
    log(f"ITIncident      {len(dfit):>9,} rows | {len(IT_CANDIDATES)} candidate features "
        f"(impact/urgency/made_sla excluded, R3.1)")

    dfmc = pd.read_csv('/kaggle/input/datasets/ziya07/multi-cloud-service-composition-dataset/'
                        'multi_cloud_service_dataset.csv')
    dfmc.columns = [c.strip().lower().replace(' ','_').replace('(','').replace(')','').replace('/','_')
                    for c in dfmc.columns]
    dfmc['servicetypeenc'] = LabelEncoder().fit_transform(dfmc['service_type'].astype(str))
    dfmc['cloudproviderenc'] = LabelEncoder().fit_transform(dfmc['cloud_provider'].astype(str))
    dfmc['edgenodeenc'] = LabelEncoder().fit_transform(dfmc['edge_node_id'].astype(str))
    norm_cpu = dfmc['cpu_utilization_%'] / 100.0
    norm_lat = dfmc['service_latency_ms'] / dfmc['service_latency_ms'].max()
    norm_thr = 1 - dfmc['throughput_requests_sec'] / dfmc['throughput_requests_sec'].max()
    norm_bw  = 1 - dfmc['network_bandwidth_mbps'] / dfmc['network_bandwidth_mbps'].max()
    norm_wv  = dfmc['workload_variability'] / dfmc['workload_variability'].max()
    composite = (0.30*norm_cpu + 0.25*norm_lat + 0.20*norm_thr + 0.15*norm_bw + 0.10*norm_wv)
    dfmc['priority_label'] = pd.qcut(composite, q=3, labels=['Low','Medium','High']).astype(str)
    MC_CANDIDATES = ['memory_usage_mb','storage_usage_gb','response_time_ms','load_balancing_%',
        'optimal_service_placement','servicetypeenc','cloudproviderenc','edgenodeenc']
    MC_CANDIDATES = [c for c in MC_CANDIDATES if c in dfmc.columns]
    log(f"MultiCloud      {len(dfmc):>9,} rows | {len(MC_CANDIDATES)} candidate features")

    dfcic_raw = pd.read_csv('/kaggle/input/datasets/ericanacletoribeiro/cicids2017-cleaned-and-preprocessed/'
                             'cicids2017_cleaned.csv', low_memory=False)
    dfcic_raw.columns = [c.strip().lower().replace(' ', '_') for c in dfcic_raw.columns]
    LABEL_COL = 'attack_type'
    SEVERITY_MAP = {'normal traffic': 'Low', 'port scanning': 'Medium', 'brute force': 'Medium',
                    'dos': 'High', 'ddos': 'High', 'web attacks': 'High', 'bots': 'High'}
    def map_severity(lbl):
        return SEVERITY_MAP.get(str(lbl).strip().lower(), 'High')
    dfcic_raw['priority_label'] = dfcic_raw[LABEL_COL].apply(map_severity)
    DOWNSAMPLE_FRAC = 0.05
    parts = [grp.sample(frac=DOWNSAMPLE_FRAC, random_state=SEED)
              for _, grp in dfcic_raw.groupby('priority_label')]
    dfcic = pd.concat(parts).reset_index(drop=True)
    exclude_cols = {LABEL_COL, 'priority_label'}
    CIC_CANDIDATES = [c for c in dfcic.columns
                       if c not in exclude_cols and pd.api.types.is_numeric_dtype(dfcic[c])]
    dfcic[CIC_CANDIDATES] = dfcic[CIC_CANDIDATES].replace([np.inf, -np.inf], np.nan).fillna(0)
    log(f"CICIDS2017      {len(dfcic):>9,} rows (5% downsample) | {len(CIC_CANDIDATES)} candidate features")

    return {
        'CloudTask':     (dfcloud, CLOUD_CANDIDATES),
        'GoogleCluster': (dfgoogle, GOOGLE_CANDIDATES),
        'ITIncident':    (dfit, IT_CANDIDATES),
        'MultiCloud':    (dfmc, MC_CANDIDATES),
        'CICIDS2017':    (dfcic, CIC_CANDIDATES),
    }, google_n_full, dfit_raw

DATASETS_RAW, GOOGLE_N_FULL, DFIT_RAW = load_all_datasets()  # fast, no checkpoint needed

log("="*70); log("SECTION 1.5: LEAKAGE AUDIT"); log("="*70)
DATASETS = {}
for name, (df, cands) in DATASETS_RAW.items():
    key = f"leakaudit_{name}"
    clean_feats = run_or_resume(key, lambda df=df, cands=cands, name=name:
                                 leakage_audit(df, cands, 'priority_label', name)[0])
    DATASETS[name] = (cap_dataset_size(df, 'priority_label'), clean_feats)
    y_enc, le, _ = encode_labels(DATASETS[name][0]['priority_label'])
    log(f"  {name:<16} n={len(DATASETS[name][0]):>9,}  features={len(clean_feats):<3}  "
        f"IR={compute_ir(y_enc):6.2f}")

# ════════════════════════════════════════════════════════════════
# SECTION 2 — E1: FULL CLASSIFICATION COMPARISON, CHECKPOINTED PER
# (dataset, seed). If this crashes or Kaggle disconnects, re-running
# this cell skips every (dataset, seed) pair already completed.
# ════════════════════════════════════════════════════════════════
log("="*70); log("SECTION 2: E1 — KATS vs 7 BASELINES (checkpointed per dataset x seed)"); log("="*70)

def run_one_seed_e1(ds_name, df, feats, seed):
    X = df[feats].fillna(0).astype(float)
    y, le, hi = encode_labels(df['priority_label'])
    ir = compute_ir(y)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=seed, stratify=y)
    cw = make_class_weights(y_tr, hi, alpha=5)
    seed_result = {}

    try:
        kats_raw = get_kats(cw, seed, ir=ir)
        kats_raw.fit(X_tr, y_tr)
        m = compute_metrics(y_te, kats_raw.predict(X_te), kats_raw.predict_proba(X_te), le, high_idx=hi)
        seed_result['KATS_raw'] = m
    except Exception as e:
        log(f"    !! KATS_raw failed on {ds_name}/seed{seed}: {e}")
        seed_result['KATS_raw'] = None

    try:
        kats_model = get_kats(cw, seed, ir=ir)
        t0 = time.perf_counter()
        kats_model, thresh = optimize_high_threshold(kats_model, X_tr, y_tr, hi, seed=seed)
        pred, proba = predict_with_threshold(kats_model, X_te, hi, thresh, len(le.classes_))
        infer_t = (time.perf_counter() - t0) / max(1, len(X_te))
        m = compute_metrics(y_te, pred, proba, le, high_idx=hi, save_cm=True)
        m['threshold'] = thresh; m['latency_us'] = infer_t * 1e6
        m['_pred'] = pred.tolist(); m['_ytrue'] = y_te.tolist()
        seed_result['KATS'] = m
    except Exception as e:
        log(f"    !! KATS failed on {ds_name}/seed{seed}: {e}")
        seed_result['KATS'] = None

    baselines = get_baselines(cw, seed)
    for bname, bmodel in baselines.items():
        try:
            t0 = time.perf_counter(); bmodel.fit(X_tr, y_tr); _ = time.perf_counter() - t0
            t0 = time.perf_counter()
            bproba = bmodel.predict_proba(X_te); bpred = bmodel.predict(X_te)
            infer_t_b = (time.perf_counter() - t0) / max(1, len(X_te))
            m = compute_metrics(y_te, bpred, bproba, le, high_idx=hi, save_cm=True)
            m['latency_us'] = infer_t_b * 1e6
            m['_pred'] = bpred.tolist(); m['_ytrue'] = y_te.tolist()
            seed_result[bname] = m
        except Exception as e:
            log(f"    !! {bname} failed on {ds_name}/seed{seed}: {e}")
            seed_result[bname] = None
    return seed_result

e1_all_results = {}
for ds_name, (df, feats) in DATASETS.items():
    log(f"--- E1: {ds_name} ---")
    e1_all_results[ds_name] = {}
    for seed in SEEDS:
        ckpt_key = f"E1_{ds_name}_seed{seed}"
        result = run_or_resume(ckpt_key, lambda df=df, feats=feats, ds=ds_name, s=seed:
                                run_one_seed_e1(ds, df, feats, s))
        e1_all_results[ds_name][seed] = result
        rh = result.get('KATS', {}) or {}
        log(f"  [{ds_name}] seed={seed} KATS RecallH={rh.get('RecallHigh', float('nan')):.4f} "
            f"MacroF1={rh.get('MacroF1', float('nan')):.4f}")

# ---- Aggregate across seeds into Table 5 / Table 7 / Table 12 ----
e1_rows, mcnemar_rows, latency_rows, confusion_by_ds = [], [], [], {}
for ds_name, (df, feats) in DATASETS.items():
    y_full, _, _ = encode_labels(df['priority_label'])
    ir = compute_ir(y_full)
    model_names = ['KATS', 'KATS_raw'] + list(get_baselines({}, 42).keys())
    for mname in model_names:
        vals = {k: [] for k in ['RecallHigh','PrecHigh','F1High','MacroF1','Kappa','AUC','Brier','PR_AUC_High']}
        lat_vals = []
        for seed in SEEDS:
            r = e1_all_results[ds_name][seed].get(mname)
            if r is None:
                continue
            for k in vals:
                vals[k].append(r.get(k, np.nan))
            if 'latency_us' in r:
                lat_vals.append(r['latency_us'])
        row = {'Dataset': ds_name, 'Model': mname, 'IR': ir}
        for k, v in vals.items():
            row[k] = float(np.nanmean(v)) if len(v) else np.nan
        e1_rows.append(row)
    latency_rows.append({'Dataset': ds_name, **{m: float(np.mean(
        [e1_all_results[ds_name][s][m]['latency_us'] for s in SEEDS
         if e1_all_results[ds_name][s].get(m) is not None])) for m in model_names
        if any(e1_all_results[ds_name][s].get(m) is not None for s in SEEDS)}})
    last_seed = SEEDS[-1]
    kats_last = e1_all_results[ds_name][last_seed].get('KATS')
    if kats_last is not None:
        confusion_by_ds[ds_name] = {'KATS': kats_last.get('ConfusionMatrix')}
        for bname in get_baselines({}, 42).keys():
            b = e1_all_results[ds_name][last_seed].get(bname)
            if b is None:
                continue
            confusion_by_ds[ds_name][bname] = b.get('ConfusionMatrix')
            if kats_last.get('_pred') is not None and b.get('_pred') is not None:
                pval, b10, b01, direction = mcnemar_pvalue_directional(
                    np.array(kats_last['_ytrue']), np.array(kats_last['_pred']), np.array(b['_pred']))
                mcnemar_rows.append([ds_name, bname, b10, b01, pval, direction])

e1_df = pd.DataFrame(e1_rows)
e1_df.to_csv(f"{RESULTS_DIR}/Table5_E1_full_results.csv", index=False)
pd.DataFrame(latency_rows).to_csv(f"{RESULTS_DIR}/Table7_latency_measured.csv", index=False)
with open(f"{RESULTS_DIR}/confusion_matrices.json", 'w') as f:
    json.dump(confusion_by_ds, f, indent=2)
mcnemar_df = pd.DataFrame(mcnemar_rows, columns=['Dataset','Baseline','b10_KATSbetter',
                                                  'b01_Basebetter','p_raw','direction'])
log("Saved Table5_E1_full_results.csv, Table7_latency_measured.csv, confusion_matrices.json")

# ════════════════════════════════════════════════════════════════
# SECTION 3 — HOLM-BONFERRONI (fast, no checkpoint needed)
# ════════════════════════════════════════════════════════════════
log("="*70); log("SECTION 3: HOLM-BONFERRONI CORRECTION"); log("="*70)
valid_mask = mcnemar_df['p_raw'].notna()
if valid_mask.sum() > 0:
    reject, p_corrected, _, _ = multipletests(mcnemar_df.loc[valid_mask, 'p_raw'].values, method='holm')
    mcnemar_df.loc[valid_mask, 'p_holm'] = p_corrected
    mcnemar_df.loc[valid_mask, 'significant_holm_0.05'] = reject
    log(f"  Total tests: {valid_mask.sum()} | Significant (Holm p<0.05): {int(reject.sum())}")
mcnemar_df.to_csv(f"{RESULTS_DIR}/Table12_McNemar_directional.csv", index=False)

# ════════════════════════════════════════════════════════════════
# SECTION 4 — M2 ABLATION, CHECKPOINTED PER (dataset, seed)
# ════════════════════════════════════════════════════════════════
log("="*70); log("SECTION 4: M2 ABLATION (checkpointed per dataset x seed)"); log("="*70)

def run_one_seed_ablation(ds_name, df, feats, seed):
    X = df[feats].fillna(0).astype(float)
    y, le, hi = encode_labels(df['priority_label'])
    ir = compute_ir(y)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=seed, stratify=y)
    cw = make_class_weights(y_tr, hi, alpha=5)
    cw_a1 = make_class_weights(y_tr, hi, alpha=1)
    out = {}
    variant_builders = {
        'T_Full': lambda: get_kats(cw, seed, ir=ir),
        'T_NoSMOTE': lambda: get_kats(cw, seed, ir=0.0),
        'T_NoAsymLoss': lambda: get_kats(cw_a1, seed, ir=ir),
    }
    use_smote = ir > IR_THRESHOLD
    def wrap(est):
        return ImbPipeline([('smote', SMOTE(random_state=seed)), ('est', est)]) if use_smote else est
    variant_builders['T_NoCalibNB'] = lambda: StackingClassifier(
        estimators=[('lgb', wrap(lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05,
                        class_weight=cw, random_state=seed, verbose=-1, n_jobs=N_JOBS))),
                    ('rf', wrap(RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                                   random_state=seed, n_jobs=N_JOBS))),
                    ('nb', wrap(GaussianNB()))],
        final_estimator=LogisticRegression(C=1.0, max_iter=2000, random_state=seed, class_weight=cw),
        stack_method='predict_proba', passthrough=True, cv=3, n_jobs=N_JOBS)
    variant_builders['T_NoStacking'] = lambda: wrap(lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, class_weight=cw, random_state=seed, verbose=-1, n_jobs=N_JOBS))

    for vname, builder in variant_builders.items():
        try:
            m = builder(); m.fit(X_tr, y_tr)
            out[vname] = compute_metrics(y_te, m.predict(X_te), m.predict_proba(X_te), le, high_idx=hi)
        except Exception as e:
            log(f"    !! ablation variant {vname} failed on {ds_name}/seed{seed}: {e}")
            out[vname] = None
    return out

ablation_all_results = {}
for ds_name, (df, feats) in DATASETS.items():
    log(f"--- Ablation: {ds_name} ---")
    ablation_all_results[ds_name] = {}
    for seed in SEEDS:
        ckpt_key = f"ABL_{ds_name}_seed{seed}"
        result = run_or_resume(ckpt_key, lambda df=df, feats=feats, ds=ds_name, s=seed:
                                run_one_seed_ablation(ds, df, feats, s))
        ablation_all_results[ds_name][seed] = result

ablation_rows = []
for ds_name, (df, feats) in DATASETS.items():
    y_full, _, _ = encode_labels(df['priority_label'])
    ir = compute_ir(y_full)
    variants = ['T_Full','T_NoSMOTE','T_NoAsymLoss','T_NoCalibNB','T_NoStacking']
    per_variant_rh = {v: [] for v in variants}
    per_variant_f1 = {v: [] for v in variants}
    per_variant_kap = {v: [] for v in variants}
    for seed in SEEDS:
        for v in variants:
            r = ablation_all_results[ds_name][seed].get(v)
            if r is None:
                continue
            per_variant_rh[v].append(r['RecallHigh'])
            per_variant_f1[v].append(r['MacroF1'])
            per_variant_kap[v].append(r['Kappa'])
    base_rh = np.mean(per_variant_rh['T_Full']) if per_variant_rh['T_Full'] else np.nan
    base_f1 = np.mean(per_variant_f1['T_Full']) if per_variant_f1['T_Full'] else np.nan
    for v in variants:
        rh = np.mean(per_variant_rh[v]) if per_variant_rh[v] else np.nan
        f1 = np.mean(per_variant_f1[v]) if per_variant_f1[v] else np.nan
        kap = np.mean(per_variant_kap[v]) if per_variant_kap[v] else np.nan
        try:
            if len(per_variant_f1['T_Full']) == len(per_variant_f1[v]) and len(per_variant_f1[v]) > 0:
                diff = np.array(per_variant_f1['T_Full']) - np.array(per_variant_f1[v])
                p_wil = 1.0 if np.all(diff == 0) else wilcoxon(
                    per_variant_f1['T_Full'], per_variant_f1[v], alternative='greater')[1]
            else:
                p_wil = np.nan
        except Exception:
            p_wil = np.nan
        ablation_rows.append({'Dataset': ds_name, 'Variant': v, 'RecallH': rh, 'MacroF1': f1,
                               'Kappa': kap, 'DeltaRecallH_vs_ThisRun_TFull': rh - base_rh,
                               'Wilcoxon_p_onesided': p_wil, 'IR': ir})
        log(f"    {v:<14} RecallH={rh:.4f} MacroF1={f1:.4f} DeltaRecallH={rh-base_rh:+.4f}")

pd.DataFrame(ablation_rows).to_csv(f"{RESULTS_DIR}/Table6_M2_ablation.csv", index=False)
log("Saved Table6_M2_ablation.csv")

# ════════════════════════════════════════════════════════════════
# SECTION 5 — NetBenefit FROM THIS RUN'S OWN E1 RESULTS (checkpointed
# as one unit — cheap, just aggregation, no model fitting)
# ════════════════════════════════════════════════════════════════
log("="*70); log("SECTION 5: NetBenefit — ILLUSTRATIVE ERROR-COST MODEL"); log("="*70)

def compute_netbenefit():
    rows = []
    for ds_name, (df, feats) in DATASETS.items():
        sub = e1_df[e1_df['Dataset'] == ds_name]
        kats_rh = sub[sub['Model'] == 'KATS']['RecallHigh'].values[0]
        others = sub[~sub['Model'].isin(['KATS', 'KATS_raw'])]
        best_row = others.loc[others['RecallHigh'].idxmax()]
        best_name, best_rh = best_row['Model'], best_row['RecallHigh']
        y_ds, le_ds, hi_ds = encode_labels(df['priority_label'])
        n_high_test = int(round((y_ds == hi_ds).mean() * len(y_ds) * 0.20))
        lat_row = [r for r in latency_rows if r['Dataset'] == ds_name][0]
        lat_kats = lat_row.get('KATS', np.nan)
        lat_best = lat_row.get(best_name, np.nan)
        savings = (kats_rh - best_rh) * n_high_test * SLA_PENALTY_PER_BREACH_USD
        overhead = 0.0
        if not (np.isnan(lat_kats) or np.isnan(lat_best)):
            overhead = (lat_kats - lat_best) * n_high_test / 1e6 / 3600.0 * COMPUTE_HOURLY_RATE_USD
        rows.append({'Dataset': ds_name, 'KATS_RecallH': kats_rh, 'BestBaseline': best_name,
                     'Baseline_RecallH': best_rh, 'N_HighTest_approx': n_high_test,
                     'Expected_Savings_USD': savings, 'Compute_Overhead_USD': overhead,
                     'Net_Benefit_USD': savings - overhead,
                     'MODEL_LABEL': 'ILLUSTRATIVE ERROR-COST MODEL (R3.12)'})
    return rows

cost_rows = run_or_resume('netbenefit', compute_netbenefit)
pd.DataFrame(cost_rows).to_csv(f"{RESULTS_DIR}/Table13_NetBenefit.csv", index=False)
for r in cost_rows:
    log(f"  {r['Dataset']:<16} KATS_RH={r['KATS_RecallH']:.4f} vs {r['BestBaseline']}_RH="
        f"{r['Baseline_RecallH']:.4f} | Net=${r['Net_Benefit_USD']:.2f}")
log("Saved Table13_NetBenefit.csv (computed from Section 2's own E1 results, R3.11 fix)")

# ════════════════════════════════════════════════════════════════
# SECTION 6 — TEMPORAL / SCHEDULER ROBUSTNESS (checkpointed as units)
# ════════════════════════════════════════════════════════════════
log("="*70); log("SECTION 6: TEMPORAL + SCHEDULER ROBUSTNESS CHECKS"); log("="*70)

def run_temporal_check():
    time_cols = [c for c in DFIT_RAW.columns if 'open' in c.lower() and 'at' in c.lower()]
    if not time_cols:
        return None
    dfit, IT_FEATURES = DATASETS['ITIncident']
    time_col = time_cols[0]
    dfit_time = dfit.copy()
    dfit_time[time_col] = pd.to_datetime(DFIT_RAW.loc[DFIT_RAW.index.intersection(dfit.index), time_col],
                                          errors='coerce')
    dfit_time = dfit_time.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)
    n_split = int(len(dfit_time) * 0.80)
    X_full = dfit_time[IT_FEATURES].fillna(0).astype(float)
    y_full, le_it, hi_it = encode_labels(dfit_time['priority_label'])
    rows = []
    X_tr_c, X_te_c, y_tr_c, y_te_c = X_full[:n_split], X_full[n_split:], y_full[:n_split], y_full[n_split:]
    ir_c = compute_ir(y_tr_c)
    cw = make_class_weights(y_tr_c, hi_it, alpha=5)
    for mname, model in {'KATS': get_kats(cw, SEED, ir=ir_c),
                          'LightGBM': get_baselines(cw, SEED)['LightGBM'],
                          'LogReg': get_baselines(cw, SEED)['LogReg']}.items():
        model.fit(X_tr_c, y_tr_c)
        met = compute_metrics(y_te_c, model.predict(X_te_c), model.predict_proba(X_te_c), le_it, high_idx=hi_it)
        rows.append(['Chronological', mname, met['RecallHigh'], met['Kappa']])
    X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(X_full, y_full, test_size=0.20,
                                                         random_state=SEED, stratify=y_full)
    ir_r = compute_ir(y_tr_r)
    cw_r = make_class_weights(y_tr_r, hi_it, alpha=5)
    for mname, model in {'KATS': get_kats(cw_r, SEED, ir=ir_r),
                          'LightGBM': get_baselines(cw_r, SEED)['LightGBM'],
                          'LogReg': get_baselines(cw_r, SEED)['LogReg']}.items():
        model.fit(X_tr_r, y_tr_r)
        met = compute_metrics(y_te_r, model.predict(X_te_r), model.predict_proba(X_te_r), le_it, high_idx=hi_it)
        rows.append(['Random', mname, met['RecallHigh'], met['Kappa']])
    return rows

temporal_rows = run_or_resume('temporal_check', run_temporal_check)
if temporal_rows:
    pd.DataFrame(temporal_rows, columns=['SplitType','Model','RecallHigh','Kappa']).to_csv(
        f"{RESULTS_DIR}/temporal_split_robustness.csv", index=False)
    log("Saved temporal_split_robustness.csv")

def run_scheduler_check():
    dfgoogle, GOOGLE_FEATURES = DATASETS['GoogleCluster']
    y_g, le_g, hi_g = encode_labels(dfgoogle['priority_label'])
    ir_g = compute_ir(y_g)
    rows = []
    GOOGLE_NO_SCHED = [f for f in GOOGLE_FEATURES if f != 'scheduler']
    for feat_set, tag in [(GOOGLE_FEATURES, 'With_Scheduler'), (GOOGLE_NO_SCHED, 'Without_Scheduler')]:
        Xg = dfgoogle[feat_set].fillna(0).astype(float)
        X_tr, X_te, y_tr, y_te = train_test_split(Xg, y_g, test_size=0.20, random_state=SEED, stratify=y_g)
        cw = make_class_weights(y_tr, hi_g, alpha=5)
        for mname, model in {'KATS': get_kats(cw, SEED, ir=ir_g),
                              'LightGBM': get_baselines(cw, SEED)['LightGBM']}.items():
            model.fit(X_tr, y_tr)
            met = compute_metrics(y_te, model.predict(X_te), model.predict_proba(X_te), le_g, high_idx=hi_g)
            rows.append([tag, mname, met['RecallHigh'], met['Kappa']])
    return rows

sched_rows = run_or_resume('scheduler_check', run_scheduler_check)
pd.DataFrame(sched_rows, columns=['FeatureSet','Model','RecallHigh','Kappa']).to_csv(
    f"{RESULTS_DIR}/scheduler_circularity_check.csv", index=False)
log("Saved scheduler_circularity_check.csv")

# ════════════════════════════════════════════════════════════════
# SECTION 7 — CICIDS2017 STACKING-COLLAPSE DIAGNOSIS (R3.6)
# ════════════════════════════════════════════════════════════════
log("="*70); log("SECTION 7: CICIDS2017 STACKING COLLAPSE DIAGNOSIS (R3.6)"); log("="*70)

def run_stacking_diagnosis():
    df_diag, feats_diag = DATASETS['CICIDS2017']
    X = df_diag[feats_diag].fillna(0).astype(float).values
    y, le_d, hi_d = encode_labels(df_diag['priority_label'])
    ir_d = compute_ir(y)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=SEED, stratify=y)
    cw_d = make_class_weights(y_tr, hi_d, alpha=5)
    kats_d = get_kats(cw_d, SEED, ir=ir_d); kats_d.fit(X_tr, y_tr)
    pred_kats_d = kats_d.predict(X_te)
    lgb_solo = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, class_weight=cw_d,
                                   random_state=SEED, verbose=-1, n_jobs=N_JOBS)
    lgb_solo.fit(X_tr, y_tr); pred_lgb_d = lgb_solo.predict(X_te)
    b2_d = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=SEED, n_jobs=N_JOBS)
    b3_d = CalibratedClassifierCV(GaussianNB(), cv=3, method='isotonic')
    b2_d.fit(X_tr, y_tr); b3_d.fit(X_tr, y_tr)
    proba_avg_d = (lgb_solo.predict_proba(X_te) + b2_d.predict_proba(X_te) + b3_d.predict_proba(X_te)) / 3
    pred_avg_d = np.argmax(proba_avg_d, axis=1)
    return {
        'KATS_stacked': {'confusion_matrix': confusion_matrix(y_te, pred_kats_d).tolist(),
                          'MacroF1': f1_score(y_te, pred_kats_d, average='macro'),
                          'RecallHigh': recall_score(y_te, pred_kats_d, labels=[hi_d], average=None)[0]},
        'LightGBM_solo': {'confusion_matrix': confusion_matrix(y_te, pred_lgb_d).tolist(),
                           'MacroF1': f1_score(y_te, pred_lgb_d, average='macro'),
                           'RecallHigh': recall_score(y_te, pred_lgb_d, labels=[hi_d], average=None)[0]},
        'ProbabilityAveraging': {'confusion_matrix': confusion_matrix(y_te, pred_avg_d).tolist(),
                                  'MacroF1': f1_score(y_te, pred_avg_d, average='macro'),
                                  'RecallHigh': recall_score(y_te, pred_avg_d, labels=[hi_d], average=None)[0]},
    }

diag_report = run_or_resume('stacking_diagnosis', run_stacking_diagnosis)
with open(f"{RESULTS_DIR}/CICIDS2017_stacking_diagnosis.json", 'w') as f:
    json.dump(diag_report, f, indent=2)
for k, v in diag_report.items():
    log(f"  {k:<22} MacroF1={v['MacroF1']:.4f} RecallH={v['RecallHigh']:.4f}")

# ════════════════════════════════════════════════════════════════
# SECTION 8 — CONTROLLED IR-SWEEP (R3.13)
# ════════════════════════════════════════════════════════════════
log("="*70); log("SECTION 8: CONTROLLED IR-SWEEP (R3.13)"); log("="*70)

def make_ir_variant_fixed_n(X, y, target_ir, minority_class, fixed_n, seed=42):
    rng = np.random.RandomState(seed)
    idx_min = np.where(y == minority_class)[0]
    idx_maj = np.where(y != minority_class)[0]
    n_min = min(max(10, int(fixed_n / (1 + target_ir))), len(idx_min))
    n_maj = min(fixed_n - n_min, len(idx_maj))
    keep = np.concatenate([rng.choice(idx_min, n_min, replace=False),
                            rng.choice(idx_maj, n_maj, replace=False)])
    rng.shuffle(keep)
    return X[keep], y[keep]

def run_ir_sweep():
    df_ir, feats_ir = DATASETS['CICIDS2017']
    X_ir = df_ir[feats_ir].fillna(0).astype(float).values
    y_ir, le_ir, hi_ir = encode_labels(df_ir['priority_label'])
    FIXED_N = 8000
    rows = []
    for target_ir in [2, 5, 10, 20, 30]:
        Xv, yv = make_ir_variant_fixed_n(X_ir, y_ir, target_ir, hi_ir, FIXED_N, seed=SEED)
        achieved_ir = compute_ir(yv)
        X_tr, X_te, y_tr, y_te = train_test_split(Xv, yv, test_size=0.20, random_state=SEED, stratify=yv)
        cw = make_class_weights(y_tr, hi_ir, alpha=5)
        scores = {}
        for mname, model in {'KATS': get_kats(cw, SEED, ir=achieved_ir),
                              'LightGBM': lgb.LGBMClassifier(n_estimators=300, class_weight=cw,
                                                              random_state=SEED, verbose=-1, n_jobs=N_JOBS),
                              'LogReg': LogisticRegression(max_iter=2000, class_weight='balanced',
                                                            random_state=SEED)}.items():
            model.fit(X_tr, y_tr)
            scores[mname] = f1_score(y_te, model.predict(X_te), average='macro')
        best_model = max(scores, key=scores.get)
        rows.append({'target_IR': target_ir, 'achieved_IR': achieved_ir, 'n': len(yv),
                     **{f'MacroF1_{k}': v for k, v in scores.items()}, 'BestModel': best_model})
    return rows

ir_sweep_rows = run_or_resume('ir_sweep', run_ir_sweep)
pd.DataFrame(ir_sweep_rows).to_csv(f"{RESULTS_DIR}/Table_IR_sweep_controlled.csv", index=False)
for r in ir_sweep_rows:
    log(f"  target_IR={r['target_IR']:<4} achieved_IR={r['achieved_IR']:.2f} best={r['BestModel']}")

# ════════════════════════════════════════════════════════════════
# SECTION 9 — FULL-STACK SHAP vs B1-ONLY SHAP (R3.15, optional)
# ════════════════════════════════════════════════════════════════
log("="*70); log("SECTION 9: FULL-STACK SHAP vs B1-ONLY SHAP (R3.15)"); log("="*70)
if ckpt_exists('shap_check'):
    log("  [RESUME] SHAP check already completed.")
else:
    try:
        import shap
        df_s, feats_s = DATASETS['ITIncident']
        X = df_s[feats_s].fillna(0).astype(float).values
        y, le_s, hi_s = encode_labels(df_s['priority_label'])
        ir_s = compute_ir(y)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=SEED, stratify=y)
        cw_s = make_class_weights(y_tr, hi_s, alpha=5)
        kats_s = get_kats(cw_s, SEED, ir=ir_s); kats_s.fit(X_tr, y_tr)
        b1_s = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, class_weight=cw_s,
                                   random_state=SEED, verbose=-1, n_jobs=N_JOBS)
        b1_s.fit(X_tr, y_tr)
        sample_idx = np.random.RandomState(SEED).choice(len(X_te), size=min(100, len(X_te)), replace=False)
        X_sample = X_te[sample_idx]
        explainer_b1 = shap.TreeExplainer(b1_s)
        shap_b1 = explainer_b1.shap_values(X_sample)
        shap_b1_high = shap_b1[hi_s] if isinstance(shap_b1, list) else shap_b1[:, :, hi_s]
        rank_b1 = np.argsort(-np.abs(shap_b1_high).mean(axis=0))
        explainer_full = shap.KernelExplainer(lambda x: kats_s.predict_proba(x)[:, hi_s],
                                               shap.sample(X_tr, 50, random_state=SEED))
        shap_full = explainer_full.shap_values(X_sample, nsamples=100)
        rank_full = np.argsort(-np.abs(shap_full).mean(axis=0))
        rho, _ = spearmanr(rank_b1, rank_full)
        log(f"  Spearman rho (B1 vs full-stack)={rho:.4f}")
        pd.DataFrame({'feature': feats_s, 'rank_B1': rank_b1.argsort(),
                      'rank_full_stack': rank_full.argsort()}).to_csv(
            f"{RESULTS_DIR}/SHAP_B1_vs_fullstack_agreement.csv", index=False)
        ckpt_save('shap_check', {'rho': rho})
    except ImportError:
        log("  'shap' not installed - run `pip install shap` then re-run this section only.")
    except Exception as e:
        log(f"  Full-stack SHAP failed ({e}) - non-fatal, continuing.")

# ════════════════════════════════════════════════════════════════
log("="*70); log(f"ALL SECTIONS COMPLETE - outputs in: {RESULTS_DIR}"); log("="*70)
for f in sorted(os.listdir(RESULTS_DIR)):
    if not f.startswith('checkpoints'):
        log(f"  - {f}")
log("")
log(">>> IMPORTANT: use Kaggle 'Save Version -> Save & Run All (Commit)' so this run persists")
log(">>> as a permanent Output even if your browser disconnects. If it DOES disconnect mid-run,")
log(">>> just re-run this same cell/script - checkpoints in results_v12/checkpoints/ mean")
log(">>> already-completed (dataset, seed) pairs are skipped, not recomputed.")
