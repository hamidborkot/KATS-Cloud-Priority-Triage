"""
KATS MASTER PIPELINE v14 — CONFIRMED PATHS, NO MORE GUESSING
====================================================================
Paths below are copied EXACTLY from your diagnostic output, not
guessed from old logs:

  GoogleCluster: /kaggle/input/datasets/derrickmwiti/google-2019-cluster-sample/borg_traces_data.csv
  ITIncident:    /kaggle/input/datasets/shamiulislamshifat/it-incident-log-dataset/incident_event_log.csv
  MultiCloud:    /kaggle/input/datasets/ziya07/multi-cloud-service-composition-dataset/multi_cloud_service_dataset.csv
  CICIDS2017:    /kaggle/input/datasets/ericanacletoribeiro/cicids2017-cleaned-and-preprocessed/cicids2017_cleaned.csv

CloudTask: the original source (programmer3/cloud-task-scheduling-dataset)
is confirmed REMOVED from Kaggle - there is no path to point to. This
version uses a SYNTHETIC generator (tested standalone before delivery)
that reproduces the same 14-feature schema and sector-conditional
priority logic already present in your own notebook's history
(seed=42, fully reproducible). This is flagged in the output and must
be disclosed in the manuscript's data-availability note as: "the
original CloudTask source dataset was removed from its host platform
after submission; results were regenerated using an equivalent
synthetic negative-control generator (seed=42), described in
Appendix X." If you recover the original file or a saved derived copy,
set CLOUDTASK_CSV_PATH below and set USE_SYNTHETIC_CLOUDTASK = False.

Sections included: data loading, leakage audit, E1 (5-seed KATS vs 7
baselines), Holm-Bonferroni McNemar, M2 ablation, NetBenefit. All
checkpointed per (dataset, seed) - safe to interrupt and resume.
Robustness sections (temporal/scheduler checks, IR-sweep, SHAP) will
be added back once this runs clean end-to-end in your environment.
"""

import os
os.environ['PYTHONWARNINGS'] = 'ignore'
import warnings, time, ast, json, pickle, re
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
                              average_precision_score, confusion_matrix)
import lightgbm as lgb
import xgboost as xgb
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests
from scipy.stats import wilcoxon

log(f"xgboost={xgb.__version__} lightgbm={lgb.__version__} pandas={pd.__version__}")

SEED, SEEDS = 42, [42, 7, 13, 99, 2026]
np.random.seed(SEED)
IR_THRESHOLD = 3.0
MAX_TRAIN_ROWS = 60000
LEAK_THRESH = 0.75
N_JOBS = -1
SLA_PENALTY_PER_BREACH_USD = 50.0
COMPUTE_HOURLY_RATE_USD = 0.50
RESULTS_DIR = '/kaggle/working/results_v14'
CKPT_DIR = f"{RESULTS_DIR}/checkpoints"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

# ==== CONFIRMED PATHS (copied verbatim from your diagnostic output) ====
GOOGLE_PATH = '/kaggle/input/datasets/derrickmwiti/google-2019-cluster-sample/borg_traces_data.csv'
IT_PATH     = '/kaggle/input/datasets/shamiulislamshifat/it-incident-log-dataset/incident_event_log.csv'
MC_PATH     = '/kaggle/input/datasets/ziya07/multi-cloud-service-composition-dataset/multi_cloud_service_dataset.csv'
CIC_PATH    = '/kaggle/input/datasets/ericanacletoribeiro/cicids2017-cleaned-and-preprocessed/cicids2017_cleaned.csv'

# CloudTask: set this if you recover the original/derived file, and flip the flag below.
CLOUDTASK_CSV_PATH = None
USE_SYNTHETIC_CLOUDTASK = True

# ════════════════════════════════════════════════════════════════
# ROBUST COLUMN MATCHING (kept from v13 - still needed for the 4
# confirmed files, since a confirmed PATH does not guarantee I know
# the exact header spelling/case inside it)
# ════════════════════════════════════════════════════════════════

def _norm(s):
    return re.sub(r'[^a-z0-9]', '', str(s).lower())

def find_col(df, *aliases, required=True):
    norm_cols = {_norm(c): c for c in df.columns}
    for alias in aliases:
        na = _norm(alias)
        if na in norm_cols:
            return norm_cols[na]
    for alias in aliases:
        na = _norm(alias)
        for nc, orig in norm_cols.items():
            if na in nc or nc in na:
                return orig
    if required:
        raise KeyError(f"Could not find any column matching {aliases!r}.\n"
                        f"Actual columns: {list(df.columns)}")
    return None

def get_col(df, *aliases, default_value=None):
    col = find_col(df, *aliases, required=(default_value is None))
    return pd.Series(default_value, index=df.index) if col is None else df[col]

# ════════════════════════════════════════════════════════════════
# CHECKPOINT INFRASTRUCTURE
# ════════════════════════════════════════════════════════════════

def ckpt_path(name):
    return f"{CKPT_DIR}/{name.replace('/', '_').replace(' ', '_')}.pkl"

def ckpt_exists(name):
    return os.path.exists(ckpt_path(name))

def ckpt_save(name, obj):
    with open(ckpt_path(name), 'wb') as f:
        pickle.dump(obj, f)

def ckpt_load(name):
    with open(ckpt_path(name), 'rb') as f:
        return pickle.load(f)

def run_or_resume(name, compute_fn):
    if ckpt_exists(name):
        log(f"  [RESUME] {name} already completed - loading from checkpoint.")
        return ckpt_load(name)
    result = compute_fn()
    ckpt_save(name, result)
    return result

# ════════════════════════════════════════════════════════════════
# STEP 0 — PROBE the 4 confirmed files + report CloudTask status
# ════════════════════════════════════════════════════════════════
log("="*70); log("STEP 0: VERIFYING THE 4 CONFIRMED FILES LOAD"); log("="*70)
for name, path in [('GoogleCluster', GOOGLE_PATH), ('ITIncident', IT_PATH),
                    ('MultiCloud', MC_PATH), ('CICIDS2017', CIC_PATH)]:
    df_probe = pd.read_csv(path, nrows=100, low_memory=False)
    log(f"  {name:<15} OK  columns={list(df_probe.columns)[:10]}"
        f"{'...' if len(df_probe.columns) > 10 else ''}")
if USE_SYNTHETIC_CLOUDTASK:
    log("  CloudTask       USING SYNTHETIC GENERATOR (original source removed from Kaggle)")
else:
    df_probe = pd.read_csv(CLOUDTASK_CSV_PATH, nrows=100, low_memory=False)
    log(f"  CloudTask       OK (from {CLOUDTASK_CSV_PATH}) columns={list(df_probe.columns)}")

# ════════════════════════════════════════════════════════════════
# SECTION 0 — MODEL UTILITIES
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
    if counts.max() / counts.min() > IR_THRESHOLD:
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
    try:
        auc = roc_auc_score(ytrue, yproba, multi_class='ovr', average='macro', labels=np.arange(nc))
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
               Kappa=cohen_kappa_score(ytrue, ypred), AUC=auc, Brier=brier, PR_AUC_High=pr_auc)
    if save_cm:
        out['ConfusionMatrix'] = confusion_matrix(ytrue, ypred).tolist()
    return out

def get_kats(cw, seed=42, ir=None, smote_k=5):
    use_smote = (ir is None) or (ir > IR_THRESHOLD)
    def wrap(est):
        return ImbPipeline([('smote', SMOTE(random_state=seed, k_neighbors=smote_k)), ('est', est)]) \
            if use_smote else est
    return StackingClassifier(
        estimators=[
            ('lgb', wrap(lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                                             num_leaves=31, class_weight=cw, random_state=seed,
                                             verbose=-1, n_jobs=N_JOBS))),
            ('rf', wrap(RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                                random_state=seed, n_jobs=N_JOBS))),
            ('nb', wrap(CalibratedClassifierCV(GaussianNB(), cv=3, method='isotonic'))),
        ],
        final_estimator=LogisticRegression(C=1.0, max_iter=2000, random_state=seed, class_weight=cw),
        stack_method='predict_proba', passthrough=True, cv=3, n_jobs=N_JOBS)

def optimize_high_threshold(model, X_train, y_train, high_idx, seed=42, val_frac=0.15):
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=val_frac, random_state=seed, stratify=y_train)
    model.fit(X_fit, y_fit)
    proba_val = model.predict_proba(X_val)[:, high_idx]
    true_high_all = (y_val == high_idx).astype(int)
    precision_floor = max(0.30, 1.5 * true_high_all.mean())
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
    pred[proba[:, high_idx] >= threshold] = high_idx
    return pred, proba

def get_baselines(cw, seed=42):
    return {
        'LightGBM': lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                                        class_weight=cw, random_state=seed, verbose=-1, n_jobs=N_JOBS),
        'XGBoost': xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                                      eval_metric='mlogloss', random_state=seed, verbosity=0, n_jobs=N_JOBS),
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
    bal_scorer = make_scorer(balanced_accuracy_score)
    rows = []
    for feat in candidate_features:
        X_single = df[[feat]].fillna(0).astype(float).values
        try:
            stump = DecisionTreeClassifier(max_depth=1, random_state=SEED, class_weight='balanced')
            bal_acc = cross_val_score(stump, X_single, y_enc, cv=5, scoring=bal_scorer).mean()
        except Exception:
            bal_acc = np.nan
        rows.append((feat, bal_acc))
    rdf = pd.DataFrame(rows, columns=['feature', 'balanced_stump_accuracy']).sort_values(
        'balanced_stump_accuracy', ascending=False).reset_index(drop=True)
    suspects = rdf[rdf['balanced_stump_accuracy'] > threshold]['feature'].tolist()
    clean = [f for f in candidate_features if f not in suspects]
    log(f"--- LEAKAGE AUDIT: {dataset_name} (thresh={threshold}) ---")
    for _, r in rdf.iterrows():
        log(f"    {r['feature']:<40} {r['balanced_stump_accuracy']:.4f}  "
            f"{'REMOVED' if r['feature'] in suspects else ''}")
    rdf.to_csv(f"{RESULTS_DIR}/leakage_audit_{dataset_name}.csv", index=False)
    return clean, rdf

def mcnemar_pvalue_directional(y_true, pred_a, pred_b):
    a_correct = (pred_a == y_true); b_correct = (pred_b == y_true)
    b10 = int(np.sum(a_correct & ~b_correct)); b01 = int(np.sum(~a_correct & b_correct))
    try:
        pval = mcnemar([[0, b10], [b01, 0]], exact=(b10 + b01 < 25), correction=True).pvalue
    except Exception:
        pval = np.nan
    direction = "TIED" if b10 == b01 else ("KATS_BETTER" if b10 > b01 else "BASELINE_BETTER")
    return pval, b10, b01, direction

# ════════════════════════════════════════════════════════════════
# SYNTHETIC CLOUDTASK GENERATOR (tested standalone; matches the
# sector-conditional logic already in your notebook's history)
# ════════════════════════════════════════════════════════════════

def generate_synthetic_cloudtask(n=6000, seed=SEED):
    rng = np.random.default_rng(seed)
    sectors = ['banking','health','government','retail','ai_inference','batch_analytics']
    weights = [0.20,0.15,0.10,0.20,0.20,0.15]
    sector_labels = rng.choice(sectors, size=n, p=weights)
    rows = []
    for s in sector_labels:
        if s == 'banking': crit=rng.integers(7,11); rto=rng.uniform(1,30); reg=1
        elif s == 'health': crit=rng.integers(6,10); rto=rng.uniform(5,60); reg=1
        elif s == 'government': crit=rng.integers(6,10); rto=rng.uniform(10,60); reg=1
        elif s == 'ai_inference': crit=rng.integers(5,9); rto=rng.uniform(5,90); reg=0
        elif s == 'retail': crit=rng.integers(3,7); rto=rng.uniform(30,240); reg=0
        else: crit=rng.integers(1,4); rto=rng.uniform(120,1440); reg=0
        rows.append(dict(
            service_criticality=int(crit), data_volume_gb=rng.exponential(50), rto_minutes=rto,
            rpo_minutes=rto*rng.uniform(0.3,0.8), dependency_count=rng.poisson(3),
            downstream_critical=rng.binomial(1, 0.3 if crit>6 else 0.05),
            redundancy_level=int(rng.integers(0,4)), regulatory_flag=reg,
            active_sessions=rng.exponential(500), bandwidth_required_mbps=rng.exponential(100),
            latency_sensitivity=rng.binomial(1, 0.7 if crit>6 else 0.2), az_risk_score=rng.beta(2,5),
            multi_region_deployed=rng.binomial(1,0.4), migration_complexity=int(rng.integers(1,6))))
    df = pd.DataFrame(rows)
    score = (0.35*df.service_criticality/10 + 0.20*(1-df.rto_minutes/df.rto_minutes.max())
             + 0.15*df.regulatory_flag + 0.15*df.downstream_critical + 0.10*df.az_risk_score
             + 0.05*(1-df.redundancy_level/3))
    df['priority_label'] = pd.cut(score, bins=[-np.inf,0.35,0.55,np.inf], labels=['Low','Medium','High']).astype(str)
    return df

# ════════════════════════════════════════════════════════════════
# SECTION 1 — LOAD ALL 5 DATASETS
# ════════════════════════════════════════════════════════════════

def load_all_datasets():
    log("="*70); log("SECTION 1: LOADING ALL 5 DATASETS"); log("="*70)

    if USE_SYNTHETIC_CLOUDTASK:
        dfcloud = generate_synthetic_cloudtask()
        log("CloudTask       USING SYNTHETIC GENERATOR (original Kaggle source removed) - "
            "DISCLOSE THIS in the manuscript's data-availability note.")
    else:
        raw1 = pd.read_csv(CLOUDTASK_CSV_PATH)
        # (original real-CSV feature engineering would go here if you recover the file)
        raise NotImplementedError("Set USE_SYNTHETIC_CLOUDTASK=True or implement real-CSV mapping.")
    CLOUD_CANDIDATES = [c for c in dfcloud.columns if c != 'priority_label']
    log(f"CloudTask       {len(dfcloud):>9,} rows | {len(CLOUD_CANDIDATES)} candidate features")

    dfgoogle = pd.read_csv(GOOGLE_PATH, low_memory=False)
    google_n_full = len(dfgoogle)
    def parse_dict_col(series, key):
        def pval(val):
            try:
                d = ast.literal_eval(str(val))
                return d.get(key, np.nan) if isinstance(d, dict) else np.nan
            except Exception:
                return np.nan
        return series.apply(pval)
    rr_col = find_col(dfgoogle, 'resource_request', required=False)
    au_col = find_col(dfgoogle, 'average_usage', required=False)
    mu_col = find_col(dfgoogle, 'maximum_usage', required=False)
    for k in ['cpus', 'memory']:
        dfgoogle[f'req{k}'] = parse_dict_col(dfgoogle[rr_col], k) if rr_col else np.nan
        dfgoogle[f'avg{k}'] = parse_dict_col(dfgoogle[au_col], k) if au_col else np.nan
        dfgoogle[f'max{k}'] = parse_dict_col(dfgoogle[mu_col], k) if mu_col else np.nan
    priority_col = find_col(dfgoogle, 'priority')
    dfgoogle['priority_label'] = dfgoogle[priority_col].apply(
        lambda p: 'Low' if p < 100 else ('Medium' if p < 200 else 'High'))
    event_col = find_col(dfgoogle, 'event', required=False)
    dfgoogle['eventenc'] = LabelEncoder().fit_transform(dfgoogle[event_col].astype(str)) if event_col else 0
    for logical in ['cycles_per_instruction','memory_accesses_per_instruction','reqcpus','reqmemory',
                     'avgcpus','avgmemory','maxcpus','maxmemory']:
        actual = find_col(dfgoogle, logical, required=False)
        if actual and dfgoogle[actual].isna().any():
            dfgoogle[actual] = dfgoogle[actual].fillna(dfgoogle[actual].median())
    for logical, fillval in [('scheduler', 0), ('vertical_scaling', 1)]:
        actual = find_col(dfgoogle, logical, required=False)
        if actual:
            dfgoogle[actual] = dfgoogle[actual].fillna(fillval)
    GOOGLE_CANDIDATES = []
    for logical in ['scheduling_class','collection_type','instance_index','assigned_memory',
                     'page_cache_memory','cycles_per_instruction','memory_accesses_per_instruction',
                     'sample_rate','scheduler','vertical_scaling','reqcpus','reqmemory','avgcpus',
                     'avgmemory','maxcpus','maxmemory','failed']:
        actual = find_col(dfgoogle, logical, required=False)
        if actual:
            GOOGLE_CANDIDATES.append(actual)
    GOOGLE_CANDIDATES.append('eventenc')
    log(f"GoogleCluster   {google_n_full:>9,} rows (full) | {len(GOOGLE_CANDIDATES)} candidate features")

    dfit_raw = pd.read_csv(IT_PATH, low_memory=False)
    sys_mod_col = find_col(dfit_raw, 'sys_mod_count')
    number_col = find_col(dfit_raw, 'number')
    dfit = dfit_raw.sort_values(sys_mod_col).groupby(number_col).last().reset_index()
    priority_col_it = find_col(dfit, 'priority')
    dfit['priority_label'] = dfit[priority_col_it].map({'1 - Critical':'High','2 - High':'High',
        '3 - Moderate':'Medium','4 - Low':'Low'})
    dfit.dropna(subset=['priority_label'], inplace=True)
    for logical_raw, enc_name in [('category','categoryenc'),('location','locationenc'),
                                   ('contact_type','contactenc'),('assignment_group','assignenc'),
                                   ('cmdb_ci','cmdbenc'),('subcategory','subcatenc')]:
        actual = find_col(dfit, logical_raw, required=False)
        if actual:
            dfit[enc_name] = LabelEncoder().fit_transform(dfit[actual].astype(str))
    knowledge_col = find_col(dfit, 'knowledge', required=False)
    dfit['knowledgeenc'] = dfit[knowledge_col].astype(int) if knowledge_col else 0
    reopen_col = find_col(dfit, 'reopen_count')
    dfit['reopenflag'] = (dfit[reopen_col] > 0).astype(int)
    IT_CANDIDATES = []
    for logical in ['reassignment_count', 'reopen_count', 'sys_mod_count']:
        actual = find_col(dfit, logical, required=False)
        if actual:
            IT_CANDIDATES.append(actual)
    for extra in ['categoryenc','locationenc','contactenc','knowledgeenc','reopenflag',
                  'assignenc','cmdbenc','subcatenc']:
        if extra in dfit.columns:
            IT_CANDIDATES.append(extra)
    log(f"ITIncident      {len(dfit):>9,} rows | {len(IT_CANDIDATES)} candidate features "
        f"(impact/urgency/made_sla excluded by design, R3.1)")

    dfmc = pd.read_csv(MC_PATH)
    st_col = find_col(dfmc, 'ServiceType')
    cp_col = find_col(dfmc, 'CloudProvider')
    en_col = find_col(dfmc, 'EdgeNodeID')
    dfmc['servicetypeenc'] = LabelEncoder().fit_transform(dfmc[st_col].astype(str))
    dfmc['cloudproviderenc'] = LabelEncoder().fit_transform(dfmc[cp_col].astype(str))
    dfmc['edgenodeenc'] = LabelEncoder().fit_transform(dfmc[en_col].astype(str))
    cpu_col = find_col(dfmc, 'CPUUtilization')
    lat_col = find_col(dfmc, 'ServiceLatency')
    thr_col = find_col(dfmc, 'Throughput')
    bw_col = find_col(dfmc, 'NetworkBandwidth')
    wv_col = find_col(dfmc, 'WorkloadVariability')
    norm_cpu = dfmc[cpu_col] / 100.0
    norm_lat = dfmc[lat_col] / dfmc[lat_col].max()
    norm_thr = 1 - dfmc[thr_col] / dfmc[thr_col].max()
    norm_bw = 1 - dfmc[bw_col] / dfmc[bw_col].max()
    norm_wv = dfmc[wv_col] / dfmc[wv_col].max()
    composite = 0.30*norm_cpu + 0.25*norm_lat + 0.20*norm_thr + 0.15*norm_bw + 0.10*norm_wv
    dfmc['priority_label'] = pd.qcut(composite, q=3, labels=['Low','Medium','High']).astype(str)
    MC_CANDIDATES = []
    for logical in ['MemoryUsage', 'StorageUsage', 'ResponseTime', 'LoadBalancing', 'OptimalServicePlacement']:
        actual = find_col(dfmc, logical, required=False)
        if actual:
            MC_CANDIDATES.append(actual)
    MC_CANDIDATES += ['servicetypeenc', 'cloudproviderenc', 'edgenodeenc']
    log(f"MultiCloud      {len(dfmc):>9,} rows | {len(MC_CANDIDATES)} candidate features")

    dfcic_raw = pd.read_csv(CIC_PATH, low_memory=False)
    label_col = find_col(dfcic_raw, 'attack_type', 'label')
    def map_severity(lbl):
        key = str(lbl).strip().lower()
        if 'benign' in key or 'normal' in key: return 'Low'
        if 'scan' in key or 'patator' in key or 'brute' in key: return 'Medium'
        return 'High'
    dfcic_raw['priority_label'] = dfcic_raw[label_col].apply(map_severity)
    parts = [grp.sample(frac=0.05, random_state=SEED) for _, grp in dfcic_raw.groupby('priority_label')]
    dfcic = pd.concat(parts).reset_index(drop=True)
    exclude = {label_col, 'priority_label'}
    CIC_CANDIDATES = [c for c in dfcic.columns if c not in exclude and pd.api.types.is_numeric_dtype(dfcic[c])]
    dfcic[CIC_CANDIDATES] = dfcic[CIC_CANDIDATES].replace([np.inf, -np.inf], np.nan).fillna(0)
    log(f"CICIDS2017      {len(dfcic):>9,} rows (5% downsample) | {len(CIC_CANDIDATES)} candidate features")

    return {
        'CloudTask':     (dfcloud, CLOUD_CANDIDATES),
        'GoogleCluster': (dfgoogle, GOOGLE_CANDIDATES),
        'ITIncident':    (dfit, IT_CANDIDATES),
        'MultiCloud':    (dfmc, MC_CANDIDATES),
        'CICIDS2017':    (dfcic, CIC_CANDIDATES),
    }, google_n_full

DATASETS_RAW, GOOGLE_N_FULL = load_all_datasets()

log("="*70); log("SECTION 1.5: LEAKAGE AUDIT"); log("="*70)
DATASETS = {}
for name, (df, cands) in DATASETS_RAW.items():
    clean_feats = run_or_resume(f"leakaudit_{name}", lambda df=df, cands=cands, name=name:
                                 leakage_audit(df, cands, 'priority_label', name)[0])
    DATASETS[name] = (cap_dataset_size(df, 'priority_label'), clean_feats)
    y_enc, _, _ = encode_labels(DATASETS[name][0]['priority_label'])
    log(f"  {name:<16} n={len(DATASETS[name][0]):>9,}  features={len(clean_feats):<3}  "
        f"IR={compute_ir(y_enc):6.2f}")

# ════════════════════════════════════════════════════════════════
# SECTION 2 — E1 (checkpointed per dataset x seed)
# ════════════════════════════════════════════════════════════════
log("="*70); log("SECTION 2: E1 — KATS vs 7 BASELINES"); log("="*70)

def run_one_seed_e1(ds_name, df, feats, seed):
    X = df[feats].fillna(0).astype(float)
    y, le, hi = encode_labels(df['priority_label'])
    ir = compute_ir(y)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=seed, stratify=y)
    cw = make_class_weights(y_tr, hi, alpha=5)
    seed_result = {}
    try:
        kats_raw = get_kats(cw, seed, ir=ir); kats_raw.fit(X_tr, y_tr)
        seed_result['KATS_raw'] = compute_metrics(y_te, kats_raw.predict(X_te),
                                                   kats_raw.predict_proba(X_te), le, high_idx=hi)
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
    for bname, bmodel in get_baselines(cw, seed).items():
        try:
            bmodel.fit(X_tr, y_tr)
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
        result = run_or_resume(f"E1_{ds_name}_seed{seed}",
                                lambda df=df, feats=feats, ds=ds_name, s=seed: run_one_seed_e1(ds, df, feats, s))
        e1_all_results[ds_name][seed] = result
        rh = result.get('KATS') or {}
        log(f"  [{ds_name}] seed={seed} KATS RecallH={rh.get('RecallHigh', float('nan')):.4f} "
            f"MacroF1={rh.get('MacroF1', float('nan')):.4f}")

e1_rows, mcnemar_rows, latency_rows, confusion_by_ds = [], [], [], {}
for ds_name, (df, feats) in DATASETS.items():
    y_full, _, _ = encode_labels(df['priority_label'])
    ir = compute_ir(y_full)
    model_names = ['KATS', 'KATS_raw'] + list(get_baselines({}, 42).keys())
    for mname in model_names:
        vals = {k: [] for k in ['RecallHigh','PrecHigh','F1High','MacroF1','Kappa','AUC','Brier','PR_AUC_High']}
        for seed in SEEDS:
            r = e1_all_results[ds_name][seed].get(mname)
            if r is None:
                continue
            for k in vals:
                vals[k].append(r.get(k, np.nan))
        row = {'Dataset': ds_name, 'Model': mname, 'IR': ir}
        for k, v in vals.items():
            row[k] = float(np.nanmean(v)) if len(v) else np.nan
        e1_rows.append(row)
    lat_row = {'Dataset': ds_name}
    for m in model_names:
        lv = [e1_all_results[ds_name][s][m]['latency_us'] for s in SEEDS
              if e1_all_results[ds_name][s].get(m) is not None and 'latency_us' in e1_all_results[ds_name][s][m]]
        if lv:
            lat_row[m] = float(np.mean(lv))
    latency_rows.append(lat_row)
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
mcnemar_df = pd.DataFrame(mcnemar_rows, columns=['Dataset','Baseline','b10_KATSbetter','b01_Basebetter','p_raw','direction'])
log("Saved Table5_E1_full_results.csv, Table7_latency_measured.csv, confusion_matrices.json")

# ════════════════════════════════════════════════════════════════
# SECTION 3 — HOLM-BONFERRONI
# ════════════════════════════════════════════════════════════════
log("="*70); log("SECTION 3: HOLM-BONFERRONI CORRECTION"); log("="*70)
valid_mask = mcnemar_df['p_raw'].notna()
if valid_mask.sum() > 0:
    reject, p_corrected, _, _ = multipletests(mcnemar_df.loc[valid_mask, 'p_raw'].values, method='holm')
    mcnemar_df.loc[valid_mask, 'p_holm'] = p_corrected
    mcnemar_df.loc[valid_mask, 'significant_holm_0.05'] = reject
    log(f"  Total tests: {valid_mask.sum()} | Significant: {int(reject.sum())}")
mcnemar_df.to_csv(f"{RESULTS_DIR}/Table12_McNemar_directional.csv", index=False)

# ════════════════════════════════════════════════════════════════
# SECTION 4 — M2 ABLATION
# ════════════════════════════════════════════════════════════════
log("="*70); log("SECTION 4: M2 ABLATION"); log("="*70)

def run_one_seed_ablation(ds_name, df, feats, seed):
    X = df[feats].fillna(0).astype(float)
    y, le, hi = encode_labels(df['priority_label'])
    ir = compute_ir(y)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=seed, stratify=y)
    cw = make_class_weights(y_tr, hi, alpha=5)
    cw_a1 = make_class_weights(y_tr, hi, alpha=1)
    use_smote = ir > IR_THRESHOLD
    def wrap(est):
        return ImbPipeline([('smote', SMOTE(random_state=seed)), ('est', est)]) if use_smote else est
    builders = {
        'T_Full': lambda: get_kats(cw, seed, ir=ir),
        'T_NoSMOTE': lambda: get_kats(cw, seed, ir=0.0),
        'T_NoAsymLoss': lambda: get_kats(cw_a1, seed, ir=ir),
        'T_NoCalibNB': lambda: StackingClassifier(
            estimators=[('lgb', wrap(lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05,
                            class_weight=cw, random_state=seed, verbose=-1, n_jobs=N_JOBS))),
                        ('rf', wrap(RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                                       random_state=seed, n_jobs=N_JOBS))),
                        ('nb', wrap(GaussianNB()))],
            final_estimator=LogisticRegression(C=1.0, max_iter=2000, random_state=seed, class_weight=cw),
            stack_method='predict_proba', passthrough=True, cv=3, n_jobs=N_JOBS),
        'T_NoStacking': lambda: wrap(lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05,
                            class_weight=cw, random_state=seed, verbose=-1, n_jobs=N_JOBS)),
    }
    out = {}
    for vname, builder in builders.items():
        try:
            m = builder(); m.fit(X_tr, y_tr)
            out[vname] = compute_metrics(y_te, m.predict(X_te), m.predict_proba(X_te), le, high_idx=hi)
        except Exception as e:
            log(f"    !! ablation {vname} failed on {ds_name}/seed{seed}: {e}")
            out[vname] = None
    return out

ablation_all_results = {}
for ds_name, (df, feats) in DATASETS.items():
    log(f"--- Ablation: {ds_name} ---")
    ablation_all_results[ds_name] = {}
    for seed in SEEDS:
        result = run_or_resume(f"ABL_{ds_name}_seed{seed}",
                                lambda df=df, feats=feats, ds=ds_name, s=seed: run_one_seed_ablation(ds, df, feats, s))
        ablation_all_results[ds_name][seed] = result

ablation_rows = []
for ds_name, (df, feats) in DATASETS.items():
    y_full, _, _ = encode_labels(df['priority_label'])
    ir = compute_ir(y_full)
    variants = ['T_Full','T_NoSMOTE','T_NoAsymLoss','T_NoCalibNB','T_NoStacking']
    per_v_rh = {v: [] for v in variants}; per_v_f1 = {v: [] for v in variants}; per_v_kap = {v: [] for v in variants}
    for seed in SEEDS:
        for v in variants:
            r = ablation_all_results[ds_name][seed].get(v)
            if r is None:
                continue
            per_v_rh[v].append(r['RecallHigh']); per_v_f1[v].append(r['MacroF1']); per_v_kap[v].append(r['Kappa'])
    base_rh = np.mean(per_v_rh['T_Full']) if per_v_rh['T_Full'] else np.nan
    for v in variants:
        rh = np.mean(per_v_rh[v]) if per_v_rh[v] else np.nan
        f1 = np.mean(per_v_f1[v]) if per_v_f1[v] else np.nan
        kap = np.mean(per_v_kap[v]) if per_v_kap[v] else np.nan
        try:
            if len(per_v_f1['T_Full']) == len(per_v_f1[v]) and per_v_f1[v]:
                diff = np.array(per_v_f1['T_Full']) - np.array(per_v_f1[v])
                p_wil = 1.0 if np.all(diff == 0) else wilcoxon(per_v_f1['T_Full'], per_v_f1[v], alternative='greater')[1]
            else:
                p_wil = np.nan
        except Exception:
            p_wil = np.nan
        ablation_rows.append({'Dataset': ds_name, 'Variant': v, 'RecallH': rh, 'MacroF1': f1, 'Kappa': kap,
                               'DeltaRecallH_vs_ThisRun_TFull': rh - base_rh, 'Wilcoxon_p_onesided': p_wil, 'IR': ir})
        log(f"    {v:<14} RecallH={rh:.4f} MacroF1={f1:.4f} DeltaRecallH={rh-base_rh:+.4f}")

pd.DataFrame(ablation_rows).to_csv(f"{RESULTS_DIR}/Table6_M2_ablation.csv", index=False)

# ════════════════════════════════════════════════════════════════
# SECTION 5 — NetBenefit FROM THIS RUN'S OWN E1 RESULTS
# ════════════════════════════════════════════════════════════════
log("="*70); log("SECTION 5: NetBenefit (illustrative error-cost model)"); log("="*70)

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
        lat_kats, lat_best = lat_row.get('KATS', np.nan), lat_row.get(best_name, np.nan)
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

log("="*70); log(f"ALL SECTIONS COMPLETE - outputs in: {RESULTS_DIR}"); log("="*70)
for f in sorted(os.listdir(RESULTS_DIR)):
    if not f.startswith('checkpoints'):
        log(f"  - {f}")
log("")
log(">>> REMINDER: CloudTask used the synthetic generator because the original Kaggle")
log(">>> dataset was removed. Add one sentence to your manuscript's data-availability")
log(">>> note disclosing this before the next reviewer round.")
