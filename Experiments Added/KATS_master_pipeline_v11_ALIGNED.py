"""
KATS MASTER PIPELINE v11 — ALIGNED WITH VERIFIED v9_SUBMISSION.py
====================================================================
This script is built DIRECTLY on top of your verified
`KATS_master_pipeline_v9_SUBMISSION.py` (the script whose
E1_full_results_5datasets_FINAL.csv and M2_ablation_5datasets_FINAL.csv
match your published Table 5 / Table 6 exactly, confirmed by direct
cell-by-cell comparison). Hyperparameters, feature lists, and function
logic are PRESERVED VERBATIM wherever they already produce your
published numbers. Only the specific, confirmed problems below are
changed — nothing else.

CONFIRMED FIXES IN THIS VERSION (mapped to Reviewer 3 comments):

  R3.1  ITIncident: made_sla_enc REMOVED from IT_CANDIDATES (residual
        post-outcome field; impact/urgency were ALREADY correctly
        excluded in v9_SUBMISSION.py — that part was never broken).
        Figure 4 SHAP panel will be regenerated from THIS script only,
        replacing the stale figure that still shows impact/urgency.

  R3.2  SMOTE moved INSIDE an imblearn Pipeline step for each base
        learner (was: applied to the full training partition BEFORE
        StackingClassifier's internal cv=3 fold splitting). This WILL
        shift ITIncident/CICIDS2017 numbers slightly vs. the old
        FINAL files — that is expected and correct; it removes a real
        leak, not a new mistake.

  R3.3  Class-weight formula unchanged (verified correct in code);
        Eq. (2) in the manuscript must be rewritten to match it
        (text-only fix, tracked here so you don't forget it).

  R3.7  Table 6 fix identified: change T_Full RecallHigh from 0.8471
        to 0.8441 in the manuscript (single-cell fix, no rerun needed
        for this specific number — but rerun anyway because of R3.2).

  R3.8  mcnemar_pvalue() now returns an explicit direction label.

  R3.9  Wilcoxon one-sided test now lives INSIDE this same ablation
        loop (was: computed by a disconnected script). Per-seed values
        saved for the appendix.

  R3.11 NetBenefit is now computed from the SAME 5-seed-averaged E1
        results produced in Section 2 of THIS run — not a separate
        single-seed re-fit. This is the confirmed fix for the
        CloudTask best-baseline mismatch (LogReg@single-seed=0.433 vs
        LogReg@5-seed-avg=0.375).

  R3.13 NEW Section 9: controlled IR-sweep (fixed n, fixed features).

  R3.14 compute_metrics() extended with PrecHigh, PR_AUC_High, and a
        saved confusion matrix per dataset/model/seed.

  R3.15 NEW: per-dataset inference latency actually measured (was:
        borrowed from a disconnected script and reused as one flat
        pair across every dataset). GoogleCluster full-size (405,894,
        printed BEFORE capping) vs. training subsample (60,000) both
        logged explicitly for Table 3.

  ALSO FIXED (found during verification, not in Reviewer 3's list but
  will cause a future round if left alone):
  - Table 2 documentation note: your ACTUAL hyperparameters are
    n_estimators=300 (LightGBM/XGBoost), 200 (Random Forest), cv=3
    (stacking) — NOT 500/300/5 as Table 2 currently states. This
    script keeps the verified 300/200/3 values (to avoid re-deriving
    every number in the paper); Table 2's TEXT must be corrected to
    say 300/200/3, not the other way around.

Run this ONE script top-to-bottom on Kaggle. It supersedes
v9_SUBMISSION.py, v8_CLEAN.py, and every _FINAL/v2.2/v2.3 CSV in
results/ and results/Updated Results/. Once this run completes, those
older files should be treated as deprecated.
"""

import os
os.environ['PYTHONWARNINGS'] = 'ignore'
import warnings, time, ast, itertools, json
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

SEED, SEEDS = 42, [42, 7, 13, 99, 2026]
np.random.seed(SEED)
IR_THRESHOLD = 3.0
MAX_TRAIN_ROWS = 60000
SMOTE_MAX_RATIO = 3.0
LEAK_THRESH = 0.75
N_JOBS = -1
SLA_PENALTY_PER_BREACH_USD = 50.0
COMPUTE_HOURLY_RATE_USD = 0.50
os.makedirs('/kaggle/working/results_v11', exist_ok=True)
RESULTS_DIR = '/kaggle/working/results_v11'

# ════════════════════════════════════════════════════════════════
# SECTION 0 — UTILITIES (verified-correct pieces kept verbatim;
# fixes clearly marked)
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
    """UNCHANGED from v9_SUBMISSION.py (verified correct). This EXACT
    formula must replace Eq. (2) in the manuscript (R3.3 text fix)."""
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
    """EXTENDED per R3.14: adds PrecHigh (was already computed but
    never surfaced to any table), PR_AUC_High, and an optional
    confusion matrix."""
    rep = classification_report(ytrue, ypred, target_names=le.classes_.tolist(),
                                 output_dict=True, zero_division=0)
    nc = len(le.classes_)
    try:
        auc = roc_auc_score(label_binarize(ytrue, classes=np.arange(nc)),
                             yproba, multi_class='ovr', average='macro')
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
    """R3.2 FIX: SMOTE now lives INSIDE an imblearn Pipeline for each
    base learner. sklearn's StackingClassifier builds its cv=3 OOF
    meta-features via cross_val_predict internally; because SMOTE is a
    pipeline step, it refits on EACH fold's training rows only — never
    on data that later appears in a held-out fold. This is the ONLY
    behavioral change vs. v9_SUBMISSION.py's get_kats(); every
    hyperparameter (300 LGB trees, 200 RF trees, cv=3, isotonic NB) is
    otherwise IDENTICAL to your verified submission script.
    SMOTE activates only when ir > IR_THRESHOLD, matching
    apply_smote_if_needed()'s original gate.
    """
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
    """UNCHANGED from v9_SUBMISSION.py (verified correct, already
    documents its own IR=1.0 degenerate-collapse fix in its docstring).
    """
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
        true_high = true_high_all
        tp = np.sum((pred_high == 1) & (true_high == 1))
        fp = np.sum((pred_high == 1) & (true_high == 0))
        fn = np.sum((pred_high == 0) & (true_high == 1))
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
    """UNCHANGED from v9_SUBMISSION.py — these hyperparameters (300
    LGB/XGB trees, 200 RF/BalancedRF trees) are what actually produced
    your published Table 5. Table 2's text must be corrected to match
    THESE values, not the other way around."""
    return {
        'LightGBM': lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                                        class_weight=cw, random_state=seed, verbose=-1, n_jobs=N_JOBS),
        'XGBoost': xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                                      use_label_encoder=False, eval_metric='mlogloss',
                                      random_state=seed, verbosity=0, n_jobs=N_JOBS),
        'RandomForest': RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                                random_state=seed, n_jobs=N_JOBS),
        'BalancedRF': BalancedRandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=N_JOBS),
        'MLP': MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=300,
                              early_stopping=True, random_state=seed, learning_rate_init=0.001),
        'LogReg': LogisticRegression(max_iter=2000, random_state=seed, class_weight='balanced'),
        'NaiveBayes': CalibratedClassifierCV(GaussianNB(), cv=3, method='isotonic'),
    }

def leakage_audit(df, candidate_features, label_col, dataset_name, threshold=LEAK_THRESH):
    """UNCHANGED (verified correct method: decision-stump balanced
    accuracy, NOT Spearman correlation — R3.5's text-fix target)."""
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
    log(f"--- LEAKAGE AUDIT: {dataset_name} (chance={chance:.4f}, thresh={threshold}, "
        f"decision-stump balanced accuracy) ---")
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
    """R3.8 FIX: adds explicit direction label to the original
    mcnemar_pvalue() from v9_SUBMISSION.py."""
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
# SECTION 1 — LOAD ALL 5 DATASETS (verbatim from v9_SUBMISSION.py,
# ONLY IT_CANDIDATES changed per R3.1)
# ════════════════════════════════════════════════════════════════
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
GOOGLE_N_FULL = len(dfgoogle)  # R3.15: log the TRUE full-trace size BEFORE any capping
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
log(f"GoogleCluster   {GOOGLE_N_FULL:>9,} rows (full trace) | {len(GOOGLE_CANDIDATES)} candidate features "
    f"| will be capped to {MAX_TRAIN_ROWS:,} for training (R3.15: report BOTH numbers in Table 3)")

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
# R3.1 FIX: made_sla_enc REMOVED. impact/urgency were ALREADY excluded in
# v9_SUBMISSION.py (that part of the reviewer's concern was based on an
# older script, not this one) -- made_sla is the one remaining
# post-outcome field, now dropped as well.
IT_CANDIDATES = ['reassignment_count','reopen_count','sys_mod_count','categoryenc',
    'locationenc','contactenc','knowledgeenc','reopenflag']
IT_CANDIDATES = [c for c in IT_CANDIDATES if c in dfit.columns]
for extra in ['assignenc','cmdbenc','subcatenc']:
    if extra in dfit.columns:
        IT_CANDIDATES.append(extra)
log(f"ITIncident      {len(dfit):>9,} rows | {len(IT_CANDIDATES)} candidate features "
    f"(impact/urgency/made_sla ALL excluded by design, R3.1)")

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
log(f"CICIDS2017 severity mapping (R3.4, publish verbatim in Sec 4.1): {SEVERITY_MAP}")
DOWNSAMPLE_FRAC = 0.05
parts = [grp.sample(frac=DOWNSAMPLE_FRAC, random_state=SEED)
          for _, grp in dfcic_raw.groupby('priority_label')]
dfcic = pd.concat(parts).reset_index(drop=True)
exclude_cols = {LABEL_COL, 'priority_label'}
CIC_CANDIDATES = [c for c in dfcic.columns
                   if c not in exclude_cols and pd.api.types.is_numeric_dtype(dfcic[c])]
dfcic[CIC_CANDIDATES] = dfcic[CIC_CANDIDATES].replace([np.inf, -np.inf], np.nan).fillna(0)
log(f"CICIDS2017      {len(dfcic):>9,} rows (5% stratified downsample) | {len(CIC_CANDIDATES)} candidate features")

# ════════════════════════════════════════════════════════════════
# SECTION 1.5 — LEAKAGE AUDIT ON ALL 5 DATASETS
# ════════════════════════════════════════════════════════════════
log("="*70); log("SECTION 1.5: LEAKAGE AUDIT — ALL 5 DATASETS"); log("="*70)
CLOUD_FEATURES, _  = leakage_audit(dfcloud, CLOUD_CANDIDATES, 'priority_label', 'CloudTask')
GOOGLE_FEATURES, _ = leakage_audit(dfgoogle, GOOGLE_CANDIDATES, 'priority_label', 'GoogleCluster')
IT_FEATURES, _     = leakage_audit(dfit, IT_CANDIDATES, 'priority_label', 'ITIncident')
MC_FEATURES, _     = leakage_audit(dfmc, MC_CANDIDATES, 'priority_label', 'MultiCloud')
CIC_FEATURES, _    = leakage_audit(dfcic, CIC_CANDIDATES, 'priority_label', 'CICIDS2017')

DATASETS = {
    'CloudTask':     (dfcloud, CLOUD_FEATURES),
    'GoogleCluster': (dfgoogle, GOOGLE_FEATURES),
    'ITIncident':    (dfit, IT_FEATURES),
    'MultiCloud':    (dfmc, MC_FEATURES),
    'CICIDS2017':    (dfcic, CIC_FEATURES),
}
log(f"Capping large datasets to MAX_TRAIN_ROWS={MAX_TRAIN_ROWS} (stratified, IR-preserving)...")
DATASETS = {name: (cap_dataset_size(df, 'priority_label'), feats)
            for name, (df, feats) in DATASETS.items()}

ir_table = []
for name, (df, feats) in DATASETS.items():
    y_enc, le, _ = encode_labels(df['priority_label'])
    ir = compute_ir(y_enc)
    ir_table.append((name, len(df), len(feats), ir))
    log(f"  {name:<16} n={len(df):>9,}  features={len(feats):<3}  IR={ir:6.2f}")
pd.DataFrame(ir_table, columns=['Dataset','N','N_Features','IR']).to_csv(
    f"{RESULTS_DIR}/Table3_dataset_summary.csv", index=False)

# ════════════════════════════════════════════════════════════════
# SECTION 2 — E1: FULL CLASSIFICATION COMPARISON, ALL 5 DATASETS, 5 SEEDS
#             (now also saves per-seed data for Section 7's NetBenefit
#              and Section 4's Wilcoxon, and confusion matrices)
# ════════════════════════════════════════════════════════════════
log("="*70); log("SECTION 2: E1 — KATS vs 7 BASELINES, ALL 5 DATASETS, 5 SEEDS"); log("="*70)

e1_rows, e1_per_seed_rows, mcnemar_rows = [], [], []
per_dataset_seed_recall = {}   # ds -> model -> list of per-seed RecallHigh (for R3.11 NetBenefit)
per_dataset_latency = {}       # R3.15
confusion_by_ds_model = {}     # R3.14

for ds_name, (df, feats) in DATASETS.items():
    log(f"--- E1: {ds_name} starting ({len(df):,} rows) ---")
    X = df[feats].fillna(0).astype(float)
    y, le, hi = encode_labels(df['priority_label'])
    ir = compute_ir(y)
    all_model_names = ['KATS', 'KATS_raw'] + list(get_baselines({}, 42).keys())
    per_model_metrics = {m: [] for m in all_model_names}
    per_model_recall_by_seed = {m: [] for m in all_model_names}
    latency_samples = {m: [] for m in all_model_names}
    last_seed_preds = {}
    kats_thresholds = []

    for seed in SEEDS:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20,
                                                    random_state=seed, stratify=y)
        cw = make_class_weights(y_tr, hi, alpha=5)

        # KATS_raw
        kats_raw = get_kats(cw, seed, ir=ir)
        t0 = time.perf_counter(); kats_raw.fit(X_tr, y_tr); _ = time.perf_counter() - t0
        proba_raw = kats_raw.predict_proba(X_te)
        pred_raw = kats_raw.predict(X_te)
        m = compute_metrics(y_te, pred_raw, proba_raw, le, high_idx=hi)
        per_model_metrics['KATS_raw'].append(m)
        per_model_recall_by_seed['KATS_raw'].append(m['RecallHigh'])

        # KATS (threshold-calibrated) — SMOTE now safely inside get_kats()'s Pipeline
        kats_model = get_kats(cw, seed, ir=ir)
        kats_model, thresh = optimize_high_threshold(kats_model, X_tr, y_tr, hi, seed=seed)
        t0 = time.perf_counter()
        pred, proba = predict_with_threshold(kats_model, X_te, hi, thresh, len(le.classes_))
        infer_t = (time.perf_counter() - t0) / max(1, len(X_te))
        kats_thresholds.append(thresh)
        m = compute_metrics(y_te, pred, proba, le, high_idx=hi, save_cm=(seed == SEEDS[-1]))
        per_model_metrics['KATS'].append(m)
        per_model_recall_by_seed['KATS'].append(m['RecallHigh'])
        latency_samples['KATS'].append(infer_t)
        last_seed_preds['KATS'] = (y_te, pred)

        baselines = get_baselines(cw, seed)
        for bname, bmodel in baselines.items():
            t0 = time.perf_counter(); bmodel.fit(X_tr, y_tr); train_t = time.perf_counter() - t0
            t0 = time.perf_counter()
            bproba = bmodel.predict_proba(X_te); bpred = bmodel.predict(X_te)
            infer_t_b = (time.perf_counter() - t0) / max(1, len(X_te))
            m = compute_metrics(y_te, bpred, bproba, le, high_idx=hi, save_cm=(seed == SEEDS[-1]))
            per_model_metrics[bname].append(m)
            per_model_recall_by_seed[bname].append(m['RecallHigh'])
            latency_samples[bname].append(infer_t_b)
            last_seed_preds[bname] = (y_te, bpred)

        e1_per_seed_rows.append({'Dataset': ds_name, 'Seed': seed,
                                  'KATS_RecallHigh': per_model_metrics['KATS'][-1]['RecallHigh'],
                                  'KATS_MacroF1': per_model_metrics['KATS'][-1]['MacroF1'],
                                  'KATS_Kappa': per_model_metrics['KATS'][-1]['Kappa']})

    log(f"  [{ds_name}] KATS calibrated thresholds across seeds: {[round(t,2) for t in kats_thresholds]}")
    per_dataset_seed_recall[ds_name] = per_model_recall_by_seed
    per_dataset_latency[ds_name] = {m: float(np.mean(latency_samples[m]) * 1e6) for m in all_model_names}
    confusion_by_ds_model[ds_name] = {m: per_model_metrics[m][-1].get('ConfusionMatrix')
                                       for m in all_model_names if 'ConfusionMatrix' in per_model_metrics[m][-1]}

    for mname, mlist in per_model_metrics.items():
        row = {'Dataset': ds_name, 'Model': mname, 'IR': ir}
        for key in ['RecallHigh', 'PrecHigh', 'F1High', 'MacroF1', 'Kappa', 'AUC', 'Brier', 'PR_AUC_High']:
            vals = [x[key] for x in mlist]
            row[key] = np.nanmean(vals)
        e1_rows.append(row)
        log(f"    {mname:<14} RecallH={row['RecallHigh']:.4f} MacroF1={row['MacroF1']:.4f} "
            f"Kappa={row['Kappa']:.4f} PrecH={row['PrecHigh']:.4f} PR-AUC-H={row['PR_AUC_High']:.4f}")

    y_te_ref, pred_kats = last_seed_preds['KATS']
    for bname in get_baselines({}, 42).keys():
        _, pred_b = last_seed_preds[bname]
        pval, b10, b01, direction = mcnemar_pvalue_directional(y_te_ref, pred_kats, pred_b)
        mcnemar_rows.append([ds_name, bname, b10, b01, pval, direction])

e1_df = pd.DataFrame(e1_rows)
e1_df.to_csv(f"{RESULTS_DIR}/Table5_E1_full_results.csv", index=False)
pd.DataFrame(e1_per_seed_rows).to_csv(f"{RESULTS_DIR}/Table5_appendix_per_seed.csv", index=False)
pd.DataFrame(per_dataset_latency).T.to_csv(f"{RESULTS_DIR}/Table7_latency_measured_per_dataset.csv")
with open(f"{RESULTS_DIR}/confusion_matrices.json", "w") as f:
    json.dump(confusion_by_ds_model, f, indent=2)
mcnemar_df = pd.DataFrame(mcnemar_rows, columns=['Dataset','Baseline','b10_KATSbetter',
                                                  'b01_Basebetter','p_raw','direction'])
log("Saved Table5_E1_full_results.csv, Table5_appendix_per_seed.csv, "
    "Table7_latency_measured_per_dataset.csv, confusion_matrices.json (R3.8/9/14/15)")

# ════════════════════════════════════════════════════════════════
# SECTION 3 — HOLM-BONFERRONI FAMILY-WISE CORRECTION (unchanged logic,
# now carries the direction column)
# ════════════════════════════════════════════════════════════════
log("="*70); log("SECTION 3: HOLM-BONFERRONI FAMILY-WISE CORRECTION"); log("="*70)
valid_mask = mcnemar_df['p_raw'].notna()
reject, p_corrected, _, _ = multipletests(mcnemar_df.loc[valid_mask, 'p_raw'].values, method='holm')
mcnemar_df.loc[valid_mask, 'p_holm'] = p_corrected
mcnemar_df.loc[valid_mask, 'significant_holm_0.05'] = reject
mcnemar_df.to_csv(f"{RESULTS_DIR}/Table12_McNemar_directional.csv", index=False)
log(f"  Total tests: {valid_mask.sum()} | Significant after Holm (p<0.05): {int(reject.sum())}")

# ════════════════════════════════════════════════════════════════
# SECTION 4 — M2 ABLATION (unchanged model-building logic from
# v9_SUBMISSION.py, EXCEPT get_kats() now has leakage-safe SMOTE;
# Wilcoxon added directly here per R3.9, so it's never orphaned again)
# ════════════════════════════════════════════════════════════════
log("="*70); log("SECTION 4: M2 KATS COMPONENT ABLATION (+ Wilcoxon, R3.9)"); log("="*70)

ablation_rows, ablation_per_seed = [], []
for ds_name, (df, feats) in DATASETS.items():
    log(f"--- Ablation: {ds_name} starting ({len(df):,} rows) ---")
    X = df[feats].fillna(0).astype(float)
    y, le, hi = encode_labels(df['priority_label'])
    ir = compute_ir(y)
    variants = {v: [] for v in ['T_Full','T_NoSMOTE','T_NoAsymLoss','T_NoCalibNB','T_NoStacking']}

    for seed in SEEDS:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20,
                                                    random_state=seed, stratify=y)
        cw = make_class_weights(y_tr, hi, alpha=5)
        cw_a1 = make_class_weights(y_tr, hi, alpha=1)

        m = get_kats(cw, seed, ir=ir); m.fit(X_tr, y_tr)
        met = compute_metrics(y_te, m.predict(X_te), m.predict_proba(X_te), le, high_idx=hi)
        variants['T_Full'].append(met)

        m = get_kats(cw, seed, ir=0.0)  # ir=0 forces the SMOTE gate off -> T_NoSMOTE
        m.fit(X_tr, y_tr)
        met = compute_metrics(y_te, m.predict(X_te), m.predict_proba(X_te), le, high_idx=hi)
        variants['T_NoSMOTE'].append(met)

        m = get_kats(cw_a1, seed, ir=ir); m.fit(X_tr, y_tr)
        met = compute_metrics(y_te, m.predict(X_te), m.predict_proba(X_te), le, high_idx=hi)
        variants['T_NoAsymLoss'].append(met)

        use_smote = ir > IR_THRESHOLD
        def wrap(est, s=seed):
            return ImbPipeline([('smote', SMOTE(random_state=s)), ('est', est)]) if use_smote else est
        m = StackingClassifier(
            estimators=[('lgb', wrap(lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05,
                            class_weight=cw, random_state=seed, verbose=-1, n_jobs=N_JOBS))),
                        ('rf', wrap(RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                                       random_state=seed, n_jobs=N_JOBS))),
                        ('nb', wrap(GaussianNB()))],
            final_estimator=LogisticRegression(C=1.0, max_iter=2000, random_state=seed, class_weight=cw),
            stack_method='predict_proba', passthrough=True, cv=3, n_jobs=N_JOBS)
        m.fit(X_tr, y_tr)
        met = compute_metrics(y_te, m.predict(X_te), m.predict_proba(X_te), le, high_idx=hi)
        variants['T_NoCalibNB'].append(met)

        m = wrap(lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, class_weight=cw,
                                     random_state=seed, verbose=-1, n_jobs=N_JOBS))
        m.fit(X_tr, y_tr)
        met = compute_metrics(y_te, m.predict(X_te), m.predict_proba(X_te), le, high_idx=hi)
        variants['T_NoStacking'].append(met)

        for vname in variants:
            ablation_per_seed.append({'Dataset': ds_name, 'Variant': vname, 'Seed': seed,
                                       'RecallHigh': variants[vname][-1]['RecallHigh'],
                                       'MacroF1': variants[vname][-1]['MacroF1'],
                                       'Kappa': variants[vname][-1]['Kappa']})

    base_rh = np.mean([x['RecallHigh'] for x in variants['T_Full']])
    base_f1 = np.mean([x['MacroF1'] for x in variants['T_Full']])
    for vname, vlist in variants.items():
        rh = np.mean([x['RecallHigh'] for x in vlist])
        f1 = np.mean([x['MacroF1'] for x in vlist])
        kap = np.mean([x['Kappa'] for x in vlist])
        full_f1s = [x['MacroF1'] for x in variants['T_Full']]
        var_f1s = [x['MacroF1'] for x in vlist]
        try:
            diff = np.array(full_f1s) - np.array(var_f1s)
            p_wil = 1.0 if np.all(diff == 0) else wilcoxon(full_f1s, var_f1s, alternative='greater')[1]
        except Exception:
            p_wil = np.nan
        ablation_rows.append({'Dataset': ds_name, 'Variant': vname, 'RecallH': rh, 'MacroF1': f1,
                               'Kappa': kap, 'DeltaRecallH_vs_ThisRun_TFull': rh - base_rh,
                               'DeltaMacroF1_vs_ThisRun_TFull': f1 - base_f1,
                               'Wilcoxon_p_onesided': p_wil,
                               'Wilcoxon_min_attainable_p_n5': 1/32, 'IR': ir})
        log(f"    {vname:<14} RecallH={rh:.4f} MacroF1={f1:.4f} DeltaRecallH={rh-base_rh:+.4f} "
            f"Wilcoxon_p(one-sided)={p_wil}")

ablation_df = pd.DataFrame(ablation_rows)
ablation_df.to_csv(f"{RESULTS_DIR}/Table6_M2_ablation.csv", index=False)
pd.DataFrame(ablation_per_seed).to_csv(f"{RESULTS_DIR}/Table6_appendix_per_seed.csv", index=False)
log("Saved Table6_M2_ablation.csv — R3.7 note: this run's own T_Full RecallHigh IS the number "
    "that belongs in Table 6 (do not substitute Table 5's KATS value).")

# ════════════════════════════════════════════════════════════════
# SECTION 5 — TEMPORAL LEAKAGE ROBUSTNESS (ITIncident) — unchanged
# ════════════════════════════════════════════════════════════════
log("="*70); log("SECTION 5: TEMPORAL SPLIT ROBUSTNESS — ITIncident"); log("="*70)
TIME_COL_CANDIDATES = [c for c in dfit_raw.columns if 'open' in c.lower() and 'at' in c.lower()]
temporal_rows = []
if TIME_COL_CANDIDATES:
    time_col = TIME_COL_CANDIDATES[0]
    dfit_time = dfit.copy()
    dfit_time[time_col] = pd.to_datetime(dfit_raw.loc[dfit.index, time_col], errors='coerce')
    dfit_time = dfit_time.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)
    n_split = int(len(dfit_time) * 0.80)
    X_full = dfit_time[IT_FEATURES].fillna(0).astype(float)
    y_full, le_it, hi_it = encode_labels(dfit_time['priority_label'])
    X_tr_c, X_te_c = X_full[:n_split], X_full[n_split:]
    y_tr_c, y_te_c = y_full[:n_split], y_full[n_split:]
    ir_c = compute_ir(y_tr_c)
    cw = make_class_weights(y_tr_c, hi_it, alpha=5)
    for mname, model in {'KATS': get_kats(cw, SEED, ir=ir_c),
                          'LightGBM': get_baselines(cw, SEED)['LightGBM'],
                          'LogReg': get_baselines(cw, SEED)['LogReg']}.items():
        model.fit(X_tr_c, y_tr_c)
        met = compute_metrics(y_te_c, model.predict(X_te_c), model.predict_proba(X_te_c), le_it, high_idx=hi_it)
        temporal_rows.append(['Chronological', mname, met['RecallHigh'], met['Kappa'], met['MacroF1']])
    X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(X_full, y_full, test_size=0.20,
                                                         random_state=SEED, stratify=y_full)
    ir_r = compute_ir(y_tr_r)
    cw_r = make_class_weights(y_tr_r, hi_it, alpha=5)
    for mname, model in {'KATS': get_kats(cw_r, SEED, ir=ir_r),
                          'LightGBM': get_baselines(cw_r, SEED)['LightGBM'],
                          'LogReg': get_baselines(cw_r, SEED)['LogReg']}.items():
        model.fit(X_tr_r, y_tr_r)
        met = compute_metrics(y_te_r, model.predict(X_te_r), model.predict_proba(X_te_r), le_it, high_idx=hi_it)
        temporal_rows.append(['Random', mname, met['RecallHigh'], met['Kappa'], met['MacroF1']])
    pd.DataFrame(temporal_rows, columns=['SplitType','Model','RecallHigh','Kappa','MacroF1']).to_csv(
        f"{RESULTS_DIR}/temporal_split_robustness.csv", index=False)
    log("Saved temporal_split_robustness.csv")
else:
    log("  WARNING: no opened_at-style timestamp column found.")

# ════════════════════════════════════════════════════════════════
# SECTION 6 — SCHEDULER CIRCULARITY CHECK (GoogleCluster) — unchanged
# ════════════════════════════════════════════════════════════════
log("="*70); log("SECTION 6: SCHEDULER CIRCULARITY CHECK — GoogleCluster"); log("="*70)
y_g, le_g, hi_g = encode_labels(dfgoogle['priority_label'])
ir_g = compute_ir(y_g)
sched_rows = []
GOOGLE_NO_SCHED = [f for f in GOOGLE_FEATURES if f != 'scheduler']
for feat_set, tag in [(GOOGLE_FEATURES, 'With_Scheduler'), (GOOGLE_NO_SCHED, 'Without_Scheduler')]:
    Xg = dfgoogle[feat_set].fillna(0).astype(float)
    X_tr, X_te, y_tr, y_te = train_test_split(Xg, y_g, test_size=0.20, random_state=SEED, stratify=y_g)
    cw = make_class_weights(y_tr, hi_g, alpha=5)
    for mname, model in {'KATS': get_kats(cw, SEED, ir=ir_g),
                          'LightGBM': get_baselines(cw, SEED)['LightGBM']}.items():
        model.fit(X_tr, y_tr)
        met = compute_metrics(y_te, model.predict(X_te), model.predict_proba(X_te), le_g, high_idx=hi_g)
        sched_rows.append([tag, mname, met['RecallHigh'], met['Kappa'], met['MacroF1']])
pd.DataFrame(sched_rows, columns=['FeatureSet','Model','RecallHigh','Kappa','MacroF1']).to_csv(
    f"{RESULTS_DIR}/scheduler_circularity_check.csv", index=False)
log("Saved scheduler_circularity_check.csv")

# ════════════════════════════════════════════════════════════════
# SECTION 7 — NetBenefit: NOW COMPUTED FROM THIS RUN'S OWN 5-SEED
# E1 RESULTS (R3.11 FIX — was a disconnected single-seed re-fit)
# ════════════════════════════════════════════════════════════════
log("="*70); log("SECTION 7: NetBenefit — ILLUSTRATIVE ERROR-COST MODEL (R3.11/R3.12)"); log("="*70)
cost_rows = []
for ds_name, (df, feats) in DATASETS.items():
    sub = e1_df[e1_df['Dataset'] == ds_name]
    kats_rh = sub[sub['Model'] == 'KATS']['RecallHigh'].values[0]
    others = sub[~sub['Model'].isin(['KATS', 'KATS_raw'])]
    best_row = others.loc[others['RecallHigh'].idxmax()]
    best_name, best_rh = best_row['Model'], best_row['RecallHigh']
    y_ds, le_ds, hi_ds = encode_labels(df['priority_label'])
    n_high_test = int(round((y_ds == hi_ds).mean() * len(y_ds) * 0.20))
    lat_kats = per_dataset_latency[ds_name]['KATS']
    lat_best = per_dataset_latency[ds_name][best_name]
    expected_savings = (kats_rh - best_rh) * n_high_test * SLA_PENALTY_PER_BREACH_USD
    compute_overhead = (lat_kats - lat_best) * n_high_test / 1e6 / 3600.0 * COMPUTE_HOURLY_RATE_USD
    net_benefit = expected_savings - compute_overhead
    cost_rows.append({'Dataset': ds_name, 'KATS_RecallH_5seedAvg': kats_rh, 'BestBaseline': best_name,
                       'Baseline_RecallH_5seedAvg': best_rh, 'N_HighTest_approx': n_high_test,
                       'Expected_Savings_USD': expected_savings, 'Compute_Overhead_USD': compute_overhead,
                       'Net_Benefit_USD': net_benefit,
                       'MODEL_LABEL': 'ILLUSTRATIVE ERROR-COST MODEL (R3.12)'})
    log(f"  {ds_name:<16} KATS_RH={kats_rh:.4f} vs {best_name}_RH={best_rh:.4f} (5-seed avg, "
        f"same numbers as Table 5) | Net=${net_benefit:.2f}")
pd.DataFrame(cost_rows).to_csv(f"{RESULTS_DIR}/Table13_NetBenefit.csv", index=False)
log("Saved Table13_NetBenefit.csv — computed from THIS run's Section 2 E1 results ONLY, "
    "same numbers as Table 5, no separate single-seed re-fit (R3.11 fix)")

# ════════════════════════════════════════════════════════════════
# SECTION 8 — CICIDS2017 STACKING-COLLAPSE DIAGNOSIS (R3.6, NEW)
# ════════════════════════════════════════════════════════════════
log("="*70); log("SECTION 8: CICIDS2017 STACKING COLLAPSE DIAGNOSIS (R3.6)"); log("="*70)
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
diag_report = {
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
with open(f"{RESULTS_DIR}/CICIDS2017_stacking_diagnosis.json", "w") as f:
    json.dump(diag_report, f, indent=2)
for k, v in diag_report.items():
    log(f"  {k:<22} MacroF1={v['MacroF1']:.4f} RecallH={v['RecallHigh']:.4f}")
log("Saved CICIDS2017_stacking_diagnosis.json")

# ════════════════════════════════════════════════════════════════
# SECTION 9 — CONTROLLED IR-SWEEP (R3.13, NEW)
# ════════════════════════════════════════════════════════════════
log("="*70); log("SECTION 9: CONTROLLED IR-SWEEP (fixed n, fixed features, R3.13)"); log("="*70)
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

df_ir, feats_ir = DATASETS['CICIDS2017']
X_ir = df_ir[feats_ir].fillna(0).astype(float).values
y_ir, le_ir, hi_ir = encode_labels(df_ir['priority_label'])
FIXED_N = 8000
ir_sweep_rows = []
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
    ir_sweep_rows.append({'target_IR': target_ir, 'achieved_IR': achieved_ir, 'n': len(yv),
                           'n_features': Xv.shape[1], **{f'MacroF1_{k}': v for k, v in scores.items()},
                           'BestModel': best_model})
    log(f"  target_IR={target_ir:<4} achieved_IR={achieved_ir:.2f} n={len(yv)} best={best_model}")
pd.DataFrame(ir_sweep_rows).to_csv(f"{RESULTS_DIR}/Table_IR_sweep_controlled.csv", index=False)
log("Saved Table_IR_sweep_controlled.csv")

# ════════════════════════════════════════════════════════════════
# SECTION 10 — FULL-STACK SHAP vs B1-ONLY SHAP (R3.15, NEW)
# ════════════════════════════════════════════════════════════════
log("="*70); log("SECTION 10: FULL-STACK SHAP vs B1-ONLY SHAP (R3.15)"); log("="*70)
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
    log(f"  B1-only vs full-stack SHAP rank agreement: Spearman rho={rho:.4f} (n={len(X_sample)})")
    pd.DataFrame({'feature': feats_s, 'rank_B1': rank_b1.argsort(), 'rank_full_stack': rank_full.argsort()}
                 ).to_csv(f"{RESULTS_DIR}/SHAP_B1_vs_fullstack_agreement.csv", index=False)
except ImportError:
    log("  'shap' not installed — run `pip install shap` on Kaggle and re-run Section 10 only.")
except Exception as e:
    log(f"  Full-stack SHAP failed ({e}) — investigate before finalizing R3.15.")

# ════════════════════════════════════════════════════════════════
log("="*70); log(f"ALL SECTIONS COMPLETE — outputs saved to: {RESULTS_DIR}"); log("="*70)
for f in sorted(os.listdir(RESULTS_DIR)):
    log(f"  - {f}")
log("")
log(f"REMINDER for Table 3: GoogleCluster full trace = {GOOGLE_N_FULL:,} rows; "
    f"training/eval used a stratified {MAX_TRAIN_ROWS:,}-row subsample (state BOTH in Sec 4.1).")
log("REMINDER for Table 2: correct the hyperparameter text to n_estimators=300 (LGB/XGB), "
    "200 (RF), cv=3 (stacking) -- these are what actually produced every number in this run.")
