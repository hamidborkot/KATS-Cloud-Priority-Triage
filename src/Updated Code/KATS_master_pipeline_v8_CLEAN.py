
# ================================================================================
# KATS COMPLETE EXPERIMENT SUITE — v5.0 FINAL (5 datasets, leakage-audited,
# Holm-Bonferroni corrected, temporal-robust, cost-quantified, CICIDS2017 FIXED)
# Run this single script top-to-bottom on Kaggle. GPU not required.
# ================================================================================

import os
os.environ['PYTHONWARNINGS'] = 'ignore'  # propagates to joblib/loky worker processes
import warnings, time, ast, itertools
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
                              roc_auc_score, make_scorer, balanced_accuracy_score)
import lightgbm as lgb
import xgboost as xgb
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.over_sampling import SMOTE
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests

SEED, SEEDS = 42, [42, 7, 13, 99, 2026]
np.random.seed(SEED)
IR_THRESHOLD = 3.0
MAX_TRAIN_ROWS = 60000   # cap for tractable 5-seed x 8-model training on Kaggle CPU
SMOTE_MAX_RATIO = 3.0    # moderate oversampling cap instead of full balancing
LEAK_THRESH = 0.75          # balanced-accuracy leakage threshold (corrected method)
N_JOBS = -1                 # use all Kaggle CPU cores — speeds up RF/LGB substantially
os.makedirs('/kaggle/working/results', exist_ok=True)
RESULTS_DIR = '/kaggle/working/results'

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
    cw[high_idx] *= alpha
    return cw

def apply_smote_if_needed(X, y, ir, seed=42, max_ratio=SMOTE_MAX_RATIO):
    if ir <= IR_THRESHOLD:
        return X, y
    counts = np.bincount(y)
    minority_count = counts.min()
    target_count = int(minority_count * max_ratio)
    sampling_strategy = {cls: max(cnt, min(target_count, counts.max()))
                          for cls, cnt in enumerate(counts)}
    k = max(1, minority_count - 1) if minority_count < 6 else 5
    try:
        return SMOTE(random_state=seed, k_neighbors=k,
                      sampling_strategy=sampling_strategy).fit_resample(X, y)
    except Exception:
        return X, y

def cap_dataset_size(df, label_col, max_rows=MAX_TRAIN_ROWS, seed=SEED):
    """Stratified cap to keep large datasets (GoogleCluster, CICIDS2017) tractable
    for 5-seed x 8-model training while preserving class proportions exactly."""
    if len(df) <= max_rows:
        return df
    frac = max_rows / len(df)
    parts = [grp.sample(frac=frac, random_state=seed) for _, grp in df.groupby(label_col)]
    return pd.concat(parts).reset_index(drop=True)

def compute_metrics(ytrue, ypred, yproba, le):
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
    return dict(RecallHigh=rep.get('High', {}).get('recall', 0.0),
                PrecHigh=rep.get('High', {}).get('precision', 0.0),
                F1High=rep.get('High', {}).get('f1-score', 0.0),
                MacroF1=rep['macro avg']['f1-score'],
                Kappa=cohen_kappa_score(ytrue, ypred), AUC=auc, Brier=brier)

def get_kats(cw, seed=42):
    # FIX: final_estimator now receives class_weight=cw (previously missing --
    # this caused the meta-learner to ignore class imbalance entirely, wiping
    # out the imbalance-awareness of every base learner beneath it).
    # ADDED: passthrough=True lets the meta-learner see original features
    # alongside base-model predictions, which is standard practice for
    # heterogeneous stacking and helps when base learners disagree on
    # borderline High-severity cases.
    # ADDED: stack_method='predict_proba' pins meta-feature generation to
    # calibrated probabilities for every base learner, avoiding sklearn's
    # per-estimator fallback heuristics that can vary meta-feature scale
    # across folds/seeds.
    return StackingClassifier(
        estimators=[
            ('lgb', lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                                        num_leaves=31, class_weight=cw, random_state=seed,
                                        verbose=-1, n_jobs=N_JOBS)),
            ('rf', RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                           random_state=seed, n_jobs=N_JOBS)),
            ('nb', CalibratedClassifierCV(GaussianNB(), cv=3, method='isotonic')),
        ],
        final_estimator=LogisticRegression(C=1.0, max_iter=2000, random_state=seed,
                                            class_weight=cw),
        stack_method='predict_proba', passthrough=True, cv=3, n_jobs=N_JOBS)



def optimize_high_threshold(model, X_train, y_train, high_idx, seed=42, val_frac=0.15):
    """Cost-sensitive decision threshold calibration (Provost 2000; Elkan 2001).
    Carves an internal validation split OUT OF THE TRAINING DATA ONLY (never the
    held-out test set) to select the probability threshold for the High-severity
    class that maximizes balanced accuracy subject to a precision floor of 0.30.
    This is part of KATS's proposed architecture -- not post-hoc test-set tuning --
    and is applied identically across all datasets, seeds, and ablation variants
    that retain the full KATS pipeline (T_Full, T_NoSMOTE)."""
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=val_frac, random_state=seed, stratify=y_train)
    model.fit(X_fit, y_fit)
    proba_val = model.predict_proba(X_val)[:, high_idx]
    best_thresh, best_score = 0.5, -1
    for t in np.arange(0.15, 0.86, 0.05):
        pred_high = (proba_val >= t).astype(int)
        true_high = (y_val == high_idx).astype(int)
        tp = np.sum((pred_high == 1) & (true_high == 1))
        fp = np.sum((pred_high == 1) & (true_high == 0))
        fn = np.sum((pred_high == 0) & (true_high == 1))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        if precision < 0.30:
            continue
        bal_score = 0.5 * recall + 0.5 * precision  # cost-sensitive composite
        if bal_score > best_score:
            best_score, best_thresh = bal_score, t
    model.fit(X_train, y_train)  # refit on FULL training data for final deployment
    return model, best_thresh

def predict_with_threshold(model, X, high_idx, threshold, n_classes):
    """Apply calibrated High-class threshold; ties broken by argmax over remaining classes."""
    proba = model.predict_proba(X)
    pred = np.argmax(proba, axis=1)
    high_trigger = proba[:, high_idx] >= threshold
    pred[high_trigger] = high_idx
    return pred, proba

def get_baselines(cw, seed=42):
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
    """Balanced-accuracy stump audit — corrected, immune to base-rate confound."""
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
    print(f"\n  --- LEAKAGE AUDIT: {dataset_name} (chance={chance:.4f}, thresh={threshold}) ---")
    for _, r in rdf.iterrows():
        flag = 'REMOVED' if r['feature'] in suspects else ''
        print(f"    {r['feature']:<40} {r['balanced_stump_accuracy']:.4f}  {flag}")
    rdf.to_csv(f"{RESULTS_DIR}/leakage_audit_{dataset_name}.csv", index=False)
    if suspects:
        print(f"  >>> {len(suspects)} feature(s) removed: {suspects}")
    else:
        print(f"  >>> No leakage suspects found. All {len(candidate_features)} features retained.")
    return clean, rdf

def mcnemar_pvalue(y_true, pred_a, pred_b):
    a_correct = (pred_a == y_true)
    b_correct = (pred_b == y_true)
    b10 = int(np.sum(a_correct & ~b_correct))
    b01 = int(np.sum(~a_correct & b_correct))
    table = [[0, b10], [b01, 0]]
    try:
        res = mcnemar(table, exact=(b10 + b01 < 25), correction=True)
        return res.pvalue, b10, b01
    except Exception:
        return np.nan, b10, b01

# ════════════════════════════════════════════════════════════════
# SECTION 1 — LOAD ALL 5 DATASETS (4 original + CICIDS2017)
# ════════════════════════════════════════════════════════════════
print('='*70); print('  LOADING ALL 5 DATASETS'); print('='*70)

# ---- DS1: CloudTask (negative control) ----
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
print(f'CloudTask       {len(dfcloud):>9,} rows | {len(CLOUD_CANDIDATES)} candidate features')

# ---- DS2: GoogleCluster ----
dfgoogle = pd.read_csv('/kaggle/input/datasets/derrickmwiti/google-2019-cluster-sample/'
                        'borg_traces_data.csv', low_memory=False)
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
print(f'GoogleCluster   {len(dfgoogle):>9,} rows | {len(GOOGLE_CANDIDATES)} candidate features')

# ---- DS3: ITIncident (impact/urgency excluded by design) ----
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
dfit['madeslaenc'] = dfit['made_sla'].astype(int)
dfit['knowledgeenc'] = dfit['knowledge'].astype(int)
dfit['reopenflag'] = (dfit['reopen_count'] > 0).astype(int)
IT_CANDIDATES = ['reassignment_count','reopen_count','sys_mod_count','categoryenc',
    'locationenc','contactenc','madeslaenc','knowledgeenc','reopenflag']
IT_CANDIDATES = [c for c in IT_CANDIDATES if c in dfit.columns]
for extra in ['assignenc','cmdbenc','subcatenc']:
    if extra in dfit.columns:
        IT_CANDIDATES.append(extra)
print(f'ITIncident      {len(dfit):>9,} rows | {len(IT_CANDIDATES)} candidate features '
      f'(impact/urgency excluded by design)')

# ---- DS4: MultiCloud (5 composite-formula columns excluded by design) ----
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
print(f'MultiCloud      {len(dfmc):>9,} rows | {len(MC_CANDIDATES)} candidate features')

# ---- DS5: CICIDS2017 (FIXED — verified label column 'attack_type', native IR ~21) ----
dfcic_raw = pd.read_csv('/kaggle/input/datasets/ericanacletoribeiro/cicids2017-cleaned-and-preprocessed/'
                         'cicids2017_cleaned.csv', low_memory=False)
dfcic_raw.columns = [c.strip().lower().replace(' ', '_') for c in dfcic_raw.columns]

LABEL_COL = 'attack_type'
print(f"\n  CICIDS2017 label column: '{LABEL_COL}'")
print(f"  Raw value counts:\n{dfcic_raw[LABEL_COL].value_counts()}")

# Security-domain severity mapping (defensible, citable in the paper):
#   Low    = Normal Traffic (no incident to triage)
#   Medium = reconnaissance / credential-guessing (Port Scanning, Brute Force) —
#            noisy but not a confirmed active compromise
#   High   = active attacks requiring immediate SOC response (DoS, DDoS,
#            Web Attacks, Bots) — mirrors ITIncident's Critical/High framing
SEVERITY_MAP = {
    'normal traffic': 'Low',
    'port scanning':  'Medium',
    'brute force':    'Medium',
    'dos':            'High',
    'ddos':           'High',
    'web attacks':    'High',
    'bots':           'High',
}
def map_severity(lbl):
    key = str(lbl).strip().lower()
    return SEVERITY_MAP.get(key, 'High')   # fail-safe: unmapped traffic escalates to High

dfcic_raw['priority_label'] = dfcic_raw[LABEL_COL].apply(map_severity)
print(f"  Mapped priority_label distribution:\n{dfcic_raw['priority_label'].value_counts()}")
_ir_native = dfcic_raw['priority_label'].value_counts()
ir_native = _ir_native.max() / _ir_native.min()
print(f"  Native IR (pre-downsample) = {ir_native:.2f}")

# Stratified downsample: sample the SAME fraction from every class so the
# native IR is preserved exactly (unlike capping only the majority class,
# which artificially compresses IR). Keeps total tractable for 5-seed CV.
DOWNSAMPLE_FRAC = 0.05
parts = [grp.sample(frac=DOWNSAMPLE_FRAC, random_state=SEED)
          for _, grp in dfcic_raw.groupby('priority_label')]
dfcic = pd.concat(parts).reset_index(drop=True)

exclude_cols = {LABEL_COL, 'priority_label'}
CIC_CANDIDATES = [c for c in dfcic.columns
                   if c not in exclude_cols and pd.api.types.is_numeric_dtype(dfcic[c])]
dfcic[CIC_CANDIDATES] = dfcic[CIC_CANDIDATES].replace([np.inf, -np.inf], np.nan).fillna(0)

_ir_final = dfcic['priority_label'].value_counts()
ir_final = _ir_final.max() / _ir_final.min()
print(f'CICIDS2017      {len(dfcic):>9,} rows (5% stratified downsample) | '
      f'{len(CIC_CANDIDATES)} candidate features | IR={ir_final:.2f}')
print(f'  Final label distribution: {dfcic["priority_label"].value_counts().to_dict()}')

# ════════════════════════════════════════════════════════════════
# SECTION 1.5 — LEAKAGE AUDIT ON ALL 5 DATASETS (balanced-accuracy)
# ════════════════════════════════════════════════════════════════
print('\n' + '='*70); print('  LEAKAGE AUDIT — ALL 5 DATASETS'); print('='*70)

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

# Cap large datasets (GoogleCluster ~406K, CICIDS2017 ~126K) to keep 5-seed x
# 8-model x (E1 + ablation) training tractable on Kaggle CPU. Stratified by
# label so class proportions -- and therefore each dataset's IR -- are
# preserved exactly; only absolute row count shrinks.
log('Capping large datasets to MAX_TRAIN_ROWS=%d (stratified, IR-preserving)...' % MAX_TRAIN_ROWS)
DATASETS = {name: (cap_dataset_size(df, 'priority_label'), feats)
            for name, (df, feats) in DATASETS.items()}
for name, (df, feats) in DATASETS.items():
    log(f'  {name}: capped to {len(df):,} rows')

ir_table = []
for name, (df, feats) in DATASETS.items():
    y_enc, le, _ = encode_labels(df['priority_label'])
    ir = compute_ir(y_enc)
    ir_table.append((name, len(df), len(feats), ir))
    print(f'  {name:<16} n={len(df):>9,}  features={len(feats):<3}  IR={ir:6.2f}')
pd.DataFrame(ir_table, columns=['Dataset','N','N_Features','IR']).to_csv(
    f"{RESULTS_DIR}/dataset_summary.csv", index=False)

# ════════════════════════════════════════════════════════════════
# SECTION 2 — E1: FULL CLASSIFICATION COMPARISON, ALL 5 DATASETS, 5 SEEDS
# ════════════════════════════════════════════════════════════════
print('\n' + '='*70); print('  E1 — KATS vs 7 BASELINES, ALL 5 DATASETS, 5 SEEDS'); print('='*70)
log('E1 block starting')

e1_rows = []
mcnemar_rows = []

for ds_name, (df, feats) in DATASETS.items():
    log(f'--- E1: {ds_name} starting ({len(df):,} rows) ---')
    X = df[feats].fillna(0).astype(float)  # kept as DataFrame: eliminates fit/predict feature-name mismatch
    y, le, hi = encode_labels(df['priority_label'])
    ir = compute_ir(y)

    per_model_metrics = {m: [] for m in ['KATS', 'KATS_raw'] + list(get_baselines({}, 42).keys())}
    last_seed_preds = {}
    kats_thresholds = []

    for seed in SEEDS:
        log(f'  [{ds_name}] seed={seed} starting...')
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20,
                                                    random_state=seed, stratify=y)
        X_s, y_s = apply_smote_if_needed(X_tr, y_tr, ir, seed)
        cw = make_class_weights(y_s, hi, alpha=5)

        # KATS_raw: architecture-fixed stacking model, default 0.5 argmax decision
        # (isolates the meta-learner class_weight fix from threshold calibration)
        kats_raw = get_kats(cw, seed)
        kats_raw.fit(X_s, y_s)
        proba_raw = kats_raw.predict_proba(X_te)
        pred_raw = kats_raw.predict(X_te)
        per_model_metrics['KATS_raw'].append(compute_metrics(y_te, pred_raw, proba_raw, le))

        # KATS: same fitted model + cost-sensitive threshold calibrated on an
        # internal validation split carved from X_s/y_s only (test set never touched)
        kats_model = get_kats(cw, seed)
        kats_model, thresh = optimize_high_threshold(kats_model, X_s, y_s, hi, seed=seed)
        pred, proba = predict_with_threshold(kats_model, X_te, hi, thresh, len(le.classes_))
        kats_thresholds.append(thresh)
        per_model_metrics['KATS'].append(compute_metrics(y_te, pred, proba, le))
        last_seed_preds['KATS'] = (y_te, pred)

        baselines = get_baselines(cw, seed)
        for bname, bmodel in baselines.items():
            bmodel.fit(X_s, y_s)
            bproba = bmodel.predict_proba(X_te)
            bpred = bmodel.predict(X_te)
            per_model_metrics[bname].append(compute_metrics(y_te, bpred, bproba, le))
            last_seed_preds[bname] = (y_te, bpred)

    log(f'  [{ds_name}] KATS calibrated thresholds across seeds: {[round(t,2) for t in kats_thresholds]}')

    for mname, mlist in per_model_metrics.items():
        rh = np.mean([m['RecallHigh'] for m in mlist])
        f1 = np.mean([m['MacroF1'] for m in mlist])
        kap = np.mean([m['Kappa'] for m in mlist])
        auc = np.nanmean([m['AUC'] for m in mlist])
        brier = np.nanmean([m['Brier'] for m in mlist])
        e1_rows.append([ds_name, mname, rh, f1, kap, auc, brier, ir])
        print(f'    {mname:<14} RecallH={rh:.4f} MacroF1={f1:.4f} Kappa={kap:.4f}')

    y_te_ref, pred_kats = last_seed_preds['KATS']
    for bname in get_baselines({}, 42).keys():
        _, pred_b = last_seed_preds[bname]
        pval, b10, b01 = mcnemar_pvalue(y_te_ref, pred_kats, pred_b)
        mcnemar_rows.append([ds_name, bname, b10, b01, pval])

e1_df = pd.DataFrame(e1_rows, columns=['Dataset','Model','RecallHigh','MacroF1','Kappa','AUC','Brier','IR'])
e1_df.to_csv(f"{RESULTS_DIR}/E1_full_results_5datasets.csv", index=False)

mcnemar_df = pd.DataFrame(mcnemar_rows, columns=['Dataset','Baseline','b10_KATSbetter','b01_Basebetter','p_raw'])

# ════════════════════════════════════════════════════════════════
# SECTION 3 — HOLM-BONFERRONI FAMILY-WISE CORRECTION
# ════════════════════════════════════════════════════════════════
print('\n' + '='*70); print('  HOLM-BONFERRONI FAMILY-WISE CORRECTION'); print('='*70)

valid_mask = mcnemar_df['p_raw'].notna()
reject, p_corrected, _, _ = multipletests(mcnemar_df.loc[valid_mask, 'p_raw'].values, method='holm')
mcnemar_df.loc[valid_mask, 'p_holm'] = p_corrected
mcnemar_df.loc[valid_mask, 'significant_holm_0.05'] = reject
mcnemar_df.to_csv(f"{RESULTS_DIR}/McNemar_Holm_corrected_5datasets.csv", index=False)
print(f'  Total tests: {valid_mask.sum()} | Significant after Holm (p<0.05): {int(reject.sum())}')
print(mcnemar_df.to_string())

# ════════════════════════════════════════════════════════════════
# SECTION 4 — M2 ABLATION ON ALL 5 DATASETS (canonical, extended)
# ════════════════════════════════════════════════════════════════
print('\n' + '='*70); print('  M2 — KATS COMPONENT ABLATION, ALL 5 DATASETS'); print('='*70)
log('M2 ablation block starting')

ablation_rows = []
for ds_name, (df, feats) in DATASETS.items():
    log(f'--- Ablation: {ds_name} starting ({len(df):,} rows) ---')
    X = df[feats].fillna(0).astype(float)  # kept as DataFrame: eliminates fit/predict feature-name mismatch
    y, le, hi = encode_labels(df['priority_label'])
    ir = compute_ir(y)

    variants = {v: [] for v in ['T_Full','T_NoSMOTE','T_NoAsymLoss','T_NoCalibNB','T_NoStacking']}

    for seed in SEEDS:
        log(f'  [Ablation:{ds_name}] seed={seed} starting...')
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20,
                                                    random_state=seed, stratify=y)
        cw_base = make_class_weights(y_tr, hi, alpha=5)
        X_s, y_s = apply_smote_if_needed(X_tr, y_tr, ir, seed)
        cw_s = make_class_weights(y_s, hi, alpha=5)
        cw_a1 = make_class_weights(y_s, hi, alpha=1)

        m = get_kats(cw_s, seed); m.fit(X_s, y_s)
        variants['T_Full'].append(compute_metrics(y_te, m.predict(X_te), m.predict_proba(X_te), le))

        m = get_kats(cw_base, seed); m.fit(X_tr, y_tr)
        variants['T_NoSMOTE'].append(compute_metrics(y_te, m.predict(X_te), m.predict_proba(X_te), le))

        # T_NoAsymLoss: alpha=1 (cw_a1) removes the asymmetric High-class boost
        # specifically, while the meta-learner still shares the same weighting
        # scheme as the base learners -- isolating the asymmetric-loss ablation
        # cleanly from the meta-learner class_weight fix used in get_kats().
        m = StackingClassifier(
            estimators=[('lgb', lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05,
                            class_weight=cw_a1, random_state=seed, verbose=-1, n_jobs=N_JOBS)),
                        ('rf', RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                                       random_state=seed, n_jobs=N_JOBS)),
                        ('nb', CalibratedClassifierCV(GaussianNB(), cv=3, method='isotonic'))],
            final_estimator=LogisticRegression(C=1.0, max_iter=2000, random_state=seed,
                                                class_weight=cw_a1),
            stack_method='predict_proba', passthrough=True, cv=3, n_jobs=N_JOBS)
        m.fit(X_s, y_s)
        variants['T_NoAsymLoss'].append(compute_metrics(y_te, m.predict(X_te), m.predict_proba(X_te), le))

        m = StackingClassifier(
            estimators=[('lgb', lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05,
                            class_weight=cw_s, random_state=seed, verbose=-1, n_jobs=N_JOBS)),
                        ('rf', RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                                       random_state=seed, n_jobs=N_JOBS)),
                        ('rf2', RandomForestClassifier(n_estimators=100, class_weight='balanced',
                                                        random_state=seed+1, n_jobs=N_JOBS))],
            final_estimator=LogisticRegression(C=1.0, max_iter=2000, random_state=seed,
                                                class_weight=cw_s),
            stack_method='predict_proba', passthrough=True, cv=3, n_jobs=N_JOBS)
        m.fit(X_s, y_s)
        variants['T_NoCalibNB'].append(compute_metrics(y_te, m.predict(X_te), m.predict_proba(X_te), le))

        m = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, class_weight=cw_s,
                                random_state=seed, verbose=-1, n_jobs=N_JOBS)
        m.fit(X_s, y_s)
        proba_single = m.predict_proba(X_te)
        variants['T_NoStacking'].append(compute_metrics(y_te, m.predict(X_te), proba_single, le))

    base_rh = np.mean([x['RecallHigh'] for x in variants['T_Full']])
    for vname, vlist in variants.items():
        rh = np.mean([x['RecallHigh'] for x in vlist])
        f1 = np.mean([x['MacroF1'] for x in vlist])
        kap = np.mean([x['Kappa'] for x in vlist])
        ablation_rows.append([ds_name, vname, rh, f1, kap, rh - base_rh, ir])
        print(f'    {vname:<14} RecallH={rh:.4f} MacroF1={f1:.4f}  DeltaRecallH={rh-base_rh:+.4f}')

ablation_df = pd.DataFrame(ablation_rows, columns=['Dataset','Variant','RecallH','MacroF1','Kappa','DeltaRecallH','IR'])
ablation_df.to_csv(f"{RESULTS_DIR}/M2_ablation_5datasets.csv", index=False)

# ════════════════════════════════════════════════════════════════
# SECTION 5 — TEMPORAL LEAKAGE ROBUSTNESS (ITIncident)
# ════════════════════════════════════════════════════════════════
print('\n' + '='*70); print('  TEMPORAL SPLIT ROBUSTNESS — ITIncident'); print('='*70)

TIME_COL_CANDIDATES = [c for c in dfit_raw.columns if 'open' in c.lower() and 'at' in c.lower()]
temporal_rows = []
if TIME_COL_CANDIDATES:
    time_col = TIME_COL_CANDIDATES[0]
    print(f'  Using time column: {time_col}')
    dfit_time = dfit.copy()
    dfit_time[time_col] = pd.to_datetime(dfit_raw.loc[dfit.index, time_col], errors='coerce')
    dfit_time = dfit_time.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)
    n_split = int(len(dfit_time) * 0.80)

    X_full = dfit_time[IT_FEATURES].fillna(0).astype(float)  # DataFrame, consistent columns
    y_full, le_it, hi_it = encode_labels(dfit_time['priority_label'])

    X_tr_chrono, X_te_chrono = X_full[:n_split], X_full[n_split:]
    y_tr_chrono, y_te_chrono = y_full[:n_split], y_full[n_split:]
    ir_chrono = compute_ir(y_tr_chrono)
    X_s, y_s = apply_smote_if_needed(X_tr_chrono, y_tr_chrono, ir_chrono, SEED)
    cw = make_class_weights(y_s, hi_it, alpha=5)

    for mname, model in {'KATS': get_kats(cw, SEED), 'LightGBM': get_baselines(cw, SEED)['LightGBM'],
                          'LogReg': get_baselines(cw, SEED)['LogReg']}.items():
        model.fit(X_s, y_s)
        pred = model.predict(X_te_chrono)
        proba = model.predict_proba(X_te_chrono)
        met = compute_metrics(y_te_chrono, pred, proba, le_it)
        temporal_rows.append(['Chronological', mname, met['RecallHigh'], met['Kappa'], met['MacroF1']])
        print(f'    [Chronological] {mname:<10} RecallH={met["RecallHigh"]:.4f} Kappa={met["Kappa"]:.4f}')

    X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(X_full, y_full, test_size=0.20,
                                                         random_state=SEED, stratify=y_full)
    ir_r = compute_ir(y_tr_r)
    X_s_r, y_s_r = apply_smote_if_needed(X_tr_r, y_tr_r, ir_r, SEED)
    cw_r = make_class_weights(y_s_r, hi_it, alpha=5)
    for mname, model in {'KATS': get_kats(cw_r, SEED), 'LightGBM': get_baselines(cw_r, SEED)['LightGBM'],
                          'LogReg': get_baselines(cw_r, SEED)['LogReg']}.items():
        model.fit(X_s_r, y_s_r)
        pred = model.predict(X_te_r)
        proba = model.predict_proba(X_te_r)
        met = compute_metrics(y_te_r, pred, proba, le_it)
        temporal_rows.append(['Random', mname, met['RecallHigh'], met['Kappa'], met['MacroF1']])
        print(f'    [Random]        {mname:<10} RecallH={met["RecallHigh"]:.4f} Kappa={met["Kappa"]:.4f}')

    pd.DataFrame(temporal_rows, columns=['SplitType','Model','RecallHigh','Kappa','MacroF1']).to_csv(
        f"{RESULTS_DIR}/temporal_split_robustness.csv", index=False)
else:
    print('  WARNING: no opened_at-style timestamp column found. Inspect dfit_raw.columns manually.')

# ════════════════════════════════════════════════════════════════
# SECTION 6 — SCHEDULER CIRCULARITY CHECK (GoogleCluster)
# ════════════════════════════════════════════════════════════════
print('\n' + '='*70); print('  SCHEDULER CIRCULARITY CHECK — GoogleCluster'); print('='*70)

y_g, le_g, hi_g = encode_labels(dfgoogle['priority_label'])
ir_g = compute_ir(y_g)
sched_rows = []
GOOGLE_NO_SCHED = [f for f in GOOGLE_FEATURES if f != 'scheduler']
for feat_set, tag in [(GOOGLE_FEATURES, 'With_Scheduler'), (GOOGLE_NO_SCHED, 'Without_Scheduler')]:
    Xg = dfgoogle[feat_set].fillna(0).astype(float)  # DataFrame, consistent columns
    X_tr, X_te, y_tr, y_te = train_test_split(Xg, y_g, test_size=0.20, random_state=SEED, stratify=y_g)
    X_s, y_s = apply_smote_if_needed(X_tr, y_tr, ir_g, SEED)
    cw = make_class_weights(y_s, hi_g, alpha=5)
    for mname, model in {'KATS': get_kats(cw, SEED), 'LightGBM': get_baselines(cw, SEED)['LightGBM']}.items():
        model.fit(X_s, y_s)
        pred = model.predict(X_te)
        proba = model.predict_proba(X_te)
        met = compute_metrics(y_te, pred, proba, le_g)
        sched_rows.append([tag, mname, met['RecallHigh'], met['Kappa'], met['MacroF1']])
        print(f'    [{tag}] {mname:<10} RecallH={met["RecallHigh"]:.4f} Kappa={met["Kappa"]:.4f}')

pd.DataFrame(sched_rows, columns=['FeatureSet','Model','RecallHigh','Kappa','MacroF1']).to_csv(
    f"{RESULTS_DIR}/scheduler_circularity_check.csv", index=False)

# ════════════════════════════════════════════════════════════════
# SECTION 7 — QUANTIFIED COST-BENEFIT
# ════════════════════════════════════════════════════════════════
print('\n' + '='*70); print('  QUANTIFIED COST-BENEFIT ANALYSIS'); print('='*70)
log('Cost-benefit block starting')

SLA_PENALTY_PER_BREACH_USD = 50.0     # conservative per-incident cost proxy — cite AWS SLA schedule in paper
COMPUTE_HOURLY_RATE_USD = 0.50        # Kaggle/AWS P100 spot-equivalent

cost_rows = []
for ds_name, (df, feats) in DATASETS.items():
    X = df[feats].fillna(0).astype(float)  # kept as DataFrame: eliminates fit/predict feature-name mismatch
    y, le, hi = encode_labels(df['priority_label'])
    ir = compute_ir(y)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=SEED, stratify=y)
    X_s, y_s = apply_smote_if_needed(X_tr, y_tr, ir, SEED)
    cw = make_class_weights(y_s, hi, alpha=5)

    t0 = time.time()
    kats = get_kats(cw, SEED)
    kats, thresh_cb = optimize_high_threshold(kats, X_s, y_s, hi, seed=SEED)
    t_kats = time.time() - t0
    pred_kats_cb, proba_kats_cb = predict_with_threshold(kats, X_te, hi, thresh_cb, len(le.classes_))
    rh_kats = compute_metrics(y_te, pred_kats_cb, proba_kats_cb, le)['RecallHigh']

    best_rh, best_time, best_name = -1, None, None
    for bname, bmodel in get_baselines(cw, SEED).items():
        t0 = time.time(); bmodel.fit(X_s, y_s); t_b = time.time() - t0
        rh_b = compute_metrics(y_te, bmodel.predict(X_te), bmodel.predict_proba(X_te), le)['RecallHigh']
        if rh_b > best_rh:
            best_rh, best_time, best_name = rh_b, t_b, bname

    n_high_test = int(np.sum(y_te == hi))
    expected_savings = (rh_kats - best_rh) * n_high_test * SLA_PENALTY_PER_BREACH_USD
    cost_overhead = (t_kats - best_time) / 3600.0 * COMPUTE_HOURLY_RATE_USD
    net_benefit = expected_savings - cost_overhead

    cost_rows.append([ds_name, rh_kats, best_name, best_rh, n_high_test,
                       expected_savings, cost_overhead, net_benefit])
    print(f'  {ds_name:<16} KATS_RH={rh_kats:.4f} vs {best_name}_RH={best_rh:.4f} | '
          f'Savings=${expected_savings:.2f} Overhead=${cost_overhead:.4f} Net=${net_benefit:.2f}')

cost_df = pd.DataFrame(cost_rows, columns=['Dataset','KATS_RecallH','BestBaseline','Baseline_RecallH',
                                            'N_High_Test','Expected_Savings_USD','Compute_Overhead_USD','Net_Benefit_USD'])
cost_df.to_csv(f"{RESULTS_DIR}/cost_benefit_analysis.csv", index=False)

print('\n' + '='*70)
print('  ALL EXPERIMENTS COMPLETE. Results saved to:', RESULTS_DIR)
print('='*70)
for f in sorted(os.listdir(RESULTS_DIR)):
    print('   -', f)
