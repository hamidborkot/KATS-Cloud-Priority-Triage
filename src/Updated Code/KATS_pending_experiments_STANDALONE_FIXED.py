
# ================================================================================
# KATS PENDING EXPERIMENTS — STANDALONE VERSION (SHAP stability + training-time
# multiplier for all 5 datasets, including CICIDS2017). Self-contained.
# FIX: handles all 3 possible shap_values() output shapes across shap versions
# (list-of-arrays, 3D ndarray (n,features,classes), or 2D ndarray).
# ================================================================================

import warnings, os, time, ast, datetime
warnings.filterwarnings('ignore')

def log(msg):
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

import numpy as np
import pandas as pd
import shap
from itertools import combinations
from scipy.stats import spearmanr
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb
from imblearn.over_sampling import SMOTE

SEED, SEEDS = 42, [42, 7, 13, 99, 2026]
np.random.seed(SEED)
IR_THRESHOLD = 3.0
MAX_TRAIN_ROWS = 60000
SMOTE_MAX_RATIO = 3.0
N_JOBS = -1
RESULTS_DIR = '/kaggle/working/results'
os.makedirs(RESULTS_DIR, exist_ok=True)

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
    if len(df) <= max_rows:
        return df
    frac = max_rows / len(df)
    parts = [grp.sample(frac=frac, random_state=seed) for _, grp in df.groupby(label_col)]
    return pd.concat(parts).reset_index(drop=True)

def get_kats(cw, seed=42):
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

def shap_mean_abs_importance(sv, n_features):
    """Robustly reduce shap_values() output to a 1D length-n_features vector,
    regardless of shap version / output shape:
      - list of length n_classes, each (n_samples, n_features)          -> old API
      - ndarray shape (n_samples, n_features, n_classes)                -> new API multiclass
      - ndarray shape (n_samples, n_features)                            -> binary/regression
    """
    if isinstance(sv, list):
        mean_abs = np.mean([np.abs(s).mean(axis=0) for s in sv], axis=0)
    else:
        sv = np.asarray(sv)
        if sv.ndim == 3:
            # (n_samples, n_features, n_classes) -> average over samples AND classes
            mean_abs = np.abs(sv).mean(axis=(0, 2))
        elif sv.ndim == 2:
            mean_abs = np.abs(sv).mean(axis=0)
        else:
            raise ValueError(f"Unexpected SHAP values shape: {sv.shape}")
    mean_abs = np.asarray(mean_abs).flatten()
    if mean_abs.shape[0] != n_features:
        raise ValueError(f"SHAP importance length {mean_abs.shape[0]} != n_features {n_features}")
    return mean_abs

# ════════════════════════════════════════════════════════════════
# SECTION 1 — LOAD ALL 5 DATASETS
# ════════════════════════════════════════════════════════════════
print('='*70); print(' LOADING ALL 5 DATASETS'); print('='*70)

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
CLOUD_FEATURES = ['service_criticality','data_volume_gb','rto_minutes','rpo_minutes',
    'dependency_count','downstream_critical','redundancy_level','regulatory_flag',
    'active_sessions','bandwidth_required_mbps','latency_sensitivity','az_risk_score',
    'multi_region_deployed','migration_complexity']
print(f'CloudTask {len(dfcloud):>9,} rows | {len(CLOUD_FEATURES)} features')

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
GOOGLE_FEATURES = ['scheduling_class','collection_type','instance_index','assigned_memory',
    'page_cache_memory','cycles_per_instruction','memory_accesses_per_instruction','sample_rate',
    'scheduler','vertical_scaling','reqcpus','reqmemory','avgcpus','avgmemory','maxcpus',
    'maxmemory','failed','eventenc']
print(f'GoogleCluster {len(dfgoogle):>9,} rows | {len(GOOGLE_FEATURES)} features')

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
IT_FEATURES = ['reassignment_count','reopen_count','sys_mod_count','categoryenc',
    'locationenc','contactenc','madeslaenc','knowledgeenc','reopenflag']
IT_FEATURES = [c for c in IT_FEATURES if c in dfit.columns]
for extra in ['assignenc','cmdbenc','subcatenc']:
    if extra in dfit.columns:
        IT_FEATURES.append(extra)
print(f'ITIncident {len(dfit):>9,} rows | {len(IT_FEATURES)} features')

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
norm_bw = 1 - dfmc['network_bandwidth_mbps'] / dfmc['network_bandwidth_mbps'].max()
norm_wv = dfmc['workload_variability'] / dfmc['workload_variability'].max()
composite = (0.30*norm_cpu + 0.25*norm_lat + 0.20*norm_thr + 0.15*norm_bw + 0.10*norm_wv)
dfmc['priority_label'] = pd.qcut(composite, q=3, labels=['Low','Medium','High']).astype(str)
MC_FEATURES = ['memory_usage_mb','storage_usage_gb','response_time_ms','load_balancing_%',
    'optimal_service_placement','servicetypeenc','cloudproviderenc','edgenodeenc']
MC_FEATURES = [c for c in MC_FEATURES if c in dfmc.columns]
print(f'MultiCloud {len(dfmc):>9,} rows | {len(MC_FEATURES)} features')

dfcic_raw = pd.read_csv('/kaggle/input/datasets/ericanacletoribeiro/cicids2017-cleaned-and-preprocessed/'
                         'cicids2017_cleaned.csv', low_memory=False)
dfcic_raw.columns = [c.strip().lower().replace(' ', '_') for c in dfcic_raw.columns]
LABEL_COL = 'attack_type'
SEVERITY_MAP = {
    'normal traffic': 'Low', 'port scanning': 'Medium', 'brute force': 'Medium',
    'dos': 'High', 'ddos': 'High', 'web attacks': 'High', 'bots': 'High',
}
def map_severity(lbl):
    key = str(lbl).strip().lower()
    return SEVERITY_MAP.get(key, 'High')
dfcic_raw['priority_label'] = dfcic_raw[LABEL_COL].apply(map_severity)
DOWNSAMPLE_FRAC = 0.05
parts = [grp.sample(frac=DOWNSAMPLE_FRAC, random_state=SEED)
         for _, grp in dfcic_raw.groupby('priority_label')]
dfcic = pd.concat(parts).reset_index(drop=True)
exclude_cols = {LABEL_COL, 'priority_label'}
CIC_FEATURES = [c for c in dfcic.columns
                if c not in exclude_cols and pd.api.types.is_numeric_dtype(dfcic[c])]
dfcic[CIC_FEATURES] = dfcic[CIC_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)
print(f'CICIDS2017 {len(dfcic):>9,} rows | {len(CIC_FEATURES)} features')

DATASETS = {
    'CloudTask': (dfcloud, CLOUD_FEATURES),
    'GoogleCluster': (dfgoogle, GOOGLE_FEATURES),
    'ITIncident': (dfit, IT_FEATURES),
    'MultiCloud': (dfmc, MC_FEATURES),
    'CICIDS2017': (dfcic, CIC_FEATURES),
}

log('Capping large datasets to MAX_TRAIN_ROWS (stratified, IR-preserving)...')
DATASETS = {name: (cap_dataset_size(df, 'priority_label'), feats)
            for name, (df, feats) in DATASETS.items()}
for name, (df, feats) in DATASETS.items():
    log(f' {name}: capped to {len(df):,} rows')

# ════════════════════════════════════════════════════════════════
# SECTION 8 — SHAP STABILITY ACROSS ALL 5 DATASETS (fixed aggregation)
# ════════════════════════════════════════════════════════════════
print('\n' + '='*70); print(' SHAP STABILITY — ALL 5 DATASETS (closing CICIDS2017 gap)'); print('='*70)
log('SHAP stability block starting')

shap_stability_rows = []
shap_rankings_store = {}

for ds_name, (df, feats) in DATASETS.items():
    log(f'--- SHAP stability: {ds_name} starting ({len(df):,} rows, {len(feats)} feats) ---')
    X = df[feats].fillna(0).astype(float).values
    y, le, hi = encode_labels(df['priority_label'])
    ir = compute_ir(y)

    seed_rankings = {}
    for seed in SEEDS:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.20, random_state=seed, stratify=y)
        X_s, y_s = apply_smote_if_needed(X_tr, y_tr, ir, seed)
        cw = make_class_weights(y_s, hi, alpha=5)

        b1 = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                                 num_leaves=31, class_weight=cw, random_state=seed,
                                 verbose=-1, n_jobs=N_JOBS)
        b1.fit(X_s, y_s)

        n_shap_sample = min(2000, len(X_te))
        rng = np.random.default_rng(seed)
        sample_idx = rng.choice(len(X_te), size=n_shap_sample, replace=False)
        X_shap = X_te[sample_idx]

        explainer = shap.TreeExplainer(b1)
        sv = explainer.shap_values(X_shap)

        mean_abs = shap_mean_abs_importance(sv, len(feats))
        ranking = pd.Series(mean_abs, index=feats).sort_values(ascending=False)
        seed_rankings[seed] = ranking

    shap_rankings_store[ds_name] = seed_rankings

    rhos = []
    for s_a, s_b in combinations(SEEDS, 2):
        rank_a = seed_rankings[s_a].reindex(feats).rank(ascending=False)
        rank_b = seed_rankings[s_b].reindex(feats).rank(ascending=False)
        rho, pval = spearmanr(rank_a.values, rank_b.values)
        rhos.append(rho)

    mean_rho = float(np.mean(rhos))
    min_rho = float(np.min(rhos))
    shap_stability_rows.append([ds_name, mean_rho, min_rho, len(rhos), ir])
    print(f' {ds_name:<16} mean_rho={mean_rho:.4f} min_rho={min_rho:.4f} '
          f'(n_pairs={len(rhos)}) IR={ir:.2f}')

    avg_rank_df = pd.concat(
        [seed_rankings[s].rename(f'seed_{s}') for s in SEEDS], axis=1)
    avg_rank_df['mean_importance'] = avg_rank_df.mean(axis=1)
    avg_rank_df = avg_rank_df.sort_values('mean_importance', ascending=False)
    avg_rank_df.to_csv(f"{RESULTS_DIR}/shap_ranking_{ds_name}.csv")

shap_stability_df = pd.DataFrame(
    shap_stability_rows,
    columns=['Dataset', 'Mean_Spearman_rho', 'Min_Spearman_rho', 'N_SeedPairs', 'IR'])
shap_stability_df.to_csv(f"{RESULTS_DIR}/shap_stability_5datasets.csv", index=False)
print('\\nFull SHAP stability table (now includes CICIDS2017):')
print(shap_stability_df.to_string(index=False))

# ════════════════════════════════════════════════════════════════
# SECTION 9 — TRAINING-TIME MULTIPLIER, ALL 5 DATASETS
# ════════════════════════════════════════════════════════════════
print('\n' + '='*70); print(' TRAINING-TIME MULTIPLIER — ALL 5 DATASETS'); print('='*70)
log('Training-time isolation block starting')

timing_rows = []
for ds_name, (df, feats) in DATASETS.items():
    log(f'--- Timing: {ds_name} starting ({len(df):,} rows) ---')
    X = df[feats].fillna(0).astype(float).values
    y, le, hi = encode_labels(df['priority_label'])
    ir = compute_ir(y)

    kats_times, lgb_times = [], []
    for seed in SEEDS:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.20, random_state=seed, stratify=y)
        X_s, y_s = apply_smote_if_needed(X_tr, y_tr, ir, seed)
        cw = make_class_weights(y_s, hi, alpha=5)

        t0 = time.time()
        lgb_solo = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                                       num_leaves=31, class_weight=cw, random_state=seed,
                                       verbose=-1, n_jobs=N_JOBS)
        lgb_solo.fit(X_s, y_s)
        lgb_times.append(time.time() - t0)

        t0 = time.time()
        kats_model = get_kats(cw, seed)
        kats_model.fit(X_s, y_s)
        kats_times.append(time.time() - t0)

    mean_kats_t = float(np.mean(kats_times))
    mean_lgb_t = float(np.mean(lgb_times))
    multiplier = mean_kats_t / mean_lgb_t if mean_lgb_t > 0 else np.nan
    timing_rows.append([ds_name, mean_lgb_t, mean_kats_t, multiplier,
                         np.std(kats_times), np.std(lgb_times), ir])
    print(f' {ds_name:<16} LGB={mean_lgb_t:7.2f}s  KATS={mean_kats_t:7.2f}s  '
          f'multiplier={multiplier:5.1f}x  IR={ir:.2f}')

timing_df = pd.DataFrame(
    timing_rows,
    columns=['Dataset', 'LightGBM_seconds', 'KATS_seconds', 'Training_Time_Multiplier',
             'KATS_std', 'LightGBM_std', 'IR'])
timing_df.to_csv(f"{RESULTS_DIR}/training_time_multiplier_5datasets.csv", index=False)
print('\\nFull training-time multiplier table (now includes CICIDS2017):')
print(timing_df.to_string(index=False))

print('\n' + '='*70)
print(' PENDING EXPERIMENTS COMPLETE. New files in:', RESULTS_DIR)
print('='*70)
for f in ['shap_stability_5datasets.csv', 'training_time_multiplier_5datasets.csv']:
    print(' -', f)
