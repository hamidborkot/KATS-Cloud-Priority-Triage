"""
KATS ADD-ON v16 — R3.6, R3.13, R3.15, and the Table 6 threshold-
calibration fix. Fully self-contained (reloads the 4 live datasets
itself; does not depend on prior checkpoints surviving). Run this
AFTER v15 has already produced your main Tables 5/6/7/12/13 -
this script only adds what was still missing.

WHAT THIS FIXES, AND WHY (grounded in your own code, not assumed):

  TABLE 6 FIX (resolves R3.7 completely):
  Your own optimize_high_threshold() docstring states threshold
  calibration is "applied identically across all datasets, seeds,
  and ablation variants that retain the full KATS pipeline (T_Full,
  T_NoSMOTE)." The ablation code never actually called it for either
  variant - a real implementation bug, not a stylistic choice.
  Fix: T_Full = Table 5's already-computed calibrated KATS values
  (same model, same procedure, no rerun). T_NoSMOTE is recomputed
  WITH calibration added. T_NoAsymLoss/T_NoCalibNB/T_NoStacking stay
  as plain argmax, exactly as your docstring scopes them (they don't
  "retain the full KATS pipeline"). Add one paragraph to Section 3.4/
  3.5 and one step to Algorithm 1 documenting the calibration step,
  since it currently appears nowhere in the manuscript's methods.

  R3.6 - CICIDS2017 stacking-collapse diagnosis: confusion matrices +
  classwise breakdown for KATS-stacked vs LightGBM-solo vs simple
  probability-averaging, isolating whether the collapse is in the
  meta-learner or the base learners.

  R3.13 - Controlled IR-sweep: CICIDS2017 subsampled to five IR levels
  (2, 5, 10, 20, 30) at a FIXED n=8000 and FIXED 52-feature space, so
  IR is the only thing that varies - a genuinely controlled test of
  whether IR alone drives model selection.

  R3.15 (SHAP part) - B1-only TreeSHAP vs full-stack KernelSHAP on a
  100-row ITIncident sample, reporting Spearman rank agreement - the
  reviewer's "local disagreement" check.
"""

import os
os.environ['PYTHONWARNINGS'] = 'ignore'
import warnings, time, ast, json, re
warnings.filterwarnings('ignore')
import datetime
def log(msg):
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix, f1_score, recall_score
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

log(f"lightgbm={lgb.__version__} pandas={pd.__version__}")

SEED, SEEDS = 42, [42, 7, 13, 99, 2026]
IR_THRESHOLD = 3.0
MAX_TRAIN_ROWS = 60000
N_JOBS = -1
RESULTS_DIR = '/kaggle/working/results_v16_addon'
os.makedirs(RESULTS_DIR, exist_ok=True)

GOOGLE_PATH = '/kaggle/input/datasets/derrickmwiti/google-2019-cluster-sample/borg_traces_data.csv'
IT_PATH     = '/kaggle/input/datasets/shamiulislamshifat/it-incident-log-dataset/incident_event_log.csv'
MC_PATH     = '/kaggle/input/datasets/ziya07/multi-cloud-service-composition-dataset/multi_cloud_service_dataset.csv'
CIC_PATH    = '/kaggle/input/datasets/ericanacletoribeiro/cicids2017-cleaned-and-preprocessed/cicids2017_cleaned.csv'

# ==== PRESERVED CLOUDTASK TABLE 6 ROWS (unchanged - no calibration
# data exists for the removed dataset; kept exactly as before) ====
CLOUDTASK_ABLATION_PRESERVED = pd.DataFrame([
    ['CloudTask', 'T_Full',       0.4080, 0.3382,  0.0000],
    ['CloudTask', 'T_NoSMOTE',    0.4080, 0.3382,  0.0000],
    ['CloudTask', 'T_NoAsymLoss', 0.3720, 0.3394, -0.0360],
    ['CloudTask', 'T_NoCalibNB',  0.3780, 0.3369, -0.0300],
    ['CloudTask', 'T_NoStacking', 0.3165, 0.3338, -0.0915],
], columns=['Dataset', 'Variant', 'RecallH', 'MacroF1', 'DeltaRecallH_vs_ThisRun_TFull'])
CLOUDTASK_ABLATION_PRESERVED['CalibrationApplied'] = 'N/A (original dataset removed from Kaggle)'
CLOUDTASK_ABLATION_PRESERVED['Source'] = 'PRESERVED_ORIGINAL_VERIFIED'

# ════════════════════════════════════════════════════════════════
# ROBUST COLUMN MATCHING (same as v15)
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
        raise KeyError(f"Could not find any column matching {aliases!r}.\nActual columns: {list(df.columns)}")
    return None

# ════════════════════════════════════════════════════════════════
# MODEL UTILITIES (identical to v15)
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

def compute_metrics(ytrue, ypred, yproba, le, high_idx=None):
    from sklearn.metrics import classification_report, cohen_kappa_score
    rep = classification_report(ytrue, ypred, target_names=le.classes_.tolist(),
                                 output_dict=True, zero_division=0)
    return dict(RecallHigh=rep.get('High', {}).get('recall', 0.0),
                MacroF1=rep['macro avg']['f1-score'],
                Kappa=cohen_kappa_score(ytrue, ypred))

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

# ════════════════════════════════════════════════════════════════
# LOAD THE 4 LIVE DATASETS (identical loading logic to v15)
# ════════════════════════════════════════════════════════════════

def load_live_datasets():
    log("="*70); log("LOADING 4 LIVE DATASETS"); log("="*70)

    dfgoogle = pd.read_csv(GOOGLE_PATH, low_memory=False)
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

    dfmc = pd.read_csv(MC_PATH)
    st_col = find_col(dfmc, 'ServiceType')
    cp_col = find_col(dfmc, 'CloudProvider')
    en_col = find_col(dfmc, 'EdgeNodeID')
    dfmc['servicetypeenc'] = LabelEncoder().fit_transform(dfmc[st_col].astype(str))
    dfmc['cloudproviderenc'] = LabelEncoder().fit_transform(dfmc[cp_col].astype(str))
    dfmc['edgenodeenc'] = LabelEncoder().fit_transform(dfmc[en_col].astype(str))
    cpu_col = find_col(dfmc, 'CPUUtilization'); lat_col = find_col(dfmc, 'ServiceLatency')
    thr_col = find_col(dfmc, 'Throughput'); bw_col = find_col(dfmc, 'NetworkBandwidth')
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

    return {
        'GoogleCluster': (cap_dataset_size(dfgoogle, 'priority_label'), GOOGLE_CANDIDATES),
        'ITIncident':    (cap_dataset_size(dfit, 'priority_label'), IT_CANDIDATES),
        'MultiCloud':    (cap_dataset_size(dfmc, 'priority_label'), MC_CANDIDATES),
        'CICIDS2017':    (cap_dataset_size(dfcic, 'priority_label'), CIC_CANDIDATES),
    }

DATASETS = load_live_datasets()
for name, (df, feats) in DATASETS.items():
    y_enc, _, _ = encode_labels(df['priority_label'])
    log(f"  {name:<16} n={len(df):>9,}  features={len(feats):<3}  IR={compute_ir(y_enc):6.2f}")

# ════════════════════════════════════════════════════════════════
# PART A — TABLE 6 FIX: T_Full (=Table 5's KATS) and T_NoSMOTE (newly
# calibrated) for all 4 live datasets, 5 seeds each
# ════════════════════════════════════════════════════════════════
log("="*70); log("PART A: TABLE 6 FIX - threshold-calibrated T_Full and T_NoSMOTE"); log("="*70)

table6_fix_rows = []
for ds_name, (df, feats) in DATASETS.items():
    log(f"--- Table 6 fix: {ds_name} ---")
    X = df[feats].fillna(0).astype(float)
    y, le, hi = encode_labels(df['priority_label'])
    ir = compute_ir(y)
    tfull_rh, tfull_f1, tfull_k = [], [], []
    tnosmote_rh, tnosmote_f1, tnosmote_k = [], [], []
    for seed in SEEDS:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=seed, stratify=y)
        cw = make_class_weights(y_tr, hi, alpha=5)
        # T_Full: identical procedure to Table 5's "KATS" (calibrated)
        m = get_kats(cw, seed, ir=ir)
        m, thresh = optimize_high_threshold(m, X_tr, y_tr, hi, seed=seed)
        pred, proba = predict_with_threshold(m, X_te, hi, thresh, len(le.classes_))
        met = compute_metrics(y_te, pred, proba, le, high_idx=hi)
        tfull_rh.append(met['RecallHigh']); tfull_f1.append(met['MacroF1']); tfull_k.append(met['Kappa'])
        # T_NoSMOTE: SAME calibration procedure, SMOTE gate forced off (ir=0.0)
        m2 = get_kats(cw, seed, ir=0.0)
        m2, thresh2 = optimize_high_threshold(m2, X_tr, y_tr, hi, seed=seed)
        pred2, proba2 = predict_with_threshold(m2, X_te, hi, thresh2, len(le.classes_))
        met2 = compute_metrics(y_te, pred2, proba2, le, high_idx=hi)
        tnosmote_rh.append(met2['RecallHigh']); tnosmote_f1.append(met2['MacroF1']); tnosmote_k.append(met2['Kappa'])
    base_rh = np.mean(tfull_rh)
    table6_fix_rows.append({'Dataset': ds_name, 'Variant': 'T_Full', 'RecallH': np.mean(tfull_rh),
                             'MacroF1': np.mean(tfull_f1), 'Kappa': np.mean(tfull_k),
                             'DeltaRecallH_vs_ThisRun_TFull': 0.0, 'CalibrationApplied': True,
                             'Source': 'LIVE_RERUN_v16_CALIBRATED'})
    table6_fix_rows.append({'Dataset': ds_name, 'Variant': 'T_NoSMOTE', 'RecallH': np.mean(tnosmote_rh),
                             'MacroF1': np.mean(tnosmote_f1), 'Kappa': np.mean(tnosmote_k),
                             'DeltaRecallH_vs_ThisRun_TFull': np.mean(tnosmote_rh) - base_rh,
                             'CalibrationApplied': True, 'Source': 'LIVE_RERUN_v16_CALIBRATED'})
    log(f"    T_Full(calibrated)    RecallH={np.mean(tfull_rh):.4f} MacroF1={np.mean(tfull_f1):.4f}")
    log(f"    T_NoSMOTE(calibrated) RecallH={np.mean(tnosmote_rh):.4f} MacroF1={np.mean(tnosmote_f1):.4f} "
        f"DeltaRecallH={np.mean(tnosmote_rh)-base_rh:+.4f}")

table6_fix_df = pd.DataFrame(table6_fix_rows)
table6_fix_df.to_csv(f"{RESULTS_DIR}/Table6_TFull_TNoSMOTE_calibrated_FIX.csv", index=False)
log("Saved Table6_TFull_TNoSMOTE_calibrated_FIX.csv - splice these two rows per dataset into your")
log("  existing Table 6, replacing the uncalibrated T_Full/T_NoSMOTE rows. Leave T_NoAsymLoss/")
log("  T_NoCalibNB/T_NoStacking exactly as they are (uncalibrated, per the docstring's scope).")

# ════════════════════════════════════════════════════════════════
# PART B — R3.6: CICIDS2017 stacking-collapse diagnosis
# ════════════════════════════════════════════════════════════════
log("="*70); log("PART B: R3.6 - CICIDS2017 STACKING COLLAPSE DIAGNOSIS"); log("="*70)

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
                      'classes_order': le_d.classes_.tolist(),
                      'MacroF1': f1_score(y_te, pred_kats_d, average='macro'),
                      'RecallHigh': recall_score(y_te, pred_kats_d, labels=[hi_d], average=None)[0]},
    'LightGBM_solo': {'confusion_matrix': confusion_matrix(y_te, pred_lgb_d).tolist(),
                       'MacroF1': f1_score(y_te, pred_lgb_d, average='macro'),
                       'RecallHigh': recall_score(y_te, pred_lgb_d, labels=[hi_d], average=None)[0]},
    'ProbabilityAveraging': {'confusion_matrix': confusion_matrix(y_te, pred_avg_d).tolist(),
                              'MacroF1': f1_score(y_te, pred_avg_d, average='macro'),
                              'RecallHigh': recall_score(y_te, pred_avg_d, labels=[hi_d], average=None)[0]},
}
with open(f"{RESULTS_DIR}/R3_6_CICIDS2017_stacking_diagnosis.json", 'w') as f:
    json.dump(diag_report, f, indent=2)
for k, v in diag_report.items():
    log(f"  {k:<22} MacroF1={v['MacroF1']:.4f} RecallH={v['RecallHigh']:.4f}")
log(f"  Class order for confusion matrices: {le_d.classes_.tolist()}")
log("Saved R3_6_CICIDS2017_stacking_diagnosis.json - use confusion matrices to write the")
log("  R3.6 diagnostic paragraph: check whether KATS_stacked's errors concentrate on a")
log("  specific class relative to LightGBM_solo (architectural) or are diffuse (implementation).")

# ════════════════════════════════════════════════════════════════
# PART C — R3.13: Controlled IR-sweep on CICIDS2017 (fixed n, fixed
# features, IR is the ONLY thing that varies)
# ════════════════════════════════════════════════════════════════
log("="*70); log("PART C: R3.13 - CONTROLLED IR-SWEEP"); log("="*70)

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
    log(f"  target_IR={target_ir:<4} achieved_IR={achieved_ir:.2f} n={len(yv)} n_features={Xv.shape[1]} "
        f"best={best_model}  scores={ {k: round(v,4) for k,v in scores.items()} }")

pd.DataFrame(ir_sweep_rows).to_csv(f"{RESULTS_DIR}/R3_13_IR_sweep_controlled.csv", index=False)
log("Saved R3_13_IR_sweep_controlled.csv - n and features held fixed, ONLY IR varies.")

# ════════════════════════════════════════════════════════════════
# PART D — R3.15: Full-stack SHAP vs B1-only SHAP on ITIncident
# ════════════════════════════════════════════════════════════════
log("="*70); log("PART D: R3.15 - FULL-STACK SHAP vs B1-ONLY SHAP"); log("="*70)
try:
    import shap
except ImportError:
    log("  'shap' not installed. Installing now...")
    os.system("pip install -q shap")
    import shap

try:
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
    mean_abs_b1 = np.abs(shap_b1_high).mean(axis=0)
    rank_b1 = np.argsort(-mean_abs_b1)

    log("  Running KernelSHAP on the full stack (this is the slow part, ~5-15 min)...")
    explainer_full = shap.KernelExplainer(lambda x: kats_s.predict_proba(x)[:, hi_s],
                                           shap.sample(X_tr, 50, random_state=SEED))
    shap_full = explainer_full.shap_values(X_sample, nsamples=100)
    mean_abs_full = np.abs(shap_full).mean(axis=0)
    rank_full = np.argsort(-mean_abs_full)

    from scipy.stats import spearmanr
    rho, pval = spearmanr(rank_b1, rank_full)
    log(f"  B1-only vs full-stack SHAP feature-rank agreement: Spearman rho={rho:.4f} (p={pval:.4g}, n={len(X_sample)})")

    shap_compare = pd.DataFrame({'feature': feats_s, 'mean_abs_SHAP_B1': mean_abs_b1,
                                  'mean_abs_SHAP_fullstack': mean_abs_full,
                                  'rank_B1': pd.Series(mean_abs_b1).rank(ascending=False).astype(int).values,
                                  'rank_fullstack': pd.Series(mean_abs_full).rank(ascending=False).astype(int).values})
    shap_compare = shap_compare.sort_values('rank_B1')
    shap_compare.to_csv(f"{RESULTS_DIR}/R3_15_SHAP_B1_vs_fullstack.csv", index=False)
    with open(f"{RESULTS_DIR}/R3_15_SHAP_summary.json", 'w') as f:
        json.dump({'spearman_rho': float(rho), 'p_value': float(pval), 'n_sample': len(X_sample)}, f, indent=2)
    log("Saved R3_15_SHAP_B1_vs_fullstack.csv and R3_15_SHAP_summary.json")
except Exception as e:
    log(f"  Full-stack SHAP failed: {e}")
    log("  If this is a memory/timeout issue, reduce nsamples or the KernelExplainer background")
    log("  sample size (shap.sample(X_tr, 50, ...) -> shap.sample(X_tr, 20, ...)) and rerun Part D only.")

log("="*70); log(f"ADD-ON COMPLETE - outputs in: {RESULTS_DIR}"); log("="*70)
for f in sorted(os.listdir(RESULTS_DIR)):
    log(f"  - {f}")
