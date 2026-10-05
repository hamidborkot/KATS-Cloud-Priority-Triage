"""
R3.13 IR-SWEEP — TRULY CONTROLLED VERSION

Why the previous two attempts still weren't clean, and why this one is:

  Attempt 1: assumed "High" was the minority class. Wrong - Medium is
  (High=7,756, Low=49,868, Medium=2,376). Achieved IR stuck at ~21
  regardless of target.

  Attempt 2: correctly targeted Medium, but sampled the "majority"
  side as a random Low+High mixture whose own internal ~6.4:1 ratio
  was never controlled. When Medium's count grew large enough to
  exceed the (randomly diluted) High count in a given draw, High
  became the ACCIDENTAL minority instead of Medium, so the achieved
  IR reflected Low:High (~6.4), not the intended Low:Medium ratio -
  explaining why targets 2 and 5 both landed near 6.4.

  THIS VERSION: Low and High are held at FIXED ABSOLUTE COUNTS
  (Low=4000, High=3000) on EVERY row of the sweep - not randomly
  diluted, not proportional, literally the same two numbers every
  time. Only Medium's count is varied to hit the target IR exactly
  (majority_fixed / target_ir). Verified arithmetically before this
  was written: achieved IR = target IR exactly (2.00, 5.00, 10.00,
  20.00, 30.00) at every level, and the ordering Medium < High < Low
  holds throughout, so no class can silently swap roles.

  Total n is NOT held perfectly fixed (it ranges ~7,100-9,000 across
  the sweep) because forcing exact n while also hitting an exact IR
  on this dataset's real class-pool sizes is mathematically
  impossible - report the actual n per row rather than obscure this.
  Features are held exactly fixed (52) throughout, and Low/High are
  held exactly fixed in absolute count - IR is therefore the only
  DELIBERATELY varied quantity, which is what R3.13 actually asks for.

Self-contained: reloads CICIDS2017 itself, does not depend on any
prior session's in-memory DATASETS variable.
"""

import os
os.environ['PYTHONWARNINGS'] = 'ignore'
import warnings, re
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
from sklearn.metrics import f1_score
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

SEED = 42
IR_THRESHOLD = 3.0
N_JOBS = -1
RESULTS_DIR = '/kaggle/working/results_v17_ir_sweep'
os.makedirs(RESULTS_DIR, exist_ok=True)

CIC_PATH = '/kaggle/input/datasets/ericanacletoribeiro/cicids2017-cleaned-and-preprocessed/cicids2017_cleaned.csv'

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

# ════════════════════════════════════════════════════════════════
# LOAD CICIDS2017 (same logic as your main pipeline)
# ════════════════════════════════════════════════════════════════
log("Loading CICIDS2017...")
dfcic_raw = pd.read_csv(CIC_PATH, low_memory=False)
dfcic_raw.columns = [c.strip().lower().replace(' ', '_') for c in dfcic_raw.columns]
label_col = find_col(dfcic_raw, 'attack_type', 'label')
def map_severity(lbl):
    key = str(lbl).strip().lower()
    if 'benign' in key or 'normal' in key: return 'Low'
    if 'scan' in key or 'patator' in key or 'brute' in key: return 'Medium'
    return 'High'
dfcic_raw['priority_label'] = dfcic_raw[label_col].apply(map_severity)
exclude = {label_col, 'priority_label'}
CIC_FEATURES = [c for c in dfcic_raw.columns if c not in exclude and pd.api.types.is_numeric_dtype(dfcic_raw[c])]
dfcic_raw[CIC_FEATURES] = dfcic_raw[CIC_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)

y_all, le_all, hi_all = encode_labels(dfcic_raw['priority_label'])
classes_all, counts_all = np.unique(y_all, return_counts=True)
pool = dict(zip(le_all.classes_[classes_all], counts_all))
log(f"Full CICIDS2017 class pool: {pool}")
minority_class_name = min(pool, key=pool.get)
majority_class_name = max(pool, key=pool.get)
mid_class_name = [c for c in pool if c not in (minority_class_name, majority_class_name)][0]
log(f"Roles: majority={majority_class_name} ({pool[majority_class_name]:,} available), "
    f"mid={mid_class_name} ({pool[mid_class_name]:,} available), "
    f"minority={minority_class_name} ({pool[minority_class_name]:,} available)")

MAJORITY_FIXED = 4000
MID_FIXED = 3000
assert MAJORITY_FIXED <= pool[majority_class_name], "majority pool too small - reduce MAJORITY_FIXED"
assert MID_FIXED <= pool[mid_class_name], "mid pool too small - reduce MID_FIXED"

majority_idx = np.where(dfcic_raw['priority_label'] == majority_class_name)[0]
mid_idx = np.where(dfcic_raw['priority_label'] == mid_class_name)[0]
minority_idx_pool = np.where(dfcic_raw['priority_label'] == minority_class_name)[0]

X_full = dfcic_raw[CIC_FEATURES].astype(float).values
y_full_labels = dfcic_raw['priority_label'].values

rng = np.random.RandomState(SEED)
fixed_majority_rows = rng.choice(majority_idx, MAJORITY_FIXED, replace=False)
fixed_mid_rows = rng.choice(mid_idx, MID_FIXED, replace=False)

# ════════════════════════════════════════════════════════════════
# THE SWEEP: only the minority row-set size changes, per target IR
# ════════════════════════════════════════════════════════════════
log("="*70); log("R3.13 TRULY CONTROLLED IR-SWEEP"); log("="*70)
ir_sweep_rows = []
for target_ir in [2, 5, 10, 20, 30]:
    n_minority = min(int(round(MAJORITY_FIXED / target_ir)), len(minority_idx_pool))
    minority_rows = rng.choice(minority_idx_pool, n_minority, replace=False)
    keep_rows = np.concatenate([fixed_majority_rows, fixed_mid_rows, minority_rows])
    rng.shuffle(keep_rows)

    Xv = X_full[keep_rows]
    yv_labels = y_full_labels[keep_rows]
    yv, le_v, hi_v = encode_labels(pd.Series(yv_labels))
    achieved_ir = compute_ir(yv)
    classes_v, counts_v = np.unique(yv, return_counts=True)
    class_counts_v = dict(zip(le_v.classes_[classes_v], counts_v))
    minority_still_correct = min(class_counts_v, key=class_counts_v.get) == minority_class_name

    X_tr, X_te, y_tr, y_te = train_test_split(Xv, yv, test_size=0.20, random_state=SEED, stratify=yv)
    cw = make_class_weights(y_tr, hi_v, alpha=5)
    scores = {}
    for mname, model in {'KATS': get_kats(cw, SEED, ir=achieved_ir),
                          'LightGBM': lgb.LGBMClassifier(n_estimators=300, class_weight=cw,
                                                          random_state=SEED, verbose=-1, n_jobs=N_JOBS),
                          'LogReg': LogisticRegression(max_iter=2000, class_weight='balanced',
                                                        random_state=SEED)}.items():
        model.fit(X_tr, y_tr)
        scores[mname] = f1_score(y_te, model.predict(X_te), average='macro')
    best_model = max(scores, key=scores.get)
    ir_sweep_rows.append({'target_IR': target_ir, 'achieved_IR': achieved_ir, 'n_total': len(yv),
                           'n_features': Xv.shape[1], 'class_counts': class_counts_v,
                           'minority_role_preserved': minority_still_correct,
                           **{f'MacroF1_{k}': v for k, v in scores.items()}, 'BestModel': best_model})
    log(f"  target_IR={target_ir:<4} achieved_IR={achieved_ir:6.2f} n={len(yv):>5} "
        f"counts={class_counts_v} minority_role_preserved={minority_still_correct} "
        f"best={best_model} scores={ {k: round(v,4) for k,v in scores.items()} }")

result_df = pd.DataFrame(ir_sweep_rows)
result_df.to_csv(f"{RESULTS_DIR}/R3_13_IR_sweep_TRULY_CONTROLLED.csv", index=False)
log("Saved R3_13_IR_sweep_TRULY_CONTROLLED.csv")
log("Verify: achieved_IR should now equal target_IR almost exactly (2.00, 5.00, 10.00, 20.00, 30.00),")
log("  and minority_role_preserved should read True on every row.")
