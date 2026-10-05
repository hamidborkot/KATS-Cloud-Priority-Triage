#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
print_pipeline_spec.py -- prints, in seconds and WITHOUT any data, the exact constructor arguments and class weights of the FINAL (cluster-A) pipeline.
The function bodies below are copied VERBATIM from your notebook cell 'KATS CLUSTER A FINAL' (weights, resampler, kats, baselines, threshold_fit).
Use the printout for Table 4 (estimator arguments), Section 5.3 (what T_NoAsymLoss removes) and Section 4.3 (threshold rule).
Run:  python print_pipeline_spec.py            (needs lightgbm, xgboost, imbalanced-learn, scikit-learn)
"""
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb, xgboost as xgb
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, SMOTENC
from imblearn.ensemble import BalancedRandomForestClassifier
IR_THRESHOLD = 3.0; MAX_ROWS = 60000; N_JOBS = 1
# ---------------------------------------------------------------- VERBATIM FROM THE NOTEBOOK ----------------------------------------------------------------
def weights(y, high, alpha=5.0):
    classes, counts = np.unique(y, return_counts=True)
    out = {int(c): float(len(y) / (len(classes) * n)) for c, n in zip(classes, counts)}
    if counts.max() / counts.min() > IR_THRESHOLD:
        out[int(high)] *= alpha
    return out



def resampler(dataset, features, cats, seed, enabled=True):
    if not enabled:
        return "passthrough"
    if dataset == "ITIncident":
        return SMOTENC(categorical_features=[features.index(x) for x in cats if x in features], random_state=seed, k_neighbors=5, sampling_strategy="not majority")
    if dataset == "CICIDS2017":
        return SMOTE(random_state=seed, k_neighbors=5, sampling_strategy="not majority")
    return "passthrough"


def kats(dataset, features, cats, cw, seed, smote_on=True, asym_on=True, cal_nb=True, stacking=True):
    if not asym_on:
        cw = {k: 1.0 for k in cw}
    def wrap(est, offset=0):
        return ImbPipeline([
            ("resample", resampler(dataset, features, cats, seed + offset, smote_on)),
            ("model", est),
        ])
    b1 = wrap(lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, num_leaves=31, class_weight=cw, random_state=seed, n_jobs=N_JOBS, verbose=-1), 0)
    if not stacking:
        return b1
    b2 = wrap(RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=seed, n_jobs=N_JOBS), 1000)
    b3 = CalibratedClassifierCV(GaussianNB(), cv=3, method="isotonic") if cal_nb else GaussianNB()
    return StackingClassifier(
        estimators=[("lgb", b1), ("rf", b2), ("nb", b3)],
        final_estimator=LogisticRegression(C=1.0, max_iter=2000, class_weight=cw, random_state=seed),
        stack_method="predict_proba", passthrough=True, cv=3, n_jobs=N_JOBS,
    )


def baselines(cw, seed):
    return {
        "LightGBM": lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, class_weight=cw, random_state=seed, n_jobs=N_JOBS, verbose=-1),
        "XGBoost": xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, eval_metric="mlogloss", random_state=seed, n_jobs=N_JOBS, verbosity=0),
        "RandomForest": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=seed, n_jobs=N_JOBS),
        "BalancedRF": BalancedRandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=N_JOBS),
        "MLP": Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", MLPClassifier(hidden_layer_sizes=(128,64,32), max_iter=300, early_stopping=True, random_state=seed))]),
        "LogReg": Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced", random_state=seed))]),
        "NaiveBayes": CalibratedClassifierCV(GaussianNB(), cv=3, method="isotonic"),
    }


def threshold_fit(model, X, y, high, seed):
    xf, xv, yf, yv = train_test_split(X, y, test_size=0.15, random_state=seed, stratify=y)
    model.fit(xf, yf)
    p = model.predict_proba(xv)[:, high]
    yt = (yv == high).astype(int)
    floor = max(0.30, 1.5 * yt.mean())
    best_t, best = 0.5, -1.0
    for t in np.arange(0.15, 0.86, 0.05):
        z = (p >= t).astype(int)
        tp = ((z == 1) & (yt == 1)).sum(); fp = ((z == 1) & (yt == 0)).sum(); fn = ((z == 0) & (yt == 1)).sum()
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        if prec >= floor and (prec + rec)/2 > best:
            best, best_t = (prec + rec)/2, float(t)
    model.fit(X, y)
    return model, best_t


def pred_threshold(p, high, t):
    z = np.argmax(p, axis=1)
    z[p[:, high] >= t] = high
    return z


# ---------------------------------------------------------------- PRINT THE SPEC ----------------------------------------------------------------
def show(obj, indent=0):
    pad = " " * indent
    if isinstance(obj, (ImbPipeline, Pipeline)):
        print(f"{pad}{type(obj).__name__}:")
        for n, s in obj.steps: print(f"{pad}  step '{n}':"); show(s, indent + 4)
    elif isinstance(obj, StackingClassifier):
        print(f"{pad}StackingClassifier(stack_method={obj.stack_method!r}, passthrough={obj.passthrough}, cv={obj.cv!r}, n_jobs={obj.n_jobs})")
        for n, e in obj.estimators: print(f"{pad}  base '{n}':"); show(e, indent + 4)
        print(f"{pad}  final_estimator:"); show(obj.final_estimator, indent + 4)
    elif isinstance(obj, str): print(f"{pad}{obj!r}  (no resampling)")
    else:
        keep = {k: v for k, v in obj.get_params(deep=False).items() if k in ("n_estimators","learning_rate","max_depth","num_leaves","class_weight","random_state","C","max_iter","cv","method","hidden_layer_sizes","early_stopping","k_neighbors","sampling_strategy","categorical_features","eval_metric","strategy","penalty","solver")}
        print(f"{pad}{type(obj).__name__}({keep})")
# class counts of the 80% training partitions (from the reported test sizes: ITIncident 4984 test, CICIDS2017 12000, GoogleCluster 12000, MultiCloud 200)
COUNTS = {"ITIncident": {0: 542, 1: 619, 2: 18773}, "CICIDS2017": {0: 6204, 1: 39896, 2: 1900}, "GoogleCluster": {0: 18480, 1: 20256, 2: 9264}, "MultiCloud": {0: 267, 1: 267, 2: 266}}
CATS = ["category_enc","location_enc","contact_type_enc","assignment_group_enc","cmdb_ci_enc","subcategory_enc","knowledge_enc","reopen_flag"]
FEATS_IT = ["reassignment_count","reopen_count","sys_mod_count"] + CATS
for ds, cnt in COUNTS.items():
    y = np.concatenate([np.full(n, k) for k, n in cnt.items()]); high = 0
    cw = weights(y, high)
    print("=" * 100); print(f"{ds}: IR = {max(cnt.values())/min(cnt.values()):.2f}  ->  weights(y, high, alpha=5.0) = { {k: round(v, 4) for k, v in cw.items()} }")
    print(f"   alpha applied to High: {max(cnt.values())/min(cnt.values()) > IR_THRESHOLD}   (IR_THRESHOLD = {IR_THRESHOLD})")
    feats = FEATS_IT if ds == "ITIncident" else [f"f{i}" for i in range(10)]
    for label, kw in [("T_Full", {}), ("T_NoSMOTE", dict(smote_on=False)), ("T_NoAsymLoss", dict(asym_on=False)), ("T_NoCalibNB", dict(cal_nb=False)), ("T_NoStacking", dict(stacking=False))]:
        m = kats(ds, feats, CATS if ds == "ITIncident" else [], cw, 42, **kw); print(f"--- {label} ---"); show(m, 2)
        if label == "T_NoAsymLoss":
            print("   >>> class_weight passed to B1 and to the meta-learner:", {k: 1.0 for k in cw}, " | B2 (RandomForest) keeps class_weight='balanced'")
print("=" * 100); print("BASELINES"); 
for n, mdl in baselines(cw, 42).items(): print(f"--- {n} ---"); show(mdl, 2)
print("=" * 100); print("THRESHOLD RULE (threshold_fit): 15% stratified validation split of the training partition; grid 0.15..0.85 step 0.05; precision floor = max(0.30, 1.5*validation prevalence);")
print("   score = (precision+recall)/2 among thresholds meeting the floor; if NO threshold meets the floor the threshold stays 0.5 (fallback).")
print("   prediction rule: argmax, then overridden to High whenever P(High) >= threshold.")
