#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
it_creation_time.py -- ITIncident creation-time sensitivity (reviewer 3, comment 1).
Question answered: does KATS still work when ONLY information available when the incident is OPENED is used?
 * one row per incident; label = FINAL priority (same rule as the paper: '1 - Critical' and '2 - High' -> High, '3 - Moderate' -> Medium, '4 - Low' -> Low)
 * feature set A "creation_time": category, subcategory, assignment_group, location, contact_type, cmdb_ci taken from the FIRST logged event
   (lowest sys_mod_count). impact, urgency, made_sla, priority, knowledge, reopen_*, reassignment_count, sys_mod_count are NOT used.
 * feature set B "final_state_11": the paper's 11 features from the LAST event (reproduces the main-table setting for a paired comparison)
 * splits: random stratified 80/20 (seeds 42,7,13) AND chronological 80/20 by incident opening time
 * KATS resampling: all creation-time features are categorical -> SMOTEN (SMOTENC cannot run without a continuous feature); final_state_11 -> SMOTENC as in the paper
Run (Kaggle):  python it_creation_time.py         (QUICK=1: smoke test)   Env: IT_PATH, OUT_DIR, SEEDS
Outputs: appendix_it_creation_time.csv (per seed), it_creation_time_summary.csv, confusion_matrices_it_creation_time.csv, latex_it_creation_time.tex
"""

import os, sys, time, json, glob, warnings, datetime, re
import numpy as np, pandas as pd
from scipy import stats
warnings.filterwarnings("ignore")
import lightgbm as lgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (roc_auc_score, average_precision_score, cohen_kappa_score, f1_score, confusion_matrix)
from imblearn.pipeline import Pipeline as ImbPipeline
def log(m): print(f"[{datetime.datetime.now():%H:%M:%S}] {m}", flush=True)
CLASSES = ["High", "Low", "Medium"]; HI = 0          # LabelEncoder order used everywhere in the study
ALPHA = 5.0; N_LGB, N_RF = int(os.environ.get("N_LGB", 300)), int(os.environ.get("N_RF", 200))
def make_cw(y):
    cl, ct = np.unique(y, return_counts=True); cw = {int(c): len(y) / (len(cl) * k) for c, k in zip(cl, ct)}; cw[HI] *= ALPHA; return cw   # Eq.(2)
def kats(cw, seed, sampler=None, nj=1):
    def wrap(est): return ImbPipeline([("smote", sampler()), ("est", est)]) if sampler else est      # nested: refit inside every stacking fold
    return StackingClassifier(
        estimators=[("lgb", wrap(lgb.LGBMClassifier(n_estimators=N_LGB, learning_rate=0.05, max_depth=6, num_leaves=31, class_weight=cw, random_state=seed, verbose=-1, n_jobs=nj))),
                    ("rf", wrap(RandomForestClassifier(n_estimators=N_RF, class_weight="balanced", random_state=seed, n_jobs=nj))),
                    ("nb", wrap(CalibratedClassifierCV(GaussianNB(), cv=3, method="isotonic")))],
        final_estimator=LogisticRegression(C=1.0, max_iter=2000, random_state=seed, class_weight=cw), stack_method="predict_proba", passthrough=True,
        cv=StratifiedKFold(3, shuffle=True, random_state=seed), n_jobs=nj)
def select_threshold(build, Xtr, ytr, seed):
    """15% stratified validation hold-out INSIDE the training partition; grid 0.05..0.95; precision floor dropped if unreachable."""
    Xf, Xv, yf, yv = train_test_split(Xtr, ytr, test_size=0.15, random_state=seed, stratify=ytr)
    m = build(yf); m.fit(Xf, yf); pv = m.predict_proba(Xv)[:, HI]; th = yv == HI; floor = max(0.30, 1.5 * th.mean())
    def scan(use_floor):
        bt, bs = None, -1
        for t in np.round(np.arange(0.05, 0.951, 0.05), 2):
            pr = pv >= t; tp = (pr & th).sum(); fp = (pr & ~th).sum(); fn = (~pr & th).sum(); rec = tp / max(1, tp + fn); pre = tp / max(1, tp + fp)
            if use_floor and pre < floor: continue
            if 0.5 * rec + 0.5 * pre > bs: bs, bt = 0.5 * rec + 0.5 * pre, float(t)
        return bt
    t = scan(True); return t if t is not None else scan(False)
def fit_predict(model_name, Xtr, ytr, Xte, seed, sampler=None):
    cw = make_cw(ytr)
    if model_name == "KATS":
        thr = select_threshold(lambda y: kats(make_cw(y), seed, sampler), Xtr, ytr, seed); m = kats(cw, seed, sampler).fit(Xtr, ytr)
        P = m.predict_proba(Xte); pred = P.argmax(1); pred[P[:, HI] >= thr] = HI; return pred, P, thr
    if model_name == "LightGBM": m = lgb.LGBMClassifier(n_estimators=N_LGB, learning_rate=0.05, max_depth=6, class_weight=cw, random_state=seed, verbose=-1, n_jobs=1)
    elif model_name == "LogReg": m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced", random_state=seed))
    m.fit(Xtr, ytr); P = m.predict_proba(Xte); return P.argmax(1), P, np.nan
def score(y, pred, P):
    present = sorted(set(y)); Y = np.stack([(y == k).astype(int) for k in range(3)], 1); hi_in = (y == HI).any()
    tp = int(((pred == HI) & (y == HI)).sum()); fp = int(((pred == HI) & (y != HI)).sum()); fn = int(((pred != HI) & (y == HI)).sum())
    try: auc = roc_auc_score(Y[:, present], P[:, present], average="macro") if len(present) > 1 else np.nan
    except Exception: auc = np.nan
    return dict(n_test=len(y), n_High_test=int((y == HI).sum()), n_Medium_test=int((y == 2).sum()), n_Low_test=int((y == 1).sum()),
                Recall_H=(tp / (tp + fn)) if hi_in else np.nan, Precision_H=(tp / (tp + fp)) if (tp + fp) else 0.0,
                MacroF1=f1_score(y, pred, labels=present, average="macro", zero_division=0), Kappa=cohen_kappa_score(y, pred),
                PRAUC_H=average_precision_score(Y[:, HI], P[:, HI]) if hi_in else np.nan, AUC_macro_ovr_present_classes=auc, FP_H=fp, FN_H=fn, TP_H=tp,
                frac_pred_High=float((pred == HI).mean()))
def tci(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]; return stats.t.ppf(.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan

from imblearn.over_sampling import SMOTEN, SMOTENC
QUICK = os.environ.get("QUICK", "0") == "1"
IT_PATH = os.environ.get("IT_PATH", "/kaggle/input/datasets/shamiulislamshifat/it-incident-log-dataset/incident_event_log.csv")
OUT = os.environ.get("OUT_DIR", "/kaggle/working/it_creation_time"); os.makedirs(OUT, exist_ok=True)
SEEDS = [int(s) for s in os.environ.get("SEEDS", "42,7").split(",")] if QUICK else [int(s) for s in os.environ.get("SEEDS", "42,7,13").split(",")]
if QUICK: N_LGB, N_RF = 30, 20
raw = pd.read_csv(IT_PATH, low_memory=False); raw.columns = [c.strip().lower() for c in raw.columns]
col = lambda n: next(c for c in raw.columns if c == n or c.replace("_", "") == n.replace("_", ""))
num, smc, opened = col("number"), col("sys_mod_count"), col("opened_at")
raw[smc] = pd.to_numeric(raw[smc], errors="coerce")
first = raw.sort_values(smc).groupby(num).first(); last = raw.sort_values(smc).groupby(num).last()
lab = last[col("priority")].map({"1 - Critical": "High", "2 - High": "High", "3 - Moderate": "Medium", "4 - Low": "Low"})
keep = lab.notna(); first, last, lab = first[keep], last[keep], lab[keep]
le = LabelEncoder().fit(CLASSES); y = le.transform(lab.values); log(f"incidents={len(y)} class counts={dict(zip(le.classes_, np.bincount(y)))}")
CAT = ["category", "subcategory", "assignment_group", "location", "contact_type", "cmdb_ci"]
XA = pd.DataFrame({c: LabelEncoder().fit_transform(first[col(c)].astype(str)) for c in CAT}, index=first.index)                 # creation-time snapshot
cat_final = {c: LabelEncoder().fit_transform(last[col(c)].astype(str)) for c in CAT}
XB = pd.DataFrame({"reassignment_count": pd.to_numeric(last[col("reassignment_count")]).values, "reopen_count": pd.to_numeric(last[col("reopen_count")]).values,
                   "sys_mod_count": last[smc].values, **{c + "_enc": v for c, v in cat_final.items()},
                   "knowledge_enc": last[col("knowledge")].astype(int).values, "reopen_flag": (pd.to_numeric(last[col("reopen_count")]) > 0).astype(int).values}, index=last.index)
t_open = pd.to_datetime(first[opened], errors="coerce", dayfirst=True); order = np.argsort(t_open.fillna(t_open.min()).values)
sets = {"creation_time": (XA, lambda: SMOTEN(random_state=0, k_neighbors=5)),
        "final_state_11": (XB, lambda: SMOTENC(categorical_features=[XB.columns.get_loc(c) for c in XB.columns if c.endswith("_enc") or c == "reopen_flag"], random_state=0, k_neighbors=5))}
rows, cms = [], []
for split in ["random", "chronological"]:
    for seed in SEEDS:
        if split == "random": tr, te = train_test_split(np.arange(len(y)), test_size=0.2, random_state=seed, stratify=y)
        else:
            cut = int(0.8 * len(y)); tr, te = order[:cut], order[cut:]
            if seed != SEEDS[0]: pass                                    # chronological split is deterministic; seeds vary only model randomness
        for fs, (X, sampler) in sets.items():
            Xtr, Xte, ytr, yte = X.iloc[tr].reset_index(drop=True), X.iloc[te].reset_index(drop=True), y[tr], y[te]
            for mname in ["KATS", "LightGBM", "LogReg"]:
                t0 = time.time(); pred, P, thr = fit_predict(mname, Xtr, ytr, Xte, seed, sampler if mname == "KATS" else None)
                r = dict(split=split, feature_set=fs, model=mname, seed=seed, threshold=thr, fit_s=time.time() - t0, **score(yte, pred, P)); rows.append(r)
                cm = confusion_matrix(yte, pred, labels=[0, 1, 2])
                for i, tc in enumerate(CLASSES):
                    for j, pc in enumerate(CLASSES): cms.append(dict(split=split, feature_set=fs, model=mname, seed=seed, true_class=tc, predicted_class=pc, count=int(cm[i, j])))
                log(f"{split} seed={seed} {fs} {mname}: R_H={r['Recall_H']:.4f} P_H={r['Precision_H']:.4f} F1m={r['MacroF1']:.4f} kappa={r['Kappa']:.4f}")
                pd.DataFrame(rows).to_csv(f"{OUT}/appendix_it_creation_time.csv", index=False)
R = pd.DataFrame(rows); pd.DataFrame(cms).to_csv(f"{OUT}/confusion_matrices_it_creation_time.csv", index=False)
keys = ["Recall_H", "Precision_H", "MacroF1", "Kappa", "PRAUC_H", "FP_H", "FN_H", "frac_pred_High"]
S = R.groupby(["split", "feature_set", "model"]).agg(**{f"{k}_{a}": (k, f) for k in keys for a, f in (("mean", "mean"), ("sd", "std"), ("ci95", tci))}).reset_index(); S.to_csv(f"{OUT}/it_creation_time_summary.csv", index=False)
L = [r"\begin{table}[t]", r"\caption{ITIncident creation-time sensitivity. \emph{creation\_time}: six categorical fields from the first logged event only; \emph{final\_state\_11}: the main-table features. Mean $\pm$ SD over seeds; chronological rows use the first 80\% of incidents by opening time for training.}",
     r"\label{tab:it-creation-time}", r"\scriptsize", r"\begin{tabular}{@{}lllrrrr@{}}", r"\toprule", r"Split & Features & Model & Recall$_H$ & Prec.$_H$ & Macro-F1 & $\kappa$ \\ \midrule"]
for (sp, fs, m), g in R.groupby(["split", "feature_set", "model"], sort=False):
    f = lambda k: f"${g[k].mean():.4f}\\pm{g[k].std(ddof=1) if len(g) > 1 else 0:.4f}$"
    L.append(f"{sp} & {fs.replace('_', chr(92) + '_')} & {m} & {f('Recall_H')} & {f('Precision_H')} & {f('MacroF1')} & {f('Kappa')} \\\\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]; open(f"{OUT}/latex_it_creation_time.tex", "w").write("\n".join(L)); log("done")
