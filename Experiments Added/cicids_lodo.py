#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cicids_lodo.py -- CICIDS2017 leave-one-capture-day-out (reviewer 3, comment 5).
IMPORTANT: the Kaggle "cleaned and preprocessed" CICIDS2017 file used in the main experiments has NO timestamp/day column, so a day-grouped split is
impossible on it. This script needs the ORIGINAL per-day files (CIC-IDS-2017 'MachineLearningCVE' or 'TrafficLabelling' CSVs) in RAW_DIR, e.g.
  Monday-WorkingHours.pcap_ISCX.csv, Tuesday-WorkingHours.pcap_ISCX.csv, Wednesday-workingHours.pcap_ISCX.csv,
  Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv, Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv,
  Friday-WorkingHours-Morning.pcap_ISCX.csv, Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv, Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
Design: Monday (benign only) is always in training; each of Tuesday..Friday is held out in turn; the model never sees that day. Severity mapping is the paper's
(benign->Low; scan/patator/brute-force -> Medium; web attacks and everything else -> High). Metrics are undefined for a class absent from the held-out day and are
reported as NaN (Recall_H is NaN on Tuesday because Tuesday has no High traffic; Macro-F1 is computed over the classes present).
The same 52 features as the main study (features_CICIDS2017.csv) are used. Models: KATS (nested SMOTE, IR>3), LightGBM, LogReg; seeds 42,7,13.
Run: RAW_DIR=/kaggle/input/.../MachineLearningCVE python cicids_lodo.py     (TRAIN_CAP=30000 keeps it to roughly 45 minutes on a Kaggle CPU; QUICK=1 smoke test)
Outputs: appendix_cicids_lodo.csv (per seed), cicids_lodo_summary.csv, latex_cicids_lodo.tex, class_composition_by_day.csv
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

from imblearn.over_sampling import SMOTE
QUICK = os.environ.get("QUICK", "0") == "1"
RAW_DIR = os.environ.get("RAW_DIR", "/kaggle/input/cicids2017/MachineLearningCVE"); OUT = os.environ.get("OUT_DIR", "/kaggle/working/cicids_lodo"); os.makedirs(OUT, exist_ok=True)
TRAIN_CAP = int(os.environ.get("TRAIN_CAP", 30000)); TEST_CAP = int(os.environ.get("TEST_CAP", 60000)); SEEDS = [42, 7] if QUICK else [42, 7, 13]
if QUICK: N_LGB, N_RF = 30, 20
FEATS52 = [l.split(",")[0].strip() for l in open(os.environ["FEATURES_CSV"]).read().split("\n")[1:] if l.strip()] if os.environ.get("FEATURES_CSV") else None   # New Results/features_CICIDS2017.csv
norm = lambda c: c.strip().lower().replace(" ", "_")
def severity(l):
    k = str(l).strip().lower()
    if "web attack" in k: return "High"                      # keep the paper's rule: web attacks -> High even though 'brute force' appears in the name
    if "benign" in k or "normal" in k: return "Low"
    if "scan" in k or "patator" in k or "brute" in k: return "Medium"
    return "High"
def day_of(f):
    b = os.path.basename(f).lower()
    for d in ["monday", "tuesday", "wednesday", "thursday", "friday"]:
        if b.startswith(d): return d
frames = []
for f in sorted(glob.glob(os.path.join(RAW_DIR, "**", "*.csv"), recursive=True)):
    d = day_of(f)
    if d is None: continue
    df = pd.read_csv(f, low_memory=False, encoding="latin-1"); df.columns = [norm(c) for c in df.columns]; df = df.loc[:, ~df.columns.duplicated()]
    lab = next(c for c in df.columns if c == "label"); df["priority_label"] = df[lab].apply(severity); df["day"] = d; frames.append(df); log(f"{os.path.basename(f)} -> {d}: {len(df):,} rows")
data = pd.concat(frames, ignore_index=True)
if FEATS52 is None: FEATS52 = [c for c in data.columns if c not in {"label", "priority_label", "day"} and pd.api.types.is_numeric_dtype(data[c])]
miss = [c for c in FEATS52 if c not in data.columns]
if miss: sys.exit(f"features missing from raw files after name normalisation: {miss}")
X_all = data[FEATS52].replace([np.inf, -np.inf], np.nan).fillna(0).astype(float); y_all = LabelEncoder().fit(CLASSES).transform(data["priority_label"]); days = data["day"].values
comp = pd.crosstab(data["day"], data["priority_label"]); comp.to_csv(f"{OUT}/class_composition_by_day.csv"); log("class composition by day:\n" + comp.to_string())
def strat_sample(idx, cap, rng):
    if len(idx) <= cap: return idx
    parts = []
    for c in (0, 1, 2):
        ic = idx[y_all[idx] == c]; parts.append(rng.choice(ic, max(1, int(round(len(ic) * cap / len(idx)))) if len(ic) else 0, replace=False) if len(ic) else ic)
    return np.concatenate(parts)
rows = []
for held in ["tuesday", "wednesday", "thursday", "friday"]:
    for seed in SEEDS:
        rng = np.random.default_rng(seed); tr_idx = strat_sample(np.where(days != held)[0], TRAIN_CAP, rng); te_idx = strat_sample(np.where(days == held)[0], TEST_CAP, rng)
        Xtr, ytr, Xte, yte = X_all.iloc[tr_idx].reset_index(drop=True), y_all[tr_idx], X_all.iloc[te_idx].reset_index(drop=True), y_all[te_idx]
        ir = np.bincount(ytr, minlength=3); ir = ir.max() / max(1, ir[ir > 0].min())
        for mname in ["KATS", "LightGBM", "LogReg"]:
            sampler = (lambda: SMOTE(random_state=seed, k_neighbors=5)) if (mname == "KATS" and ir > 3) else None
            t0 = time.time(); pred, P, thr = fit_predict(mname, Xtr, ytr, Xte, seed, sampler)
            r = dict(held_out_day=held, model=mname, seed=seed, train_IR=ir, n_train=len(ytr), threshold=thr, fit_s=time.time() - t0, **score(yte, pred, P)); rows.append(r)
            log(f"held-out {held} seed={seed} {mname}: R_H={r['Recall_H']:.4f} P_H={r['Precision_H']:.4f} F1m={r['MacroF1']:.4f} kappa={r['Kappa']:.4f}")
            pd.DataFrame(rows).to_csv(f"{OUT}/appendix_cicids_lodo.csv", index=False)
R = pd.DataFrame(rows); keys = ["Recall_H", "Precision_H", "MacroF1", "Kappa", "FP_H", "FN_H"]
S = R.groupby(["held_out_day", "model"]).agg(**{f"{k}_{a}": (k, f) for k in keys for a, f in (("mean", "mean"), ("sd", "std"), ("ci95", tci))}, n_High_test=("n_High_test", "first"), n_Medium_test=("n_Medium_test", "first")).reset_index()
S.to_csv(f"{OUT}/cicids_lodo_summary.csv", index=False)
L = [r"\begin{table}[t]", r"\caption{CICIDS2017 leave-one-capture-day-out (train on the other four days; Monday is benign only and is always in training). NaN: class absent on the held-out day. Mean $\pm$ SD over three seeds.}",
     r"\label{tab:cicids-lodo}", r"\scriptsize", r"\begin{tabular}{@{}llrrrr@{}}", r"\toprule", r"Held-out day & Model & Recall$_H$ & Prec.$_H$ & Macro-F1 & $\kappa$ \\ \midrule"]
for (d, m), g in R.groupby(["held_out_day", "model"], sort=False):
    f = lambda k: ("NaN" if g[k].isna().all() else f"${g[k].mean():.4f}\\pm{g[k].std(ddof=1) if len(g) > 1 else 0:.4f}$")
    L.append(f"{d.capitalize()} & {m} & {f('Recall_H')} & {f('Precision_H')} & {f('MacroF1')} & {f('Kappa')} \\\\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]; open(f"{OUT}/latex_cicids_lodo.tex", "w").write("\n".join(L)); log("done")
