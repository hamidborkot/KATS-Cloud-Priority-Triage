#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cicids_lodo_v2.py -- CICIDS2017 leave-one-capture-day-out (reviewer 3, comment 5)  [fixes "No objects to concatenate"]

WHY IT FAILED: the "cleaned and preprocessed" CSV used in the main study has no Timestamp/day column, and RAW_DIR pointed at a folder that holds no
per-day files, so nothing was loaded. This version
  1. auto-discovers per-day files under /kaggle/input (CSV or parquet; any file whose name contains monday..friday),
  2. prints exactly what it found, and stops with clear instructions if nothing is found,
  3. offers MODE=family (runs on the cleaned CSV you already have): leave-one-ATTACK-FAMILY-out. This is NOT day-grouped; report it as a
     supplementary grouped/zero-day-style check and say so.

MODE=day (default): Monday (benign only) always in training; Tuesday..Friday held out in turn.
   Per-day sources that work: Kaggle 'chethuhn/network-intrusion-dataset' (MachineLearningCVE CSVs) or the official MachineLearningCSV.zip;
   'dhoogla/cicids2017' parquet files also work (day is read from the file name, class from the file name if there is no Label column).
MODE=family: CLEANED_CSV=/kaggle/input/datasets/ericanacletoribeiro/cicids2017-cleaned-and-preprocessed/cicids2017_cleaned.csv
Env: RAW_DIR, MODE, FEATURES_CSV (repo: New Results/features_CICIDS2017.csv), TRAIN_CAP=30000, TEST_CAP=60000, QUICK=1, OUT_DIR
Outputs: appendix_cicids_lodo.csv (per seed), cicids_lodo_summary.csv, latex_cicids_lodo.tex, class_composition_by_group.csv, run_log.txt
"""
import os, sys, time, glob, warnings, datetime, re
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
from sklearn.metrics import roc_auc_score, average_precision_score, cohen_kappa_score, f1_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

QUICK = os.environ.get("QUICK", "0") == "1"; MODE = os.environ.get("MODE", "day")
OUT = os.environ.get("OUT_DIR", "/kaggle/working/cicids_lodo"); os.makedirs(OUT, exist_ok=True)
LOGF = open(os.path.join(OUT, "run_log.txt"), "a")
def log(m):
    s = f"[{datetime.datetime.now():%H:%M:%S}] {m}"; print(s, flush=True); LOGF.write(s + "\n"); LOGF.flush()
CLASSES = ["High", "Low", "Medium"]; HI = 0                 # LabelEncoder order used everywhere in the study
ALPHA = 5.0; N_LGB, N_RF = (30, 20) if QUICK else (300, 200)
TRAIN_CAP = int(os.environ.get("TRAIN_CAP", 30000)); TEST_CAP = int(os.environ.get("TEST_CAP", 60000)); SEEDS = [42, 7] if QUICK else [42, 7, 13]
DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday"]
norm = lambda c: re.sub(r"\s+", "_", str(c).strip().lower())

def make_cw(y):
    cl, ct = np.unique(y, return_counts=True); cw = {int(c): len(y) / (len(cl) * k) for c, k in zip(cl, ct)}
    if HI in cw: cw[HI] *= ALPHA
    return cw                                                # Eq.(2)
def kats(cw, seed, sampler=None, nj=1):
    def wrap(est): return ImbPipeline([("smote", sampler()), ("est", est)]) if sampler else est        # nested inside every stacking fold
    return StackingClassifier(
        estimators=[("lgb", wrap(lgb.LGBMClassifier(n_estimators=N_LGB, learning_rate=0.05, max_depth=6, num_leaves=31, class_weight=cw, random_state=seed, verbose=-1, n_jobs=nj))),
                    ("rf", wrap(RandomForestClassifier(n_estimators=N_RF, class_weight="balanced", random_state=seed, n_jobs=nj))),
                    ("nb", wrap(CalibratedClassifierCV(GaussianNB(), cv=3, method="isotonic")))],
        final_estimator=LogisticRegression(C=1.0, max_iter=2000, random_state=seed, class_weight=cw), stack_method="predict_proba", passthrough=True,
        cv=StratifiedKFold(3, shuffle=True, random_state=seed), n_jobs=nj)
def select_threshold(build, Xtr, ytr, seed):
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
def fit_predict(name, Xtr, ytr, Xte, seed, sampler=None):
    cw = make_cw(ytr)
    if name == "KATS":
        thr = select_threshold(lambda y: kats(make_cw(y), seed, sampler), Xtr, ytr, seed); m = kats(cw, seed, sampler).fit(Xtr, ytr)
        P = m.predict_proba(Xte); pred = P.argmax(1); pred[P[:, HI] >= thr] = HI; return pred, P, thr
    if name == "LightGBM": m = lgb.LGBMClassifier(n_estimators=N_LGB, learning_rate=0.05, max_depth=6, class_weight=cw, random_state=seed, verbose=-1, n_jobs=1)
    else: m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced", random_state=seed))
    m.fit(Xtr, ytr); P = m.predict_proba(Xte); return P.argmax(1), P, np.nan
def full_proba(P, classes_present):
    """models trained without a class return fewer columns: expand to the 3-column layout"""
    if P.shape[1] == 3: return P
    out = np.zeros((len(P), 3)); out[:, classes_present] = P; return out
def score(y, pred, P):
    present = sorted(set(y)); Y = np.stack([(y == k).astype(int) for k in range(3)], 1); hi_in = (y == HI).any()
    tp = int(((pred == HI) & (y == HI)).sum()); fp = int(((pred == HI) & (y != HI)).sum()); fn = int(((pred != HI) & (y == HI)).sum())
    try: auc = roc_auc_score(Y[:, present], P[:, present], average="macro") if len(present) > 1 else np.nan
    except Exception: auc = np.nan
    return dict(n_test=len(y), n_High_test=int((y == HI).sum()), n_Medium_test=int((y == 2).sum()), n_Low_test=int((y == 1).sum()),
                Recall_H=(tp / (tp + fn)) if hi_in else np.nan, Precision_H=(tp / (tp + fp)) if (tp + fp) else 0.0,
                MacroF1=f1_score(y, pred, labels=present, average="macro", zero_division=0), Kappa=cohen_kappa_score(y, pred),
                PRAUC_H=average_precision_score(Y[:, HI], P[:, HI]) if hi_in else np.nan, AUC_macro_present=auc, FP_H=fp, FN_H=fn, TP_H=tp,
                frac_pred_High=float((pred == HI).mean()), alert_rate_nonLow=float((pred != 1).mean()))
def tci(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]; return stats.t.ppf(.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan
def severity(l):
    k = re.sub(r"[^a-z]", "", str(l).lower())               # tolerant to '\ufffd', spaces, hyphens
    if "webattack" in k: return "High"                       # paper's rule (name contains 'brute force' but it is a web attack)
    if "benign" in k or "normal" in k: return "Low"
    if "scan" in k or "patator" in k or "brute" in k: return "Medium"
    return "High"

# ------------------------------------------------------------------ load
FEATS = None
fc = os.environ.get("FEATURES_CSV")
if fc and os.path.exists(fc): FEATS = [norm(l.split(",")[0]) for l in open(fc).read().split("\n")[1:] if l.strip()]
def read_any(f):
    return pd.read_parquet(f) if f.lower().endswith(".parquet") else pd.read_csv(f, low_memory=False, encoding="latin-1")

if MODE == "day":
    RAW = os.environ.get("RAW_DIR", "/kaggle/input"); cand = []
    for f in glob.glob(os.path.join(RAW, "**", "*"), recursive=True):
        b = os.path.basename(f).lower()
        if b.endswith((".csv", ".parquet")) and "cleaned" not in b and any(d in b for d in DAYS): cand.append(f)
    if not cand and RAW != "/kaggle/input":                  # RAW_DIR wrong -> search everything
        for f in glob.glob("/kaggle/input/**/*", recursive=True):
            b = os.path.basename(f).lower()
            if b.endswith((".csv", ".parquet")) and "cleaned" not in b and any(d in b for d in DAYS): cand.append(f)
    log(f"per-day candidate files found: {len(cand)}")
    for f in sorted(cand): log("   " + f)
    if not cand:
        log("NO per-day CICIDS2017 files found. Add one of these as a Kaggle input and re-run:")
        log("   kaggle.com/datasets/chethuhn/network-intrusion-dataset   (Monday-WorkingHours.pcap_ISCX.csv, ...)")
        log("   kaggle.com/datasets/dhoogla/cicids2017                   (parquet files named <Class>-<Day>-no-metadata.parquet)")
        log("   or the official MachineLearningCSV.zip from unb.ca/cic/datasets/ids-2017.html")
        log("If you cannot add raw data: run MODE=family, or use the limitation statement in the answer. Nothing was fabricated.")
        sys.exit(1)
    frames = []
    for f in sorted(cand):
        b = os.path.basename(f).lower(); day = next(d for d in DAYS if d in b)
        df = read_any(f); df.columns = [norm(c) for c in df.columns]; df = df.loc[:, ~df.columns.duplicated()]
        lab = next((c for c in df.columns if c in ("label", "attack_type", "attack")), None)
        raw_label = df[lab] if lab else pd.Series([re.sub(r"(?i)[-_ ]?(monday|tuesday|wednesday|thursday|friday).*", "", os.path.basename(f))] * len(df))
        df["priority_label"] = raw_label.apply(severity); df["group"] = day; frames.append(df); log(f"{os.path.basename(f)} -> {day}: {len(df):,} rows")
    data = pd.concat(frames, ignore_index=True, sort=False); HOLD = ["tuesday", "wednesday", "thursday", "friday"]
else:
    CL = os.environ.get("CLEANED_CSV", "/kaggle/input/datasets/ericanacletoribeiro/cicids2017-cleaned-and-preprocessed/cicids2017_cleaned.csv")
    data = pd.read_csv(CL, low_memory=False); data.columns = [norm(c) for c in data.columns]
    lab = next(c for c in data.columns if c in ("attack_type", "label")); data["priority_label"] = data[lab].apply(severity)
    data["group"] = data[lab].astype(str).str.strip()
    data = pd.concat([g.sample(frac=min(1.0, 0.05), random_state=42) for _, g in data.groupby("priority_label")], ignore_index=True)   # same 5% per-class sample as the main study
    HOLD = [g for g in data["group"].unique() if severity(g) != "Low" and (data["group"] == g).sum() >= 50]
    log(f"MODE=family (NOT day-grouped). families held out in turn: {HOLD}")
if FEATS is None: FEATS = [c for c in data.columns if c not in {"label", "attack_type", "priority_label", "group"} and pd.api.types.is_numeric_dtype(data[c])]
have = [c for c in FEATS if c in data.columns]; miss = [c for c in FEATS if c not in data.columns]
log(f"features used: {len(have)} / {len(FEATS)}; missing: {miss[:10]}")
if len(have) < 30: sys.exit("fewer than 30 of the study features found after name normalisation; check FEATURES_CSV / column names")
X_all = data[have].replace([np.inf, -np.inf], np.nan).fillna(0).astype(float).reset_index(drop=True)
y_all = LabelEncoder().fit(CLASSES).transform(data["priority_label"]); grp = data["group"].values
comp = pd.crosstab(data["group"], data["priority_label"]); comp.to_csv(f"{OUT}/class_composition_by_group.csv"); log("composition:\n" + comp.to_string())

def strat_sample(idx, cap, rng):
    if len(idx) <= cap: return idx
    parts = []
    for c in (0, 1, 2):
        ic = idx[y_all[idx] == c]
        if len(ic): parts.append(rng.choice(ic, max(1, int(round(len(ic) * cap / len(idx)))), replace=False))
    return np.concatenate(parts)

rows = []
for held in HOLD:
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        if MODE == "day": tr_pool, te_pool = np.where(grp != held)[0], np.where(grp == held)[0]
        else:
            fam = np.where(grp == held)[0]; ben = np.where(y_all == 1)[0]; rng.shuffle(ben); cut = int(0.8 * len(ben))
            tr_pool = np.concatenate([np.where((grp != held) & (y_all != 1))[0], ben[:cut]]); te_pool = np.concatenate([fam, ben[cut:]])
        tr_idx = strat_sample(tr_pool, TRAIN_CAP, rng); te_idx = strat_sample(te_pool, TEST_CAP, rng)
        Xtr, ytr, Xte, yte = X_all.iloc[tr_idx].reset_index(drop=True), y_all[tr_idx], X_all.iloc[te_idx].reset_index(drop=True), y_all[te_idx]
        cnt = np.bincount(ytr, minlength=3); present_tr = [k for k in range(3) if cnt[k] > 0]
        if len(present_tr) < 3 or cnt[cnt > 0].min() < 10: log(f"skip {held}: training set lacks a class {cnt}"); continue
        ir = cnt.max() / cnt.min()
        for mname in ["KATS", "LightGBM", "LogReg"]:
            sampler = (lambda s=seed: SMOTE(random_state=s, k_neighbors=5)) if (mname == "KATS" and ir > 3) else None
            t0 = time.time(); pred, P, thr = fit_predict(mname, Xtr, ytr, Xte, seed, sampler)
            sc = score(yte, pred, P)
            if MODE == "family": fam_mask = np.isin(te_idx, fam); sc["heldout_family_alert_rate"] = float((pred[fam_mask] != 1).mean()); sc["heldout_family_class_correct"] = float((pred[fam_mask] == yte[fam_mask]).mean())
            r = dict(held_out=held, model=mname, seed=seed, train_IR=ir, n_train=len(ytr), threshold=thr, fit_s=time.time() - t0, **sc); rows.append(r)
            log(f"held-out {held} seed={seed} {mname}: R_H={r['Recall_H']:.4f} P_H={r['Precision_H']:.4f} F1m={r['MacroF1']:.4f} kappa={r['Kappa']:.4f}")
            pd.DataFrame(rows).to_csv(f"{OUT}/appendix_cicids_lodo.csv", index=False)
if not rows: sys.exit("no result rows produced")
R = pd.DataFrame(rows); keys = ["Recall_H", "Precision_H", "MacroF1", "Kappa", "FP_H", "FN_H"] + (["heldout_family_alert_rate"] if MODE == "family" else [])
S = R.groupby(["held_out", "model"]).agg(**{f"{k}_{a}": (k, f) for k in keys for a, f in (("mean", "mean"), ("sd", "std"), ("ci95", tci))}).reset_index(); S.to_csv(f"{OUT}/cicids_lodo_summary.csv", index=False)
cap = ("CICIDS2017 leave-one-capture-day-out (train on the other four days; Monday is benign only and is always in training)." if MODE == "day" else
       "CICIDS2017 leave-one-attack-family-out (supplementary; NOT day-grouped: the cleaned file has no timestamp). Test = held-out family + 20\\% of benign held out from training.")
L = [r"\begin{table}[t]", r"\caption{" + cap + r" NaN: class absent. Mean $\pm$ SD over " + str(len(SEEDS)) + r" seeds.}", r"\label{tab:cicids-lodo}", r"\scriptsize",
     r"\begin{tabular}{@{}llrrrr@{}}", r"\toprule", r"Held-out & Model & Recall$_H$ & Prec.$_H$ & Macro-F1 & $\kappa$ \\ \midrule"]
for (d, m), g in R.groupby(["held_out", "model"], sort=False):
    f = lambda k: ("NaN" if g[k].isna().all() else f"${g[k].mean():.4f}\\pm{(g[k].std(ddof=1) if len(g) > 1 else 0):.4f}$")
    L.append(f"{str(d).replace('_', ' ').capitalize()} & {m} & {f('Recall_H')} & {f('Precision_H')} & {f('MacroF1')} & {f('Kappa')} \\\\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]; open(f"{OUT}/latex_cicids_lodo.tex", "w").write("\n".join(L)); log("done")
