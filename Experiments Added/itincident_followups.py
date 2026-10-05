#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
itincident_followups.py  -- two missing ITIncident analyses, built from the RAW incident_event_log.csv exactly as the main pipeline does
(events sorted by sys_mod_count, one row per incident; High = '1 - Critical' + '2 - High', Medium = '3 - Moderate', Low = '4 - Low').

PART A  availability-at-prediction-time sensitivity (reviewer comment 1)
   FINAL11      the paper's 11 final-state features (incl. lifecycle counters: reassignment_count, reopen_count, sys_mod_count, reopen_flag, knowledge_enc)
   FINAL6_CAT   final-state categoricals only (category, subcategory, assignment_group, location, contact_type, cmdb_ci)  -> isolates the counters
   CREATION6    the same six fields from the FIRST event of each incident (sys_mod_count minimum) -> what is known when the ticket is opened
   Models KATS (nested SMOTENC, threshold rule as in the main study), LightGBM (alpha-weighted), LogReg; seeds 42,7,13.  Label = FINAL priority in all three.
PART B  per-row SHAP fidelity (reviewer comment 15): B1 TreeSHAP vs full-stack KernelSHAP on 50 High + 50 non-High test rows (seed 42),
   per-row Spearman / top-1 / top-3 agreement, global rank rho with p-value, and a control (KernelSHAP on B1 vs TreeSHAP on B1) that detects ordering/scale bugs.
Run:  pip install shap ; IT_PATH=/kaggle/input/datasets/shamiulislamshifat/it-incident-log-dataset/incident_event_log.csv python itincident_followups.py
Env: QUICK=1 (smoke test), SKIP_SHAP=1, OUT_DIR
Outputs: it_sensitivity_per_seed.csv, it_sensitivity_summary.csv, latex_it_sensitivity.tex, it_shap_fidelity_rows.csv, it_shap_fidelity_summary.csv, run_log.txt
"""
import os, sys, time, re, warnings, datetime, json
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
from sklearn.metrics import average_precision_score, cohen_kappa_score, f1_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTENC

QUICK = os.environ.get("QUICK", "0") == "1"
OUT = os.environ.get("OUT_DIR", "/kaggle/working/it_followups"); os.makedirs(OUT, exist_ok=True)
IT_PATH = os.environ.get("IT_PATH", "/kaggle/input/datasets/shamiulislamshifat/it-incident-log-dataset/incident_event_log.csv")
LOGF = open(os.path.join(OUT, "run_log.txt"), "a")
def log(m):
    s = f"[{datetime.datetime.now():%H:%M:%S}] {m}"; print(s, flush=True); LOGF.write(s + "\n"); LOGF.flush()
SEEDS = [42, 7] if QUICK else [42, 7, 13]; ALPHA = 5.0; N_LGB, N_RF = (30, 20) if QUICK else (300, 200)
CLASSES = ["High", "Low", "Medium"]; HI = 0
norm = lambda c: re.sub(r"[^a-z0-9]", "_", str(c).strip().lower())

# ------------------------------------------------------------------ data (same construction as the main pipeline)
raw = pd.read_csv(IT_PATH, low_memory=False); raw.columns = [norm(c) for c in raw.columns]
need = ["number", "sys_mod_count", "priority", "reassignment_count", "reopen_count", "knowledge", "category", "subcategory", "assignment_group", "location", "contact_type", "cmdb_ci"]
miss = [c for c in need if c not in raw.columns]
if miss: sys.exit(f"missing columns {miss}; columns are {list(raw.columns)}")
raw["sys_mod_count"] = pd.to_numeric(raw["sys_mod_count"], errors="coerce")
srt = raw.sort_values(["number", "sys_mod_count"], kind="mergesort")
final = srt.groupby("number").last().reset_index()          # paper: final logged state per incident
first = srt.groupby("number").first().reset_index()         # creation-time snapshot (lowest sys_mod_count event)
prio_map = {"1 - Critical": "High", "2 - High": "High", "3 - Moderate": "Medium", "4 - Low": "Low"}
final["priority_label"] = final["priority"].map(prio_map); keep = final["priority_label"].notna(); final = final[keep].reset_index(drop=True)
first = first.set_index("number").loc[final["number"]].reset_index()
log(f"incidents={len(final)}  label counts={final['priority_label'].value_counts().to_dict()}  High share={(final['priority_label']=='High').mean():.4f}")
CAT6 = ["category", "subcategory", "assignment_group", "location", "contact_type", "cmdb_ci"]
def enc(df, cols): return pd.DataFrame({c: LabelEncoder().fit_transform(df[c].astype(str)) for c in cols})
Xf_cat = enc(final, CAT6); Xc_cat = enc(first, CAT6)
final11 = pd.concat([final[["reassignment_count", "reopen_count", "sys_mod_count"]].astype(float).reset_index(drop=True), Xf_cat.add_suffix("_enc"),
                     pd.DataFrame({"knowledge_enc": final["knowledge"].astype(int).values, "reopen_flag": (final["reopen_count"] > 0).astype(int).values})], axis=1)
SETS = {"FINAL11": (final11, [c for c in final11.columns if c.endswith("_enc") or c == "reopen_flag"]),      # 7 *_enc + reopen_flag = the 8 categoricals of the paper
        "FINAL6_CAT": (Xf_cat.add_suffix("_enc"), list(Xf_cat.add_suffix("_enc").columns)),
        "CREATION6": (Xc_cat.add_suffix("_enc"), list(Xc_cat.add_suffix("_enc").columns))}
y_all = LabelEncoder().fit(CLASSES).transform(final["priority_label"])
log("FINAL11 columns: " + ", ".join(final11.columns))

# ------------------------------------------------------------------ models
def make_cw(y):
    cl, ct = np.unique(y, return_counts=True); cw = {int(c): len(y) / (len(cl) * k) for c, k in zip(cl, ct)}; cw[HI] *= ALPHA; return cw
def kats(cw, seed, cat_idx, nj=1, smote=True):
    def wrap(est): return ImbPipeline([("smote", SMOTENC(categorical_features=cat_idx, random_state=seed, k_neighbors=5)), ("est", est)]) if smote else est
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
def fit_eval(model, X, y, Xte, yte, seed, cat_idx):
    cw = make_cw(y)
    if model == "KATS":
        thr = select_threshold(lambda yy: kats(make_cw(yy), seed, cat_idx), X, y, seed); m = kats(cw, seed, cat_idx).fit(X, y)
        P = m.predict_proba(Xte); pred = P.argmax(1); pred[P[:, HI] >= thr] = HI
    else:
        thr = np.nan
        m = (lgb.LGBMClassifier(n_estimators=N_LGB, learning_rate=0.05, max_depth=6, class_weight=cw, random_state=seed, verbose=-1, n_jobs=1) if model == "LightGBM"
             else make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced", random_state=seed)))
        m.fit(X, y); P = m.predict_proba(Xte); pred = P.argmax(1)
    tp = int(((pred == HI) & (yte == HI)).sum()); fp = int(((pred == HI) & (yte != HI)).sum()); fn = int(((pred != HI) & (yte == HI)).sum())
    return dict(threshold=thr, Recall_H=tp / max(1, tp + fn), Precision_H=tp / max(1, tp + fp), MacroF1=f1_score(yte, pred, average="macro"), Kappa=cohen_kappa_score(yte, pred),
                PRAUC_H=average_precision_score((yte == HI).astype(int), P[:, HI]), TP_H=tp, FP_H=fp, FN_H=fn, n_test=len(yte), n_High_test=int((yte == HI).sum()))

# ------------------------------------------------------------------ PART A
rows = []; ck = os.path.join(OUT, "it_sensitivity_per_seed.csv")
for sname, (X, cats) in SETS.items():
    cat_idx = [list(X.columns).index(c) for c in cats]
    for seed in SEEDS:
        Xtr, Xte, ytr, yte = train_test_split(X, y_all, test_size=0.20, random_state=seed, stratify=y_all)
        Xtr, Xte = Xtr.reset_index(drop=True), Xte.reset_index(drop=True)
        for mname in ["KATS", "LightGBM", "LogReg"]:
            t0 = time.time(); r = fit_eval(mname, Xtr, ytr, Xte, yte, seed, cat_idx); rows.append(dict(feature_set=sname, n_features=X.shape[1], model=mname, seed=seed, fit_s=time.time() - t0, **r))
            log(f"{sname:10s} seed={seed} {mname:8s} R_H={r['Recall_H']:.4f} P_H={r['Precision_H']:.4f} F1m={r['MacroF1']:.4f} kappa={r['Kappa']:.4f} PR-AUC={r['PRAUC_H']:.4f}")
            pd.DataFrame(rows).to_csv(ck, index=False)
R = pd.DataFrame(rows)
def tci(x): x = np.asarray(x, float); return stats.t.ppf(.975, len(x) - 1) * x.std(ddof=1) / len(x) ** .5 if len(x) > 1 else np.nan
keys = ["Recall_H", "Precision_H", "MacroF1", "Kappa", "PRAUC_H", "FP_H", "FN_H"]
S = R.groupby(["feature_set", "model"]).agg(**{f"{k}_{a}": (k, f) for k in keys for a, f in (("mean", "mean"), ("sd", "std"), ("ci95", tci))}).reset_index(); S.to_csv(os.path.join(OUT, "it_sensitivity_summary.csv"), index=False)
L = [r"\begin{table}[t]", r"\caption{ITIncident feature-availability sensitivity (mean $\pm$ SD over " + str(len(SEEDS)) + r" seeds; label = final priority in all rows). FINAL11 = paper feature set; FINAL6\_CAT removes the lifecycle counters; CREATION6 uses the same six categorical fields from each incident's first logged event.}",
     r"\label{tab:it-creation-time}", r"\scriptsize", r"\begin{tabular}{@{}llrrrr@{}}", r"\toprule", r"Feature set & Model & Recall$_H$ & Prec.$_H$ & Macro-F1 & $\kappa$ \\ \midrule"]
for (fs, m), g in R.groupby(["feature_set", "model"], sort=False):
    f = lambda k: f"${g[k].mean():.4f}\\pm{g[k].std(ddof=1) if len(g) > 1 else 0:.4f}$"
    L.append(f"{fs.replace('_', chr(92) + '_')} & {m} & {f('Recall_H')} & {f('Precision_H')} & {f('MacroF1')} & {f('Kappa')} \\\\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]; open(os.path.join(OUT, "latex_it_sensitivity.tex"), "w").write("\n".join(L)); log("PART A done")

# ------------------------------------------------------------------ PART B
def shap_part():
    if os.environ.get("SKIP_SHAP") == "1": return
    try: import shap
    except Exception: log("shap missing -> pip install shap; PART B skipped"); return
    X, cats = SETS["FINAL11"]; cat_idx = [list(X.columns).index(c) for c in cats]; seed = 42; feats = list(X.columns)
    Xtr, Xte, ytr, yte = train_test_split(X, y_all, test_size=0.20, random_state=seed, stratify=y_all); Xtr, Xte = Xtr.reset_index(drop=True), Xte.reset_index(drop=True)
    cw = make_cw(ytr); b1 = lgb.LGBMClassifier(n_estimators=N_LGB, learning_rate=0.05, max_depth=6, class_weight=cw, random_state=seed, verbose=-1, n_jobs=1).fit(Xtr, ytr)
    stack = kats(cw, seed, cat_idx).fit(Xtr, ytr)
    rng = np.random.default_rng(seed); hi_i = np.where(yte == HI)[0]; lo_i = np.where(yte != HI)[0]; nh = min(50 if not QUICK else 5, len(hi_i)); nl = (100 if not QUICK else 10) - nh
    ix = np.concatenate([rng.choice(hi_i, nh, replace=False), rng.choice(lo_i, nl, replace=False)]); Xs = Xte.iloc[ix]; is_high = np.r_[np.ones(nh, bool), np.zeros(nl, bool)]
    tv = shap.TreeExplainer(b1).shap_values(Xs)
    if isinstance(tv, list): tv = tv[HI]
    elif np.ndim(tv) == 3: tv = tv[:, :, HI]                     # (rows, features, classes) in recent shap versions
    assert tv.shape == (len(Xs), len(feats)), f"TreeSHAP shape {tv.shape} != {(len(Xs), len(feats))} -> class/feature axes mixed up"
    bg = shap.kmeans(Xtr, 10); ns = 100 if QUICK else 300
    fS = lambda x: stack.predict_proba(pd.DataFrame(x, columns=feats))[:, HI]; fB = lambda x: b1.predict_proba(pd.DataFrame(x, columns=feats))[:, HI]
    ks = np.asarray(shap.KernelExplainer(fS, bg).shap_values(Xs, nsamples=ns, silent=True)); kb = np.asarray(shap.KernelExplainer(fB, bg).shap_values(Xs, nsamples=ns, silent=True))
    if ks.ndim == 3: ks = ks[:, :, 0]
    if kb.ndim == 3: kb = kb[:, :, 0]
    assert ks.shape == tv.shape == kb.shape, (ks.shape, tv.shape, kb.shape)
    def compare(A, B, name):
        rho = np.array([stats.spearmanr(np.abs(A[i]), np.abs(B[i]))[0] for i in range(len(A))]); t1 = np.array([np.argmax(np.abs(A[i])) == np.argmax(np.abs(B[i])) for i in range(len(A))])
        t3 = np.array([len(set(np.argsort(-np.abs(A[i]))[:3]) & set(np.argsort(-np.abs(B[i]))[:3])) / 3 for i in range(len(A))])
        ga, gb = np.abs(A).mean(0), np.abs(B).mean(0); grho, gp = stats.spearmanr(ga, gb)
        return pd.DataFrame(dict(comparison=name, row=np.arange(len(A)), is_High=is_high, rho=rho, top1_match=t1, top3_overlap=t3)), dict(
            comparison=name, n_rows=len(A), n_features=A.shape[1], global_rho=grho, global_p=gp, rho_row_median=np.nanmedian(rho), rho_row_q25=np.nanpercentile(rho, 25), rho_row_q75=np.nanpercentile(rho, 75),
            rho_row_median_High=np.nanmedian(rho[is_high]), rho_row_median_nonHigh=np.nanmedian(rho[~is_high]), top1_match_rate=t1.mean(), top3_overlap_mean=t3.mean(), frac_rows_rho_lt_0=float(np.nanmean(rho < 0)))
    parts, summ = [], []
    for nm, (A, B) in {"B1_TreeSHAP_vs_FullStack_KernelSHAP": (tv, ks), "B1_TreeSHAP_vs_B1_KernelSHAP_(control)": (tv, kb), "B1_KernelSHAP_vs_FullStack_KernelSHAP": (kb, ks)}.items():
        d, s = compare(A, B, nm); parts.append(d); summ.append(s)
    pd.concat(parts).to_csv(os.path.join(OUT, "it_shap_fidelity_rows.csv"), index=False); SS = pd.DataFrame(summ); SS.to_csv(os.path.join(OUT, "it_shap_fidelity_summary.csv"), index=False)
    gtab = pd.DataFrame({"feature": feats, "meanabs_B1_Tree": np.abs(tv).mean(0), "meanabs_B1_Kernel": np.abs(kb).mean(0), "meanabs_FullStack_Kernel": np.abs(ks).mean(0)})
    for c in ["meanabs_B1_Tree", "meanabs_B1_Kernel", "meanabs_FullStack_Kernel"]: gtab["rank_" + c[8:]] = gtab[c].rank(ascending=False)
    gtab.sort_values("rank_FullStack_Kernel").to_csv(os.path.join(OUT, "it_shap_global_rank_table.csv"), index=False); log("PART B done\n" + SS.round(4).to_string())
try: shap_part()
except Exception as e: log(f"!! PART B failed: {e!r}")
log("ALL DONE -> " + OUT)
