#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
meta_coefficients.py -- recompute Table 10 (meta-learner coefficients) on the FINAL pipeline:
   3-fold stacking (cv=3), passthrough=True, nested resampling inside each fold when IR>3, alpha-weighted multinomial LogisticRegression meta-learner, five seeds.
The notebook cell that produced the old numbers used MultiCloud only, seed 42, SMOTE applied BEFORE stacking, and its saved output is empty, so 3.42/1.85/0.73 cannot be
traced to the final pipeline.  Coefficient layout: final_estimator_.coef_ has shape (3, 9 + p): columns 0-2 = B1 (High,Low,Medium probabilities), 3-5 = B2, 6-8 = B3, 9.. = raw features.
Input: model-matrix CSVs (feature columns + 'priority_label'), e.g. dumped from the notebook:   MATRICES="ITIncident=/kaggle/working/mm_ITIncident.csv,CICIDS2017=..."
Env: SEEDS="42,7,13,99,2026", CAT_ITIncident="col1,col2,..." (categorical columns -> SMOTENC), QUICK=1, OUT_DIR
Outputs: meta_coef_per_seed.csv (every coefficient), meta_coef_summary.csv (Table 10 layout, mean +/- SD over seeds), meta_coef_signed_own_class.csv, latex_table10.tex
"""
import os, sys, warnings, re
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, SMOTENC
QUICK = os.environ.get("QUICK", "0") == "1"; OUT = os.environ.get("OUT_DIR", "/kaggle/working/meta_coef"); os.makedirs(OUT, exist_ok=True)
SEEDS = [int(s) for s in os.environ.get("SEEDS", "42,7" if QUICK else "42,7,13,99,2026").split(",")]
N_LGB, N_RF = (30, 20) if QUICK else (300, 200); ALPHA = 5.0; CLASSES = ["High", "Low", "Medium"]; HI = 0
def make_cw(y, gated):
    cl, ct = np.unique(y, return_counts=True); cw = {int(c): len(y) / (len(cl) * k) for c, k in zip(cl, ct)}
    if (not gated) or ct.max() / ct.min() > 3.0: cw[HI] *= ALPHA
    return cw                                   # gated=True reproduces the v14/v15 notebook (alpha only when IR>3); False = Eq.(2) literally
def kats(cw, seed, sampler):
    wrap = (lambda est: ImbPipeline([("smote", sampler()), ("est", est)])) if sampler else (lambda est: est)
    return StackingClassifier(
        estimators=[("lgb", wrap(lgb.LGBMClassifier(n_estimators=N_LGB, learning_rate=0.05, max_depth=6, num_leaves=31, class_weight=cw, random_state=seed, verbose=-1, n_jobs=1))),
                    ("rf", wrap(RandomForestClassifier(n_estimators=N_RF, class_weight="balanced", random_state=seed, n_jobs=1))),
                    ("nb", wrap(CalibratedClassifierCV(GaussianNB(), cv=3, method="isotonic")))],
        final_estimator=LogisticRegression(C=1.0, max_iter=2000, random_state=seed, class_weight=cw), stack_method="predict_proba", passthrough=True, cv=3, n_jobs=1)
rows = []
for item in os.environ["MATRICES"].split(","):
    name, path = item.split("=", 1); df = pd.read_csv(path); cats = [c for c in os.environ.get("CAT_" + name, "").split(",") if c]
    y = LabelEncoder().fit(CLASSES).transform(df["priority_label"].astype(str)); X = df.drop(columns=["priority_label"]).astype(float).fillna(0)
    cat_idx = [list(X.columns).index(c) for c in cats]
    for seed in SEEDS:
        Xtr, _, ytr, _ = train_test_split(X, y, test_size=0.20, random_state=seed, stratify=y)
        ir = np.bincount(ytr).max() / np.bincount(ytr).min(); gated = os.environ.get("ALPHA_MODE", "gated") == "gated"
        sampler = None
        if ir > 3.0: sampler = (lambda s=seed: SMOTENC(categorical_features=cat_idx, random_state=s, k_neighbors=5)) if cat_idx else (lambda s=seed: SMOTE(random_state=s, k_neighbors=5))
        m = kats(make_cw(ytr, gated), seed, sampler).fit(Xtr, ytr); W = m.final_estimator_.coef_; p = Xtr.shape[1]
        assert W.shape == (3, 9 + p), f"unexpected coef shape {W.shape}; expected (3, {9 + p}) with passthrough=True"
        assert list(m.final_estimator_.classes_) == [0, 1, 2]
        sd = Xtr.std(0).replace(0, np.nan).values
        for b, bn in enumerate(["B1_LightGBM", "B2_RandomForest", "B3_CalibratedNB"]):
            for r, rc in enumerate(CLASSES):
                for k, kc in enumerate(CLASSES): rows.append(dict(dataset=name, seed=seed, base=bn, out_class=rc, prob_col=kc, coef=W[r, 3 * b + k]))
        for r, rc in enumerate(CLASSES): rows.append(dict(dataset=name, seed=seed, base="PASSTHROUGH_standardised", out_class=rc, prob_col="mean_abs_over_features", coef=np.nanmean(np.abs(W[r, 9:] * sd))))
        print(name, seed, "ok", flush=True)
C = pd.DataFrame(rows); C.to_csv(f"{OUT}/meta_coef_per_seed.csv", index=False)
# Table-10 layout: for each base learner and OUTPUT class row, mean |coef| over the three probability columns; then mean +/- SD over seeds
A = C[C.base.str.startswith("B")].copy(); A["abs"] = A.coef.abs()
T = A.groupby(["dataset", "seed", "base", "out_class"]).abs.mean().reset_index()
S = T.groupby(["dataset", "base", "out_class"]).abs.agg(["mean", "std"]).reset_index(); S.to_csv(f"{OUT}/meta_coef_summary.csv", index=False)
# SIGNED own-class coefficients (row High x column High etc.): is "the meta-learner amplifies High signals" true?
own = C[(C.base.str.startswith("B")) & (C.out_class == C.prob_col)].groupby(["dataset", "base", "out_class"]).coef.agg(["mean", "std", "min", "max"]).reset_index()
own.to_csv(f"{OUT}/meta_coef_signed_own_class.csv", index=False)
L = [r"\begin{table}[t]", r"\caption{Meta-learner coefficients recomputed on the final 3-fold, passthrough pipeline: mean absolute value of the coefficients on each base learner's three-probability block, per output class; mean over " + str(len(SEEDS)) + r" seeds.}", r"\label{tab:meta-coefs}", r"\small",
     r"\begin{tabular}{@{}llrrr@{}}", r"\toprule", r"Dataset & Base learner & High & Medium & Low \\ \midrule"]
for (d, b), g in S.groupby(["dataset", "base"], sort=False):
    v = {r.out_class: f"${r['mean']:.2f}\\pm{(0 if np.isnan(r['std']) else r['std']):.2f}$" for _, r in g.iterrows()}
    L.append(f"{d} & {b.replace('_', chr(92) + '_')} & {v['High']} & {v['Medium']} & {v['Low']} \\\\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]; open(f"{OUT}/latex_table10.tex", "w").write("\n".join(L))
print(S.round(3).to_string()); print(own.round(3).to_string())
