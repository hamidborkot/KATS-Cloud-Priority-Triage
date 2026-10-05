#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cicids_multiseed_sensitivity.py

Real, checkpointed reviewer-13 sensitivity suite. No fabricated results.
It runs three linked experiments on the SAME final CICIDS mapping:
 A) fixed-test / fixed-N class-composition IR sweep: IR=2,5,10,20,30; seeds 42,7,13; Current-KATS and LightGBM.
 B) activation-gate test: fixed N=8000 and IR≈3; gates 2,3,5; seeds 42,7,13; Current-KATS only.
 C) oversampling-ratio test: fixed N=8000 and IR≈20; ratios .50,.75,1.00; seeds 42,7,13; Current-KATS only.

Each seed's test split is fixed across all conditions within that seed and is never resampled.
All output rows are checkpointed immediately to OUT.

Run on Kaggle:
  python cicids_multiseed_sensitivity.py
Optional: QUICK=1 for a smoke test only -- NEVER use QUICK results in manuscript.
"""
import os, time, json, re, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
warnings.filterwarnings("ignore")

import lightgbm as lgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score, cohen_kappa_score, precision_score, recall_score, average_precision_score, brier_score_loss
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# ---------------- Configuration ----------------
CIC_PATH = Path(os.environ.get("CIC_PATH", "/kaggle/input/datasets/ericanacletoribeiro/cicids2017-cleaned-and-preprocessed/cicids2017_cleaned.csv"))
OUT = Path(os.environ.get("OUT_DIR", "/kaggle/working/CICIDS_MULTISEED_SENSITIVITY"))
OUT.mkdir(parents=True, exist_ok=True)
QUICK = os.environ.get("QUICK", "0") == "1"
SEEDS = [42] if QUICK else [42, 7, 13]
IR_LEVELS = [2, 5, 10] if QUICK else [2, 5, 10, 20, 30]
GATES = [2, 3, 5]
RATIOS = [0.50, 0.75, 1.00]
N_TRAIN = 2000 if QUICK else 8000
N_EST = 50 if QUICK else 300
N_RF = 40 if QUICK else 200
ALPHA = 5.0
N_JOBS = -1

LOG = open(OUT / "run_log.txt", "a", encoding="utf-8")
def log(msg):
    text = time.strftime("[%H:%M:%S] ") + str(msg)
    print(text, flush=True)
    LOG.write(text + "\n")
    LOG.flush()

def norm(x):
    return re.sub(r"[^a-z0-9]", "", str(x).lower())

def find_col(df, *candidates):
    d = {norm(c): c for c in df.columns}
    for c in candidates:
        if norm(c) in d:
            return d[norm(c)]
    for c in candidates:
        q = norm(c)
        for k, v in d.items():
            if q in k or k in q:
                return v
    raise KeyError(f"Cannot find one of {candidates}; columns={df.columns.tolist()}")

def encode_labels(s):
    classes = np.array(["High", "Low", "Medium"])
    lookup = {x:i for i,x in enumerate(classes)}
    y = s.astype(str).map(lookup).values
    if np.any(pd.isna(y)):
        raise ValueError("Unexpected mapped labels")
    return y.astype(int), 0, classes

def map_priority(v):
    x = str(v).lower()
    if "benign" in x or "normal" in x:
        return "Low"
    if "scan" in x or "patator" in x or "brute" in x:
        return "Medium"
    return "High"

def cap_stratified(df, label, n=60000):
    if len(df) <= n:
        return df.reset_index(drop=True)
    frac = n / len(df)
    return pd.concat([g.sample(frac=frac, random_state=42) for _, g in df.groupby(label)], ignore_index=True)

def class_weights(y):
    cls, cnt = np.unique(y, return_counts=True)
    cw = {int(c): len(y)/(len(cls)*n) for c,n in zip(cls,cnt)}
    if cnt.max()/cnt.min() > 3:
        cw[0] *= ALPHA
    return cw

def smote_strategy(y, ratio):
    cls, cnt = np.unique(y, return_counts=True)
    max_n = int(cnt.max())
    target = int(np.ceil(ratio * max_n))
    out = {}
    for c, n in zip(cls, cnt):
        if n < target:
            out[int(c)] = target
    return out

def make_sampler(y, ratio):
    st = smote_strategy(y, ratio)
    if not st:
        return "passthrough"
    min_n = min(np.sum(y == c) for c in st)
    k = min(5, max(1, min_n - 1))
    return SMOTE(random_state=42, k_neighbors=k, sampling_strategy=st)

def make_kats(cw, seed, y_train, smote_enabled, ratio):
    def wrap(est, offset):
        if not smote_enabled:
            return est
        sampler = make_sampler(y_train, ratio)
        if sampler == "passthrough":
            return est
        sampler.set_params(random_state=seed + offset)
        return ImbPipeline([("smote", sampler), ("model", est)])
    b1 = wrap(lgb.LGBMClassifier(n_estimators=N_EST, learning_rate=.05, max_depth=6, num_leaves=31,
                                  class_weight=cw, random_state=seed, verbose=-1, n_jobs=N_JOBS), 0)
    b2 = wrap(RandomForestClassifier(n_estimators=N_RF, class_weight="balanced", random_state=seed, n_jobs=N_JOBS), 1000)
    b3 = CalibratedClassifierCV(GaussianNB(), cv=3, method="isotonic")
    return StackingClassifier(
        estimators=[("lgb", b1), ("rf", b2), ("nb", b3)],
        final_estimator=LogisticRegression(C=1.0, max_iter=2000, class_weight=cw, random_state=seed),
        stack_method="predict_proba", passthrough=True,
        cv=StratifiedKFold(3, shuffle=True, random_state=seed), n_jobs=N_JOBS)

def choose_threshold(model, Xtr, ytr, seed):
    Xf, Xv, yf, yv = train_test_split(Xtr, ytr, test_size=.15, random_state=seed, stratify=ytr)
    model.fit(Xf, yf)
    p = model.predict_proba(Xv)[:,0]
    truth = yv == 0
    floor = max(.30, 1.5*truth.mean())
    best_t, best_score, admissible = .5, -1., 0
    for t in np.arange(.15,.851,.05):
        ph = p >= t; tp=(ph & truth).sum(); fp=(ph & ~truth).sum(); fn=(~ph & truth).sum()
        prec=tp/(tp+fp) if tp+fp else 0.; rec=tp/(tp+fn) if tp+fn else 0.
        if prec >= floor:
            admissible += 1
            score=(prec+rec)/2
            if score > best_score:
                best_score, best_t=score,float(t)
    model.fit(Xtr,ytr)
    return model,best_t,(admissible==0),admissible

def ece(y, p, n_classes=3, bins=10):
    ans=[]
    for c in range(n_classes):
        yc=(y==c).astype(int); pc=p[:,c]; total=0.
        for b in range(bins):
            lo,hi=b/bins,(b+1)/bins
            mask=(pc>=lo)&((pc<=hi) if b==bins-1 else (pc<hi))
            if mask.any(): total += mask.mean()*abs(yc[mask].mean()-pc[mask].mean())
        ans.append(total)
    return float(np.mean(ans))

def score(y, pred, p):
    yh=y==0; ph=pred==0;tp=int((yh&ph).sum());fp=int((~yh&ph).sum());fn=int((yh&~ph).sum())
    return dict(Recall_H=tp/(tp+fn) if tp+fn else np.nan, Precision_H=tp/(tp+fp) if tp+fp else 0.,
                F1_H=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0., MacroF1=f1_score(y,pred,average="macro"),
                Kappa=cohen_kappa_score(y,pred), PRAUC_H=average_precision_score(yh.astype(int),p[:,0]),
                Brier=float(np.mean([brier_score_loss((y==c).astype(int),p[:,c]) for c in range(3)])), ECE=ece(y,p),
                TP_H=tp,FP_H=fp,FN_H=fn,predicted_High=int(ph.sum()))

def fixed_n_distribution(target_ir, n, pool_counts):
    # Low is largest, Medium is smallest, High lies at geometric midpoint; all three sum exactly n.
    m = int(round(n / (target_ir + np.sqrt(target_ir) + 1)))
    h = int(round(np.sqrt(target_ir)*m))
    l = n-m-h
    counts={1:l, 0:h, 2:m} # classes High=0 Low=1 Medium=2
    if any(counts[c] > pool_counts[c] for c in counts):
        raise ValueError(f"Infeasible target IR={target_ir}, N={n}, wanted={counts}, pool={pool_counts}")
    return counts

def sample_training(pool_idx, y, counts, seed):
    rng=np.random.default_rng(seed)
    take=[]
    for c,n in counts.items():
        idx=pool_idx[y[pool_idx]==c]
        take.append(rng.choice(idx,size=n,replace=False))
    z=np.concatenate(take);rng.shuffle(z);return z

def save_rows(rows, name):
    pd.DataFrame(rows).to_csv(OUT/name,index=False)

# -------- Load exact mapped/capped CICIDS final dataset --------
log("Loading CICIDS")
df=pd.read_csv(CIC_PATH,low_memory=False)
df.columns=[str(c).strip().lower().replace(" ","_") for c in df.columns]
label=find_col(df,"attack_type","label")
df["priority_label"]=df[label].apply(map_priority)
df=pd.concat([g.sample(frac=.05,random_state=42) for _,g in df.groupby("priority_label")],ignore_index=True)
features=[c for c in df.columns if c not in {label,"priority_label"} and pd.api.types.is_numeric_dtype(df[c])]
df[features]=df[features].replace([np.inf,-np.inf],np.nan).fillna(0)
df=cap_stratified(df,"priority_label",60000).reset_index(drop=True)
X=df[features].astype(float); y,HI,classes=encode_labels(df.priority_label)
log(f"Final CICIDS n={len(df)}, features={len(features)}, class counts={dict(zip(classes,np.bincount(y)))}, IR={np.bincount(y).max()/np.bincount(y).min():.4f}")

rows=[]; ck=OUT/"cicids_multiseed_sensitivity_per_seed.csv"
if ck.exists():
    old=pd.read_csv(ck);rows=old.to_dict("records");done=set(zip(old.experiment,old.seed,old.condition,old.model));log(f"Resuming {len(done)} rows")
else: done=set()

def evaluate(experiment, seed, condition, Xtr, ytr, Xte, yte, use_smote, ratio, model_name):
    key=(experiment,seed,str(condition),model_name)
    if key in done:return
    t0=time.time();cw=class_weights(ytr)
    if model_name=="KATS":
        model=make_kats(cw,seed,ytr,use_smote,ratio)
        model,thr,fallback,nadm=choose_threshold(model,Xtr,ytr,seed)
        p=model.predict_proba(Xte);pred=p.argmax(1);pred[p[:,0]>=thr]=0
    else:
        thr=.5;fallback=False;nadm=np.nan
        model=lgb.LGBMClassifier(n_estimators=N_EST,learning_rate=.05,max_depth=6,num_leaves=31,class_weight=cw,random_state=seed,verbose=-1,n_jobs=N_JOBS)
        model.fit(Xtr,ytr);p=model.predict_proba(Xte);pred=model.predict(Xte)
    r=dict(experiment=experiment,seed=seed,condition=condition,model=model_name,n_train=len(ytr),n_test=len(yte),
           train_IR=float(np.bincount(ytr).max()/np.bincount(ytr).min()),threshold=thr,fallback_used=fallback,n_admissible_thresholds=nadm,
           smote_enabled=use_smote,smote_ratio=ratio,fit_seconds=time.time()-t0,**score(yte,pred,p))
    rows.append(r);save_rows(rows,"cicids_multiseed_sensitivity_per_seed.csv")
    log(f"{experiment} seed={seed} condition={condition} {model_name}: IR={r['train_IR']:.3f}, MacroF1={r['MacroF1']:.4f}, kappa={r['Kappa']:.4f}")

# A. Fixed test and fixed N. Same test split for all target IR conditions within each seed.
for seed in SEEDS:
    pool_idx,test_idx=train_test_split(np.arange(len(y)),test_size=.20,random_state=seed,stratify=y)
    pool_counts={c:int(np.sum(y[pool_idx]==c)) for c in range(3)}
    for target in IR_LEVELS:
        counts=fixed_n_distribution(target,N_TRAIN,pool_counts)
        tr_idx=sample_training(pool_idx,y,counts,seed*100+int(target))
        for model in ["KATS","LightGBM"]:
            evaluate("fixed_test_fixed_n_ir",seed,target,X.iloc[tr_idx],y[tr_idx],X.iloc[test_idx],y[test_idx],target>3,1.0,model)

# B. Gate test at IR approximately 3. The only difference is activation gate.
for seed in SEEDS:
    pool_idx,test_idx=train_test_split(np.arange(len(y)),test_size=.20,random_state=seed,stratify=y)
    counts=fixed_n_distribution(3,N_TRAIN,{c:int(np.sum(y[pool_idx]==c)) for c in range(3)})
    tr_idx=sample_training(pool_idx,y,counts,seed*1000+3)
    actual_ir=np.bincount(y[tr_idx]).max()/np.bincount(y[tr_idx]).min()
    for gate in GATES:
        evaluate("activation_gate",seed,f"gate_{gate}",X.iloc[tr_idx],y[tr_idx],X.iloc[test_idx],y[test_idx],actual_ir>gate,1.0,"KATS")

# C. Ratio test at IR approximately 20. The only difference is resampling ratio.
for seed in SEEDS:
    pool_idx,test_idx=train_test_split(np.arange(len(y)),test_size=.20,random_state=seed,stratify=y)
    counts=fixed_n_distribution(20,N_TRAIN,{c:int(np.sum(y[pool_idx]==c)) for c in range(3)})
    tr_idx=sample_training(pool_idx,y,counts,seed*1000+20)
    for ratio in RATIOS:
        evaluate("oversampling_ratio",seed,f"ratio_{ratio:.2f}",X.iloc[tr_idx],y[tr_idx],X.iloc[test_idx],y[test_idx],True,ratio,"KATS")

R=pd.DataFrame(rows)
def ci(x):
    x=np.asarray(x,float);return stats.t.ppf(.975,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x)) if len(x)>1 else np.nan
metrics=["Recall_H","Precision_H","MacroF1","Kappa","PRAUC_H","FP_H","FN_H","Brier","ECE"]
S=R.groupby(["experiment","condition","model"],as_index=False).agg(**{f"{m}_{a}":(m,f) for m in metrics for a,f in [("mean","mean"),("sd","std"),("ci95",ci)]},n_seeds=("seed","nunique"),train_IR=("train_IR","mean"),n_train=("n_train","mean"),n_test=("n_test","mean"))
S.to_csv(OUT/"cicids_multiseed_sensitivity_summary.csv",index=False)
log("="*90);log("FINAL MULTI-SEED SENSITIVITY SUMMARY");print(S.round(5).to_string(index=False));log(f"Saved: {OUT}")
