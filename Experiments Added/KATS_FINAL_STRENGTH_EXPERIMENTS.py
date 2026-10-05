#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
KATS_FINAL_STRENGTH_EXPERIMENTS.py

Self-contained final reviewer experiments for comments 6, 13 and 15.
No CloudTask. Uses the four live datasets and final cluster-A feature construction.

Run one mode at a time in Kaggle:
  MODE=stack python KATS_FINAL_STRENGTH_EXPERIMENTS.py
  MODE=ir python KATS_FINAL_STRENGTH_EXPERIMENTS.py
  MODE=latency python KATS_FINAL_STRENGTH_EXPERIMENTS.py
  MODE=all python KATS_FINAL_STRENGTH_EXPERIMENTS.py

Each result is checkpointed after every completed configuration.
Do NOT use QUICK=1 for manuscript results. QUICK is only a smoke test.
"""
import os, re, ast, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score, cohen_kappa_score, precision_score, recall_score, average_precision_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, SMOTENC
import lightgbm as lgb
warnings.filterwarnings("ignore")

# ----------------------------- configuration -----------------------------
MODE = os.environ.get("MODE", "stack").lower()
QUICK = os.environ.get("QUICK", "0") == "1"
OUT = Path(os.environ.get("OUT_DIR", "/kaggle/working/KATS_FINAL_STRENGTH"))
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = [42, 7] if QUICK else [42, 7, 13, 99, 2026]
IR_SEEDS = [42, 7] if QUICK else [42, 7, 13]
N_JOBS = 1
MAX_ROWS = 60000
IR_THRESHOLD = 3.0
ALPHA = 5.0
N_LGB = 30 if QUICK else 300
N_RF = 20 if QUICK else 200
LAT_REPS = 5 if QUICK else 30

GC_PATH = Path("/kaggle/input/datasets/derrickmwiti/google-2019-cluster-sample/borg_traces_data.csv")
IT_PATH = Path("/kaggle/input/datasets/shamiulislamshifat/it-incident-log-dataset/incident_event_log.csv")
MC_CANDIDATES = [
    Path("/kaggle/input/datasets/ziya07/multi-cloud-service-composition-dataset/multi_cloud_service_dataset.csv"),
    Path("/kaggle/input/datasets/ziya07/multi-cloud-service-composition-dataset/multicloudservicedataset.csv"),
]
CIC_PATH = Path("/kaggle/input/datasets/ericanacletoribeiro/cicids2017-cleaned-and-preprocessed/cicids2017_cleaned.csv")

def log(msg):
    print(msg, flush=True)

def norm(x):
    return re.sub(r"[^a-z0-9]", "", str(x).lower())

def col(df, *candidates, required=True):
    lookup = {norm(c): c for c in df.columns}
    for candidate in candidates:
        key = norm(candidate)
        if key in lookup:
            return lookup[key]
    for candidate in candidates:
        key = norm(candidate)
        for k, original in lookup.items():
            if key in k or k in key:
                return original
    if required:
        raise KeyError(f"Missing {candidates}. Available={df.columns.tolist()}")
    return None

def cap(df, label_col):
    if len(df) <= MAX_ROWS:
        return df.reset_index(drop=True)
    frac = MAX_ROWS / len(df)
    return pd.concat([g.sample(frac=frac, random_state=42) for _,g in df.groupby(label_col)], ignore_index=True)

def labels(series):
    le = LabelEncoder().fit(["High", "Low", "Medium"])
    y = le.transform(series.astype(str))
    high = int(np.where(le.classes_ == "High")[0][0])
    return y, le, high

def ir(y):
    _, counts = np.unique(y, return_counts=True)
    return float(counts.max() / counts.min())

def weights(y, high):
    classes, counts = np.unique(y, return_counts=True)
    out = {int(c): float(len(y)/(len(classes)*n)) for c,n in zip(classes,counts)}
    if counts.max()/counts.min() > IR_THRESHOLD:
        out[int(high)] *= ALPHA
    return out

def matrix(df, features):
    return df[features].replace([np.inf,-np.inf],np.nan).fillna(0).astype(float)

def parse_dict(series, key):
    def f(v):
        try:
            z = ast.literal_eval(str(v))
            return z.get(key, np.nan) if isinstance(z, dict) else np.nan
        except Exception:
            return np.nan
    return series.apply(f)

# ----------------------------- exact final loaders -----------------------------
def load_gc():
    d = pd.read_csv(GC_PATH, low_memory=False)
    rr, au, mu = col(d,"resource_request",required=False), col(d,"average_usage",required=False), col(d,"maximum_usage",required=False)
    for k in ["cpus","memory"]:
        d[f"req{k}"] = parse_dict(d[rr],k) if rr else np.nan
        d[f"avg{k}"] = parse_dict(d[au],k) if au else np.nan
        d[f"max{k}"] = parse_dict(d[mu],k) if mu else np.nan
    pri = col(d,"priority")
    d["priority_label"] = d[pri].apply(lambda x: "Low" if x < 100 else "Medium" if x < 200 else "High")
    ev = col(d,"event",required=False)
    d["event_enc"] = LabelEncoder().fit_transform(d[ev].astype(str)) if ev else 0
    names=["scheduling_class","collection_type","instance_index","assigned_memory","page_cache_memory","cycles_per_instruction","memory_accesses_per_instruction","sample_rate","scheduler","vertical_scaling","reqcpus","reqmemory","avgcpus","avgmemory","maxcpus","maxmemory","failed"]
    features=[]
    for n in names:
        c=col(d,n,required=False)
        if c:
            if pd.api.types.is_numeric_dtype(d[c]): d[c]=d[c].replace([np.inf,-np.inf],np.nan).fillna(d[c].median())
            features.append(c)
    features.append("event_enc")
    return cap(d,"priority_label"), features, []

def load_it():
    raw = pd.read_csv(IT_PATH, low_memory=False)
    raw.columns=[re.sub(r"[^a-z0-9]","_",str(x).strip().lower()) for x in raw.columns]
    n, mod, pri = col(raw,"number"), col(raw,"sys_mod_count"), col(raw,"priority")
    d = raw.sort_values([n,mod],kind="mergesort").groupby(n,group_keys=False).tail(1).copy()
    d["priority_label"] = d[pri].map({"1 - Critical":"High","2 - High":"High","3 - Moderate":"Medium","4 - Low":"Low"})
    d=d.dropna(subset=["priority_label"]).copy()
    cmap={"category":"category_enc","location":"location_enc","contact_type":"contact_type_enc","assignment_group":"assignment_group_enc","cmdb_ci":"cmdb_ci_enc","subcategory":"subcategory_enc","knowledge":"knowledge_enc"}
    cats=[]
    for src,dst in cmap.items():
        c=col(d,src,required=False)
        if c:
            d[dst]=LabelEncoder().fit_transform(d[c].astype(str));cats.append(dst)
    reopen=col(d,"reopen_count")
    d["reopen_flag"]=(d[reopen]>0).astype(int);cats.append("reopen_flag")
    features=[col(d,"reassignment_count"),col(d,"reopen_count"),col(d,"sys_mod_count")]+cats
    return d,features,cats

def load_mc():
    path=next((p for p in MC_CANDIDATES if p.exists()),None)
    if path is None: raise FileNotFoundError("MultiCloud source not found")
    d=pd.read_csv(path,low_memory=False)
    st,cp,en=col(d,"ServiceType"),col(d,"CloudProvider"),col(d,"EdgeNodeID")
    d["service_type_enc"]=LabelEncoder().fit_transform(d[st].astype(str));d["cloud_provider_enc"]=LabelEncoder().fit_transform(d[cp].astype(str));d["edge_node_enc"]=LabelEncoder().fit_transform(d[en].astype(str))
    cpu,lat,thr,bw,wv=[col(d,x) for x in ["CPUUtilization","ServiceLatency","Throughput","NetworkBandwidth","WorkloadVariability"]]
    score=.30*d[cpu]/100+.25*d[lat]/d[lat].max()+.20*(1-d[thr]/d[thr].max())+.15*(1-d[bw]/d[bw].max())+.10*d[wv]/d[wv].max()
    d["priority_label"]=pd.qcut(score,3,labels=["Low","Medium","High"]).astype(str)
    f=[col(d,x,required=False) for x in ["MemoryUsage","StorageUsage","ResponseTime","LoadBalancing","OptimalServicePlacement"]]
    return d,[x for x in f if x]+["service_type_enc","cloud_provider_enc","edge_node_enc"],["service_type_enc","cloud_provider_enc","edge_node_enc"]

def load_cic():
    d=pd.read_csv(CIC_PATH,low_memory=False)
    d.columns=[str(x).strip().lower().replace(" ","_") for x in d.columns]
    lc=col(d,"attack_type","label")
    def mp(v):
        x=str(v).lower()
        if "benign" in x or "normal" in x:return "Low"
        if "scan" in x or "patator" in x or "brute" in x:return "Medium"
        return "High"
    d["priority_label"]=d[lc].apply(mp)
    d=pd.concat([g.sample(frac=.05,random_state=42) for _,g in d.groupby("priority_label")],ignore_index=True)
    f=[x for x in d if x not in {lc,"priority_label"} and pd.api.types.is_numeric_dtype(d[x])]
    d[f]=d[f].replace([np.inf,-np.inf],np.nan).fillna(0)
    return cap(d,"priority_label"),f,[]

# ----------------------------- final pipeline components -----------------------------
def make_resampler(dataset, features, cats, seed, enabled=True, ratio=1.0):
    if not enabled: return "passthrough"
    # `not majority` is the final pipeline's current ratio=1.0 behavior.
    if dataset=="ITIncident":
        return SMOTENC(categorical_features=[features.index(x) for x in cats if x in features],random_state=seed,k_neighbors=5,sampling_strategy="not majority")
    if dataset=="CICIDS2017":
        return SMOTE(random_state=seed,k_neighbors=5,sampling_strategy="not majority")
    return "passthrough"

def make_stack(dataset, features, cats, cw, seed, *, meta_weighted=True, passthrough=True, scaled_meta=False, resampling=True):
    def wrap(est,offset):
        return ImbPipeline([("resample",make_resampler(dataset,features,cats,seed+offset,resampling)),("model",est)])
    b1=wrap(lgb.LGBMClassifier(n_estimators=N_LGB,learning_rate=.05,max_depth=6,num_leaves=31,class_weight=cw,random_state=seed,n_jobs=N_JOBS,verbose=-1),0)
    b2=wrap(RandomForestClassifier(n_estimators=N_RF,class_weight="balanced",random_state=seed,n_jobs=N_JOBS),1000)
    b3=CalibratedClassifierCV(GaussianNB(),cv=3,method="isotonic")
    meta=LogisticRegression(C=1.0,max_iter=2000,class_weight=cw if meta_weighted else None,random_state=seed)
    if scaled_meta:
        meta=Pipeline([("scale",StandardScaler()),("meta",meta)])
    return StackingClassifier(estimators=[("lgb",b1),("rf",b2),("nb",b3)],final_estimator=meta,stack_method="predict_proba",passthrough=passthrough,cv=3,n_jobs=N_JOBS)

def score(y,pred,proba,high):
    yh=(y==high).astype(int);ph=(pred==high).astype(int);tp=int(((yh==1)&(ph==1)).sum());fp=int(((yh==0)&(ph==1)).sum());fn=int(((yh==1)&(ph==0)).sum())
    return dict(RecallH=tp/(tp+fn) if tp+fn else np.nan,PrecH=tp/(tp+fp) if tp+fp else 0.,MacroF1=f1_score(y,pred,average="macro"),Kappa=cohen_kappa_score(y,pred),PRAUC_High=average_precision_score(yh,proba[:,high]),FP_High=fp,FN_High=fn,pred_High_fraction=ph.mean())

def save_checkpoint(rows,path): pd.DataFrame(rows).to_csv(path,index=False)
def summarize(df,group_cols):
    out=[]
    for key,g in df.groupby(group_cols):
        if not isinstance(key,tuple):key=(key,)
        row=dict(zip(group_cols,key))
        for c in ["RecallH","PrecH","MacroF1","Kappa","PRAUC_High","FP_High","FN_High","pred_High_fraction"]:
            row[c+"_mean"]=g[c].mean();row[c+"_sd"]=g[c].std(ddof=1)
        out.append(row)
    return pd.DataFrame(out)

# ----------------------------- comment 6 -----------------------------
def run_stack():
    d,f,cats=load_cic();X=matrix(d,f);y,le,hi=labels(d.priority_label)
    ck=OUT/"R3_6_cicids_stack_diagnostic_per_seed.csv"; rows=pd.read_csv(ck).to_dict("records") if ck.exists() else [];done={(r['seed'],r['variant']) for r in rows}
    variants={
        "KATS_current":dict(meta_weighted=True,passthrough=True,scaled_meta=False),
        "KATS_unweighted_meta":dict(meta_weighted=False,passthrough=True,scaled_meta=False),
        "KATS_scaled_meta":dict(meta_weighted=True,passthrough=True,scaled_meta=True),
        "KATS_no_passthrough":dict(meta_weighted=True,passthrough=False,scaled_meta=False),
    }
    for seed in SEEDS:
        Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=seed,stratify=y);cw=weights(ytr,hi)
        fitted=None
        for name,kw in variants.items():
            if (seed,name) in done: continue
            t=time.time();m=make_stack("CICIDS2017",f,cats,cw,seed,**kw).fit(Xtr,ytr);p=m.predict_proba(Xte);pred=m.predict(Xte);r=score(yte,pred,p,hi);r.update(seed=seed,variant=name,fit_seconds=time.time()-t);rows.append(r);save_checkpoint(rows,ck);print(seed,name,r,flush=True)
            if name=="KATS_current": fitted=m
        # Fit/reload current model if needed for probability average. The average uses original argmax, as in your existing single-seed diagnostic.
        if (seed,"Probability_average_argmax") not in done:
            if fitted is None: fitted=make_stack("CICIDS2017",f,cats,cw,seed,meta_weighted=True,passthrough=True,scaled_meta=False).fit(Xtr,ytr)
            p=(fitted.named_estimators_["lgb"].predict_proba(Xte)+fitted.named_estimators_["rf"].predict_proba(Xte)+fitted.named_estimators_["nb"].predict_proba(Xte))/3
            r=score(yte,np.argmax(p,axis=1),p,hi);r.update(seed=seed,variant="Probability_average_argmax",fit_seconds=0.0);rows.append(r);save_checkpoint(rows,ck);print(seed,"Probability_average_argmax",r,flush=True)
        if (seed,"LightGBM_standalone") not in done:
            t=time.time();m=lgb.LGBMClassifier(n_estimators=N_LGB,learning_rate=.05,max_depth=6,num_leaves=31,class_weight=cw,random_state=seed,n_jobs=N_JOBS,verbose=-1).fit(Xtr,ytr);p=m.predict_proba(Xte);r=score(yte,m.predict(Xte),p,hi);r.update(seed=seed,variant="LightGBM_standalone",fit_seconds=time.time()-t);rows.append(r);save_checkpoint(rows,ck);print(seed,"LightGBM_standalone",r,flush=True)
    R=pd.DataFrame(rows);S=summarize(R,["variant"]);S.to_csv(OUT/"R3_6_cicids_stack_diagnostic_summary.csv",index=False);print("\nCOMMENT 6 SUMMARY\n",S.round(4).to_string(index=False))

# ----------------------------- comment 13 -----------------------------
def controlled_counts(N,target_ir):
    # Low is majority; High and Medium are equal minorities. N remains fixed exactly.
    m=max(20,int(round(N/(target_ir+2))));low=N-2*m
    return {"Low":low,"High":m,"Medium":m}

def sampled_training(Xpool,ypool,counts,le,seed):
    rng=np.random.default_rng(seed);idx=[]
    for label,n in counts.items():
        c=int(np.where(le.classes_==label)[0][0]);pool=np.where(ypool==c)[0]
        if n>len(pool): raise ValueError(f"Need {n} {label} rows but only {len(pool)} available")
        idx.extend(rng.choice(pool,n,replace=False))
    idx=np.asarray(idx);rng.shuffle(idx);return Xpool.iloc[idx].reset_index(drop=True),ypool[idx]

def run_ir():
    d,f,cats=load_cic();X=matrix(d,f);y,le,hi=labels(d.priority_label)
    ck=OUT/"R3_13_fixed_test_fixed_n_per_seed.csv";rows=pd.read_csv(ck).to_dict("records") if ck.exists() else [];done={(r['seed'],r['N_train'],r['target_IR'],r['model']) for r in rows}
    for seed in IR_SEEDS:
        Xpool,Xtest,ypool,ytest=train_test_split(X,y,test_size=.2,random_state=seed,stratify=y)
        for N in [4000,7000]:
            for target in [2,5,10,20,30]:
                counts=controlled_counts(N,target);Xtr,ytr=sampled_training(Xpool,ypool,counts,le,seed);ach=ir(ytr);cw=weights(ytr,hi)
                for model in ["KATS_gate3_ratio1","LightGBM"]:
                    if (seed,N,target,model) in done:continue
                    if model=="LightGBM":m=lgb.LGBMClassifier(n_estimators=N_LGB,learning_rate=.05,max_depth=6,num_leaves=31,class_weight=cw,random_state=seed,n_jobs=N_JOBS,verbose=-1).fit(Xtr,ytr)
                    else:m=make_stack("CICIDS2017",f,cats,cw,seed,meta_weighted=True,passthrough=True,scaled_meta=False).fit(Xtr,ytr)
                    p=m.predict_proba(Xtest);pred=m.predict(Xtest);r=score(ytest,pred,p,hi);r.update(seed=seed,N_train=N,target_IR=target,achieved_IR=ach,test_n=len(ytest),test_IR=ir(ytest),model=model);rows.append(r);save_checkpoint(rows,ck);print(seed,N,target,model,r,flush=True)
    # At N=7000, IR=10, explicitly test activation gates and resampling target ratios.
    for seed in IR_SEEDS:
        Xpool,Xtest,ypool,ytest=train_test_split(X,y,test_size=.2,random_state=seed,stratify=y);Xtr,ytr=sampled_training(Xpool,ypool,controlled_counts(7000,10),le,seed);cw=weights(ytr,hi)
        for gate in [2,3,5]:
            for ratio in [.5,.75,1.0]:
                model=f"KATS_gate{gate}_ratio{ratio}" 
                if (seed,7000,10,model) in done:continue
                # locally reproduce gate/rho; no hidden use of global pipeline choices
                def sampler(seed2):
                    if ir(ytr)<=gate:return "passthrough"
                    ct=np.bincount(ytr);major=ct.max();target_counts={i:max(ct[i],min(major,int(np.ceil(ratio*major)))) for i in range(len(ct)) if ct[i]<major}
                    return SMOTE(random_state=seed2,k_neighbors=5,sampling_strategy=target_counts)
                def custom_stack():
                    def wrap(est,offset):return ImbPipeline([("resample",sampler(seed+offset)),("model",est)])
                    return StackingClassifier(estimators=[("lgb",wrap(lgb.LGBMClassifier(n_estimators=N_LGB,learning_rate=.05,max_depth=6,num_leaves=31,class_weight=cw,random_state=seed,n_jobs=N_JOBS,verbose=-1),0)),("rf",wrap(RandomForestClassifier(n_estimators=N_RF,class_weight="balanced",random_state=seed,n_jobs=N_JOBS),1000)),("nb",CalibratedClassifierCV(GaussianNB(),cv=3,method="isotonic"))],final_estimator=LogisticRegression(C=1.,max_iter=2000,class_weight=cw,random_state=seed),stack_method="predict_proba",passthrough=True,cv=3,n_jobs=N_JOBS)
                m=custom_stack().fit(Xtr,ytr);p=m.predict_proba(Xtest);r=score(ytest,m.predict(Xtest),p,hi);r.update(seed=seed,N_train=7000,target_IR=10,achieved_IR=ir(ytr),test_n=len(ytest),test_IR=ir(ytest),model=model);rows.append(r);save_checkpoint(rows,ck);print(seed,model,r,flush=True)
    R=pd.DataFrame(rows);S=summarize(R,["N_train","target_IR","model"]);S.to_csv(OUT/"R3_13_fixed_test_fixed_n_summary.csv",index=False);print("\nCOMMENT 13 SUMMARY\n",S.round(4).to_string(index=False))

# ----------------------------- comment 15 -----------------------------
def run_latency():
    loaders={"GoogleCluster":load_gc,"ITIncident":load_it,"MultiCloud":load_mc,"CICIDS2017":load_cic}
    ck=OUT/"R3_15_batch_latency.csv";rows=pd.read_csv(ck).to_dict("records") if ck.exists() else [];done={(r['dataset'],r['model'],r['batch']) for r in rows}
    for ds,loader in loaders.items():
        d,f,cats=loader();X=matrix(d,f);y,le,hi=labels(d.priority_label);Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y);cw=weights(ytr,hi)
        models={"KATS":make_stack(ds,f,cats,cw,42,meta_weighted=True,passthrough=True,scaled_meta=False),"LightGBM":lgb.LGBMClassifier(n_estimators=N_LGB,learning_rate=.05,max_depth=6,num_leaves=31,class_weight=cw,random_state=42,n_jobs=1,verbose=-1)}
        for name,m in models.items():
            t=time.perf_counter();m.fit(Xtr,ytr);fit=time.perf_counter()-t
            for batch in [1,32,128,200]:
                if batch>len(Xte) or (ds,name,batch) in done:continue
                sample=Xte.iloc[:batch];m.predict_proba(sample) # warmup
                vals=[]
                for _ in range(LAT_REPS):
                    t=time.perf_counter();m.predict_proba(sample);vals.append((time.perf_counter()-t)*1e6/batch)
                r=dict(dataset=ds,model=name,batch=batch,fit_seconds=fit,n_repeats=LAT_REPS,us_per_item_median=float(np.median(vals)),us_per_item_p95=float(np.percentile(vals,95)),us_per_item_mean=float(np.mean(vals)),us_per_item_sd=float(np.std(vals,ddof=1)),hardware="Kaggle P100 node; CPU timing; n_jobs=1");rows.append(r);save_checkpoint(rows,ck);print(r,flush=True)
    R=pd.DataFrame(rows);R.to_csv(OUT/"R3_15_batch_latency_summary.csv",index=False);print("\nCOMMENT 15 LATENCY\n",R.round(3).to_string(index=False))

if MODE in ["stack","all"]:run_stack()
if MODE in ["ir","all"]:run_ir()
if MODE in ["latency","all"]:run_latency()
print("DONE. Outputs:",OUT)
