# ============================================================
# KATS CLUSTER A FINAL
# Self-contained Kaggle script. Paste ALL of this into one cell.
# Four live datasets only: GoogleCluster, ITIncident, MultiCloud,
# CICIDS2017. CloudTask remains preserved historical output.
# ============================================================

import os, re, ast, json, pickle, warnings
from pathlib import Path
from datetime import datetime
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, cohen_kappa_score, brier_score_loss,
    average_precision_score, precision_score, recall_score,
    confusion_matrix,
)
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV

import lightgbm as lgb
import xgboost as xgb
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, SMOTENC
from imblearn.ensemble import BalancedRandomForestClassifier

# ---------------- Configuration ----------------
SEEDS = [42, 7, 13, 99, 2026]
TEMPORAL_SEEDS = [42, 7, 13]
IR_LEVELS = [2, 5, 10, 20, 30]
IR_THRESHOLD = 3.0
MAX_ROWS = 60000
N_JOBS = 1
OUT = Path("/kaggle/working/KATS_CLUSTER_A_FINAL")
CKPT = OUT / "checkpoints"
OUT.mkdir(parents=True, exist_ok=True)
CKPT.mkdir(parents=True, exist_ok=True)

IT_PATH = "/kaggle/input/datasets/shamiulislamshifat/it-incident-log-dataset/incident_event_log.csv"
GC_PATH = "/kaggle/input/datasets/derrickmwiti/google-2019-cluster-sample/borg_traces_data.csv"
MC_PATH = "/kaggle/input/datasets/ziya07/multi-cloud-service-composition-dataset/multi_cloud_service_dataset.csv"
CIC_PATH = "/kaggle/input/datasets/ericanacletoribeiro/cicids2017-cleaned-and-preprocessed/cicids2017_cleaned.csv"


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def norm(x):
    return re.sub(r"[^a-z0-9]", "", str(x).lower())


def col(df, *names, required=True):
    cols = {norm(c): c for c in df.columns}
    for name in names:
        if norm(name) in cols:
            return cols[norm(name)]
    for name in names:
        key = norm(name)
        for nk, original in cols.items():
            if key in nk or nk in key:
                return original
    if required:
        raise KeyError(f"Cannot locate {names}; columns={list(df.columns)}")
    return None


def cp(name, fn):
    path = CKPT / f"{re.sub(r'[^A-Za-z0-9_.-]', '_', name)}.pkl"
    if path.exists():
        log(f"RESUME {name}")
        with open(path, "rb") as f:
            return pickle.load(f)
    obj = fn()
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    return obj


def label_y(series):
    le = LabelEncoder()
    y = le.fit_transform(series.astype(str))
    high = int(np.where(le.classes_ == "High")[0][0])
    return y, le, high


def ir(y):
    _, c = np.unique(y, return_counts=True)
    return float(c.max() / c.min())


def weights(y, high, alpha=5.0):
    classes, counts = np.unique(y, return_counts=True)
    out = {int(c): float(len(y) / (len(classes) * n)) for c, n in zip(classes, counts)}
    if counts.max() / counts.min() > IR_THRESHOLD:
        out[int(high)] *= alpha
    return out


def cap(df, label_col):
    if len(df) <= MAX_ROWS:
        return df.reset_index(drop=True)
    f = MAX_ROWS / len(df)
    return pd.concat([g.sample(frac=f, random_state=42) for _, g in df.groupby(label_col)]).reset_index(drop=True)


def feature_matrix(df, features):
    return df[features].replace([np.inf, -np.inf], np.nan).fillna(0).astype(float)


def ece_binary(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    value, rows = 0.0, []
    for b in range(bins):
        if b == bins - 1:
            mask = (p >= edges[b]) & (p <= edges[b + 1])
        else:
            mask = (p >= edges[b]) & (p < edges[b + 1])
        n = int(mask.sum())
        if n == 0:
            rows.append((b, 0, np.nan, np.nan))
            continue
        conf = float(p[mask].mean())
        acc = float(y[mask].mean())
        value += n / len(y) * abs(acc - conf)
        rows.append((b, n, conf, acc))
    return float(value), rows


def metrics(y, pred, proba, le, high):
    rep = classification_report(y, pred, target_names=le.classes_.tolist(), output_dict=True, zero_division=0)
    ncls = len(le.classes_)
    briers, eces, binrows = [], [], []
    for c in range(ncls):
        yb = (y == c).astype(int)
        briers.append(brier_score_loss(yb, proba[:, c]))
        e, br = ece_binary(yb, proba[:, c])
        eces.append(e)
        for b, n, conf, acc in br:
            binrows.append({"class_index": c, "bin": b, "count": n, "mean_confidence": conf, "empirical_frequency": acc})
    yh = (y == high).astype(int)
    ph = (pred == high).astype(int)
    return {
        "RecallH": float(rep["High"]["recall"]),
        "PrecH": float(rep["High"]["precision"]),
        "F1H": float(rep["High"]["f1-score"]),
        "MacroF1": float(rep["macro avg"]["f1-score"]),
        "Kappa": float(cohen_kappa_score(y, pred)),
        "Brier": float(np.mean(briers)),
        "ECE": float(np.mean(eces)),
        "PRAUC_High": float(average_precision_score(yh, proba[:, high])),
        "FP_High": int(((ph == 1) & (yh == 0)).sum()),
        "FN_High": int(((ph == 0) & (yh == 1)).sum()),
        "bins": binrows,
        "cm": confusion_matrix(y, pred).tolist(),
    }


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

# ---------------- Load data ----------------
def load_it():
    raw = pd.read_csv(IT_PATH, low_memory=False)
    num, mod, opened, pri = col(raw,"number"), col(raw,"sys_mod_count"), col(raw,"opened_at"), col(raw,"priority")
    d = raw.sort_values([num, mod]).groupby(num, group_keys=False).tail(1).copy()
    d["opened_at_final"] = pd.to_datetime(d[opened], errors="coerce")
    d["priority_label"] = d[pri].map({"1 - Critical":"High", "2 - High":"High", "3 - Moderate":"Medium", "4 - Low":"Low"})
    d = d.dropna(subset=["opened_at_final","priority_label"]).copy()
    cmap = {"category":"category_enc","location":"location_enc","contact_type":"contact_type_enc","assignment_group":"assignment_group_enc","cmdb_ci":"cmdb_ci_enc","subcategory":"subcategory_enc","knowledge":"knowledge_enc"}
    cats=[]
    for src,dst in cmap.items():
        c=col(d,src,required=False)
        if c is not None:
            d[dst]=LabelEncoder().fit_transform(d[c].astype(str)); cats.append(dst)
    reopen=col(d,"reopen_count"); d["reopen_flag"]=(d[reopen]>0).astype(int); cats.append("reopen_flag")
    feats=[col(d,"reassignment_count"),col(d,"reopen_count"),col(d,"sys_mod_count")]+cats
    bad=["impact","urgency","made_sla","priority","resolved_by","resolved_at","closed_at"]
    assert not any(col(d,b,required=False) in feats for b in bad)
    return d,feats,cats

def load_gc():
    d=pd.read_csv(GC_PATH,low_memory=False)
    def pdict(s,k):
        def f(v):
            try:
                z=ast.literal_eval(str(v)); return z.get(k,np.nan) if isinstance(z,dict) else np.nan
            except: return np.nan
        return s.apply(f)
    rr,au,mu=col(d,"resource_request",required=False),col(d,"average_usage",required=False),col(d,"maximum_usage",required=False)
    for k in ["cpus","memory"]:
        d[f"req{k}"]=pdict(d[rr],k) if rr else np.nan; d[f"avg{k}"]=pdict(d[au],k) if au else np.nan; d[f"max{k}"]=pdict(d[mu],k) if mu else np.nan
    pri=col(d,"priority"); d["priority_label"]=d[pri].apply(lambda v:"Low" if v<100 else "Medium" if v<200 else "High")
    ev=col(d,"event",required=False); d["event_enc"]=LabelEncoder().fit_transform(d[ev].astype(str)) if ev else 0
    names=["scheduling_class","collection_type","instance_index","assigned_memory","page_cache_memory","cycles_per_instruction","memory_accesses_per_instruction","sample_rate","scheduler","vertical_scaling","reqcpus","reqmemory","avgcpus","avgmemory","maxcpus","maxmemory","failed"]
    feats=[]
    for n in names:
        c=col(d,n,required=False)
        if c:
            if pd.api.types.is_numeric_dtype(d[c]): d[c]=d[c].replace([np.inf,-np.inf],np.nan).fillna(d[c].median())
            feats.append(c)
    feats.append("event_enc")
    return cap(d,"priority_label"),feats,col(d,"collection_id",required=False),pri

def load_mc():
    d=pd.read_csv(MC_PATH)
    st,cp,en=col(d,"ServiceType"),col(d,"CloudProvider"),col(d,"EdgeNodeID")
    d["service_type_enc"]=LabelEncoder().fit_transform(d[st].astype(str)); d["cloud_provider_enc"]=LabelEncoder().fit_transform(d[cp].astype(str)); d["edge_node_enc"]=LabelEncoder().fit_transform(d[en].astype(str))
    cpu,lat,thr,bw,wv=[col(d,x) for x in ["CPUUtilization","ServiceLatency","Throughput","NetworkBandwidth","WorkloadVariability"]]
    score=.30*d[cpu]/100+.25*d[lat]/d[lat].max()+.20*(1-d[thr]/d[thr].max())+.15*(1-d[bw]/d[bw].max())+.10*d[wv]/d[wv].max()
    d["priority_label"]=pd.qcut(score,3,labels=["Low","Medium","High"]).astype(str)
    feats=[col(d,x,required=False) for x in ["MemoryUsage","StorageUsage","ResponseTime","LoadBalancing","OptimalServicePlacement"]]
    return d,[x for x in feats if x]+["service_type_enc","cloud_provider_enc","edge_node_enc"],["service_type_enc","cloud_provider_enc","edge_node_enc"]

def load_cic():
    d=pd.read_csv(CIC_PATH,low_memory=False); d.columns=[x.strip().lower().replace(" ","_") for x in d.columns]
    lc=col(d,"attack_type","label")
    def mp(v):
        x=str(v).lower()
        if "benign" in x or "normal" in x:return "Low"
        if "scan" in x or "patator" in x or "brute" in x:return "Medium"
        return "High"
    d["priority_label"]=d[lc].apply(mp)
    d=pd.concat([g.sample(frac=.05,random_state=42) for _,g in d.groupby("priority_label")]).reset_index(drop=True)
    feats=[x for x in d if x not in {lc,"priority_label"} and pd.api.types.is_numeric_dtype(d[x])]
    d[feats]=d[feats].replace([np.inf,-np.inf],np.nan).fillna(0)
    return cap(d,"priority_label"),feats,[]

log("Loading final data")
it_df,IT_FEATURES,IT_CATS=load_it(); gc_df,GC_FEATURES,GC_GROUP,GC_PRI=load_gc(); mc_df,MC_FEATURES,MC_CATS=load_mc(); cic_df,CIC_FEATURES,CIC_CATS=load_cic()
DATA={"GoogleCluster":(gc_df,GC_FEATURES,[]),"ITIncident":(it_df,IT_FEATURES,IT_CATS),"MultiCloud":(mc_df,MC_FEATURES,MC_CATS),"CICIDS2017":(cic_df,CIC_FEATURES,[])}
for name,(d,f,cats) in DATA.items():
    yy,_,_=label_y(d.priority_label); pd.DataFrame({"feature":f,"type":["categorical" if z in cats else "numeric" for z in f]}).to_csv(OUT/f"features_{name}.csv",index=False); log(f"{name}: n={len(d)}, features={len(f)}, IR={ir(yy):.4f}")

# ---------------- Main benchmark ----------------
metric_rows=[]; prediction_rows=[]; bin_rows=[]; cm_rows=[]
for ds,(d,feats,cats) in DATA.items():
    X=feature_matrix(d,feats); y,le,hi=label_y(d.priority_label)
    for seed in SEEDS:
        def job(ds=ds,X=X,y=y,le=le,hi=hi,feats=feats,cats=cats,seed=seed):
            Xtr,Xte,ytr,yte,itr,ite=train_test_split(X,y,np.arange(len(y)),test_size=.2,random_state=seed,stratify=y)
            cw=weights(ytr,hi)
            models={"KATS":kats(ds,feats,cats,cw,seed)}; models.update(baselines(cw,seed)); out={}
            for name,m in models.items():
                if name=="KATS": m,t=threshold_fit(m,Xtr,ytr,hi,seed)
                else: m.fit(Xtr,ytr); t=.5
                p=m.predict_proba(Xte); pred=pred_threshold(p,hi,t) if name=="KATS" else m.predict(Xte)
                out[name]={"m":metrics(yte,pred,p,le,hi),"p":p,"pred":pred,"yt":yte,"ids":ite,"t":t,"classes":le.classes_.tolist()}
            return out
        out=cp(f"main_{ds}_{seed}",job)
        for name,z in out.items():
            m=z["m"]; metric_rows.append({"dataset":ds,"seed":seed,"model":name,"threshold":z["t"],**{k:v for k,v in m.items() if k not in {"bins","cm"}}})
            for i in range(len(z["yt"])):
                prediction_rows.append({"dataset":ds,"seed":seed,"model":name,"test_row_id":int(z["ids"][i]),"true_label":int(z["yt"][i]),"predicted_label":int(z["pred"][i]),"prob_low":float(z["p"][i,0]),"prob_medium":float(z["p"][i,1]),"prob_high":float(z["p"][i,hi]),"selected_high_threshold":z["t"]})
            for b in m["bins"]: bin_rows.append({"dataset":ds,"seed":seed,"model":name,**b})
            cm=np.array(m["cm"])
            for a in range(cm.shape[0]):
                for b in range(cm.shape[1]): cm_rows.append({"dataset":ds,"seed":seed,"model":name,"true_class":le.classes_[a],"predicted_class":le.classes_[b],"count":int(cm[a,b])})
        log(f"Main done: {ds} seed {seed}")
main=pd.DataFrame(metric_rows); preds=pd.DataFrame(prediction_rows); bins=pd.DataFrame(bin_rows); cms=pd.DataFrame(cm_rows)
main.to_csv(OUT/"clusterA_main_metrics_per_seed.csv",index=False); preds.to_csv(OUT/"clusterA_predictions.csv",index=False); bins.to_csv(OUT/"clusterA_calibration_bins.csv",index=False); cms.to_csv(OUT/"clusterA_confusion_matrices.csv",index=False)
summary=main.groupby(["dataset","model"]).agg(**{f"{k}_mean":(k,"mean") for k in ["RecallH","PrecH","F1H","MacroF1","Kappa","Brier","ECE","PRAUC_High","FP_High","FN_High"]},**{f"{k}_sd":(k,"std") for k in ["RecallH","PrecH","F1H","MacroF1","Kappa","Brier","ECE","PRAUC_High","FP_High","FN_High"]}).reset_index()
summary.to_csv(OUT/"clusterA_metrics_mean_sd.csv",index=False)

# ---------------- Calibrated ablation ----------------
ab=[]
for ds,(d,feats,cats) in DATA.items():
    X=feature_matrix(d,feats); y,le,hi=label_y(d.priority_label)
    variants={"T_Full":(True,True,True,True),"T_NoSMOTE":(False,True,True,True),"T_NoAsymLoss":(True,False,True,True),"T_NoCalibNB":(True,True,False,True),"T_NoStacking":(True,True,True,False)}
    for seed in SEEDS:
        Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=seed,stratify=y); cw=weights(ytr,hi)
        for v,cfg in variants.items():
            def job(v=v,cfg=cfg):
                m=kats(ds,feats,cats,cw,seed,*cfg); m,t=threshold_fit(m,Xtr,ytr,hi,seed); p=m.predict_proba(Xte); pred=pred_threshold(p,hi,t); return {"t":t,"m":metrics(yte,pred,p,le,hi)}
            z=cp(f"abl_{ds}_{v}_{seed}",job); ab.append({"dataset":ds,"seed":seed,"variant":v,"threshold":z["t"],**{k:v for k,v in z["m"].items() if k not in {"bins","cm"}}})
ab=pd.DataFrame(ab); ab.to_csv(OUT/"clusterA_ablation_per_seed.csv",index=False)
absum=ab.groupby(["dataset","variant"]).agg(RecallH_mean=("RecallH","mean"),RecallH_sd=("RecallH","std"),MacroF1_mean=("MacroF1","mean"),MacroF1_sd=("MacroF1","std"),Kappa_mean=("Kappa","mean"),Kappa_sd=("Kappa","std")).reset_index()
for ds in absum.dataset.unique():
    ref=absum[(absum.dataset==ds)&(absum.variant=="T_Full")].iloc[0]
    ix=absum.dataset==ds; absum.loc[ix,"Delta_RecallH"]=absum.loc[ix,"RecallH_mean"]-ref.RecallH_mean; absum.loc[ix,"Delta_MacroF1"]=absum.loc[ix,"MacroF1_mean"]-ref.MacroF1_mean
absum.to_csv(OUT/"clusterA_ablation_summary.csv",index=False)

# ---------------- Valid temporal sweep ----------------
temp=[]; d=it_df.sort_values("opened_at_final").reset_index(drop=True); X=feature_matrix(d,IT_FEATURES); y,le,hi=label_y(d.priority_label)
for frac in [.6,.7,.8,.9]:
    n=int(len(d)*frac); Xtr,Xte,ytr,yte=X.iloc[:n],X.iloc[n:],y[:n],y[n:]; assert d.opened_at_final.iloc[:n].max()<=d.opened_at_final.iloc[n:].min()
    for seed in TEMPORAL_SEEDS:
        cw=weights(ytr,hi); models={"KATS":kats("ITIncident",IT_FEATURES,IT_CATS,cw,seed),"LightGBM":baselines(cw,seed)["LightGBM"],"LogReg":baselines(cw,seed)["LogReg"]}
        for name,m in models.items():
            if name=="KATS":m,t=threshold_fit(m,Xtr,ytr,hi,seed)
            else:m.fit(Xtr,ytr);t=.5
            p=m.predict_proba(Xte);pred=pred_threshold(p,hi,t) if name=="KATS" else m.predict(Xte); temp.append({"train_fraction":frac,"seed":seed,"model":name,"threshold":t,**{k:v for k,v in metrics(yte,pred,p,le,hi).items() if k not in {"bins","cm"}}})
temp=pd.DataFrame(temp);temp.to_csv(OUT/"clusterA_temporal_sweep.csv",index=False)

# ---------------- Matched recall ----------------
matched=[]
for ds in ["ITIncident","CICIDS2017"]:
    d,feats,cats=DATA[ds];X=feature_matrix(d,feats);y,le,hi=label_y(d.priority_label)
    for seed in SEEDS:
        Xo,Xte,yo,yte=train_test_split(X,y,test_size=.2,random_state=seed,stratify=y);Xf,Xv,yf,yv=train_test_split(Xo,yo,test_size=.15,random_state=seed,stratify=yo);cw=weights(yf,hi)
        models={"KATS":kats(ds,feats,cats,cw,seed),"LightGBM":baselines(cw,seed)["LightGBM"],"XGBoost":baselines(cw,seed)["XGBoost"],"LogReg":baselines(cw,seed)["LogReg"]}
        for name,m in models.items():
            m.fit(Xf,yf);pv=m.predict_proba(Xv); target=.90; bestt=min(np.arange(.05,.96,.01),key=lambda t:abs(recall_score(yv==hi,pv[:,hi]>=t,zero_division=0)-target));m.fit(Xo,yo);pt=m.predict_proba(Xte);pred=pred_threshold(pt,hi,bestt);matched.append({"dataset":ds,"seed":seed,"model":name,"target_recall":target,"threshold":bestt,**{k:v for k,v in metrics(yte,pred,pt,le,hi).items() if k not in {"bins","cm"}}})
matched=pd.DataFrame(matched);matched.to_csv(OUT/"clusterA_matched_operating_points.csv",index=False)

# ---------------- Fixed-test IR sensitivity ----------------
irrows=[];X=feature_matrix(cic_df,CIC_FEATURES);y,le,hi=label_y(cic_df.priority_label)
for seed in SEEDS:
    Xpool,Xtest,ypool,ytest=train_test_split(X,y,test_size=.2,random_state=seed,stratify=y);inds={c:np.where(ypool==c)[0] for c in np.unique(ypool)};cnt={c:len(v) for c,v in inds.items()};maj=max(cnt,key=cnt.get);minor=min(cnt,key=cnt.get);mid=[c for c in cnt if c not in {maj,minor}][0];rng=np.random.RandomState(seed)
    for level in IR_LEVELS:
        nmaj=3000;nmin=int(round(nmaj/level));nmid=6000-nmaj-nmin;chosen=np.concatenate([rng.choice(inds[maj],nmaj,False),rng.choice(inds[mid],nmid,False),rng.choice(inds[minor],nmin,False)]);rng.shuffle(chosen);Xtr=Xpool.iloc[chosen];ytr=ypool[chosen];cw=weights(ytr,hi)
        for name,m in {"KATS":kats("CICIDS2017",CIC_FEATURES,[],cw,seed),"LightGBM":baselines(cw,seed)["LightGBM"],"XGBoost":baselines(cw,seed)["XGBoost"],"LogReg":baselines(cw,seed)["LogReg"]}.items():
            if name=="KATS":m,t=threshold_fit(m,Xtr,ytr,hi,seed)
            else:m.fit(Xtr,ytr);t=.5
            p=m.predict_proba(Xtest);pred=pred_threshold(p,hi,t) if name=="KATS" else m.predict(Xtest);irrows.append({"seed":seed,"training_target_ir":level,"training_achieved_ir":ir(ytr),"training_n":len(ytr),"fixed_test_n":len(ytest),"model":name,"threshold":t,**{k:v for k,v in metrics(ytest,pred,p,le,hi).items() if k not in {"bins","cm"}}})
irout=pd.DataFrame(irrows);irout.to_csv(OUT/"clusterA_ir_fixed_test_sensitivity.csv",index=False)

# ---------------- Final checks ----------------
checks={"it_features_11":len(IT_FEATURES)==11,"main_160":len(main)==160,"ablation_100":len(ab)==100,"temporal_36":len(temp)==36,"matched_40":len(matched)==40,"ir_100":len(irout)==100,"predictions_saved":len(preds)>0}
json.dump(checks,open(OUT/"clusterA_validation_checks.json","w"),indent=2)
log("COMPLETE");log(json.dumps(checks,indent=2));log(str(OUT))
