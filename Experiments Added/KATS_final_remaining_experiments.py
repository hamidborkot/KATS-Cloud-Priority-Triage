#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
KATS final remaining experiments -- standalone, checkpointed, real results only.

Completes the two feasible remaining reviewer experiments:
 A. CICIDS training-size sensitivity at fixed test distribution and fixed IR=20.
    n_train = 4000, 8000, 12000; seeds = 42,7,13; models = current KATS and LightGBM.
 B. Live-dataset batch latency distributions for current KATS and LightGBM.
    batches = 1,32,128,200; seed 42; 30 repeats; median and p95; CPU/hardware logged.

It does NOT fabricate these impossible-with-current-data requests:
 - CICIDS capture-session grouping: cleaned derivative has no day/session ID.
 - trace replay / deadline / queue simulation: no trace scheduling fields are available.

Run: python KATS_final_remaining_experiments.py
Outputs and checkpoints: /kaggle/working/KATS_REMAINING_FINAL/
"""
import os, re, ast, glob, time, platform, warnings, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import recall_score, precision_score, f1_score, cohen_kappa_score, average_precision_score, brier_score_loss
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, SMOTENC
import lightgbm as lgb
warnings.filterwarnings("ignore")

OUT=Path('/kaggle/working/KATS_REMAINING_FINAL'); OUT.mkdir(parents=True,exist_ok=True)
SEEDS=[42,7,13]
IR_THRESHOLD=3.0; ALPHA=5.0; MAX_ROWS=60000; N_JOBS=1
N_LGB=300; N_RF=200; LAT_REPS=30; BATCHES=[1,32,128,200]
GC_PATH='/kaggle/input/datasets/derrickmwiti/google-2019-cluster-sample/borg_traces_data.csv'
IT_PATH='/kaggle/input/datasets/shamiulislamshifat/it-incident-log-dataset/incident_event_log.csv'
CIC_PATH='/kaggle/input/datasets/ericanacletoribeiro/cicids2017-cleaned-and-preprocessed/cicids2017_cleaned.csv'

def log(x): print(x,flush=True)
def norm(x): return re.sub(r'[^a-z0-9]','',str(x).lower())
def col(df,*want,required=True):
    d={norm(c):c for c in df.columns}
    for w in want:
        if norm(w) in d:return d[norm(w)]
    for w in want:
        q=norm(w)
        for k,v in d.items():
            if q in k or k in q:return v
    if required:raise KeyError(f'Missing {want}; columns={list(df.columns)}')
    return None
def label_y(s):
    le=LabelEncoder().fit(['High','Low','Medium']);y=le.transform(s.astype(str));return y,le,int(np.where(le.classes_=='High')[0][0])
def cap(df,label):
    if len(df)<=MAX_ROWS:return df.reset_index(drop=True)
    f=MAX_ROWS/len(df);return pd.concat([g.sample(frac=f,random_state=42) for _,g in df.groupby(label)],ignore_index=True)
def weights(y,hi):
    cs,ct=np.unique(y,return_counts=True);w={int(c):len(y)/(len(cs)*n) for c,n in zip(cs,ct)}
    if ct.max()/ct.min()>IR_THRESHOLD:w[hi]*=ALPHA
    return w
def ece_bin(y,p,bins=10):
    ans=0.;edges=np.linspace(0,1,bins+1)
    for b in range(bins):
        z=(p>=edges[b])&((p<=edges[b+1]) if b==bins-1 else (p<edges[b+1]))
        if z.any():ans+=z.mean()*abs(y[z].mean()-p[z].mean())
    return ans
def metrics(y,pred,proba,hi):
    yh=(y==hi).astype(int);ph=(pred==hi).astype(int)
    tp=int(((yh==1)&(ph==1)).sum());fp=int(((yh==0)&(ph==1)).sum());fn=int(((yh==1)&(ph==0)).sum())
    b=np.mean([brier_score_loss((y==k).astype(int),proba[:,k]) for k in range(3)])
    e=np.mean([ece_bin((y==k).astype(int),proba[:,k]) for k in range(3)])
    return dict(Recall_H=recall_score(yh,ph),Precision_H=precision_score(yh,ph,zero_division=0),F1_H=f1_score(yh,ph,zero_division=0),MacroF1=f1_score(y,pred,average='macro'),Kappa=cohen_kappa_score(y,pred),PRAUC_H=average_precision_score(yh,proba[:,hi]),Brier=b,ECE=e,TP_H=tp,FP_H=fp,FN_H=fn,n_H_test=int(yh.sum()),predicted_High=int(ph.sum()))
def resampler(dataset,features,cats,seed,enabled=True):
    if not enabled:return 'passthrough'
    if dataset=='ITIncident':return SMOTENC(categorical_features=[features.index(x) for x in cats if x in features],random_state=seed,k_neighbors=5,sampling_strategy='not majority')
    if dataset=='CICIDS2017':return SMOTE(random_state=seed,k_neighbors=5,sampling_strategy='not majority')
    return 'passthrough'
def kats(dataset,features,cats,cw,seed):
    def wrap(est,off):return ImbPipeline([('resample',resampler(dataset,features,cats,seed+off,True)),('model',est)])
    b1=wrap(lgb.LGBMClassifier(n_estimators=N_LGB,learning_rate=.05,max_depth=6,num_leaves=31,class_weight=cw,random_state=seed,n_jobs=N_JOBS,verbose=-1),0)
    b2=wrap(RandomForestClassifier(n_estimators=N_RF,class_weight='balanced',random_state=seed,n_jobs=N_JOBS),1000)
    b3=CalibratedClassifierCV(GaussianNB(),cv=3,method='isotonic')
    return StackingClassifier(estimators=[('lgb',b1),('rf',b2),('nb',b3)],final_estimator=LogisticRegression(C=1.,max_iter=2000,class_weight=cw,random_state=seed),stack_method='predict_proba',passthrough=True,cv=StratifiedKFold(3,shuffle=True,random_state=seed),n_jobs=N_JOBS)
def threshold_fit(m,X,y,hi,seed):
    xf,xv,yf,yv=train_test_split(X,y,test_size=.15,random_state=seed,stratify=y)
    m.fit(xf,yf);p=m.predict_proba(xv)[:,hi];yt=(yv==hi).astype(int);floor=max(.30,1.5*yt.mean());best_t=.5;best=-1; admissible=0
    for t in np.arange(.15,.86,.05):
        ph=(p>=t).astype(int);tp=((ph==1)&(yt==1)).sum();fp=((ph==1)&(yt==0)).sum();fn=((ph==0)&(yt==1)).sum();pr=tp/(tp+fp) if tp+fp else 0;rc=tp/(tp+fn) if tp+fn else 0
        if pr>=floor:
            admissible+=1
            if (pr+rc)/2>best:best=(pr+rc)/2;best_t=float(t)
    m.fit(X,y);return m,best_t,(admissible==0),admissible
def pred_thr(p,hi,t):
    z=np.argmax(p,axis=1);z[p[:,hi]>=t]=hi;return z
def parse_dict(s,key):
    def f(v):
        try:
            q=ast.literal_eval(str(v));return q.get(key,np.nan) if isinstance(q,dict) else np.nan
        except:return np.nan
    return s.apply(f)
def load_cic():
    d=pd.read_csv(CIC_PATH,low_memory=False);d.columns=[str(x).strip().lower().replace(' ','_') for x in d.columns];lc=col(d,'attack_type','label')
    def mp(v):
        x=str(v).lower()
        if 'benign' in x or 'normal' in x:return 'Low'
        if 'scan' in x or 'patator' in x or 'brute' in x:return 'Medium'
        return 'High'
    d['priority_label']=d[lc].apply(mp);d=pd.concat([g.sample(frac=.05,random_state=42) for _,g in d.groupby('priority_label')],ignore_index=True)
    f=[x for x in d if x not in {lc,'priority_label'} and pd.api.types.is_numeric_dtype(d[x])];d[f]=d[f].replace([np.inf,-np.inf],np.nan).fillna(0);return cap(d,'priority_label'),f,[]
def load_it():
    raw=pd.read_csv(IT_PATH,low_memory=False);raw.columns=[re.sub(r'[^a-z0-9]','_',str(x).strip().lower()) for x in raw.columns]
    num,mod,pri=col(raw,'number'),col(raw,'sys_mod_count'),col(raw,'priority');d=raw.sort_values([num,mod]).groupby(num,group_keys=False).tail(1).copy()
    d['priority_label']=d[pri].map({'1 - Critical':'High','2 - High':'High','3 - Moderate':'Medium','4 - Low':'Low'});d=d.dropna(subset=['priority_label']).copy()
    cmap={'category':'category_enc','location':'location_enc','contact_type':'contact_type_enc','assignment_group':'assignment_group_enc','cmdb_ci':'cmdb_ci_enc','subcategory':'subcategory_enc','knowledge':'knowledge_enc'};cats=[]
    for src,dst in cmap.items():
        q=col(d,src,required=False)
        if q is not None:d[dst]=LabelEncoder().fit_transform(d[q].astype(str));cats.append(dst)
    rp=col(d,'reopen_count');d['reopen_flag']=(d[rp]>0).astype(int);cats.append('reopen_flag')
    return d,[col(d,'reassignment_count'),col(d,'reopen_count'),col(d,'sys_mod_count')]+cats,cats
def load_gc():
    d=pd.read_csv(GC_PATH,low_memory=False);rr,au,mu=col(d,'resource_request',required=False),col(d,'average_usage',required=False),col(d,'maximum_usage',required=False)
    for k in ['cpus','memory']:
        d['req'+k]=parse_dict(d[rr],k) if rr else np.nan;d['avg'+k]=parse_dict(d[au],k) if au else np.nan;d['max'+k]=parse_dict(d[mu],k) if mu else np.nan
    p=col(d,'priority');d['priority_label']=d[p].apply(lambda x:'Low' if x<100 else 'Medium' if x<200 else 'High');ev=col(d,'event',required=False);d['event_enc']=LabelEncoder().fit_transform(d[ev].astype(str)) if ev else 0
    names=['scheduling_class','collection_type','instance_index','assigned_memory','page_cache_memory','cycles_per_instruction','memory_accesses_per_instruction','sample_rate','scheduler','vertical_scaling','reqcpus','reqmemory','avgcpus','avgmemory','maxcpus','maxmemory','failed'];f=[]
    for n in names:
        q=col(d,n,required=False)
        if q:
            if pd.api.types.is_numeric_dtype(d[q]):d[q]=d[q].replace([np.inf,-np.inf],np.nan).fillna(d[q].median())
            f.append(q)
    f.append('event_enc');return cap(d,'priority_label'),f,[]
def load_mc():
    # resolve exact source without relying on an in-memory notebook variable
    candidates=glob.glob('/kaggle/input/**/*multi*cloud*service*.csv',recursive=True)
    if not candidates:raise FileNotFoundError('MultiCloud CSV unavailable')
    d=pd.read_csv(sorted(candidates,key=lambda x:os.path.getsize(x),reverse=True)[0],low_memory=False)
    st,cp,en=col(d,'ServiceType'),col(d,'CloudProvider'),col(d,'EdgeNodeID');d['service_type_enc']=LabelEncoder().fit_transform(d[st].astype(str));d['cloud_provider_enc']=LabelEncoder().fit_transform(d[cp].astype(str));d['edge_node_enc']=LabelEncoder().fit_transform(d[en].astype(str))
    cpu,lat,thr,bw,wv=[col(d,x) for x in ['CPUUtilization','ServiceLatency','Throughput','NetworkBandwidth','WorkloadVariability']]
    score=.30*d[cpu]/100+.25*d[lat]/d[lat].max()+.20*(1-d[thr]/d[thr].max())+.15*(1-d[bw]/d[bw].max())+.10*d[wv]/d[wv].max();d['priority_label']=pd.qcut(score,3,labels=['Low','Medium','High']).astype(str)
    f=[col(d,x,required=False) for x in ['MemoryUsage','StorageUsage','ResponseTime','LoadBalancing','OptimalServicePlacement']];f=[x for x in f if x]+['service_type_enc','cloud_provider_enc','edge_node_enc'];return d,f,['service_type_enc','cloud_provider_enc','edge_node_enc']
def cpu_name():
    try:return [x.split(':',1)[1].strip() for x in open('/proc/cpuinfo') if 'model name' in x][0]
    except:return platform.processor()

# ---------- load once ----------
log('Loading final live datasets')
cic_df,CIC_FEATURES,CIC_CATS=load_cic();it_df,IT_FEATURES,IT_CATS=load_it();gc_df,GC_FEATURES,GC_CATS=load_gc();mc_df,MC_FEATURES,MC_CATS=load_mc()
DATA={'GoogleCluster':(gc_df,GC_FEATURES,GC_CATS),'ITIncident':(it_df,IT_FEATURES,IT_CATS),'MultiCloud':(mc_df,MC_FEATURES,MC_CATS),'CICIDS2017':(cic_df,CIC_FEATURES,CIC_CATS)}
for ds,(d,f,cats) in DATA.items():
    y,le,hi=label_y(d.priority_label);log(f'{ds}: n={len(d)}, features={len(f)}, IR={ir(y):.4f}')

# ============================================================ A: FIXED-TEST / FIXED-IR / TRAINING-SIZE SWEEP
# At IR=20, vary n_train only. Existing user results already cover IR, activation gate, and ratio sweeps.
# ============================================================
A_PATH=OUT/'cicids_training_size_fixed_ir_per_seed.csv';arows=pd.read_csv(A_PATH).to_dict('records') if A_PATH.exists() else []
done=set(zip(pd.DataFrame(arows).get('seed',[]),pd.DataFrame(arows).get('n_train',[]),pd.DataFrame(arows).get('model',[]))) if arows else set()
X=cic_df[CIC_FEATURES].replace([np.inf,-np.inf],np.nan).fillna(0).astype(float);y,le,hi=label_y(cic_df.priority_label)
IR_TARGET=20.;SIZES=[4000,8000,12000]
for seed in SEEDS:
    Xpool,Xtest,ypool,ytest=train_test_split(X,y,test_size=.20,random_state=seed,stratify=y)
    Xpool=Xpool.reset_index(drop=True); rng=np.random.default_rng(seed)
    # Counts: Medium=m, High=3m, Low=IR*m; n = (IR+4)m. Rounded residual goes to Low.
    idx={c:np.where(ypool==c)[0] for c in range(3)}
    for ntrain in SIZES:
        m=int(ntrain/(IR_TARGET+4)); counts={2:m,0:3*m,1:ntrain-4*m} # labels High=0, Low=1, Medium=2
        if any(counts[c]>len(idx[c]) for c in counts):log(f'SKIP n={ntrain}: unavailable class count {counts}');continue
        take=np.concatenate([rng.choice(idx[c],counts[c],replace=False) for c in [0,1,2]]);rng.shuffle(take)
        Xtr=Xpool.iloc[take].reset_index(drop=True);ytr=ypool[take];cw=weights(ytr,hi);actual=ir(ytr)
        for name in ['KATS','LightGBM']:
            if (seed,ntrain,name) in done:continue
            t=time.time()
            if name=='KATS':
                model,thr,fallback,nadm=threshold_fit(kats('CICIDS2017',CIC_FEATURES,CIC_CATS,cw,seed),Xtr,ytr,hi,seed);p=model.predict_proba(Xtest);pred=pred_thr(p,hi,thr)
            else:
                model=lgb.LGBMClassifier(n_estimators=N_LGB,learning_rate=.05,max_depth=6,num_leaves=31,class_weight=cw,random_state=seed,n_jobs=N_JOBS,verbose=-1).fit(Xtr,ytr);p=model.predict_proba(Xtest);pred=model.predict(Xtest);thr=.5;fallback=False;nadm=np.nan
            r=dict(experiment='fixed_test_fixed_ir_training_size',seed=seed,n_train=ntrain,train_IR=actual,n_test=len(ytest),model=name,threshold=thr,fallback_used=fallback,n_admissible_thresholds=nadm,fit_seconds=time.time()-t,**metrics(ytest,pred,p,hi));arows.append(r);pd.DataFrame(arows).to_csv(A_PATH,index=False);log(f'TRAIN_SIZE seed={seed} n={ntrain} {name}: IR={actual:.3f}, MacroF1={r["MacroF1"]:.4f}, kappa={r["Kappa"]:.4f}')
A=pd.DataFrame(arows)
if len(A):
    def ci(x):return stats.t.ppf(.975,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x)) if len(x)>1 else np.nan
    SA=A.groupby(['experiment','n_train','train_IR','n_test','model']).agg(**{f'{k}_{s}':(k,fn) for k in ['Recall_H','Precision_H','MacroF1','Kappa','PRAUC_H','FP_H','FN_H','Brier','ECE'] for s,fn in [('mean','mean'),('sd','std'),('ci95',ci)]}).reset_index();SA.to_csv(OUT/'cicids_training_size_fixed_ir_summary.csv',index=False)
    print('\n'+'='*96+'\nTRAINING-SIZE SWEEP SUMMARY\n'+'='*96);print(SA.round(4).to_string(index=False))

# ============================================================ B: LIVE BATCH LATENCY DISTRIBUTIONS
# KATS and LightGBM, seed 42, batch 1/32/128/200, 30 repetitions, median/p95.
# ============================================================
L_PATH=OUT/'live_batch_latency_per_repeat.csv';lrows=pd.read_csv(L_PATH).to_dict('records') if L_PATH.exists() else []
ldone=set(zip(pd.DataFrame(lrows).get('dataset',[]),pd.DataFrame(lrows).get('model',[]),pd.DataFrame(lrows).get('batch_size',[]),pd.DataFrame(lrows).get('repeat',[]))) if lrows else set()
for ds,(d,f,cats) in DATA.items():
    X=d[f].replace([np.inf,-np.inf],np.nan).fillna(0).astype(float);y,le,hi=label_y(d.priority_label);Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.20,random_state=42,stratify=y);cw=weights(ytr,hi)
    models={'KATS':kats(ds,f,cats,cw,42),'LightGBM':lgb.LGBMClassifier(n_estimators=N_LGB,learning_rate=.05,max_depth=6,num_leaves=31,class_weight=cw,random_state=42,n_jobs=1,verbose=-1)}
    for name,model in models.items():
        t=time.perf_counter();model.fit(Xtr,ytr);fit=time.perf_counter()-t
        for bs in BATCHES:
            if bs>len(Xte):continue
            # one warm-up
            model.predict_proba(Xte.iloc[:bs])
            for rep in range(LAT_REPS):
                if (ds,name,bs,rep) in ldone:continue
                ix=np.random.default_rng(rep).choice(len(Xte),size=bs,replace=False);xb=Xte.iloc[ix];t=time.perf_counter();model.predict_proba(xb);us=(time.perf_counter()-t)*1e6/bs
                lrows.append(dict(dataset=ds,model=name,seed=42,batch_size=bs,repeat=rep,us_per_prediction=us,fit_seconds=fit,n_train=len(Xtr),n_features=len(f),n_jobs=1,cpu=cpu_name(),cores=os.cpu_count(),hardware='Kaggle P100 environment'));pd.DataFrame(lrows).to_csv(L_PATH,index=False)
            subset=pd.DataFrame([r for r in lrows if r['dataset']==ds and r['model']==name and r['batch_size']==bs]);log(f'LATENCY {ds} {name} batch={bs}: median={subset.us_per_prediction.median():.2f}us p95={subset.us_per_prediction.quantile(.95):.2f}us')
L=pd.DataFrame(lrows)
if len(L):
    SL=L.groupby(['dataset','model','batch_size','n_train','n_features','n_jobs','cpu','cores','hardware']).us_per_prediction.agg(median_us='median',p95_us=lambda x:x.quantile(.95),mean_us='mean',sd_us='std',n_repeats='count').reset_index();SL.to_csv(OUT/'live_batch_latency_summary.csv',index=False)
    print('\n'+'='*96+'\nLIVE BATCH LATENCY SUMMARY\n'+'='*96);print(SL.round(3).to_string(index=False))

# Zip actual outputs
with zipfile.ZipFile(OUT/'KATS_REMAINING_FINAL_RESULTS.zip','w',zipfile.ZIP_DEFLATED) as z:
    for p in OUT.glob('*.csv'):z.write(p,p.name)
print('\nDONE. Download:',OUT/'KATS_REMAINING_FINAL_RESULTS.zip')
