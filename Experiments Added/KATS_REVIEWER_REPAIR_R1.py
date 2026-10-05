#!/usr/bin/env python
"""KATS reviewer repair R1: new, explicitly versioned evaluation; never overwrites old results."""
import os, sys, re, ast, json, time, hashlib, shutil, warnings, platform, itertools
from pathlib import Path
from datetime import datetime
import importlib.metadata as im
import numpy as np
import pandas as pd
from scipy.stats import binomtest, rankdata, spearmanr
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GroupShuffleSplit, GroupKFold, StratifiedKFold, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score, cohen_kappa_score, average_precision_score, confusion_matrix
from joblib import dump, load
try:
    import lightgbm as lgb
    import xgboost as xgb
    from imblearn.over_sampling import SMOTE, SMOTENC
    from imblearn.ensemble import BalancedRandomForestClassifier
except ImportError as e:
    raise ImportError('Install missing dependencies in a separate Kaggle cell: %pip install lightgbm xgboost imbalanced-learn shap joblib ; restart if versions were changed.') from e

VERSION = 'KATS_REVIEWER_REPAIR_R1_2026_10_02'
INPUT = Path(os.environ.get('KATS_INPUT_ROOT', '/kaggle/input'))
OUT = Path(os.environ.get('KATS_REPAIR_OUT', '/kaggle/working/KATS_REVIEWER_REPAIR_R1'))
OUT.mkdir(parents=True, exist_ok=True)
CASES = OUT / 'cases'; CASES.mkdir(exist_ok=True)
# Full is the default because learned-preprocessing/model-training changes define a NEW evaluation.
DATASETS = tuple(os.environ.get('KATS_DATASETS', 'GoogleCluster,ITIncident,MultiCloud,CICIDS2017').split(','))
SEEDS = [42, 7, 13, 99, 2026]
EXTRA_SEEDS = [42, 7, 13]
STAGES = set(os.environ.get('KATS_STAGES', 'audit,main,ablation,grouped,availability,temporal,uncertainty,shap,latency').split(','))
N_JOBS = int(os.environ.get('KATS_N_JOBS', '1'))
BOOT_N = int(os.environ.get('KATS_BOOT_N', '1000'))
SHAP_ROWS_PER_STRATUM = int(os.environ.get('KATS_SHAP_ROWS_PER_STRATUM', '50'))
SHAP_NSAMPLES = int(os.environ.get('KATS_SHAP_NSAMPLES', '300'))
DAYFIRST = os.environ.get('KATS_IT_DAYFIRST', '1') == '1'
LABELS = ['High', 'Low', 'Medium']; HIGH = 0
MODELS = ['KATS','LightGBM','XGBoost','RandomForest','BalancedRF','MLP','LogReg','NaiveBayes']
PATH_ENV = {'GoogleCluster':'GC_PATH','ITIncident':'IT_PATH','MultiCloud':'MC_PATH','CICIDS2017':'CIC_PATH'}
BASENAMES = {'GoogleCluster':'borg_traces_data.csv','ITIncident':'incident_event_log.csv',
             'MultiCloud':'multi_cloud_service_dataset.csv','CICIDS2017':'cicids2017_cleaned.csv'}
VERSIONS = {p:im.version(p) for p in ['numpy','pandas','scipy','scikit-learn','lightgbm','xgboost','imbalanced-learn','joblib']}
CONFIG = dict(version=VERSION, datasets=DATASETS, seeds=SEEDS, jobs=N_JOBS, cv=3,
              alpha_high=5, high_weight_IR_gate=3, numeric_missing='training-median-Google/fixed-zero-others',
              categorical_encoding='training-fold OrdinalEncoder unknown=-1', raw_passthrough=True,
              duplicate_split='same-X groups when X repeats; otherwise stratified rows',
              baseline_defaults='explicit BalancedRF bootstrap=False/replacement=True/all',
              threshold_grid=[round(float(t),2) for t in np.arange(.15,.86,.05)], dayfirst=DAYFIRST,
              versions=VERSIONS)

def log(s): print('['+datetime.now().strftime('%H:%M:%S')+'] '+str(s), flush=True)
def json_write(p,obj):
    p=Path(p); tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,default=str)); os.replace(tmp,p)
def csv_write(p,df):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');df.to_csv(tmp,index=False);os.replace(tmp,p)
def digest_file(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def hash_values(x):
    if isinstance(x,np.ndarray):x=pd.DataFrame(x)
    return hashlib.sha256(pd.util.hash_pandas_object(x,index=False).values.tobytes()).hexdigest()
def norm(x): return re.sub('[^a-z0-9]','',str(x).lower())
def column(d,*names,required=True):
    exact=[c for c in d if norm(c) in [norm(n) for n in names]]
    if len(exact)==1:return exact[0]
    cand=[c for c in d if any(norm(c).startswith(norm(n)) for n in names)]
    if len(cand)==1:return cand[0]
    if required:raise ValueError(f'Ambiguous/missing column {names}: candidates={cand}; columns={list(d.columns)}')
    return None

def source_path(ds):
    manual=os.environ.get(PATH_ENV[ds])
    if manual:
        p=Path(manual)
        if not p.is_file():raise FileNotFoundError(p)
        return p
    found=sorted(INPUT.rglob(BASENAMES[ds]))
    if not found:raise FileNotFoundError(f'Attach raw {BASENAMES[ds]} to Kaggle; or set {PATH_ENV[ds]} explicitly. Result CSVs do not replace raw data.')
    if len(found)>1:
        hashes=[digest_file(p) for p in found]
        if len(set(hashes))!=1:raise ValueError(f'Different raw files found for {ds}: {found}. Set {PATH_ENV[ds]}; do not guess a source.')
    return found[0]

def cap(d):
    if len(d)<=60000:return d.reset_index(drop=True)
    f=60000/len(d)
    return pd.concat([g.sample(frac=f,random_state=42) for _,g in d.groupby('_label')]).reset_index(drop=True)
def dates(s):
    try:return pd.to_datetime(s,errors='coerce',format='mixed',dayfirst=DAYFIRST,utc=True)
    except TypeError:return pd.to_datetime(s,errors='coerce',dayfirst=DAYFIRST,utc=True)

class Data:
    def __init__(self,name,d,features,cats,policy='zero',groups=None,times=None):
        self.name=name;self.features=list(features);self.cats=list(cats);self.policy=policy
        self.X=d[features].copy().reset_index(drop=True)
        for c in features:
            if c in cats:self.X[c]=self.X[c].fillna('__MISSING__').astype(str)
            else:self.X[c]=pd.to_numeric(self.X[c],errors='coerce').replace([np.inf,-np.inf],np.nan)
        self.y=d['_label'].map({v:i for i,v in enumerate(LABELS)}).to_numpy()
        if pd.isna(self.y).any():raise ValueError('Invalid label')
        self.y=self.y.astype(int);self.ids=d['_id'].astype(str).to_numpy()
        if len(np.unique(self.ids))!=len(self.ids):raise ValueError('Repeated retained source ID')
        self.catidx=[self.features.index(c) for c in cats]
        self.xgroups=pd.util.hash_pandas_object(self.X,index=False).astype(str).to_numpy()
        self.groups=np.asarray(groups) if groups is not None else None
        self.times=np.asarray(times) if times is not None else None
        self.hash=hash_values(self.X)+hash_values(self.y[:,None])+hash_values(self.ids[:,None])
        self.source_hash=''

def load_dataset(ds):
    p=source_path(ds);log(f'RAW {ds}: {p}')
    d=pd.read_csv(p,low_memory=False);d['_id']=[f'{ds}:{i}' for i in range(len(d))]
    original=len(d); rawdup=d.drop(columns=['_id']).duplicated().sum()
    extra={};groups=times=None;cats=[];policy='zero'
    if ds=='GoogleCluster':
        def parse(v,k):
            try:
                z=ast.literal_eval(str(v));return z.get(k,np.nan) if isinstance(z,dict) else np.nan
            except (ValueError,SyntaxError,TypeError):return np.nan
        for root,prefix in [('resource_request','req'),('average_usage','avg'),('maximum_usage','max')]:
            c=column(d,root)
            for k in ['cpus','memory']:d[prefix+k]=d[c].map(lambda v:parse(v,k))
        pr=column(d,'priority');d['_label']=np.where(d[pr]<100,'Low',np.where(d[pr]<200,'Medium','High'))
        ev=column(d,'event');d['event_enc']=d[ev].fillna('__MISSING__').astype(str);cats=['event_enc']
        names=['scheduling_class','collection_type','instance_index','assigned_memory','page_cache_memory',
               'cycles_per_instruction','memory_accesses_per_instruction','sample_rate','scheduler','vertical_scaling',
               'reqcpus','reqmemory','avgcpus','avgmemory','maxcpus','maxmemory','failed']
        features=[column(d,n) for n in names]+['event_enc'];policy='median'
        missing=[]
        for c in features:
            if c in cats:continue
            z=pd.to_numeric(d[c],errors='coerce').replace([np.inf,-np.inf],np.nan)
            missing.append(dict(feature=c,source_missing_or_nonfinite=int(z.isna().sum()),source_median=float(z.median())))
        d=cap(d);gcol=column(d,'collection_id');groups=d[gcol].astype(str).to_numpy()
        for r in missing:r['retained_missing_or_nonfinite']=int(pd.to_numeric(d[r['feature']],errors='coerce').replace([np.inf,-np.inf],np.nan).isna().sum())
        csv_write(OUT/'Google_source_and_retained_imputation_audit.csv',pd.DataFrame(missing))
        log(pd.DataFrame(missing).to_string(index=False));extra['median_source_audit_only_not_used_for_fitting']=True
    elif ds=='ITIncident':
        num=column(d,'number');mod=column(d,'sys_mod_count');opened=column(d,'opened_at');pri=column(d,'priority')
        updated=column(d,'sys_updated_at',required=False)
        d['_sequence']=pd.to_numeric(d[mod],errors='coerce')
        if d['_sequence'].isna().any():raise ValueError('Missing/non-numeric IT update sequence: audit before selecting snapshot')
        srt=d.sort_values([num,'_sequence','_id'],kind='mergesort')
        final=srt.groupby(num,sort=False,group_keys=False).tail(1).copy()
        first=srt.groupby(num,sort=False,group_keys=False).head(1).copy()
        mapping={'1 - Critical':'High','2 - High':'High','3 - Moderate':'Medium','4 - Low':'Low'}
        final['_label']=final[pri].map(mapping);final['_opened']=dates(final[opened])
        final=final[final['_label'].notna() & final['_opened'].notna()].copy()
        cmap={n:n+'_enc' for n in ['category','location','contact_type','assignment_group','cmdb_ci','subcategory','knowledge']}
        for src,dst in cmap.items():final[dst]=final[column(final,src)].fillna('__MISSING__').astype(str)
        reopen=column(final,'reopen_count');final['reopen_flag']=np.where(pd.to_numeric(final[reopen],errors='coerce')>0,'1','0')
        cats=list(cmap.values())+['reopen_flag']
        features=[column(final,n) for n in ['reassignment_count','reopen_count','sys_mod_count']]+cats
        if len(features)!=11:raise AssertionError('IT schema must have11 columns')
        final['_info_time']=dates(final[updated]) if updated else pd.NaT
        d=final.reset_index(drop=True);times=d['_info_time'].to_numpy()
        keep=d[num].astype(str).to_numpy();first=first.set_index(num).loc[d[num]].reset_index()
        six=[cmap[n] for n in ['category','location','contact_type','assignment_group','cmdb_ci','subcategory']]
        e=pd.DataFrame({'_id':d['_id']+'_FIRST','_label':d['_label']})
        for n in ['category','location','contact_type','assignment_group','cmdb_ci','subcategory']:
            e[cmap[n]]=first[column(first,n)].fillna('__MISSING__').astype(str).to_numpy()
        e['_info_time']=dates(first[updated]).to_numpy() if updated else pd.NaT
        extra.update(incident_ids=keep, first=e, six=six,
                     final_label_information_times=times, snapshot_ties=int(srt.duplicated([num,'_sequence'],keep=False).sum()),
                     updated_field=updated, actual_row_snapshot=True, date_parse_dayfirst=DAYFIRST)
        csv_write(OUT/'IT_snapshot_identity.csv',pd.DataFrame({'incident_id':keep,'final_source_row_id':d['_id'],
                'first_source_row_id':first['_id'].to_numpy(),'final_information_time':d['_info_time'],
                'first_information_time':e['_info_time']}))
        # Exact source-row IDs, not groupby.first/last field mosaics.
    elif ds=='MultiCloud':
        for src,dst in [('ServiceType','service_type_enc'),('CloudProvider','cloud_provider_enc'),('EdgeNodeID','edge_node_enc')]:
            d[dst]=d[column(d,src)].fillna('__MISSING__').astype(str);cats.append(dst)
        fields=[column(d,x) for x in ['CPUUtilization','ServiceLatency','Throughput','NetworkBandwidth','WorkloadVariability']]
        v=[pd.to_numeric(d[x],errors='raise') for x in fields]
        if any(vv.isna().any() for vv in v):raise ValueError('Missing target-construction input: no invented label')
        score=.30*v[0]/100+.25*v[1]/v[1].max()+.20*(1-v[2]/v[2].max())+.15*(1-v[3]/v[3].max())+.10*v[4]/v[4].max()
        d['_label']=pd.qcut(score,3,labels=['Low','Medium','High']).astype(str)
        features=[column(d,x) for x in ['MemoryUsage','StorageUsage','ResponseTime','LoadBalancing','OptimalServicePlacement']]+cats
        extra['target_definition']='corpus-relative five-term composite and qcut; not supplied urgency'
    else:
        d.columns=[str(x).strip().lower().replace(' ','_') if x!='_id' else x for x in d.columns]
        lc=column(d,'attack_type','label')
        def mapper(x):
            x=str(x).lower()
            return 'Low' if ('benign' in x or 'normal' in x) else 'Medium' if any(v in x for v in ['scan','patator','brute']) else 'High'
        d['_label']=d[lc].map(mapper)
        inv=d.groupby([lc,'_label'],dropna=False).size().rename('count').reset_index();csv_write(OUT/'CICIDS_label_inventory.csv',inv)
        d=pd.concat([g.sample(frac=.05,random_state=42) for _,g in d.groupby('_label')]).reset_index(drop=True)
        features=[x for x in d if x not in {lc,'_label','_id'} and pd.api.types.is_numeric_dtype(d[x])]
        if len(features)!=52:raise ValueError(f'CICIDS schema has {len(features)} numeric columns, expected52; check derivative; do not silently alter features.')
        d=cap(d);extra['capture_session_evaluation']='not available in this cleaned52-feature derivative; no day IDs fabricated'
    data=Data(ds,d,features,cats,policy,groups,times);data.source_hash=digest_file(p);data.extra=extra
    audit=dict(dataset=ds,source_path=str(p),raw_sha256=data.source_hash,raw_rows=original,raw_full_row_duplicates=int(rawdup),
               retained_rows=len(d),features=len(features),class_order=LABELS,
               class_counts={LABELS[k]:int((data.y==k).sum()) for k in range(3)},
               predictor_duplicate_rows=int(data.X.duplicated().sum()),
               predictor_label_duplicate_rows=int(data.X.assign(_y=data.y).duplicated().sum()),
               learned_preprocessing='per-fit training only', retained_matrix_hash=data.hash,
               availability='retrospective retained-state classification; early IT analysed separately')
    json_write(OUT/f'{ds}_data_audit.json',audit)
    csv_write(OUT/f'{ds}_feature_schema.csv',pd.DataFrame({'feature':features,'categorical':[c in cats for c in features]}))
    csv_write(OUT/f'{ds}_retained_row_labels.csv',pd.DataFrame({'row_id':data.ids,'label':data.y,'predictor_group':data.xgroups}))
    log(audit);return data

class TrainOnlyPreprocessor(BaseEstimator):
    def __init__(self,catidx=(),policy='zero'):
        self.catidx=catidx;self.policy=policy
    def fit(self,X,y=None):
        a=np.asarray(X,dtype=object);self.n_features_in_=a.shape[1];self.cats_=list(self.catidx)
        self.nums_=[i for i in range(a.shape[1]) if i not in self.cats_]
        z=np.asarray(a[:,self.nums_],dtype=float);z[~np.isfinite(z)]=np.nan
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            self.fill_=np.nanmedian(z,axis=0) if self.policy=='median' else np.zeros(z.shape[1])
        self.fill_=np.where(np.isfinite(self.fill_),self.fill_,0.)
        if self.cats_:
            self.encoder_=OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1)
            self.encoder_.fit(a[:,self.cats_].astype(str))
        self.fit_rows_hash_=hash_values(a);return self
    def transform(self,X):
        a=np.asarray(X,dtype=object);out=np.empty(a.shape,dtype=float)
        z=np.asarray(a[:,self.nums_],dtype=float);z[~np.isfinite(z)]=np.nan
        r,c=np.where(np.isnan(z));z[r,c]=self.fill_[c];out[:,self.nums_]=z
        if self.cats_:out[:,self.cats_]=self.encoder_.transform(a[:,self.cats_].astype(str))
        if not np.isfinite(out).all():raise AssertionError('Nonfinite model input')
        return out
    def fit_transform(self,X,y=None):return self.fit(X,y).transform(X)

def class_weights(y,on=True):
    counts=np.bincount(np.asarray(y,dtype=int),minlength=3)
    if (counts==0).any():raise ValueError(f'Training partition missing class: {counts}')
    if not on:return {i:1. for i in range(3)}
    w={i:len(y)/(3*counts[i]) for i in range(3)}
    if counts.max()/counts.min()>3:w[0]*=5
    return w

def cv_splits(y,groups=None,times=None):
    y=np.asarray(y)
    if times is not None:
        t=pd.to_datetime(times,utc=True);uniq=np.sort(t.unique())
        if t.isna().any() or len(uniq)<4:raise ValueError('Insufficient known information times for forward folds')
        pairs=[]
        for a,b in TimeSeriesSplit(n_splits=3).split(uniq):
            tr=np.where(np.isin(t,uniq[a]))[0];va=np.where(np.isin(t,uniq[b]))[0]
            if not t[tr].max()<t[va].min():raise AssertionError('Forward CV boundary violated')
            pairs.append((tr,va))
    elif groups is not None:
        pairs=list(GroupKFold(n_splits=3).split(np.zeros((len(y),1)),y,groups))
    else:pairs=list(StratifiedKFold(n_splits=3,shuffle=False).split(np.zeros((len(y),1)),y))
    for tr,va in pairs:
        if len(np.unique(y[tr]))!=3:raise ValueError('CV training fold missing class; no random-fold fallback')
        if groups is not None and times is None and set(np.asarray(groups)[tr])&set(np.asarray(groups)[va]):raise AssertionError('Group leakage')
    return pairs

def aligned_prob(model,X):
    raw=model.predict_proba(X);p=np.zeros((len(X),3))
    p[:,np.asarray(model.classes_,dtype=int)]=raw
    if not np.isfinite(p).all() or (p< -1e-8).any() or not np.allclose(p.sum(1),1,atol=1e-7):raise AssertionError('Invalid class probabilities')
    return p

class FoldLearner(ClassifierMixin,BaseEstimator):
    def __init__(self,kind='LightGBM',catidx=(),policy='zero',seed=42,resample=False,weighted=True,calibrated=True,scale=False):
        self.kind=kind;self.catidx=catidx;self.policy=policy;self.seed=seed
        self.resample=resample;self.weighted=weighted;self.calibrated=calibrated;self.scale=scale
    def fit(self,X,y,groups=None,times=None):
        y=np.asarray(y,dtype=int);class_weights(y);self.classes_=np.arange(3);self.n_features_in_=np.asarray(X).shape[1]
        self.fit_rows_hash_=hash_values(np.asarray(X,dtype=object))
        if self.kind=='NaiveBayes' and self.calibrated:
            child=FoldLearner('NaiveBayes',self.catidx,self.policy,self.seed,False,False,False,False)
            cv=cv_splits(y,groups,times)
            try:self.estimator_=CalibratedClassifierCV(estimator=child,method='isotonic',cv=cv,ensemble=True)
            except TypeError:self.estimator_=CalibratedClassifierCV(base_estimator=child,method='isotonic',cv=cv,ensemble=True)
            self.estimator_.fit(X,y);return self
        self.prep_=TrainOnlyPreprocessor(self.catidx,self.policy);z=self.prep_.fit_transform(X)
        cw=class_weights(y,self.weighted)
        self.original_training_counts_=np.bincount(y,minlength=3).tolist();self.class_weights_=cw
        yr=y
        if self.resample:
            if np.bincount(y,minlength=3).min()<=5:raise ValueError('Fixed k=5 not feasible in this training fold; no silent k change')
            cls=SMOTENC if self.catidx else SMOTE
            kw=dict(random_state=self.seed,k_neighbors=5,sampling_strategy='not majority')
            if self.catidx:kw['categorical_features']=list(self.catidx)
            z,yr=cls(**kw).fit_resample(z,y)
            if self.catidx and not np.equal(z[:,self.catidx],np.round(z[:,self.catidx])).all():raise AssertionError('Synthetic noninteger category')
        if self.scale:
            self.scaler_=StandardScaler().fit(z);z=self.scaler_.transform(z)
        common=dict(random_state=self.seed,n_jobs=N_JOBS)
        if self.kind=='LightGBM':m=lgb.LGBMClassifier(n_estimators=300,learning_rate=.05,max_depth=6,num_leaves=31,class_weight=cw,verbose=-1,**common)
        elif self.kind=='XGBoost':m=xgb.XGBClassifier(n_estimators=300,learning_rate=.05,max_depth=6,eval_metric='mlogloss',verbosity=0,**common)
        elif self.kind=='RandomForest':m=RandomForestClassifier(n_estimators=200,class_weight='balanced',**common)
        elif self.kind=='BalancedRF':m=BalancedRandomForestClassifier(n_estimators=200,bootstrap=False,replacement=True,sampling_strategy='all',**common)
        elif self.kind=='MLP':m=MLPClassifier(hidden_layer_sizes=(128,64,32),max_iter=300,early_stopping=True,random_state=self.seed)
        elif self.kind=='LogReg':m=LogisticRegression(C=1,max_iter=2000,class_weight='balanced',random_state=self.seed)
        elif self.kind=='NaiveBayes':m=GaussianNB()
        else:raise ValueError(self.kind)
        self.estimator_=m.fit(z,yr);return self
    def predict_proba(self,X):
        if self.kind=='NaiveBayes' and self.calibrated:return aligned_prob(self.estimator_,X)
        z=self.prep_.transform(X)
        if self.scale:z=self.scaler_.transform(z)
        return aligned_prob(self.estimator_,z)
    def predict(self,X):return self.predict_proba(X).argmax(1)

class HonestStack(ClassifierMixin,BaseEstimator):
    def __init__(self,catidx=(),policy='zero',seed=42,resample=False,weighted=True,nb_calibrated=True,passthrough=True,scaled_meta=False):
        self.catidx=catidx;self.policy=policy;self.seed=seed;self.resample=resample;self.weighted=weighted
        self.nb_calibrated=nb_calibrated;self.passthrough=passthrough;self.scaled_meta=scaled_meta
    def learners(self):
        return [FoldLearner('LightGBM',self.catidx,self.policy,self.seed,self.resample,self.weighted),
                FoldLearner('RandomForest',self.catidx,self.policy,self.seed+1000,self.resample,True),
                FoldLearner('NaiveBayes',self.catidx,self.policy,self.seed,False,False,self.nb_calibrated)]
    def fit(self,X,y,groups=None,times=None):
        a=np.asarray(X,dtype=object);y=np.asarray(y,dtype=int);self.classes_=np.arange(3);self.n_features_in_=a.shape[1]
        oof=np.full((len(y),9),np.nan);self.fold_audit_=[]
        for k,(tr,va) in enumerate(cv_splits(y,groups,times)):
            for j,b in enumerate(self.learners()):
                b.fit(a[tr],y[tr],None if groups is None else np.asarray(groups)[tr],None if times is None else np.asarray(times)[tr])
                oof[va,j*3:(j+1)*3]=b.predict_proba(a[va])
                self.fold_audit_.append(dict(fold=k,learner=j,train_n=len(tr),validation_n=len(va),
                    train_indices_hash=hash_values(tr[:,None]),validation_indices_hash=hash_values(va[:,None]),
                    fitted_preprocessing_rows_hash=b.fit_rows_hash_,resampled=j<2 and self.resample))
        valid=np.isfinite(oof).all(1)
        if not valid.any():raise ValueError('No eligible OOF rows')
        z=np.column_stack([oof[valid],a[valid]]) if self.passthrough else oof[valid]
        meta_cats=tuple(9+i for i in self.catidx) if self.passthrough else ()
        self.meta_prep_=TrainOnlyPreprocessor(meta_cats,self.policy);z=self.meta_prep_.fit_transform(z)
        if self.scaled_meta:self.meta_scaler_=StandardScaler().fit(z);z=self.meta_scaler_.transform(z)
        self.meta_=LogisticRegression(C=1,max_iter=2000,class_weight=class_weights(y[valid],self.weighted),random_state=self.seed)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always');self.meta_.fit(z,y[valid]);self.meta_warnings_=[str(v.message) for v in w]
        self.meta_iteration_count_=self.meta_.n_iter_.tolist();self.meta_training_rows_=int(valid.sum())
        self.base_=[]
        for b in self.learners():self.base_.append(b.fit(a,y,groups,times))
        return self
    def predict_proba(self,X):
        a=np.asarray(X,dtype=object);p=np.column_stack([b.predict_proba(a) for b in self.base_])
        z=np.column_stack([p,a]) if self.passthrough else p;z=self.meta_prep_.transform(z)
        if self.scaled_meta:z=self.meta_scaler_.transform(z)
        return aligned_prob(self.meta_,z)
    def predict(self,X):return self.predict_proba(X).argmax(1)

def build(data,name,seed,variant='T_Full'):
    if name=='KATS':
        use=data.name in ['ITIncident','CICIDS2017'] and variant!='T_NoResampling'
        weighted=variant!='T_NoAsymLoss'
        if variant=='T_NoStacking':return FoldLearner('LightGBM',tuple(data.catidx),data.policy,seed,use,weighted)
        return HonestStack(tuple(data.catidx),data.policy,seed,use,weighted,variant!='T_NoCalibNB',variant!='NoPassthrough',variant=='ScaledMeta')
    return FoldLearner(name,tuple(data.catidx),data.policy,seed,False,name=='LightGBM',True,name in ['MLP','LogReg'])

def split_random(y,seed,groups=None,test_size=.2):
    n=len(y)
    if groups is None:return train_test_split(np.arange(n),test_size=test_size,random_state=seed,stratify=y)
    for attempt in range(50):
        tr,te=next(GroupShuffleSplit(n_splits=1,test_size=test_size,random_state=seed+attempt*7919).split(np.zeros((n,1)),y,groups))
        if np.bincount(y[tr],minlength=3).min()>=12 and np.bincount(y[te],minlength=3).min()>=1:
            if set(groups[tr])&set(groups[te]):raise AssertionError('Split group overlap')
            return tr,te
    raise ValueError('No feasible group split with all classes; no row-split fallback')
def duplicate_groups(data):return data.xgroups if len(np.unique(data.xgroups))<len(data.y) else None

def predict_rule(p,t=None):
    z=p.argmax(1)
    if t is not None:z[p[:,0]>=t]=0
    return z

def choose_and_fit(data,model,X,y,seed,groups=None,times=None):
    if model!='KATS':
        m=build(data,model,seed).fit(X,y,groups,times);return m,None,dict(rule='argmax',fallback_used=None),[]
    if times is not None:
        t=pd.to_datetime(times,utc=True);u=np.sort(t.unique());cut=u[max(1,int(len(u)*.85))]
        f=np.where(t<cut)[0];v=np.where(t>=cut)[0]
    else:f,v=split_random(y,seed,groups,.15)
    m=build(data,'KATS',seed,data.variant).fit(X[f],y[f],None if groups is None else groups[f],None if times is None else times[f])
    p=m.predict_proba(X[v]);truth=y[v]==0;floor=max(.30,1.5*truth.mean());rows=[];best_t=.5;best=-1.;eligible=0
    for t in CONFIG['threshold_grid']:
        z=p[:,0]>=t;tp=int((z&truth).sum());fp=int((z&~truth).sum());fn=int((~z&truth).sum())
        pr=tp/(tp+fp) if tp+fp else 0.;rec=tp/(tp+fn) if tp+fn else 0.;admit=pr>=floor
        if admit:
            eligible+=1
            if (pr+rec)/2>best:best=(pr+rec)/2;best_t=t
        actual=predict_rule(p,t)==0;atp=int((actual&truth).sum());afp=int((actual&~truth).sum())
        rows.append(dict(candidate=t,precision_binary_threshold=pr,recall_binary_threshold=rec,precision_floor=floor,
                         admissible=admit,actual_override_precision=atp/(atp+afp) if atp+afp else 0,
                         actual_override_recall=atp/max(1,truth.sum())))
    final=build(data,'KATS',seed,data.variant).fit(X,y,groups,times)
    info=dict(rule='argmax_plus_High_override',selected_threshold=best_t,fallback_used=eligible==0,
              n_admissible=eligible,precision_floor=floor,validation_n=len(v),selection_metric='mean_binary_High_precision_recall',
              validation_fit_rows_hash=hash_values(f[:,None]),validation_rows_hash=hash_values(v[:,None]))
    return final,best_t,info,rows

def metrics(y,p,pred):
    cm=confusion_matrix(y,pred,labels=np.arange(3));tp=cm[0,0];fn=cm[0].sum()-tp;fp=cm[:,0].sum()-tp
    bins=[];eces=[]
    for c in range(3):
        b=np.minimum((p[:,c]*10).astype(int),9);e=0.
        for j in range(10):
            ix=b==j;n=int(ix.sum());q=float(p[ix,c].mean()) if n else np.nan;r=float((y[ix]==c).mean()) if n else np.nan
            if n:e+=n/len(y)*abs(q-r)
            bins.append(dict(class_label=LABELS[c],bin=j,n=n,mean_probability=q,positive_fraction=r))
        eces.append(e)
    return dict(RecallH=float(tp/(tp+fn)) if tp+fn else np.nan,PrecH=float(tp/(tp+fp)) if tp+fp else 0.,
        MacroF1=float(f1_score(y,pred,labels=[0,1,2],average='macro',zero_division=0)),Kappa=float(cohen_kappa_score(y,pred,labels=[0,1,2])),
        PRAUC_High=float(average_precision_score(y==0,p[:,0])),Brier=float(np.mean((p-np.eye(3)[y])**2)),ECE=float(np.mean(eces)),
        FP_High=int(fp),FN_High=int(fn),TP_High=int(tp),TN_High=int(len(y)-tp-fn-fp)),cm,bins

RESULTS=[];ERRORS=[];DATA={}
def result_path(key,suffix):return CASES/(key+suffix)
def fingerprint(data,experiment,name,seed,variant,tr,te):
    spec=dict(config=CONFIG,raw=data.source_hash,matrix=data.hash,experiment=experiment,model=name,seed=seed,variant=variant,
              train=hash_values(np.asarray(tr)[:,None]),test=hash_values(np.asarray(te)[:,None]))
    return hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()[:24],spec

def restore_case(key):
    local=result_path(key,'.json')
    if local.exists():return local
    for p in INPUT.rglob(key+'.json'):
        folder=p.parent
        if not (folder/(key+'.npz')).exists():continue
        for f in folder.glob(key+'*'):
            if f.is_file():shutil.copy2(f,CASES/f.name)
        return local
    return None

def run_case(data,experiment,name,seed,tr,te,variant='T_Full',groups=None,times=None):
    key,spec=fingerprint(data,experiment,name,seed,variant,tr,te);path=restore_case(key)
    need_model=(experiment=='main' and seed==42 and name in ['KATS','LightGBM'])
    if path and result_path(key,'.npz').exists() and (not need_model or result_path(key,'.joblib').exists()):
        r=json.loads(path.read_text());RESULTS.append(r);log(f'RESUME {experiment} {data.name} {name} seed={seed} {variant}');return r
    X=data.X.to_numpy(dtype=object);data.variant=variant
    tr=np.asarray(tr);te=np.asarray(te);gtr=None if groups is None else np.asarray(groups)[tr];ttr=None if times is None else np.asarray(times)[tr]
    started=time.perf_counter();log(f'FIT {experiment} {data.name} {name} seed={seed} {variant}; train={len(tr)} test={len(te)}')
    model,t,selection,trace=choose_and_fit(data,name,X[tr],data.y[tr],seed,gtr,ttr)
    fit_seconds=time.perf_counter()-started;p=model.predict_proba(X[te]);pred=predict_rule(p,t)
    m,cm,bins=metrics(data.y[te],p,pred)
    xoverlap=len(set(data.xgroups[tr])&set(data.xgroups[te]));gov=0 if groups is None else len(set(np.asarray(groups)[tr])&set(np.asarray(groups)[te]))
    if groups is not None and times is None and gov:raise AssertionError('Requested group separation failed')
    r=dict(dataset=data.name,experiment=experiment,model=name,seed=seed,variant=variant,case_id=key,
           n_train=len(tr),n_test=len(te),threshold=t,fit_seconds_selection_plus_refit=fit_seconds,
           threshold_selection=selection,shared_predictor_hash_groups=xoverlap,group_overlap=gov,config_spec=spec,**m)
    np.savez_compressed(result_path(key,'.npz'),row_ids=data.ids[te].astype(str),train_ids=data.ids[tr].astype(str),
                         y=data.y[te],pred=pred,p=p,groups=(data.xgroups[te] if groups is None else np.asarray(groups)[te]).astype(str),class_order=np.array(LABELS))
    csv_write(result_path(key,'_calibration.csv'),pd.DataFrame(bins))
    csv_write(result_path(key,'_confusion.csv'),pd.DataFrame(cm,index=LABELS,columns=LABELS).reset_index(names='true_class'))
    csv_write(result_path(key,'_threshold_candidates.csv'),pd.DataFrame(trace))
    if isinstance(model,HonestStack):
        json_write(result_path(key,'_fold_audit.json'),dict(folds=model.fold_audit_,meta_warnings=model.meta_warnings_,
                    meta_iterations=model.meta_iteration_count_,meta_training_n=model.meta_training_rows_))
    if need_model:
        tmp=result_path(key,'.joblib.tmp');dump(model,tmp,compress=3);os.replace(tmp,result_path(key,'.joblib'))
    json_write(result_path(key,'.json'),r);RESULTS.append(r);write_progress()
    log(f'RESULT {experiment} {data.name} {name} seed={seed}: RH={m["RecallH"]:.4f} PH={m["PrecH"]:.4f} F1={m["MacroF1"]:.4f} k={m["Kappa"]:.4f} FP={m["FP_High"]} FN={m["FN_High"]} fallback={selection.get("fallback_used")}')
    return r

def safe_case(*args,**kwargs):
    try:return run_case(*args,**kwargs)
    except Exception as e:
        d=args[0];record=dict(dataset=d.name,experiment=args[1],model=args[2],seed=args[3],variant=kwargs.get('variant','T_Full'),error=repr(e))
        ERRORS.append(record);json_write(OUT/'FAILED_CASES.json',ERRORS);log(f'FAILED NOT CLOSED: {record}');return None

def write_progress():
    if not RESULTS:return
    rows=[{k:v for k,v in r.items() if not isinstance(v,(dict,list))} for r in RESULTS]
    csv_write(OUT/'all_completed_metrics_per_seed.csv',pd.DataFrame(rows).drop_duplicates('case_id'))

def same_X_groups(data):return duplicate_groups(data)
def main_stage():
    for ds,d in DATA.items():
        groups=same_X_groups(d)
        for seed in SEEDS:
            tr,te=split_random(d.y,seed,groups)
            csv_write(OUT/f'main_split_{ds}_{seed}.csv',pd.DataFrame({'row_id':d.ids,'partition':np.where(np.isin(np.arange(len(d.y)),te),'test','train')}))
            for model in MODELS:safe_case(d,'main',model,seed,tr,te,groups=groups)

def find_result(ds,experiment,model,seed,variant='T_Full'):
    rr=[r for r in RESULTS if (r['dataset'],r['experiment'],r['model'],r['seed'],r['variant'])==(ds,experiment,model,seed,variant)]
    return rr[-1] if rr else None

def ablation_stage():
    for ds,d in DATA.items():
        g=same_X_groups(d)
        for seed in SEEDS:
            tr,te=split_random(d.y,seed,g);full=find_result(ds,'main','KATS',seed)
            if not full:full=safe_case(d,'main','KATS',seed,tr,te,groups=g)
            if full:
                # T_Full is a provenance link to the SAME actual evaluated model, not a second fit.
                record={k:v for k,v in full.items() if k!='config_spec'};record.update(experiment='ablation_reference',reference_case_id=full['case_id'])
                json_write(OUT/f'ablation_reference_{ds}_{seed}.json',record)
            for variant in ['T_NoResampling','T_NoAsymLoss','T_NoCalibNB','T_NoStacking']:
                if variant=='T_NoResampling' and ds not in ['ITIncident','CICIDS2017'] and full:
                    ref={**full,'experiment':'ablation_equivalent','variant':variant,'reference_case_id':full['case_id']}
                    RESULTS.append(ref);continue
                safe_case(d,'ablation','KATS',seed,tr,te,variant=variant,groups=g)

def grouped_stage():
    d=DATA.get('GoogleCluster')
    if d is None:return
    if d.groups is None:raise ValueError('Google collection IDs unavailable')
    for seed in EXTRA_SEEDS:
        tr,te=split_random(d.y,seed,d.groups)
        for model in ['KATS','LightGBM','RandomForest']:safe_case(d,'collection_grouped',model,seed,tr,te,groups=d.groups)

def availability_stage():
    d=DATA.get('ITIncident')
    if d is None:return
    e=d.extra;six=e['six'];first=e['first']
    final6=pd.DataFrame(d.X[six]);final6['_label']=[LABELS[i] for i in d.y];final6['_id']=d.ids
    variants={'FINAL6_CAT':Data('ITIncident',final6,six,six),'FIRST_EVENT6':Data('ITIncident',first,six,six)}
    for name,v in variants.items():
        v.source_hash=d.source_hash;v.hash+=name
        for seed in EXTRA_SEEDS:
            # Same INCIDENT partition as the revised full benchmark, not separate feature-dependent splits.
            g=same_X_groups(d);tr,te=split_random(d.y,seed,g)
            for model in ['KATS','LightGBM','LogReg']:safe_case(v,'availability_'+name,model,seed,tr,te,groups=g)
    log('Availability FINAL11 reference uses MAIN revised model rows for the same seeds. FIRST_EVENT6 predicts eventual priority; no creation-time claim.')

def temporal_stage():
    d=DATA.get('ITIncident')
    if d is None:return
    t=pd.to_datetime(d.extra['final_label_information_times'],utc=True)
    if t.isna().any():
        json_write(OUT/'temporal_unavailable.json',dict(reason='Missing final information times; no invented prospective chronology.',missing=int(t.isna().sum())))
        raise ValueError('IT final information timestamps unavailable for strict forward evaluation')
    for frac in [.6,.7,.8,.9]:
        u=np.sort(t.unique());cut=u[min(len(u)-1,max(1,int(len(u)*frac)))];tr=np.where(t<cut)[0];te=np.where(t>=cut)[0]
        assert t[tr].max()<t[te].min()
        for seed in EXTRA_SEEDS:
            for model in ['KATS','LightGBM','LogReg']:
                safe_case(d,f'temporal_final_information_{frac}',model,seed,tr,te,times=t.to_numpy())
    # Early-observation forward test: training labels already known by cutoff; test observations after cutoff.
    first=d.extra['first'];ft=pd.to_datetime(first['_info_time'],utc=True)
    if ft.notna().all():
        six=d.extra['six'];v=Data('ITIncident',first,six,six);v.source_hash=d.source_hash;v.hash+='EARLY_FORWARD'
        u=np.sort(ft.unique());cut=u[max(1,int(len(u)*.8))];tr=np.where((ft<cut)&(t<cut))[0];te=np.where(ft>=cut)[0]
        for model in ['KATS','LightGBM','LogReg']:safe_case(v,'temporal_first_event_matured_labels',model,42,tr,te,times=ft.to_numpy())

# Conditional paired uncertainty: hold fitted models fixed; bootstrap test rows or predictor/entity groups.
def bootstrap_weights(groups,rng):
    u,inv=np.unique(groups,return_inverse=True);count=np.bincount(rng.integers(0,len(u),len(u)),minlength=len(u));return count[inv].astype(float)
def weighted_scores(y,p,pred,w):
    c=np.bincount(y*3+pred,weights=w,minlength=9).reshape(3,3);rs=c.sum(1);cs=c.sum(0);total=c.sum();diag=np.diag(c)
    rh=diag[0]/rs[0] if rs[0] else np.nan;ph=diag[0]/cs[0] if cs[0] else 0.
    denom=rs+cs;mf=np.mean(np.divide(2*diag,denom,out=np.zeros(3),where=denom>0))
    pe=np.dot(rs,cs)/total**2 if total else np.nan;ka=(diag.sum()/total-pe)/(1-pe) if total and pe<1 else np.nan
    br=np.dot(w,((p-np.eye(3)[y])**2).mean(1))/total;ece=[]
    for k in range(3):
        b=np.minimum((p[:,k]*10).astype(int),9);cnt=np.bincount(b,weights=w,minlength=10)
        pp=np.bincount(b,weights=w*p[:,k],minlength=10);yy=np.bincount(b,weights=w*(y==k),minlength=10)
        ece.append(np.abs(pp-yy).sum()/total)
    return np.array([rh,ph,mf,ka,br,np.mean(ece)])
def holm(p):
    p=np.asarray(p);order=np.argsort(p);out=np.empty(len(p));adj=np.maximum.accumulate((len(p)-np.arange(len(p)))*p[order]);out[order]=np.minimum(adj,1);return out

def uncertainty_stage():
    keys=['RecallH','PrecH','MacroF1','Kappa','Brier','ECE'];diffrows=[];mcrows=[];binrows=[]
    for ds in DATA:
        ref=find_result(ds,'main','KATS',2026)
        if not ref:continue
        a=np.load(result_path(ref['case_id'],'.npz'),allow_pickle=False);y=a['y'];pa=a['p'];za=a['pred'];ids=a['row_ids'];groups=a['groups']
        if not np.array_equal(a['class_order'],LABELS):raise AssertionError('Class order mismatch')
        for model in MODELS[1:]:
            rr=find_result(ds,'main',model,2026)
            if not rr:continue
            b=np.load(result_path(rr['case_id'],'.npz'),allow_pickle=False)
            if not np.array_equal(ids,b['row_ids']) or not np.array_equal(y,b['y']):raise AssertionError('Unpaired model test IDs/labels')
            pb=b['p'];zb=b['pred'];base=weighted_scores(y,pa,za,np.ones(len(y)))-weighted_scores(y,pb,zb,np.ones(len(y)))
            vals=[];rng=np.random.default_rng(2026)
            for _ in range(BOOT_N):
                w=bootstrap_weights(groups,rng);vals.append(weighted_scores(y,pa,za,w)-weighted_scores(y,pb,zb,w))
            vals=np.asarray(vals)
            for j,k in enumerate(keys):
                v=vals[np.isfinite(vals[:,j]),j];lo,hi=np.quantile(v,[.025,.975]) if len(v) else (np.nan,np.nan)
                diffrows.append(dict(dataset=ds,seed=2026,comparison='KATS-minus-'+model,metric=k,difference=base[j],ci_lower=lo,ci_upper=hi,
                                     valid_bootstraps=len(v),resampling_unit='test predictor/entity group',estimand='conditional on fixed fitted models and evaluation set; not training-seed/population CI'))
            ca=za==y;cb=zb==y;b10=int((ca&~cb).sum());b01=int((~ca&cb).sum())
            pv=binomtest(b10,b10+b01,.5).pvalue if b10+b01 else 1.
            mcrows.append(dict(dataset=ds,baseline=model,seed=2026,b10=b10,b01=b01,accuracy_difference=float(ca.mean()-cb.mean()),p_exact=pv,
                               inference_scope='exact paired-row McNemar; correlated group dependence not corrected by this test'))
        # High-bin positive-fraction CI, group bootstrap; no pooling overlapping seed tests.
        binids=np.minimum((pa[:,0]*10).astype(int),9);samples=[[] for _ in range(10)];rng=np.random.default_rng(919)
        for _ in range(BOOT_N):
            w=bootstrap_weights(groups,rng)
            for j in range(10):
                ix=binids==j;den=w[ix].sum()
                if den:samples[j].append(float(np.dot(w[ix],y[ix]==0)/den))
        for j in range(10):
            ix=binids==j;v=samples[j];lo,hi=np.quantile(v,[.025,.975]) if len(v) else (np.nan,np.nan)
            binrows.append(dict(dataset=ds,model='KATS',seed=2026,bin=j,n=int(ix.sum()),
                                mean_probability=float(pa[ix,0].mean()) if ix.any() else np.nan,
                                positive_fraction=float((y[ix]==0).mean()) if ix.any() else np.nan,
                                ci_lower=lo,ci_upper=hi,valid_bootstraps=len(v),method='test-group bootstrap, fixed predictor; seed2026 only'))
        log(f'UNCERTAINTY {ds}: paired bootstrap and seed2026 reliability bins computed')
    csv_write(OUT/'paired_conditional_test_CIs.csv',pd.DataFrame(diffrows));csv_write(OUT/'High_reliability_group_bootstrap_CIs.csv',pd.DataFrame(binrows))
    m=pd.DataFrame(mcrows)
    if len(m):m['p_holm']=holm(m.p_exact);m['family_n']=len(m);m['complete_28_comparison_family']=len(m)==28
    csv_write(OUT/'McNemar_seed2026_exact_Holm.csv',m)

# Actual revised canonical fitted model + its actual fitted first learner; no separate surrogate.
def shap_stage():
    try:import shap
    except ImportError:raise ImportError('SHAP missing; install shap and rerun KATS_STAGES=shap,latency,uncertainty; main cases resume from own checkpoints.')
    ref=find_result('ITIncident','main','KATS',42)
    if not ref:raise ValueError('Need revised MAIN ITIncident KATS seed42 case; no substitute fitted stack')
    stack=load(result_path(ref['case_id'],'.joblib'));d=DATA['ITIncident'];stored=np.load(result_path(ref['case_id'],'.npz'),allow_pickle=False)
    loc={v:i for i,v in enumerate(d.ids)};te=np.array([loc[v] for v in stored['row_ids']]);tr=np.array([loc[v] for v in stored['train_ids']])
    X=d.X.to_numpy(dtype=object);Xtr=X[tr];Xte=X[te];y=stored['y']
    if not isinstance(stack,HonestStack) or not np.allclose(stack.predict_proba(Xte),stored['p'],rtol=0,atol=1e-10):raise AssertionError('SHAP fitted-model identity failed')
    rng=np.random.default_rng(42);ix=np.concatenate([rng.choice(np.where(y==0)[0],min(SHAP_ROWS_PER_STRATUM,(y==0).sum()),replace=False),
             rng.choice(np.where(y!=0)[0],min(SHAP_ROWS_PER_STRATUM,(y!=0).sum()),replace=False)])
    # Numeric perturbation coordinates with exact inverse for categorical states.
    # Use ACTUAL training observations as background; no kmeans fractional-category background.
    prep=TrainOnlyPreprocessor(tuple(d.catidx),d.policy).fit(Xtr);numeric=prep.transform(Xtr);rows=prep.transform(Xte[ix])
    bg=numeric[rng.choice(len(numeric),min(25,len(numeric)),replace=False)]
    def decode(z):
        z=np.asarray(z,float);raw=z.astype(object)
        if d.catidx:raw[:,d.catidx]=prep.encoder_.inverse_transform(z[:,d.catidx])
        return raw
    b1=stack.base_[0]
    fs=lambda z:stack.predict_proba(decode(z))[:,0]
    fb=lambda z:b1.predict_proba(decode(z))[:,0]
    expected=stored['p'][ix,0]
    if not np.allclose(fs(rows),expected,atol=1e-10):raise AssertionError('Explanation coordinate decoder changes predictions')
    estimates=[];sv={}
    for label,fn in [('stack_probability',fs),('actual_B1_probability',fb)]:
        ex=shap.KernelExplainer(fn,bg);v=np.asarray(ex.shap_values(rows,nsamples=SHAP_NSAMPLES,l1_reg=0,silent=True))
        if v.ndim==3:v=v[:,:,0]
        if v.shape!=rows.shape:raise AssertionError('SHAP output shape mismatch')
        sv[label]=v;residual=fn(rows)-(float(np.asarray(ex.expected_value).ravel()[0])+v.sum(1))
        estimates.append(dict(function=label,mean_abs_additivity_residual=float(np.abs(residual).mean()),max_abs_additivity_residual=float(np.abs(residual).max())))
    aa,bb=sv['actual_B1_probability'],sv['stack_probability'];records=[]
    for i in range(len(ix)):
        av,bv=np.abs(aa[i]),np.abs(bb[i]);rho=float(spearmanr(av,bv).statistic)
        records.append(dict(row_id=stored['row_ids'][ix[i]],stratum='High' if y[ix[i]]==0 else 'nonHigh',rho=rho,
                            top1_match=int(av.argmax()==bv.argmax()),top3_overlap=len(set(av.argsort()[-3:])&set(bv.argsort()[-3:]))/3))
    r=pd.DataFrame(records);summary=[]
    for label,rr in [('combined_class_enriched',r)]+list(r.groupby('stratum')):
        v=rr.rho.dropna();summary.append(dict(stratum=label,n=len(rr),valid_rho=len(v),median_rho=float(v.median()),
                 rho_q25=float(v.quantile(.25)),rho_q75=float(v.quantile(.75)),top1_match=float(rr.top1_match.mean()),top3_overlap=float(rr.top3_overlap.mean())))
    gr=spearmanr(np.abs(aa).mean(0),np.abs(bb).mean(0))
    csv_write(OUT/'canonical_fitted_stack_local_fidelity.csv',r);csv_write(OUT/'canonical_fitted_stack_fidelity_summary.csv',pd.DataFrame(summary))
    csv_write(OUT/'canonical_SHAP_additivity.csv',pd.DataFrame(estimates))
    np.savez_compressed(OUT/'canonical_probability_SHAP_values.npz',B1=aa,stack=bb,row_ids=stored['row_ids'][ix],features=np.array(d.features))
    json_write(OUT/'canonical_SHAP_identity.json',dict(case_id=ref['case_id'],actual_base_learner_index=0,prediction_identity_pass=True,
        global_rho=float(gr.statistic),global_p=float(gr.pvalue),sample='class-enriched actual test rows, one seed42',
        output='P(High), NOT discrete thresholded priority decision',background='25 actual training rows',
        nsamples=SHAP_NSAMPLES,shap_version=im.version('shap'),perturbation='marginal masking in numeric coordinate representation with exact category inverse; not conditional/causal'))
    log('ACTUAL FITTED STACK FIDELITY\n'+pd.DataFrame(summary).to_string(index=False));log(estimates)

def latency_stage():
    rows=[]
    for ds,d in DATA.items():
        for name in ['KATS','LightGBM']:
            r=find_result(ds,'main',name,42)
            if not r or not result_path(r['case_id'],'.joblib').exists():continue
            model=load(result_path(r['case_id'],'.joblib'));z=np.load(result_path(r['case_id'],'.npz'),allow_pickle=False)
            loc={v:i for i,v in enumerate(d.ids)};ix=np.array([loc[v] for v in z['row_ids']]);X=d.X.iloc[ix].to_numpy(dtype=object)
            if not np.allclose(model.predict_proba(X),z['p'],atol=1e-10):raise AssertionError('Timing fitted-model identity')
            for batch in [1,32,128,200]:
                xx=X[:batch];model.predict_proba(xx);times=[]
                for k in range(30):
                    start=time.perf_counter_ns();model.predict_proba(xx);us=(time.perf_counter_ns()-start)/1000
                    times.append(us);rows.append(dict(dataset=ds,model=name,case_id=r['case_id'],seed=42,batch_size=batch,repeat=k,total_batch_us=us,us_per_item=us/batch))
                log(f'LATENCY {ds} {name} batch={batch}: median={np.median(times)/batch:.3f}us/item p95={np.percentile(times,95)/batch:.3f}us/item')
    df=pd.DataFrame(rows);csv_write(OUT/'latency_raw_repetitions.csv',df)
    if len(df):
        agg=df.groupby(['dataset','model','case_id','batch_size']).us_per_item.agg(median_us='median',mean_us='mean',sd_us='std',n_repeats='size').reset_index()
        q=df.groupby(['dataset','model','case_id','batch_size']).us_per_item.quantile(.95).reset_index(name='p95_us');agg=agg.merge(q)
        csv_write(OUT/'latency_summary.csv',agg)
    json_write(OUT/'timing_scope.json',dict(hardware=platform.platform(),cpu=platform.processor(),logical_cores=os.cpu_count(),jobs=N_JOBS,
                  call='predict_proba of EXACT saved main model; includes its fitted preprocessing, NOT raw ingestion or priority override',
                  warmups=1,repeats=30,p95='numpy percentile95 linear interpolation',units='microseconds per item AND total microseconds per batch',
                  threads={k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']}))

def finish():
    write_progress();flat=pd.DataFrame([{k:v for k,v in r.items() if not isinstance(v,(dict,list))} for r in RESULTS]).drop_duplicates(['experiment','dataset','model','seed','variant']) if RESULTS else pd.DataFrame()
    if len(flat):
        ks=['RecallH','PrecH','MacroF1','Kappa','PRAUC_High','Brier','ECE','FP_High','FN_High']
        su=flat.groupby(['experiment','dataset','model','variant'])[ks].agg(['mean','std']).reset_index();su.columns=['_'.join(x).rstrip('_') if isinstance(x,tuple) else x for x in su.columns]
        csv_write(OUT/'all_experiment_mean_sd.csv',su)
        log('CURRENT MEAN/SD TABLE\n'+su.to_string(index=False))
    main=[r for r in RESULTS if r['experiment']=='main'];expected=len(DATASETS)*len(SEEDS)*len(MODELS)
    status=dict(version=VERSION,completed_main_cases=len({r['case_id'] for r in main}),expected_main_cases=expected,
                main_complete=len({r['case_id'] for r in main})==expected,failed_cases=ERRORS,
                canonical_main_and_ablation_reference='same case_id; no independent T_Full means overwritten',
                old_results='untouched; R1 revised results require a new manuscript source designation',
                limitations=['cleaned CICIDS session IDs unavailable','retrospective labels and engineered QoS tiers are not validated operational urgency',
                             'no SLA/queueing simulation','conditional test bootstrap is not training-seed uncertainty','exact row McNemar assumes independent paired units; grouping dependence disclosed'],
                statistical_protocol=CONFIG)
    json_write(OUT/'FINAL_STATUS.json',status)
    # Checkpoints persist only if the user saves/downloads output or reattaches it as Kaggle input.
    archive=shutil.make_archive(str(OUT)+'_RESULTS','zip',OUT)
    log(f'ARCHIVE {archive}; STATUS main_complete={status["main_complete"]}; failed={len(ERRORS)}')
    try:
        from IPython.display import display,FileLink
        display(FileLink(archive))
    except ImportError:pass

def run():
    json_write(OUT/'run_configuration.json',CONFIG)
    if '__file__' in globals() and Path(__file__).is_file():shutil.copy2(__file__,OUT/'generating_code.py')
    for ds in DATASETS:
        if ds not in BASENAMES:raise ValueError(ds)
        DATA[ds]=load_dataset(ds)
    stages=[('main',main_stage),('ablation',ablation_stage),('grouped',grouped_stage),('availability',availability_stage),
            ('temporal',temporal_stage),('uncertainty',uncertainty_stage),('shap',shap_stage),('latency',latency_stage)]
    for name,fn in stages:
        if name not in STAGES:continue
        try:
            log('STAGE '+name);fn();finish()
        except Exception as e:
            ERRORS.append(dict(stage=name,error=repr(e)));json_write(OUT/'FAILED_CASES.json',ERRORS);log(f'STAGE {name} NOT CLOSED: {e}');finish()
    finish()
if __name__=='__main__':run()
