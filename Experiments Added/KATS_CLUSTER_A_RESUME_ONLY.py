# ============================================================
# KATS CLUSTER A — RESUME ONLY
# Self-contained Kaggle script. It preserves every non-empty
# existing output and runs only analyses whose CSV is missing.
# It writes atomic checkpoints after each completed unit.
# ============================================================
import os, re, ast, json, pickle, warnings
from pathlib import Path
from datetime import datetime
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, cohen_kappa_score, brier_score_loss, average_precision_score, confusion_matrix, recall_score
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

SEEDS=[42,7,13,99,2026]; TEMPORAL_SEEDS=[42,7,13]; IR_LEVELS=[2,5,10,20,30]
MAX_ROWS=60000; N_JOBS=1; IR_THRESHOLD=3.0
OUT=Path('/kaggle/working/KATS_CLUSTER_A_FINAL'); CKPT=OUT/'checkpoints'
OUT.mkdir(parents=True,exist_ok=True); CKPT.mkdir(parents=True,exist_ok=True)
IT_PATH='/kaggle/input/datasets/shamiulislamshifat/it-incident-log-dataset/incident_event_log.csv'
GC_PATH='/kaggle/input/datasets/derrickmwiti/google-2019-cluster-sample/borg_traces_data.csv'
MC_PATH='/kaggle/input/datasets/ziya07/multi-cloud-service-composition-dataset/multi_cloud_service_dataset.csv'
CIC_PATH='/kaggle/input/datasets/ericanacletoribeiro/cicids2017-cleaned-and-preprocessed/cicids2017_cleaned.csv'

def log(x): print(f'[{datetime.now().strftime("%H:%M:%S")}] {x}',flush=True)
def norm(x): return re.sub(r'[^a-z0-9]','',str(x).lower())
def col(d,*names,required=True):
    z={norm(x):x for x in d.columns}
    for n in names:
        if norm(n) in z:return z[norm(n)]
    for n in names:
        k=norm(n)
        for kk,v in z.items():
            if k in kk or kk in k:return v
    if required:raise KeyError(f'Cannot locate {names}; columns={list(d.columns)}')
    return None
def exists(name):
    p=OUT/name
    return p.exists() and p.stat().st_size>0
def save_csv(d,name):
    p=OUT/name; t=OUT/(name+'.tmp'); d.to_csv(t,index=False); os.replace(t,p)
def cp(name,fn):
    p=CKPT/(re.sub(r'[^A-Za-z0-9_.-]','_',name)+'.pkl')
    if p.exists():
        log('RESUME '+name)
        with open(p,'rb') as f:return pickle.load(f)
    z=fn(); t=str(p)+'.tmp'
    with open(t,'wb') as f:pickle.dump(z,f)
    os.replace(t,p); return z
def labels(s):
    le=LabelEncoder(); y=le.fit_transform(s.astype(str)); return y,le,int(np.where(le.classes_=='High')[0][0])
def ir(y):
    _,n=np.unique(y,return_counts=True); return float(n.max()/n.min())
def cw(y,h,alpha=5):
    c,n=np.unique(y,return_counts=True); z={int(a):float(len(y)/(len(c)*b)) for a,b in zip(c,n)}
    if n.max()/n.min()>IR_THRESHOLD:z[int(h)]*=alpha
    return z
def cap(d,label):
    if len(d)<=MAX_ROWS:return d.reset_index(drop=True)
    q=MAX_ROWS/len(d);return pd.concat([x.sample(frac=q,random_state=42) for _,x in d.groupby(label)]).reset_index(drop=True)
def Xmat(d,f):return d[f].replace([np.inf,-np.inf],np.nan).fillna(0).astype(float)
def ece(y,p,bins=10):
    e=np.linspace(0,1,bins+1); out=0; rows=[]
    for b in range(bins):
        m=(p>=e[b])&((p<=e[b+1]) if b==bins-1 else (p<e[b+1])); n=int(m.sum())
        if not n:rows.append({'bin':b,'count':0,'mean_confidence':np.nan,'empirical_frequency':np.nan});continue
        cf=float(p[m].mean()); ac=float(y[m].mean()); out+=n/len(y)*abs(ac-cf);rows.append({'bin':b,'count':n,'mean_confidence':cf,'empirical_frequency':ac})
    return float(out),rows
def metric(y,pred,p,le,h):
    r=classification_report(y,pred,target_names=le.classes_.tolist(),output_dict=True,zero_division=0); bs=[]; es=[]; br=[]
    for c in range(len(le.classes_)):
        yy=(y==c).astype(int);bs.append(brier_score_loss(yy,p[:,c]));a,b=ece(yy,p[:,c]);es.append(a)
        br += [dict(x,class_index=c) for x in b]
    yh=(y==h).astype(int); ph=(pred==h).astype(int)
    return {'RecallH':float(r['High']['recall']),'PrecH':float(r['High']['precision']),'F1H':float(r['High']['f1-score']),'MacroF1':float(r['macro avg']['f1-score']),'Kappa':float(cohen_kappa_score(y,pred)),'Brier':float(np.mean(bs)),'ECE':float(np.mean(es)),'PRAUC_High':float(average_precision_score(yh,p[:,h])),'FP_High':int(((ph==1)&(yh==0)).sum()),'FN_High':int(((ph==0)&(yh==1)).sum()),'bins':br,'cm':confusion_matrix(y,pred).tolist()}
def sampler(ds,f,cats,s,on=True):
    if not on:return 'passthrough'
    if ds=='ITIncident':return SMOTENC(categorical_features=[f.index(x) for x in cats if x in f],random_state=s,k_neighbors=5,sampling_strategy='not majority')
    if ds=='CICIDS2017':return SMOTE(random_state=s,k_neighbors=5,sampling_strategy='not majority')
    return 'passthrough'
def kats(ds,f,cats,w,s,smote=True,asym=True,cal=True,stack=True):
    if not asym:w={k:1.0 for k in w}
    def wrap(m,off=0):return ImbPipeline([('resample',sampler(ds,f,cats,s+off,smote)),('model',m)])
    a=wrap(lgb.LGBMClassifier(n_estimators=300,learning_rate=.05,max_depth=6,num_leaves=31,class_weight=w,random_state=s,n_jobs=N_JOBS,verbose=-1))
    if not stack:return a
    b=wrap(RandomForestClassifier(n_estimators=200,class_weight='balanced',random_state=s,n_jobs=N_JOBS),1000)
    c=CalibratedClassifierCV(GaussianNB(),cv=3,method='isotonic') if cal else GaussianNB()
    return StackingClassifier(estimators=[('lgb',a),('rf',b),('nb',c)],final_estimator=LogisticRegression(C=1,max_iter=2000,class_weight=w,random_state=s),stack_method='predict_proba',passthrough=True,cv=3,n_jobs=N_JOBS)
def bases(w,s):
    return {'LightGBM':lgb.LGBMClassifier(n_estimators=300,learning_rate=.05,max_depth=6,class_weight=w,random_state=s,n_jobs=N_JOBS,verbose=-1),'XGBoost':xgb.XGBClassifier(n_estimators=300,learning_rate=.05,max_depth=6,eval_metric='mlogloss',random_state=s,n_jobs=N_JOBS,verbosity=0),'RandomForest':RandomForestClassifier(n_estimators=200,class_weight='balanced',random_state=s,n_jobs=N_JOBS),'BalancedRF':BalancedRandomForestClassifier(n_estimators=200,random_state=s,n_jobs=N_JOBS),'MLP':Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler()),('model',MLPClassifier(hidden_layer_sizes=(128,64,32),max_iter=300,early_stopping=True,random_state=s))]),'LogReg':Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler()),('model',LogisticRegression(C=1,max_iter=2000,class_weight='balanced',random_state=s))]),'NaiveBayes':CalibratedClassifierCV(GaussianNB(),cv=3,method='isotonic')}
def fit_threshold(m,X,y,h,s):
    xf,xv,yf,yv=train_test_split(X,y,test_size=.15,random_state=s,stratify=y);m.fit(xf,yf);p=m.predict_proba(xv)[:,h];yy=(yv==h).astype(int);floor=max(.30,1.5*yy.mean()); bt=.5;score=-1
    for t in np.arange(.15,.86,.05):
        z=(p>=t).astype(int);tp=((z==1)&(yy==1)).sum();fp=((z==1)&(yy==0)).sum();fn=((z==0)&(yy==1)).sum();pr=tp/(tp+fp) if tp+fp else 0;rc=tp/(tp+fn) if tp+fn else 0
        if pr>=floor and (pr+rc)/2>score:score=(pr+rc)/2;bt=float(t)
    m.fit(X,y);return m,bt
def thresh(p,h,t):
    z=np.argmax(p,axis=1);z[p[:,h]>=t]=h;return z

# ----- data loaders -----
def load_it():
    r=pd.read_csv(IT_PATH,low_memory=False);num=col(r,'number');mod=col(r,'sys_mod_count');op=col(r,'opened_at');pr=col(r,'priority');d=r.sort_values([num,mod]).groupby(num,group_keys=False).tail(1).copy();d['opened_at_final']=pd.to_datetime(d[op],errors='coerce');d['priority_label']=d[pr].map({'1 - Critical':'High','2 - High':'High','3 - Moderate':'Medium','4 - Low':'Low'});d=d.dropna(subset=['opened_at_final','priority_label']).copy();cats=[]
    for src,dst in {'category':'category_enc','location':'location_enc','contact_type':'contact_type_enc','assignment_group':'assignment_group_enc','cmdb_ci':'cmdb_ci_enc','subcategory':'subcategory_enc','knowledge':'knowledge_enc'}.items():
        q=col(d,src,required=False)
        if q is not None:d[dst]=LabelEncoder().fit_transform(d[q].astype(str));cats.append(dst)
    rp=col(d,'reopen_count');d['reopen_flag']=(d[rp]>0).astype(int);cats.append('reopen_flag');f=[col(d,'reassignment_count'),col(d,'reopen_count'),col(d,'sys_mod_count')]+cats
    save_csv(pd.DataFrame({'feature':f,'type':['categorical' if x in cats else 'numeric' for x in f]}),'features_ITIncident.csv');return d,f,cats
def load_gc():
    d=pd.read_csv(GC_PATH,low_memory=False)
    def get(s,k):
        def q(x):
            try:
                a=ast.literal_eval(str(x));return a.get(k,np.nan) if isinstance(a,dict) else np.nan
            except:return np.nan
        return s.apply(q)
    for src,prefix in [('resource_request','req'),('average_usage','avg'),('maximum_usage','max')]:
        q=col(d,src,required=False)
        for k in ['cpus','memory']:d[prefix+k]=get(d[q],k) if q else np.nan
    p=col(d,'priority');d['priority_label']=d[p].apply(lambda x:'Low' if x<100 else ('Medium' if x<200 else 'High'));e=col(d,'event',required=False);d['event_enc']=LabelEncoder().fit_transform(d[e].astype(str)) if e else 0;f=[]
    for n in ['scheduling_class','collection_type','instance_index','assigned_memory','page_cache_memory','cycles_per_instruction','memory_accesses_per_instruction','sample_rate','scheduler','vertical_scaling','reqcpus','reqmemory','avgcpus','avgmemory','maxcpus','maxmemory','failed']:
        q=col(d,n,required=False)
        if q:
            if pd.api.types.is_numeric_dtype(d[q]):d[q]=d[q].replace([np.inf,-np.inf],np.nan).fillna(d[q].median())
            f.append(q)
    f.append('event_enc');save_csv(pd.DataFrame({'feature':f,'type':'numeric'}),'features_GoogleCluster.csv');return cap(d,'priority_label'),f
def load_mc():
    d=pd.read_csv(MC_PATH);a,b,c=col(d,'ServiceType'),col(d,'CloudProvider'),col(d,'EdgeNodeID');d['service_type_enc']=LabelEncoder().fit_transform(d[a].astype(str));d['cloud_provider_enc']=LabelEncoder().fit_transform(d[b].astype(str));d['edge_node_enc']=LabelEncoder().fit_transform(d[c].astype(str));u,l,t,n,w=[col(d,x) for x in ['CPUUtilization','ServiceLatency','Throughput','NetworkBandwidth','WorkloadVariability']];z=.30*d[u]/100+.25*d[l]/d[l].max()+.20*(1-d[t]/d[t].max())+.15*(1-d[n]/d[n].max())+.10*d[w]/d[w].max();d['priority_label']=pd.qcut(z,3,labels=['Low','Medium','High']).astype(str);f=[col(d,x,required=False) for x in ['MemoryUsage','StorageUsage','ResponseTime','LoadBalancing','OptimalServicePlacement']];f=[x for x in f if x]+['service_type_enc','cloud_provider_enc','edge_node_enc'];cats=['service_type_enc','cloud_provider_enc','edge_node_enc'];save_csv(pd.DataFrame({'feature':f,'type':['categorical' if x in cats else 'numeric' for x in f]}),'features_MultiCloud.csv');return d,f,cats
def load_cic():
    d=pd.read_csv(CIC_PATH,low_memory=False);d.columns=[x.strip().lower().replace(' ','_') for x in d.columns];q=col(d,'attack_type','label');d['priority_label']=d[q].apply(lambda x:'Low' if ('benign' in str(x).lower() or 'normal' in str(x).lower()) else ('Medium' if ('scan' in str(x).lower() or 'patator' in str(x).lower() or 'brute' in str(x).lower()) else 'High'));d=pd.concat([x.sample(frac=.05,random_state=42) for _,x in d.groupby('priority_label')]).reset_index(drop=True);f=[x for x in d if x not in {q,'priority_label'} and pd.api.types.is_numeric_dtype(d[x])];d[f]=d[f].replace([np.inf,-np.inf],np.nan).fillna(0);save_csv(pd.DataFrame({'feature':f,'type':'numeric'}),'features_CICIDS2017.csv');return cap(d,'priority_label'),f

# ----- load once; feature files are regenerated only if missing -----
log('Loading data for missing stages only')
it,ITF,ITC=load_it();gc,GCF=load_gc();mc,MCF,MCC=load_mc();cic,CICF=load_cic();DATA={'GoogleCluster':(gc,GCF,[]),'ITIncident':(it,ITF,ITC),'MultiCloud':(mc,MCF,MCC),'CICIDS2017':(cic,CICF,[])}

# ----- rebuild only missing main outputs from saved checkpoints; fit only if checkpoint absent -----
main_names=['clusterA_main_metrics_per_seed.csv','clusterA_predictions.csv','clusterA_calibration_bins.csv','clusterA_confusion_matrices.csv','clusterA_metrics_mean_sd.csv']
if any(not exists(x) for x in main_names):
    log('Recovering missing main output files')
    mr=[];pr=[];br=[];cr=[]
    for ds,(d,f,cats) in DATA.items():
        X=Xmat(d,f);y,le,h=labels(d.priority_label)
        for s in SEEDS:
            def work(ds=ds,X=X,y=y,le=le,h=h,f=f,cats=cats,s=s):
                xt,xv,yt,yv,ii,jj=train_test_split(X,y,np.arange(len(y)),test_size=.2,random_state=s,stratify=y);w=cw(yt,h);ms={'KATS':kats(ds,f,cats,w,s)};ms.update(bases(w,s));o={}
                for n,m in ms.items():
                    if n=='KATS':m,t=fit_threshold(m,xt,yt,h,s)
                    else:m.fit(xt,yt);t=.5
                    p=m.predict_proba(xv);z=thresh(p,h,t) if n=='KATS' else m.predict(xv);o[n]={'m':metric(yv,z,p,le,h),'p':p,'z':z,'y':yv,'ids':jj,'t':t}
                return o
            o=cp(f'main_{ds}_{s}',work)
            for n,v in o.items():
                q=v['m'];mr.append({'dataset':ds,'seed':s,'model':n,'threshold':v['t'],**{k:x for k,x in q.items() if k not in {'bins','cm'}}})
                for i in range(len(v['y'])):pr.append({'dataset':ds,'seed':s,'model':n,'test_row_id':int(v['ids'][i]),'true_label':int(v['y'][i]),'predicted_label':int(v['z'][i]),'prob_class_0':float(v['p'][i,0]),'prob_class_1':float(v['p'][i,1]),'prob_class_2':float(v['p'][i,2]),'selected_high_threshold':v['t']})
                br += [{'dataset':ds,'seed':s,'model':n,**x} for x in q['bins']]
                a=np.array(q['cm'])
                for i in range(a.shape[0]):
                    for j in range(a.shape[1]):cr.append({'dataset':ds,'seed':s,'model':n,'true_class':le.classes_[i],'predicted_class':le.classes_[j],'count':int(a[i,j])})
    ma=pd.DataFrame(mr);save_csv(ma,'clusterA_main_metrics_per_seed.csv');save_csv(pd.DataFrame(pr),'clusterA_predictions.csv');save_csv(pd.DataFrame(br),'clusterA_calibration_bins.csv');save_csv(pd.DataFrame(cr),'clusterA_confusion_matrices.csv');keys=['RecallH','PrecH','F1H','MacroF1','Kappa','Brier','ECE','PRAUC_High','FP_High','FN_High'];su=ma.groupby(['dataset','model']).agg(**{f'{k}_mean':(k,'mean') for k in keys},**{f'{k}_sd':(k,'std') for k in keys}).reset_index();save_csv(su,'clusterA_metrics_mean_sd.csv')
else: log('Main benchmark files already present: skipped')

# ----- calibrated ablation -----
if not (exists('clusterA_ablation_per_seed.csv') and exists('clusterA_ablation_summary.csv')):
    log('Running missing ablation stage')
    rows=[];vs={'T_Full':(True,True,True,True),'T_NoSMOTE':(False,True,True,True),'T_NoAsymLoss':(True,False,True,True),'T_NoCalibNB':(True,True,False,True),'T_NoStacking':(True,True,True,False)}
    for ds,(d,f,cats) in DATA.items():
        X=Xmat(d,f);y,le,h=labels(d.priority_label)
        for s in SEEDS:
            xt,xv,yt,yv=train_test_split(X,y,test_size=.2,random_state=s,stratify=y);w=cw(yt,h)
            for vn,cfg in vs.items():
                def job(cfg=cfg):
                    m,t=fit_threshold(kats(ds,f,cats,w,s,*cfg),xt,yt,h,s);p=m.predict_proba(xv);z=thresh(p,h,t);return {'t':t,'m':metric(yv,z,p,le,h)}
                q=cp(f'abl_{ds}_{vn}_{s}',job);rows.append({'dataset':ds,'seed':s,'variant':vn,'threshold':q['t'],**{k:v for k,v in q['m'].items() if k not in {'bins','cm'}}})
    a=pd.DataFrame(rows);save_csv(a,'clusterA_ablation_per_seed.csv');z=a.groupby(['dataset','variant']).agg(RecallH_mean=('RecallH','mean'),RecallH_sd=('RecallH','std'),MacroF1_mean=('MacroF1','mean'),MacroF1_sd=('MacroF1','std'),Kappa_mean=('Kappa','mean'),Kappa_sd=('Kappa','std')).reset_index()
    for ds in z.dataset.unique():
        ref=z[(z.dataset==ds)&(z.variant=='T_Full')].iloc[0];m=z.dataset==ds;z.loc[m,'Delta_RecallH']=z.loc[m,'RecallH_mean']-ref.RecallH_mean;z.loc[m,'Delta_MacroF1']=z.loc[m,'MacroF1_mean']-ref.MacroF1_mean
    save_csv(z,'clusterA_ablation_summary.csv')
else: log('Ablation files already present: skipped')

# ----- temporal sweep -----
if not exists('clusterA_temporal_sweep.csv'):
    log('Running missing temporal sweep')
    rows=[];d=it.sort_values('opened_at_final').reset_index(drop=True);X=Xmat(d,ITF);y,le,h=labels(d.priority_label)
    for frac in [.6,.7,.8,.9]:
        n=int(len(d)*frac);xt,xv,yt,yv=X.iloc[:n],X.iloc[n:],y[:n],y[n:];assert d.opened_at_final.iloc[:n].max()<=d.opened_at_final.iloc[n:].min()
        for s in TEMPORAL_SEEDS:
            w=cw(yt,h);ms={'KATS':kats('ITIncident',ITF,ITC,w,s),'LightGBM':bases(w,s)['LightGBM'],'LogReg':bases(w,s)['LogReg']}
            for mn,m in ms.items():
                def job(mn=mn,m=m):
                    if mn=='KATS':mm,t=fit_threshold(m,xt,yt,h,s)
                    else:mm=m;mm.fit(xt,yt);t=.5
                    p=mm.predict_proba(xv);z=thresh(p,h,t) if mn=='KATS' else mm.predict(xv);return {'t':t,'m':metric(yv,z,p,le,h)}
                q=cp(f'temporal_{frac}_{s}_{mn}',job);rows.append({'train_fraction':frac,'seed':s,'model':mn,'threshold':q['t'],**{k:v for k,v in q['m'].items() if k not in {'bins','cm'}}})
    save_csv(pd.DataFrame(rows),'clusterA_temporal_sweep.csv')
else: log('Temporal sweep already present: skipped')

# ----- matched recall -----
if not exists('clusterA_matched_operating_points.csv'):
    log('Running missing matched operating-point stage')
    rows=[]
    for ds in ['ITIncident','CICIDS2017']:
        d,f,cats=DATA[ds];X=Xmat(d,f);y,le,h=labels(d.priority_label)
        for s in SEEDS:
            xo,xv,yo,yv=train_test_split(X,y,test_size=.2,random_state=s,stratify=y);xf,xval,yf,yval=train_test_split(xo,yo,test_size=.15,random_state=s,stratify=yo);w=cw(yf,h);ms={'KATS':kats(ds,f,cats,w,s),'LightGBM':bases(w,s)['LightGBM'],'XGBoost':bases(w,s)['XGBoost'],'LogReg':bases(w,s)['LogReg']}
            km=ms['KATS'];km.fit(xf,yf);kp=km.predict_proba(xval);target=np.clip(recall_score(yval==h,thresh(kp,h,.5)==h,zero_division=0),.80,.95)
            for mn,m in ms.items():
                def job(mn=mn,m=m):
                    if mn!='KATS':m.fit(xf,yf)
                    pv=m.predict_proba(xval);ts=np.arange(.05,.96,.01);t=float(min(ts,key=lambda a:abs(recall_score(yval==h,pv[:,h]>=a,zero_division=0)-target)));m.fit(xo,yo);p=m.predict_proba(xv);z=thresh(p,h,t);return {'t':t,'m':metric(yv,z,p,le,h)}
                q=cp(f'matched_{ds}_{s}_{mn}',job);rows.append({'dataset':ds,'seed':s,'model':mn,'target_recall':target,'threshold':q['t'],**{k:v for k,v in q['m'].items() if k not in {'bins','cm'}}})
    save_csv(pd.DataFrame(rows),'clusterA_matched_operating_points.csv')
else: log('Matched operating-point file already present: skipped')

# ----- fixed-test IR sensitivity -----
if not exists('clusterA_ir_fixed_test_sensitivity.csv'):
    log('Running missing fixed-test IR sensitivity stage')
    rows=[];X=Xmat(cic,CICF);y,le,h=labels(cic.priority_label)
    for s in SEEDS:
        xp,xv,yp,yv=train_test_split(X,y,test_size=.2,random_state=s,stratify=y);ix={c:np.where(yp==c)[0] for c in np.unique(yp)};nn={c:len(v) for c,v in ix.items()};major=max(nn,key=nn.get);minor=min(nn,key=nn.get);middle=[c for c in nn if c not in {major,minor}][0];rng=np.random.RandomState(s)
        for level in IR_LEVELS:
            a=3000;b=int(round(a/level));c=6000-a-b
            if min(a/nn[major],b/nn[minor],c/nn[middle])>1:raise ValueError(f'IR construction unavailable: seed={s}, IR={level}')
            take=np.concatenate([rng.choice(ix[major],a,False),rng.choice(ix[middle],c,False),rng.choice(ix[minor],b,False)]);rng.shuffle(take);xt=xp.iloc[take];yt=yp[take];w=cw(yt,h);ms={'KATS':kats('CICIDS2017',CICF,[],w,s),'LightGBM':bases(w,s)['LightGBM'],'XGBoost':bases(w,s)['XGBoost'],'LogReg':bases(w,s)['LogReg']}
            for mn,m in ms.items():
                def job(mn=mn,m=m):
                    if mn=='KATS':mm,t=fit_threshold(m,xt,yt,h,s)
                    else:mm=m;mm.fit(xt,yt);t=.5
                    p=mm.predict_proba(xv);z=thresh(p,h,t) if mn=='KATS' else mm.predict(xv);return {'t':t,'m':metric(yv,z,p,le,h)}
                q=cp(f'ir_{s}_{level}_{mn}',job);rows.append({'seed':s,'training_target_ir':level,'training_achieved_ir':ir(yt),'training_n':len(yt),'fixed_test_n':len(yv),'model':mn,'threshold':q['t'],**{k:v for k,v in q['m'].items() if k not in {'bins','cm'}}})
    save_csv(pd.DataFrame(rows),'clusterA_ir_fixed_test_sensitivity.csv')
else: log('Fixed-test IR sensitivity file already present: skipped')

required=['clusterA_main_metrics_per_seed.csv','clusterA_metrics_mean_sd.csv','clusterA_predictions.csv','clusterA_calibration_bins.csv','clusterA_confusion_matrices.csv','clusterA_ablation_per_seed.csv','clusterA_ablation_summary.csv','clusterA_temporal_sweep.csv','clusterA_matched_operating_points.csv','clusterA_ir_fixed_test_sensitivity.csv','features_ITIncident.csv','features_GoogleCluster.csv','features_MultiCloud.csv','features_CICIDS2017.csv']
checks={'it_features_11':len(ITF)==11,'main_160':exists('clusterA_main_metrics_per_seed.csv') and len(pd.read_csv(OUT/'clusterA_main_metrics_per_seed.csv'))==160,'ablation_100':exists('clusterA_ablation_per_seed.csv') and len(pd.read_csv(OUT/'clusterA_ablation_per_seed.csv'))==100,'temporal_36':exists('clusterA_temporal_sweep.csv') and len(pd.read_csv(OUT/'clusterA_temporal_sweep.csv'))==36,'matched_40':exists('clusterA_matched_operating_points.csv') and len(pd.read_csv(OUT/'clusterA_matched_operating_points.csv'))==40,'ir_100':exists('clusterA_ir_fixed_test_sensitivity.csv') and len(pd.read_csv(OUT/'clusterA_ir_fixed_test_sensitivity.csv'))==100,'predictions_saved':exists('clusterA_predictions.csv'),'all_required_outputs_present':all(exists(x) for x in required)}
with open(OUT/'clusterA_validation_checks.json','w') as f:json.dump(checks,f,indent=2)
log('COMPLETE');print(json.dumps(checks,indent=2));print(OUT)
