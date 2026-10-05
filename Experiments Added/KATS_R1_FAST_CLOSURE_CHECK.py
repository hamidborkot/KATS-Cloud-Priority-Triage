# Paste this entire script into one Kaggle notebook cell.
from pathlib import Path
import hashlib, io, json, re, zipfile, traceback
import numpy as np
import pandas as pd
from scipy.stats import binomtest

ZIP_PATH = Path('/kaggle/working/KATS_REVIEWER_REPAIR_R1_RESULTS.zip')
OUT = Path('/kaggle/working/KATS_R1_CLOSURE_CHECK')
OUT.mkdir(parents=True, exist_ok=True)
MAIN_CSV = None         # Set to an archive member name ONLY if reported candidates are ambiguous.
PREDICTIONS_CSV = None  # Same rule. Do not select an old archive/run.
SEEDS = {42, 7, 13, 99, 2026}
DATASETS = {'GoogleCluster', 'ITIncident', 'MultiCloud', 'CICIDS2017'}
MODELS = {'KATS', 'LightGBM', 'XGBoost', 'RandomForest', 'BalancedRF', 'MLP', 'LogReg', 'NaiveBayes'}
TOL = 2e-6
checks, inventory, failure_records, meta_records = [], [], [], []
tables, roles = {}, {}

def say(s): print(s, flush=True)
def check(name, state, detail):
    checks.append({'check': name, 'state': state, 'detail': str(detail)})
    say(f'[{state}] {name}: {detail}')
def key(s): return re.sub(r'[^a-z0-9]', '', str(s).lower())
ALIASES = {
 'experiment': ['experiment','experiment_name'], 'dataset': ['dataset','dataset_name'],
 'model': ['model','model_name','classifier'], 'variant': ['variant','ablation_variant'],
 'seed': ['seed','random_seed','random_state'],
 'row_id': ['test_row_id','row_id','source_row_id','sample_id','test_id'],
 'y_true': ['y_true','true_label','true_class','actual_label'],
 'y_pred': ['y_pred','predicted_label','predicted_class','prediction'],
 'RecallH': ['RecallH','Recall_H','RecallHigh','recall_high'],
 'PrecH': ['PrecH','Precision_H','PrecisionH','PrecHigh','precision_high'],
 'MacroF1': ['MacroF1','Macro_F1','F1_macro','macro_f1'],
 'Kappa': ['Kappa','cohen_kappa','cohens_kappa'],
 'FP_High': ['FP_High','FP_H','fp_high'], 'FN_High': ['FN_High','FN_H','fn_high'],
 'PRAUC_High': ['PRAUC_High','PRAUC_H','PR_AUC_H','AveragePrecision_High'],
 'Brier': ['Brier','Brier_score'], 'ECE': ['ECE','ece_score'],
 'threshold': ['threshold','selected_high_threshold','high_threshold','t_high']}

def normalise(df):
    df = df.copy()
    lookup = {key(c): c for c in df.columns}
    ren = {}
    for target, names in ALIASES.items():
        hits = [lookup[key(n)] for n in names if key(n) in lookup]
        hits = list(dict.fromkeys(hits))
        if target in df: continue
        if len(hits) == 1: ren[hits[0]] = target
    df = df.rename(columns=ren)
    if 'model' in df:
        mm = {'logisticregression':'LogReg','logreg':'LogReg','randomforest':'RandomForest',
              'randomforestclassifier':'RandomForest','balancedrf':'BalancedRF',
              'balancedrandomforest':'BalancedRF','naivebayes':'NaiveBayes','gaussiannb':'NaiveBayes',
              'lightgbm':'LightGBM','xgboost':'XGBoost','mlp':'MLP','kats':'KATS'}
        df['model'] = df.model.map(lambda x: mm.get(key(x), str(x)))
    if 'seed' in df: df['seed'] = pd.to_numeric(df.seed, errors='coerce')
    for c in ['RecallH','PrecH','MacroF1','Kappa','FP_High','FN_High','PRAUC_High','Brier','ECE','threshold']:
        if c in df: df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

def main_rows(df):
    x = df.copy()
    if 'experiment' in x:
        x = x[x.experiment.map(key).isin({'main','mainclassification','mainevaluation','mainbenchmark'})]
    if 'variant' in x:
        x = x[x.variant.map(key).isin({'tfull','full','katscurrent','current','none','nan',''})]
    return x

def digest_frame(df, cols, sort):
    x = df[cols].sort_values(sort).reset_index(drop=True)
    return hashlib.sha256(x.to_csv(index=False).encode()).hexdigest()

def task_info(obj, source, loc='$'):
    if isinstance(obj, dict):
        norm = {key(k): v for k, v in obj.items()}
        if any(k in norm for k in ['error','exception','traceback','errmsg','failuremessage']):
            failure_records.append({'source': source, 'location': loc,
                                    'record': json.dumps(obj, default=str, ensure_ascii=False)})
        for k, v in obj.items():
            nk = key(k)
            if nk in {'failed','failures','errors','failedtasks','failurecount'}:
                meta_records.append({'source':source,'location':loc+'.'+str(k),'value':json.dumps(v,default=str)})
                if isinstance(v, list):
                    for i, item in enumerate(v):
                        failure_records.append({'source':source,'location':f'{loc}.{k}[{i}]',
                                                'record':json.dumps(item,default=str,ensure_ascii=False)})
            if nk in {'runid','pipelineversion','codehash','rawsha256','classorder','learnedpreprocessing'}:
                meta_records.append({'source':source,'location':loc+'.'+str(k),'value':json.dumps(v,default=str)})
            task_info(v, source, loc+'.'+str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj): task_info(v, source, loc+f'[{i}]')

if not ZIP_PATH.is_file():
    raise FileNotFoundError(f'Archive does not exist in this Kaggle session: {ZIP_PATH}')
sha = hashlib.sha256()
with ZIP_PATH.open('rb') as f:
    for b in iter(lambda:f.read(1024*1024), b''): sha.update(b)
archive_sha = sha.hexdigest()
say(f'ZIP SHA256: {archive_sha}\nZERO TRAINING. Original archive is read-only.')

with zipfile.ZipFile(ZIP_PATH) as z:
    members = [m for m in z.infolist() if not m.is_dir()]
    for m in members:
        row = {'member':m.filename,'bytes':m.file_size,'role':'other','columns':''}
        try:
            if m.filename.lower().endswith('.csv'):
                with z.open(m) as f: h = pd.read_csv(f, nrows=0)
                nh = normalise(h)
                cols = set(nh.columns)
                if {'dataset','model','seed','RecallH','MacroF1','Kappa'} <= cols: role='per_seed_metrics'
                elif {'dataset','model','seed','row_id','y_true','y_pred'} <= cols: role='predictions'
                elif {'stratum','median_rho'} <= cols: role='fidelity_summary'
                elif 'experiment' in cols and any(str(c).endswith(('_mean','_std','_sd')) for c in cols): role='summary'
                elif re.search(r'fail|error|status',m.filename,re.I): role='task_status'
                else: role='other_csv'
                roles[m.filename] = role
                row.update(role=role,columns=' | '.join(h.columns.astype(str)))
                if role in {'per_seed_metrics','fidelity_summary','task_status'}:
                    with z.open(m) as f: d = normalise(pd.read_csv(f))
                    tables[m.filename] = d
                    if role=='task_status':
                        for i,r in d.iterrows():
                            rs = r.to_dict(); ss = str(rs)
                            if re.search(r'fail|error|exception',ss,re.I):
                                failure_records.append({'source':m.filename,'location':str(i),'record':ss})
            elif m.filename.lower().endswith('.json'):
                obj = json.loads(z.read(m).decode('utf-8-sig'))
                task_info(obj,m.filename); row['role']='json_metadata'
            elif m.filename.lower().endswith(('.txt','.log')) and m.file_size < 10_000_000:
                s = z.read(m).decode('utf-8',errors='replace')
                for i,line in enumerate(s.splitlines()):
                    if re.search(r'\b(FAIL|FAILED|ERROR|Traceback|Exception)\b',line) and 'failed=0' not in line:
                        failure_records.append({'source':m.filename,'location':str(i+1),'record':line})
                    elif 'failed=' in line:
                        meta_records.append({'source':m.filename,'location':str(i+1),'value':line})
        except Exception as e:
            row['role']='unreadable'; check('READ '+m.filename,'BLOCKED',repr(e))
        inventory.append(row)
    pd.DataFrame(inventory).to_csv(OUT/'inventory.csv',index=False)
    pd.DataFrame(failure_records,columns=['source','location','record']).drop_duplicates().to_csv(OUT/'failure_records.csv',index=False)
    pd.DataFrame(meta_records,columns=['source','location','value']).drop_duplicates().to_csv(OUT/'metadata_records.csv',index=False)
    say('\nFAILURE / STATUS RECORDS (record count is NOT number of unresolved jobs):')
    for r in failure_records: say(json.dumps(r,ensure_ascii=False))
    for r in meta_records:
        if re.search(r'fail|error',r['location']+r['value'],re.I): say(json.dumps(r,ensure_ascii=False))
    if not failure_records: check('Failure ledger','NOT_ESTABLISHED','No detailed failure records found inside ZIP; no failed task is invented or retried.')
    else: check('Failure ledger','REVIEW_REQUIRED',f'{len(failure_records)} records found; inspect task identity and final retry status.')

    candidates=[]
    for name,d in tables.items():
        if roles.get(name)=='per_seed_metrics':
            x=main_rows(d)
            if len(x) and x.dataset.isin(DATASETS).any(): candidates.append((name,x))
    say('\nMAIN PER-SEED CANDIDATES: '+str([(n,len(x)) for n,x in candidates]))
    chosen=None
    if MAIN_CSV:
        hits=[v for v in candidates if v[0]==MAIN_CSV]
        if len(hits)==1: chosen=hits[0]
        else: check('Main selection','BLOCKED','MAIN_CSV not a discovered valid candidate.')
    elif candidates:
        core=['dataset','model','seed','RecallH','MacroF1','Kappa']
        fingerprints={digest_frame(x,core,['dataset','model','seed']) for _,x in candidates}
        if len(fingerprints)==1: chosen=candidates[0]
        else: check('Main selection','BLOCKED','Distinct candidate sources: set MAIN_CSV to the correct repair-run member; no source silently chosen.')
    else: check('Main selection','BLOCKED','No recognised per-seed main metrics. Inspect inventory.csv. Summary rows cannot prove seed coverage.')

    main=None
    if chosen:
        source,main=chosen
        say('MAIN SOURCE: '+source)
        main.to_csv(OUT/'selected_main_per_seed.csv',index=False)
        duplicates=main.duplicated(['dataset','model','seed'],keep=False)
        expected={(d,m,s) for d in DATASETS for m in MODELS for s in SEEDS}
        actual=set(zip(main.dataset,main.model,main.seed))
        missing=sorted(expected-actual); extra=sorted(actual-expected,key=str)
        ok=(not duplicates.any()) and not missing and not extra and len(main)==160
        check('160-row main coverage','PASS' if ok else 'BLOCKED',
              f'rows={len(main)}, duplicate_keys={int(duplicates.sum())}, missing={missing}, extra={extra}')
        numeric=[k for k in ['RecallH','PrecH','MacroF1','Kappa','PRAUC_High','Brier','ECE','FP_High','FN_High'] if k in main]
        finite=np.isfinite(main[numeric].to_numpy(float)).all()
        check('Finite main metrics','PASS' if finite else 'BLOCKED','NaN/inf is not substituted with zero.')
        summary=main.groupby(['dataset','model'])[numeric].agg(['mean','std','count'])
        summary.columns=['_'.join(c) for c in summary.columns]
        summary.reset_index().to_csv(OUT/'recomputed_main_summary.csv',index=False)
        say('\nRECOMPUTED MAIN SUMMARY:\n'+summary.round(6).to_string())
        ref=main[main.model=='KATS'].copy()
        ab_sources=[]
        for name,d in tables.items():
            if roles.get(name)!='per_seed_metrics' or 'variant' not in d: continue
            x=d.copy()
            if 'experiment' in x: x=x[x.experiment.astype(str).str.contains('ablation',case=False,na=False)]
            else: x=x[x.variant.map(key).ne('tfull')]
            x=x[x.model.eq('KATS')&x.dataset.isin(DATASETS)]
            if not len(x): continue
            ab_sources.append(name)
            if x.duplicated(['dataset','seed','variant']).any():
                check('Ablation '+name,'BLOCKED','Duplicate dataset/seed/variant keys; original rows retained.');continue
            comparable=[c for c in numeric if c in x]
            joined=x.merge(ref[['dataset','seed']+comparable],on=['dataset','seed'],suffixes=('','_reference'),validate='many_to_one')
            for metric in comparable: joined['Delta_'+metric]=joined[metric]-joined[metric+'_reference']
            token=hashlib.sha256(name.encode()).hexdigest()[:8]
            joined.to_csv(OUT/f'ablation_deltas_{token}.csv',index=False)
            for ds in ['GoogleCluster','MultiCloud']:
                eq=joined[(joined.dataset==ds)&joined.variant.map(key).isin({'tnoresampling','tnosmote','noresampling'})]
                if len(eq):
                    diffs=eq[['Delta_'+c for c in comparable]].abs().to_numpy()
                    good=np.isfinite(diffs).all() and np.max(diffs)<=TOL and set(eq.seed)==SEEDS
                    check(ds+' no-resampling '+name,'PASS_NUMERICAL' if good else 'BLOCKED',
                          f'rows={len(eq)}, max_metric_diff={np.nanmax(diffs):.9g}; treatment/prediction identity still requires protocol evidence.')
            full=joined[joined.variant.map(key).isin({'tfull','full'})]
            if len(full):
                diffs=full[['Delta_'+c for c in comparable]].abs().to_numpy()
                good=np.isfinite(diffs).all() and np.max(diffs)<=TOL
                check('Ablation full baseline '+name,'PASS_NUMERICAL' if good else 'BLOCKED',f'max_metric_diff={np.nanmax(diffs):.9g}')
        if not ab_sources: check('Ablation sources','NOT_ESTABLISHED','No recognised variant per-seed rows found; see inventory.')

    pred_candidates=[n for n,r in roles.items() if r=='predictions']
    say('\nPREDICTION CANDIDATES: '+str(pred_candidates))
    pred_name=PREDICTIONS_CSV if PREDICTIONS_CSV in pred_candidates else (pred_candidates[0] if len(pred_candidates)==1 else None)
    if not pred_name:
        check('Prediction pairing','NOT_ESTABLISHED','Set PREDICTIONS_CSV if several distinct candidates; row-level labels cannot be reconstructed from counts.')
    elif main is not None and not main.duplicated(['dataset','model','seed']).any():
        with z.open(pred_name) as f: pred=normalise(pd.read_csv(f))
        pred=main_rows(pred)
        pred=pred[pred.dataset.isin(DATASETS)&pred.model.isin(MODELS)]
        recalc=[]; paired=[]
        from sklearn.metrics import recall_score,precision_score,f1_score,cohen_kappa_score
        labelmap={'0':'High','1':'Low','2':'Medium','High':'High','Low':'Low','Medium':'Medium'}
        for c in ['y_true','y_pred']:
            values=pred[c].astype(str)
            values=values.str.replace(r'^([012])\.0$',r'\1',regex=True)
            pred[c]=values.map(labelmap)
        if pred[['y_true','y_pred']].isna().any().any():
            check('Prediction labels','BLOCKED','Unknown label encoding. Configure from actual encoder metadata; no guessing.')
        elif pred.duplicated(['dataset','model','seed','row_id']).any():
            check('Prediction keys','BLOCKED','Duplicate test-ID within model/seed; no silent dropping.')
        else:
            for (ds,model,seed),g in pred.groupby(['dataset','model','seed']):
                y,p=g.y_true,g.y_pred; yh=y.eq('High'); ph=p.eq('High')
                recalc.append(dict(dataset=ds,model=model,seed=seed,n_test=len(g),High_support=int(yh.sum()),
                    RecallH=recall_score(yh,ph,zero_division=0),PrecH=precision_score(yh,ph,zero_division=0),
                    MacroF1=f1_score(y,p,labels=['High','Low','Medium'],average='macro',zero_division=0),
                    Kappa=cohen_kappa_score(y,p,labels=['High','Low','Medium']),
                    FP_High=int((~yh&ph).sum()),FN_High=int((yh&~ph).sum())))
            rc=pd.DataFrame(recalc);rc.to_csv(OUT/'prediction_recomputed_metrics.csv',index=False)
            merge=main.merge(rc,on=['dataset','model','seed'],suffixes=('_saved','_from_pred'),how='outer',indicator=True)
            merge.to_csv(OUT/'prediction_vs_saved_metrics.csv',index=False)
            if len(merge)==160 and merge['_merge'].eq('both').all():
                diffcols=[]
                for k in ['RecallH','PrecH','MacroF1','Kappa','FP_High','FN_High']:
                    if k+'_saved' in merge and k+'_from_pred' in merge:
                        diffcols.append(np.abs(merge[k+'_saved']-merge[k+'_from_pred']).to_numpy())
                values=np.concatenate(diffcols)
                good=np.isfinite(values).all() and np.max(values)<=TOL
                check('Prediction-derived main metrics','PASS' if good else 'BLOCKED',f'max_abs_diff={np.nanmax(values):.9g}')
            else: check('Prediction-derived main metrics','BLOCKED','Missing/extra model-seed groups; see saved comparison.')
            for (ds,seed),g in pred.groupby(['dataset','seed']):
                a=g[g.model=='KATS'].set_index('row_id')
                for model in sorted(MODELS-{'KATS'}):
                    b=g[g.model==model].set_index('row_id')
                    same=len(a)>0 and a.index.equals(b.index)
                    if not same: same=len(a)>0 and len(a)==len(b) and set(a.index)==set(b.index)
                    if same:
                        b=b.reindex(a.index)
                        same=a.y_true.equals(b.y_true)
                    if not same:
                        check(f'Pairing {ds}/{seed}/{model}','BLOCKED','Test IDs or labels differ.');continue
                    if int(seed)==2026:
                        ca=a.y_true.eq(a.y_pred); cb=b.y_true.eq(b.y_pred)
                        b10=int((ca&~cb).sum());b01=int((~ca&cb).sum());n=b10+b01
                        pval=float(binomtest(b10,n,0.5).pvalue) if n else 1.0
                        paired.append(dict(dataset=ds,baseline=model,seed=2026,b10=b10,b01=b01,
                            accuracy_difference=float(ca.mean()-cb.mean()),p_exact=pval))
            if len(paired)==28:
                mc=pd.DataFrame(paired);o=np.argsort(mc.p_exact.to_numpy());pv=mc.p_exact.to_numpy()[o]
                adj=np.minimum(1,np.maximum.accumulate(pv*(28-np.arange(28))))
                out=np.empty(28);out[o]=adj;mc['p_holm']=out
                mc['direction']=np.where(mc.accuracy_difference>0,'KATS',np.where(mc.accuracy_difference<0,'baseline','tie'))
                mc.to_csv(OUT/'mcnemar_current_exact_holm_28.csv',index=False)
                check('Current McNemar','PASS',f'{int((mc.p_holm<.05).sum())}/28 Holm-significant; computed from current aligned predictions.')
                say(mc.to_string(index=False))
            else: check('Current McNemar','NOT_ESTABLISHED',f'Only {len(paired)}/28 seed2026 pairs available.')
            check('Encoded class order','REVIEW_REQUIRED','Numeric labels were mapped with documented High=0/Low=1/Medium=2. Confirm current encoder JSON agrees.')

    fidelity=[(n,d) for n,d in tables.items() if roles.get(n)=='fidelity_summary']
    say('\nACTUAL-MODEL FIDELITY EVIDENCE:')
    for name,d in fidelity:
        say(name+'\n'+d.to_string(index=False))
        token=hashlib.sha256(name.encode()).hexdigest()[:8]
        d.to_csv(OUT/f'fidelity_evidence_{token}.csv',index=False)
    if fidelity:
        check('Fidelity result table','EVIDENCE_PRESENT',f'{len(fidelity)} tables; do not rerun just because older notes said absent.')
    else: check('Fidelity result table','NOT_DISCOVERED','Your displayed audit exists; locate its CSV in inventory or save it. Do not substitute old surrogate values.')
    check('Fidelity model identity','CONFIRM_SOURCE',
          'Link audit to current saved stack and extract its fitted B1; use the existing new audit/code. Summary titles alone cannot authenticate this.')

checks_df=pd.DataFrame(checks);checks_df.to_csv(OUT/'closure_checks.csv',index=False)
json.dump({'archive_sha256':archive_sha,'zero_training':True,'checks':checks,
           'failure_records':failure_records,'metadata_records':meta_records},
          (OUT/'closure_checks.json').open('w'),indent=2,default=str)
blocking=checks_df[checks_df.state=='BLOCKED']
report=['# KATS R1 closure audit','',f'Archive SHA256: `{archive_sha}`','',
        'No models trained. No original results overwritten. No missing metric invented.',
        '',checks_df.to_markdown(index=False),'',
        '## Decision','',
        'Resolve BLOCKED checks first. NOT_ESTABLISHED means evidence could not be verified, not that an experiment was never executed.',
        'CONFIRM_SOURCE/REVIEW_REQUIRED require source inspection. Numerical passes do not certify all reviewer comments or acceptance.',
        'Do not automatically restart completed training or repeat the actual-fitted SHAP audit.']
(OUT/'READINESS.md').write_text('\n'.join(report),encoding='utf-8')
say('\nFINAL CLOSURE CHECKS\n'+checks_df.to_string(index=False))
say(f'\nFinished. Blocking checks: {len(blocking)}. Outputs: {OUT}')
say('If failures remain unclear, return failure_records.csv + closure_checks.csv + inventory.csv, not another long mean/SD dump.')
