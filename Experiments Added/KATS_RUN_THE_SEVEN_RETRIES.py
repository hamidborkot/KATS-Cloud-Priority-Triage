# ONE CELL. Uses the repair source saved by the previous cell.
import ast, builtins, contextlib, copy, hashlib, io, json, os, re, traceback, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC, SMOTEN

SOURCE = Path('/kaggle/working/KATS_RETRY_7_ONLY/original_repair_cell.py')
ZIP = Path('/kaggle/working/KATS_REVIEWER_REPAIR_R1_RESULTS.zip')
OUT = Path('/kaggle/working/KATS_RETRY_7_EXACT')
OUT.mkdir(parents=True, exist_ok=True)
if not SOURCE.is_file():
    raise FileNotFoundError('The original_repair_cell.py saved by your previous cell is required.')
source = SOURCE.read_text(encoding='utf-8')
source_hash = hashlib.sha256(source.encode()).hexdigest()
with zipfile.ZipFile(ZIP) as z:
    member = next(n for n in z.namelist() if Path(n).name == 'FAILED_CASES.json')
    ledger = json.loads(z.read(member))
KEYS = ('experiment','model','seed','variant')
jobs = {}
for r in ledger:
    if r.get('dataset') == 'ITIncident' and 'SMOTE-NC is not designed to work only with categorical' in r.get('error',''):
        jobs[tuple(r[k] for k in KEYS)] = {k:r[k] for k in ('dataset',)+KEYS}
expected = {(e,'KATS',s,'T_Full') for e in ('availability_FINAL6_CAT','availability_FIRST_EVENT6') for s in (42,7,13)}
expected.add(('temporal_first_event_matured_labels','KATS',42,'T_Full'))
if set(jobs) != expected:
    raise RuntimeError('Ledger does not match the seven reported tasks. Nothing trained.')

captured = {}
class CallsCollected(BaseException): pass
class UnexpectedTraining(BaseException): pass
capture_mode = True

def capture_case(data, experiment, name, seed, tr, te, variant='T_Full', groups=None, times=None):
    k=(str(experiment),str(name),int(seed),str(variant))
    if k in jobs and k not in captured:
        captured[k] = (copy.deepcopy(data),experiment,name,seed,
                       copy.deepcopy(tr),copy.deepcopy(te),variant,
                       copy.deepcopy(groups),copy.deepcopy(times))
        builtins.print(f'Prepared {len(captured)}/7: {experiment}, seed={seed}',flush=True)
    if len(captured)==7: raise CallsCollected()
    return None

def fit_guard(method,*args,**kwargs):
    estimator=getattr(method,'__self__',None)
    if capture_mode and getattr(estimator,'_estimator_type',None)=='classifier':
        raise UnexpectedTraining('Source tried classifier fitting outside run_case during call collection; stopped safely.')
    return method(*args,**kwargs)

class Rewrite(ast.NodeTransformer):
    def visit_Constant(self,node):
        if isinstance(node.value,str):
            value=node.value
            if value.startswith('/kaggle/working/KATS_REVIEWER_REPAIR_R1'):
                value=value.replace('/kaggle/working/KATS_REVIEWER_REPAIR_R1',str(OUT),1)
            elif value.startswith('KATS_REVIEWER_REPAIR_R1'):
                value=value.replace('KATS_REVIEWER_REPAIR_R1',OUT.name,1)
            return ast.copy_location(ast.Constant(value=value),node)
        return node
    def visit_Call(self,node):
        node=self.generic_visit(node)
        if isinstance(node.func,ast.Attribute) and node.func.attr=='fit':
            return ast.copy_location(ast.Call(func=ast.Name(id='_retry_fit_guard',ctx=ast.Load()),
                        args=[node.func]+node.args,keywords=node.keywords),node)
        return node
    def visit_FunctionDef(self,node):
        original=ast.get_source_segment(source,node) or ''
        node=self.generic_visit(node)
        if node.name=='run_case':
            node.name='_retry_original_run_case'
            return [node,ast.Assign(targets=[ast.Name(id='run_case',ctx=ast.Store())],
                                   value=ast.Name(id='_retry_capture',ctx=ast.Load()))]
        # Do not produce fake cumulative reports/archives during dry call collection.
        if 'CURRENT MEAN/SD TABLE' in original or ('all_completed_metrics_per_seed.csv' in original and ('to_csv' in original or 'ARCHIVE' in original)):
            node.body.insert(0,ast.If(test=ast.Name(id='_retry_collecting',ctx=ast.Load()),body=[ast.Return(value=ast.Constant(None))],orelse=[]))
        return node

parsed=ast.parse(source)
if not any(isinstance(n,ast.FunctionDef) and n.name=='run_case' for n in ast.walk(parsed)):
    raise RuntimeError('Saved source has no run_case definition. Nothing trained.')
rewritten=ast.fix_missing_locations(Rewrite().visit(parsed))
ns={'__name__':'__main__','_retry_capture':capture_case,'_retry_fit_guard':fit_guard,
    '_retry_collecting':True,'__file__':str(SOURCE)}
oldcwd=os.getcwd(); changed_env={}
for k,v in list(os.environ.items()):
    if v.startswith('/kaggle/working/KATS_REVIEWER_REPAIR_R1'):
        changed_env[k]=v;os.environ[k]=v.replace('/kaggle/working/KATS_REVIEWER_REPAIR_R1',str(OUT),1)
try:
    os.chdir(OUT)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(rewritten,str(SOURCE),'exec'),ns)
    except CallsCollected:
        pass
    except BaseException:
        (OUT/'call_collection_error.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise
    if set(captured)!=expected:
        missing=sorted(expected-set(captured))
        raise RuntimeError(f'Original source did not reach all seven calls: {missing}. No alternative splits invented.')
    capture_mode=False;ns['_retry_collecting']=False
    original_runner=ns['_retry_original_run_case']
    (OUT/'RETRY_PROTOCOL.json').write_text(json.dumps({
        'original_source_sha256':source_hash,'main_benchmark_retrained':False,
        'scope':'seven IT categorical-only sensitivities','sampler':'SMOTEN',
        'calls':'original data/tr/te/groups/times construction replayed without classifier fitting',
        'qualification':'exact reproduction still depends on source determinism; inspect saved split IDs'},indent=2))

    if not hasattr(SMOTENC,'_kats_original_fit_resample'):
        SMOTENC._kats_original_fit_resample=SMOTENC.fit_resample
    old_sampler_method=SMOTENC.fit_resample
    def categorical_safe(self,X,y,**kwargs):
        cats=np.asarray(self.categorical_features);n=X.shape[1]
        if cats.dtype.kind=='b':allcat=(cats.size==n and bool(cats.all()))
        elif cats.dtype.kind in 'iu':allcat=(set(cats.tolist())==set(range(n)))
        elif hasattr(X,'columns'):allcat=(set(cats.ravel().tolist())==set(X.columns))
        else:allcat=False
        if not allcat:return SMOTENC._kats_original_fit_resample(self,X,y,**kwargs)
        sampler=SMOTEN(sampling_strategy=self.sampling_strategy,
                      random_state=self.random_state,k_neighbors=self.k_neighbors)
        result=sampler.fit_resample(X,y,**kwargs)
        self.__dict__.update(sampler.__dict__);self.effective_sampler_='SMOTEN'
        return result
    SMOTENC.fit_resample=categorical_safe

    def metric_record(obj):
        candidates=[]
        def walk(v):
            if isinstance(v,pd.Series):v=v.to_dict()
            if isinstance(v,dict):
                normal={re.sub('[^a-z0-9]','',str(k).lower()) for k in v}
                if normal & {'macrof1','f1macro'} and normal & {'recallh','recallhigh','recall'}:
                    candidates.append({k:x.item() if isinstance(x,np.generic) else x for k,x in v.items()
                                       if isinstance(x,(str,int,float,bool,np.generic)) or x is None})
                for x in v.values():
                    if isinstance(x,(dict,list,tuple,pd.Series)):walk(x)
            elif isinstance(v,(list,tuple)):
                for x in v:walk(x)
        walk(obj)
        unique={json.dumps(x,sort_keys=True,default=str):x for x in candidates}
        if len(unique)==1:return next(iter(unique.values()))
        return None

    records=[]
    try:
        for i,k in enumerate(sorted(expected),1):
            job=jobs[k];print(f'RUN {i}/7: {job}',flush=True)
            before={str(p):p.stat().st_mtime_ns for p in OUT.rglob('*.json')}
            value=original_runner(*captured[k])
            result=metric_record(value)
            if result is None:
                found=[]
                for p in OUT.rglob('*.json'):
                    if not re.search(r'metric|result',p.name,re.I) or 'validation' in p.name.lower():continue
                    if before.get(str(p))==p.stat().st_mtime_ns:continue
                    try:
                        r=metric_record(json.loads(p.read_text()))
                        if r is not None:found.append(r)
                    except (ValueError,OSError):pass
                result=metric_record(found)
            if result is None:
                raise RuntimeError('Evaluator returned no uniquely identifiable test metrics. Existing generated case files are preserved; see output folder.')
            records.append({**result,**job,'effective_sampler':'SMOTEN','retry_status':'metrics_returned'})
            pd.DataFrame(records).to_csv(OUT/'retry7_per_seed.csv',index=False)
            print(pd.DataFrame([records[-1]]).to_string(index=False),flush=True)
    finally:
        SMOTENC.fit_resample=old_sampler_method
    df=pd.DataFrame(records)
    grouping=['dataset','experiment','model','variant','effective_sampler']
    numeric=[c for c in df.select_dtypes(include=np.number) if c!='seed']
    summary=df.groupby(grouping)[numeric].agg(['mean','std','count'])
    summary.columns=['_'.join(x) for x in summary.columns]
    summary.reset_index().to_csv(OUT/'retry7_mean_sd.csv',index=False)
    print('\nSEVEN-CASE RESULTS\n'+df.to_string(index=False),flush=True)
    print('\nMEAN / SAMPLE SD / COUNT\n'+summary.to_string(),flush=True)
    print('\nSaved results:',OUT,flush=True)
    print('Original ZIP untouched. Report SMOTEN adaptation for categorical-only sensitivities; do not relabel it as unchanged SMOTENC.',flush=True)
finally:
    os.chdir(oldcwd)
    for k,v in changed_env.items():os.environ[k]=v
