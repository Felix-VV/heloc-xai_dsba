"""Check the distributed numerical evidence without raw data or fitted model files."""
from pathlib import Path
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
from hxai.common import ROOT,RESULTS,TABLES,load_json,load_config,dump_json

p=load_json(RESULTS/'predictive.json');a=load_json(RESULTS/'ablation.json')
full=pd.read_csv(TABLES/'test_predictions.csv');ab=pd.read_csv(TABLES/'ablation_test_predictions.csv')
checks=[]
def check(name,condition):
    checks.append({'name':name,'passed':bool(condition)})
    print(('PASS ' if condition else 'FAIL ')+name)
def auc(y,p):
    _,inv,n=np.unique(p,return_inverse=True,return_counts=True)
    rank=(np.cumsum(n)-n+(n+1)/2)[inv];n1=(y==1).sum()
    return (rank[y==1].sum()-n1*(n1+1)/2)/(n1*(len(y)-n1))
check('same test customers and labels in both conditions',np.array_equal(full.row_id,ab.row_id) and np.array_equal(full.y,ab.y))
check('complete binary test labels and expected sample count',len(full)==a['n_test']==1975 and set(full.y)=={0,1} and not full.isna().any().any())
y=full.y.to_numpy()
for m in ['l1_logreg','cart','xgboost']:
    for variant,values,expected in [('full',full['p_'+m].to_numpy(),p['test'][m]),('no_score',ab['no_score_'+m].to_numpy(),a['per_model'][m]['no_score'])]:
        clipped=np.clip(values,1e-12,1-1e-12)
        actual={'auc':auc(y,values),'brier':float(np.mean((values-y)**2)),
                'logloss':float(-np.mean(y*np.log(clipped)+(1-y)*np.log1p(-clipped)))}
        check(f'{m}/{variant}: saved metrics independently recomputed',all(np.isclose(actual[k],expected[k],atol=1e-7) for k in actual))
# Independent rank-based paired resampling: no call to hxai.metrics or sklearn.
cfg=load_config(); names=['l1_logreg','cart','xgboost']
pairs=[(m,n) for i,m in enumerate(names) for n in names[i+1:]]
dist={f'{m}-{n}_{metric}':[] for m,n in pairs for metric in ['auc','brier']}
rng=np.random.default_rng(cfg['seed'])
for _ in range(cfg['evaluation']['bootstrap_resamples']):
    idx=rng.integers(0,len(y),len(y)); yy=y[idx]
    if yy.min()==yy.max():continue
    vals={m:(auc(yy,full['p_'+m].to_numpy()[idx]),float(np.mean((full['p_'+m].to_numpy()[idx]-yy)**2))) for m in names}
    for m,n in pairs:
        for j,metric in enumerate(['auc','brier']):dist[f'{m}-{n}_{metric}'].append(vals[m][j]-vals[n][j])
for key,values in dist.items():
    expected=p['bootstrap'][key]
    check(f'{key}: paired interval independently recomputed',np.allclose(np.quantile(values,[.025,.975]),[expected['lo'],expected['hi']],atol=1e-7))
check('ablation is explicitly exploratory','exploratory' in a['analysis'])
rep=pd.read_csv(TABLES/'ablation_paired_repeated_splits.csv')
check('ten paired development repetitions for each model',len(rep)==30 and rep.groupby('model').repeat.nunique().eq(10).all())
check('full report numerical inputs are generated',all((RESULTS/n).exists() for n in ['report_numbers.tex','ablation_numbers.tex']))
for table in ['table1_predictive.tex','table2_ablation.tex','table3_ablation_differences.tex']:
    check('report input exists: '+table,(TABLES/table).is_file())
summary={'status':'passed' if all(x['passed'] for x in checks) else 'failed','checks':checks,
         'passed':sum(x['passed'] for x in checks),'total':len(checks),
         'scope':'Saved-output consistency. No claim that model refitting or raw-data verification ran in report-only mode.'}
dump_json(summary,RESULTS/'saved_results_validation.json')
raise SystemExit(0 if summary['status']=='passed' else 1)
