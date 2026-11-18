"""Checks for the ablation, report integration and final-project integrity."""
from pathlib import Path
import hashlib
import re
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import joblib
import numpy as np
import pandas as pd
from hxai.common import ROOT,RESULTS,MODELS,TABLES,FIGURES,LOGS,load_json,load_config,dump_json
from hxai.data import design_matrix
from hxai.models import predict_proba

cfg=load_config();a=load_json(RESULTS/'ablation.json')
b=joblib.load(MODELS/'data_bundle.joblib');pop,sp=b['pop'],b['splits']
X,fmap=design_matrix(pop.X_raw.drop(columns=['ExternalRiskEstimate']),cfg)
df=pd.read_csv(TABLES/'ablation_test_predictions.csv')
tests=[];skips=[]
def check(name,ok,detail=''):
    tests.append({'name':name,'passed':bool(ok),'detail':detail})
    print(('PASS ' if ok else 'FAIL ')+name+(f' [{detail}]' if detail else ''),flush=True)
def independent_auc(y,p):
    _,inv,cnt=np.unique(p,return_inverse=True,return_counts=True)
    ranks=(np.cumsum(cnt)-cnt+(cnt+1)/2)[inv]
    n1=(y==1).sum();n0=len(y)-n1
    return float((ranks[y==1].sum()-n1*(n1+1)/2)/(n1*n0))

check('exploratory ablation completed',a['status']=='complete' and 'exploratory' in a['analysis'])
check('same population, test positions, row IDs and labels',len(X)==9871 and
      np.array_equal(df.pos.to_numpy(),sp['test']) and np.array_equal(df.row_id.to_numpy(),pop.row_id[sp['test']]) and
      np.array_equal(df.y.to_numpy(),pop.y[sp['test']]))
check('score and all derived score columns absent',not any('ExternalRiskEstimate' in x for x in X.columns) and
      'ExternalRiskEstimate' not in set(fmap.variable) and X.shape[1]==44 and fmap.variable.nunique()==22)
check('test row-ID digest matches extension metadata',hashlib.sha256(pop.row_id[sp['test']].tobytes()).hexdigest()==a['test_row_ids_sha256'])
check('customer groups stay on one side of original split',not set(pop.groups[sp['dev']])&set(pop.groups[sp['test']]))

for name in ['l1_logreg','cart','xgboost']:
    model=joblib.load(MODELS/f'ablation_no_score_{name}.joblib')
    p=predict_proba(model,X.iloc[sp['test']]);saved=df[f'no_score_{name}'].to_numpy();y=df.y.to_numpy()
    check(f'{name}: saved no-score predictions match fitted model',np.allclose(p,saved,atol=1e-7,rtol=1e-6))
    met={'auc':independent_auc(y,saved),'brier':float(np.mean((saved-y)**2)),
         'logloss':float(-np.mean(y*np.log(saved)+(1-y)*np.log1p(-saved)))}
    check(f'{name}: AUC/Brier/log-loss independently recomputed',all(np.isclose(met[k],a['per_model'][name]['no_score'][k],atol=1e-7) for k in met))
    check(f'{name}: model input count is reduced',model.n_features_in_==44)
    cv=pd.read_csv(TABLES/f'ablation_cv_results_{name}.csv')
    selected=a['per_model'][name]['selection'];means=cv.groupby('config').brier.mean()
    check(f'{name}: selection minimizes development CV Brier',np.isclose(means.loc[selected['best_config']],means.min(),atol=1e-12))
    full=df[f'full_{name}'].to_numpy()
    changes=a['per_model'][name]['individual_change']
    check(f'{name}: individual change summaries match saved rows',
          np.isclose(changes['mean_absolute'],np.abs(full-saved).mean()) and
          np.isclose(changes['fraction_gt_0.10'],(np.abs(full-saved)>.1).mean()) and
          np.isclose(changes['label_disagreement_at_0.5'],((full>=.5)!=(saved>=.5)).mean()))
    # Independent rank-based paired bootstrap, preserving the project's seed and sampling order.
    rng=np.random.default_rng(cfg['seed']);auc_d=[];brier_d=[]
    for _ in range(cfg['evaluation']['bootstrap_resamples']):
        idx=rng.integers(0,len(y),len(y));yy=y[idx]
        if yy.min()==yy.max():continue
        auc_d.append(independent_auc(yy,full[idx])-independent_auc(yy,saved[idx]))
        brier_d.append(float(np.mean((full[idx]-yy)**2)-np.mean((saved[idx]-yy)**2)))
    bs=a['per_model'][name]['bootstrap']
    check(f'{name}: full-minus-no-score interval signs and endpoints independently verified',
          np.allclose(np.quantile(auc_d,[.025,.975]),[bs['full-no_score_auc']['lo'],bs['full-no_score_auc']['hi']],atol=1e-6) and
          np.allclose(np.quantile(brier_d,[.025,.975]),[bs['full-no_score_brier']['lo'],bs['full-no_score_brier']['hi']],atol=1e-6))

lr=joblib.load(MODELS/'ablation_no_score_l1_logreg.joblib')
numeric=list(fmap.loc[fmap.kind=='numeric','column'])
check('no-score LR imputation uses development statistics',np.allclose(lr.named_steps['impute'].named_transformers_['num'].statistics_,X.iloc[sp['dev']][numeric].median().to_numpy()))
check('no-score LR centering uses development rows',np.allclose(lr[:-1].transform(X.iloc[sp['dev']]).mean(0),0,atol=1e-8))
coef=pd.read_csv(TABLES/'ablation_lr_coefficients.csv')
check('no-score complexity maps transformed coefficients to correct original variables',
      coef.loc[coef.coef_standardized!=0,'variable'].nunique()==a['per_model']['l1_logreg']['complexity']['original_variables_used'])
rep=pd.read_csv(TABLES/'ablation_paired_repeated_splits.csv')
check('ten paired repeated splits per model',len(rep)==30 and rep.groupby('model').repeat.nunique().eq(10).all() and not rep.duplicated(['repeat','model']).any())
for metric in ['auc','brier','logloss']:
    check(f'repeated {metric}: correct full-minus-no-score differences',np.allclose(rep[f'delta_{metric}_full_minus_no_score'],rep[f'{metric}_full']-rep[f'{metric}_no_score']))

report=(ROOT/'report/report.tex').read_text()
check('report imports generated baseline and extension numbers and main table',all(x in report for x in
      ['\\input{../results/report_numbers.tex}','\\input{../results/ablation_numbers.tex}','\\input{../results/tables/table1_predictive.tex}']))
check('report contains no stale assets directory references','assets/' not in report)
check('all seven figure groups exist in PDF/PNG/SVG',all((FIGURES/f'{stem}.{suffix}').is_file() for stem in
      ['fig1_predictive','fig2_cases','fig3_stability','fig4_ablation_performance','fig5_ablation_individual','fig6_ablation_repeated','fig7_ablation_calibration'] for suffix in ['pdf','png','svg']))
full_df=pd.read_csv(TABLES/'test_predictions.csv')
check('full and no-score test membership and labels identical',np.array_equal(full_df.row_id,df.row_id) and np.array_equal(full_df.y,df.y))

failed=sum(not t['passed'] for t in tests)
summary={'status':'passed' if not failed else 'failed','passed':len(tests)-failed,'total':len(tests),'tests':tests,'skipped':skips}
dump_json(summary,RESULTS/'extended_validation.json')
with (LOGS/'09_extended_checks.txt').open('w') as f:
    for t in tests:f.write(('PASS ' if t['passed'] else 'FAIL ')+t['name']+'\n')
    for s in skips:f.write('SKIP '+s+'\n')
    f.write(f"\n{len(tests)-failed}/{len(tests)} checks passed\n")
print(f'{len(tests)-failed}/{len(tests)} extended checks passed.',flush=True)
raise SystemExit(1 if failed else 0)
