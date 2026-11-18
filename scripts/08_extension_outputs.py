"""Generate ablation figures, tables and report macros from saved results."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hxai.common import ROOT, RESULTS, TABLES, FIGURES, RunLog, load_json, dump_json

log = RunLog("08_extension_outputs")
aj = load_json(RESULTS / "ablation.json")
assert aj["status"] == "complete"
pj = load_json(RESULTS / "predictive.json")
pred = pd.read_csv(TABLES / "ablation_test_predictions.csv")
M = ("l1_logreg", "cart", "xgboost")
LABEL = {"l1_logreg": "L1 logistic", "cart": "CART", "xgboost": "XGBoost"}
NAMES = {"l1_logreg": "LR", "cart": "CART", "xgboost": "XGB"}
FULL, NO = "#2666a8", "#c76d24"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11,
                     "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "savefig.dpi": 300, "svg.fonttype": "none"})

def save(fig, stem):
    for ext in ("pdf", "png", "svg"):
        fig.savefig(FIGURES / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)

# Compact numerical tables, with full and no-score rows for every model.
rows = []
for m in M:
    d = aj["per_model"][m]
    for variant in ("full", "no_score"):
        cal = pj["calibration"][m] if variant == "full" else d["calibration"]
        rows.append({"model": m, "variant": variant, **d[variant],
                     "calibration_slope": cal["recal_slope"], "mean_observed_minus_predicted": cal["citl"]})
pd.DataFrame(rows).to_csv(TABLES / "ablation_summary.csv", index=False)

# Figure 4: paired performance with conditional test intervals.
fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), layout="constrained")
for ax, metric, title in zip(axes, ("auc", "brier"), ("(a) Discrimination: ROC-AUC", "(b) Probability quality: Brier")):
    for j, m in enumerate(M):
        d = aj["per_model"][m]
        for variant, offset, color, marker in (("full", -.12, FULL, "o"), ("no_score", .12, NO, "s")):
            value = d[variant][metric]
            ci = d["bootstrap"][f"{variant}_{metric}"]
            ax.plot([ci["lo"],ci["hi"]],[j+offset,j+offset],color=color,lw=2)
            ax.scatter([value],[j+offset],color=color,marker=marker,s=44,zorder=3,
                       label=("With upstream score" if variant=="full" else "Without upstream score") if j==0 else None)
    ax.set_yticks(range(3),[LABEL[m] for m in M]); ax.invert_yaxis()
    ax.set_xlabel("ROC-AUC (higher is better)" if metric=="auc" else "Brier score (lower is better)")
    ax.set_title(title); ax.grid(axis="x",alpha=.2)
axes[0].legend(loc="lower left",fontsize=9)
fig.suptitle("Exploratory score-removal analysis: same customers, independently retuned models",fontsize=11)
save(fig,"fig4_ablation_performance")

# Figure 5: applicant-level shifts, without implying observed treatment decisions.
fig, axes = plt.subplots(1,3,figsize=(12,4),layout="constrained")
for ax,m in zip(axes,M):
    for label,color in ((0,FULL),(1,NO)):
        sub=pred.loc[pred.y==label]
        ax.scatter(sub[f"full_{m}"],sub[f"no_score_{m}"],s=9,alpha=.18,color=color,rasterized=True,
                   label="Good" if label==0 else "Bad")
    ax.plot([0,1],[0,1],color="#555555",ls="--",lw=1)
    ax.axvline(.5,color="#bbbbbb",lw=.7); ax.axhline(.5,color="#bbbbbb",lw=.7)
    ch=aj["per_model"][m]["individual_change"]
    ax.set_title(f"{LABEL[m]}\n{100*ch['fraction_gt_0.10']:.1f}% change by >0.10",fontsize=11)
    ax.set(xlim=(0,1),ylim=(0,1),xlabel="P(Bad), with score",ylabel="P(Bad), without score")
    ax.set_aspect("equal")
axes[-1].legend(loc="upper left",markerscale=2,fontsize=9)
save(fig,"fig5_ablation_individual")

# Figure 6: repeated development splits; differences are full minus no-score.
paired=pd.read_csv(TABLES/"ablation_paired_repeated_splits.csv")
fig,axes=plt.subplots(1,2,figsize=(10.5,3.7),layout="constrained")
for ax,metric,title in zip(axes,("auc","brier"),("(a) AUC change","(b) Brier change")):
    for j,m in enumerate(M):
        values=paired.loc[paired.model==m,f"delta_{metric}_full_minus_no_score"].to_numpy()
        yy=j+np.linspace(-.16,.16,len(values))
        ax.scatter(values,yy,color=FULL,s=28,alpha=.75)
        ax.plot([values.mean()]*2,[j-.25,j+.25],color=NO,lw=3)
    ax.axvline(0,color="#555555",ls="--",lw=1)
    ax.set_yticks(range(3),[LABEL[m] for m in M]); ax.invert_yaxis()
    ax.set_title(title); ax.set_xlabel("Full minus no-score: " + (">0 favours full" if metric=="auc" else "<0 favours full"))
    ax.grid(axis="x",alpha=.2)
fig.suptitle("Ten overlapping development splits: dots = paired changes; orange bar = mean",fontsize=11)
save(fig,"fig6_ablation_repeated")

# Figure 7: actual calibration diagnostics, separated from overall Brier scores.
fig,axes=plt.subplots(1,3,figsize=(12,3.9),layout="constrained")
for ax,m in zip(axes,M):
    ax.plot([0,1],[0,1],color="#777777",ls="--",lw=1)
    for variant,color,marker in (("full",FULL,"o"),("no_score",NO,"s")):
        cal=pj["calibration"][m] if variant=="full" else aj["per_model"][m]["calibration"]
        bins=pd.DataFrame(cal["bins"])
        ax.plot(bins.mean_pred,bins.obs_rate,color=color,marker=marker,ms=4,lw=1.5,
                label=f"{'With' if variant=='full' else 'Without'} score; slope {cal['recal_slope']:.2f}")
    ax.set(xlim=(0,1),ylim=(0,1),xlabel="Mean predicted P(Bad)",ylabel="Observed share Bad",title=LABEL[m])
    ax.grid(alpha=.15); ax.legend(fontsize=8,loc="lower right")
save(fig,"fig7_ablation_calibration")

# TeX macros are regenerated from the numerical outputs on each run.
mac={"NNoScoreCols":str(aj["n_columns_no_score"]),"NNoScoreVars":str(aj["n_variables_no_score"])}
def ci(x,digits=3):return f"[{x['lo']:.{digits}f}, {x['hi']:.{digits}f}]"
for m in M:
    n=NAMES[m]; d=aj["per_model"][m]
    mac.update({f"NoScoreAUC{n}":f"{d['no_score']['auc']:.3f}",f"NoScoreBrier{n}":f"{d['no_score']['brier']:.4f}",
                f"ScoreDeltaAUC{n}":f"{d['full_minus_no_score']['auc']:+.3f}",
                f"ScoreDeltaBrier{n}":f"{d['full_minus_no_score']['brier']:+.4f}",
                f"ScoreDeltaAUCci{n}":ci(d['bootstrap']['full-no_score_auc']),
                f"ScoreDeltaBrierci{n}":ci(d['bootstrap']['full-no_score_brier'],4),
                f"NoScoreChangeTen{n}":f"{100*d['individual_change']['fraction_gt_0.10']:.1f}\\%",
                f"NoScoreMeanChange{n}":f"{d['individual_change']['mean_absolute']:.3f}",
                f"NoScoreLabelChange{n}":f"{100*d['individual_change']['label_disagreement_at_0.5']:.1f}\\%",
                f"RepScoreAUCPos{n}":str(aj['repeated']['per_model'][m]['auc']['delta_positive_count']),
                f"RepScoreAUCMean{n}":f"{aj['repeated']['per_model'][m]['auc']['delta_mean']:+.4f}"})
lr=aj['per_model']['l1_logreg']['complexity']
mac['NoScoreLRCoef']=str(lr['nonzero_coefficients']);mac['NoScoreLRVars']=str(lr['original_variables_used'])
gap_full=pj['test']['xgboost']['auc']-pj['test']['l1_logreg']['auc']
gap_no=aj['per_model']['xgboost']['no_score']['auc']-aj['per_model']['l1_logreg']['no_score']['auc']
mac['GapFullXGBLR']=f"{gap_full:+.3f}"; mac['GapNoScoreXGBLR']=f"{gap_no:+.3f}"
with (RESULTS/'ablation_numbers.tex').open('w') as f:
    f.write('% Auto-generated from the completed exploratory ablation. Do not edit.\n')
    for key,value in sorted(mac.items()):f.write('\\newcommand{\\'+key+'}{'+value.replace('-','$-$')+'}\n')

tex=[r'\begin{tabular}{@{}llrrrr@{}}',r'\toprule',r'Model & Inputs & AUC & Brier & Log-loss & Cal. slope \\',r'\midrule']
for row in rows:
    tex.append(f"{LABEL[row['model']]} & {'Full' if row['variant']=='full' else 'No score'} & {row['auc']:.3f} & {row['brier']:.4f} & {row['logloss']:.3f} & {row['calibration_slope']:.2f} \\\\")
tex.extend([r'\bottomrule',r'\end{tabular}'])
(TABLES/'table2_ablation.tex').write_text('\n'.join(tex)+'\n')

tex=[r'\begin{tabular}{@{}lrr@{}}',r'\toprule',r'Model & AUC difference [95\% CI] & Brier difference [95\% CI] \\',r'\midrule']
for m in M:
    n=NAMES[m]
    tex.append(f"{LABEL[m]} & \\ScoreDeltaAUC{n} \\ScoreDeltaAUCci{n} & \\ScoreDeltaBrier{n} \\ScoreDeltaBrierci{n} \\\\")
tex.extend([r'\bottomrule',r'\end{tabular}'])
(TABLES/'table3_ablation_differences.tex').write_text('\n'.join(tex)+'\n')

figure_index=[
 ('fig1_predictive','主展示','总体预测表现、校准和重复划分','Performance and calibration do not establish explanation quality.'),
 ('fig2_cases','主展示','两个预定个案的完整预测与局部归因','Illustrations are not population evidence or proof of causality.'),
 ('fig3_stability','主展示','重训与参考样本变化需要区分','Finite-background sensitivity is different from changing the reference population.'),
 ('fig4_ablation_performance','主展示','去掉上游评分后的公平模型比较','New score-removal findings are exploratory.'),
 ('fig5_ablation_individual','深入讨论','总体指标之外的客户概率变化','The 0.5 threshold is descriptive, not an optimal lending policy.'),
 ('fig6_ablation_repeated','深入讨论','重新选择模型后的划分敏感性','Overlapping development splits are dependent.'),
 ('fig7_ablation_calibration','深入讨论','去掉评分后的校准诊断','Brier is not a pure calibration measure.'),
]
index=['# Figure index\n','主展示素材可择取四至五组；不要求把所有面板放进 15 分钟展示。\n']
for stem,use,meaning,caution in figure_index:
    index.extend([f'## {stem}｜{use}\n',f'{meaning}。\n',f'- PDF: `../results/figures/{stem}.pdf`\n- PNG: `../results/figures/{stem}.png`\n',f'- 解释边界：{caution}\n'])
(ROOT/'docs/Figure_Index.md').write_text('\n'.join(index))
log.note('Generated four extension figures, numerical tables/macros, figure index, and figure index.')
log.save()
