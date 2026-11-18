"""Figures, LaTeX tables and number macros for the report, from saved results only."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hxai.common import FIGURES, RESULTS, ROOT, TABLES, RunLog, load_json

log = RunLog("05_figures_tables")
pj = load_json(RESULTS / "predictive.json")
rj = load_json(RESULTS / "repeated_splits.json")
sj = load_json(RESULTS / "stability.json")
ej = load_json(RESULTS / "explanations.json")
aj = load_json(RESULTS / "audit.json")

# Categorical slots 1-3 of the reference palette (validated all-pairs, light mode).
COL = {"l1_logreg": "#2a78d6", "cart": "#eb6834", "xgboost": "#1baf7a"}
MARK = {"l1_logreg": "o", "cart": "s", "xgboost": "^"}
LAB = {"l1_logreg": "L1 logistic", "cart": "CART", "xgboost": "XGBoost"}
INK, MUTED, GRID = "#2b2b2b", "#6b6b6b", "#e4e4e0"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 8.5,
                     "axes.titleweight": "bold", "font.family": "DejaVu Sans",
                     "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "savefig.dpi": 300})
M = ("l1_logreg", "cart", "xgboost")

# ---------------------------------------------------------------- Figure 1: predictive
fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.35), gridspec_kw={"width_ratios": [1.1, 1, 1]})
a = ax[0]
a.plot([0, 1], [0, 1], ls="--", lw=1, color=MUTED, zorder=0)
for m in M:
    b = pd.DataFrame(pj["calibration"][m]["bins"])
    a.plot(b.mean_pred, b.obs_rate, lw=2, color=COL[m], marker=MARK[m], ms=4.5,
           markeredgecolor="white", markeredgewidth=0.8,
           label=f"{LAB[m]}, slope {pj['calibration'][m]['recal_slope']:.2f}")
a.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted P(Bad)", ylabel="Observed share Bad",
      title="(a) Calibration, test set")
a.grid(color=GRID, lw=0.6); a.legend(loc="lower right", fontsize=6.3, handlelength=1.4, borderaxespad=0.2)

pairs = [("xgboost-l1_logreg", "XGB − LR"), ("xgboost-cart", "XGB − CART"), ("l1_logreg-cart", "LR − CART")]
for k, (metric, title, unit) in enumerate([("auc", "(b) ΔAUC per split", "ΔAUC (>0: first better)"),
                                           ("brier", "(c) ΔBrier per split", "ΔBrier (<0: first better)")]):
    a = ax[k + 1]
    for j, (key, lab) in enumerate(pairs):
        v = np.array(rj["paired_differences"][f"{key}_{metric}"]["values"])
        jitter = (np.arange(len(v)) - len(v) / 2) * 0.018
        a.scatter(v, np.full(len(v), j) + jitter, s=12, color=MUTED, alpha=0.8, zorder=2,
                  edgecolor="white", linewidth=0.4)
        a.plot([v.mean()] * 2, [j - 0.28, j + 0.28], color=INK, lw=2, zorder=3)
    a.axvline(0, color=MUTED, lw=1, ls="--", zorder=1)
    a.set_yticks(range(len(pairs))); a.set_yticklabels([p[1] for p in pairs])
    a.set_ylim(-0.6, len(pairs) - 0.4); a.invert_yaxis()
    a.set(xlabel=unit, title=title); a.grid(axis="x", color=GRID, lw=0.6)
fig.text(0.99, 0.005, "(b, c): grey dots = 10 repeated 75/25 splits of the development set, black bar = mean",
         ha="right", fontsize=6, color=MUTED)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(FIGURES / "fig1_predictive.pdf"); fig.savefig(FIGURES / "fig1_predictive.png")
fig.savefig(FIGURES / "fig1_predictive.svg")
plt.close(fig)

# ---------------------------------------------------------------- Figure 2: same-case explanations
cases = pd.read_csv(TABLES / "explanation_cases.csv").set_index("case")
byvar = pd.read_csv(TABLES / "explanation_cases_by_variable.csv")
SHORT = {"ExternalRiskEstimate": "ExternalRiskEstimate", "MSinceMostRecentInqexcl7days": "MSinceRecentInq",
         "NetFractionRevolvingBurden": "RevolvingBurden", "NumSatisfactoryTrades": "NumSatisfactoryTrades",
         "NumBank2NatlTradesWHighUtilization": "NumHighUtilTrades", "PercentInstallTrades": "PctInstallTrades",
         "AverageMInFile": "AverageMInFile", "MSinceOldestTradeOpen": "MSinceOldestTrade",
         "MSinceMostRecentTradeOpen": "MSinceRecentTrade", "PercentTradesNeverDelq": "PctNeverDelq",
         "MSinceMostRecentDelq": "MSinceRecentDelq", "NumInqLast6M": "NumInqLast6M",
         "NumRevolvingTradesWBalance": "NumRevTradesWBal", "NumInstallTradesWBalance": "NumInstTradesWBal",
         "NetFractionInstallBurden": "InstallBurden", "NumTotalTrades": "NumTotalTrades"}
show = [("disagreement", "Case D: disagreement"), ("special_codes", "Case S: special codes")]
fig, ax = plt.subplots(2, 1, figsize=(7.0, 6.7))
for a, (cname, title) in zip(ax, show):
    d = byvar[byvar.case == cname].copy()
    d["mx"] = d[["lr_contribution", "xgb_shap_sum"]].abs().max(axis=1)
    d = d.sort_values("mx", ascending=False).head(8).iloc[::-1]
    yy = np.arange(len(d))
    a.barh(yy + 0.19, d.lr_contribution, height=0.36, color=COL["l1_logreg"], label="LR contribution",
           edgecolor="white", linewidth=0.6)
    a.barh(yy - 0.19, d.xgb_shap_sum, height=0.36, color=COL["xgboost"], label="XGB TreeSHAP (sum)",
           edgecolor="white", linewidth=0.6)
    labels = [f"{'● ' if p else ''}{SHORT.get(v, v)} = {r}" for v, r, p in zip(d.variable, d.raw_value, d.on_cart_path)]
    a.set_yticks(yy); a.set_yticklabels(labels, fontsize=8.5)
    a.axvline(0, color=MUTED, lw=0.8)
    a.grid(axis="x", color=GRID, lw=0.6)
    c = cases.loc[cname]
    a.set_title(f"{title} (outcome: {'Bad' if c.y else 'Good'})\nP(Bad): LR {c.p_l1_logreg:.2f} | CART {c.p_cart:.2f} | XGB {c.p_xgboost:.2f}",
                fontsize=9.0)
    a.set_xlabel("log-odds attribution (vs. reference)")
h, l = ax[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=2, fontsize=8, bbox_to_anchor=(0.56, 0.045))
fig.text(0.01, 0.005, "● variable on the CART path. Codes −7/−8/−9 denote special states.\n"
         "Top 8 variables ranked by max(|LR|, |XGB|).", fontsize=6.5, color=MUTED, va="bottom")
fig.tight_layout(rect=(0, 0.12, 1, 1))
fig.savefig(FIGURES / "fig2_cases.pdf"); fig.savefig(FIGURES / "fig2_cases.png")
fig.savefig(FIGURES / "fig2_cases.svg")
plt.close(fig)

# ---------------------------------------------------------------- Figure 3: stability
pc = pd.read_csv(TABLES / "stability_per_case.csv")
arms = [("retrain", "l1_logreg", "LR: refit on 90% subsamples"),
        ("retrain", "xgboost", "XGB: refit on 90% subsamples"),
        ("seed_only", "xgboost", "XGB: seed only (full dev set)"),
        ("background_50", "xgboost", "XGB: background redraw, n=50"),
        ("background_200", "xgboost", "XGB: background redraw, n=200"),
        ("background_50", "l1_logreg", "LR: background redraw, n=50")]
fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.6), sharey=True)
for a, (metric, title) in zip(ax, [("top3_overlap", "(a) Top-3 variable overlap"),
                                   ("sign_agreement", "(b) Sign agreement"),
                                   ("rel_l1", "(c) Relative L1 distance")]):
    data = [pc[(pc.analysis == an) & (pc.method == me)][metric].dropna().to_numpy() for an, me, _ in arms]
    bp = a.boxplot(data, orientation="horizontal", widths=0.55, patch_artist=True, showfliers=True,
                   flierprops={"marker": ".", "markersize": 2, "markeredgecolor": MUTED},
                   medianprops={"color": INK, "lw": 1.5}, whiskerprops={"color": MUTED},
                   capprops={"color": MUTED})
    for patch, (_, me, _) in zip(bp["boxes"], arms):
        patch.set_facecolor(COL[me]); patch.set_alpha(0.55); patch.set_edgecolor(COL[me])
    a.set_title(title); a.grid(axis="x", color=GRID, lw=0.6)
ax[0].set_yticks(range(1, len(arms) + 1)); ax[0].set_yticklabels([x[2] for x in arms], fontsize=6.5)
ax[0].invert_yaxis()
ax[0].set_xlabel("mean over run pairs"); ax[1].set_xlabel("share of runs, majority sign")
ax[2].set_xlabel("0 = identical vectors")
fig.text(0.99, 0.005, f"Each box: {sj['n_eval_cases']} fixed test cases; {sj['n_runs']} runs/draws per arm; "
         "attributions grouped by original variable (log-odds).", ha="right", fontsize=6, color=MUTED)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(FIGURES / "fig3_stability.pdf"); fig.savefig(FIGURES / "fig3_stability.png")
fig.savefig(FIGURES / "fig3_stability.svg")
plt.close(fig)

# ---------------------------------------------------------------- LaTeX number macros
S = sj["summary"]


def f3(x): return f"{x:.3f}"
def f4(x): return f"{x:.4f}"
def pct(x): return f"{100 * x:.1f}\\%"


mac = {}
pop = aj["population"]; raw = aj["raw_audit"]
mac.update({"NRaw": f"{raw['n_rows']:,}", "NAllNine": str(raw["all_minus9_rows"]), "NPop": f"{pop['n_population']:,}",
            "NDev": f"{pop['n_dev']:,}", "NTest": f"{pop['n_test']:,}", "BadRatePop": pct(pop["bad_rate_population"]),
            "NCols": str(pop["n_features_design"]), "NPartialNine": str(raw["rows_with_any_minus9_not_all"])})
names = {"l1_logreg": "LR", "cart": "CART", "xgboost": "XGB"}
for m, n in names.items():
    t = pj["test"][m]; bt = pj["bootstrap"]
    mac[f"TestAUC{n}"] = f3(t["auc"]); mac[f"TestBrier{n}"] = f4(t["brier"]); mac[f"TestLL{n}"] = f3(t["logloss"])
    mac[f"AUCci{n}"] = f"[{f3(bt[m + '_auc']['lo'])}, {f3(bt[m + '_auc']['hi'])}]"
    mac[f"Brierci{n}"] = f"[{f4(bt[m + '_brier']['lo'])}, {f4(bt[m + '_brier']['hi'])}]"
    mac[f"CalSlope{n}"] = f"{pj['calibration'][m]['recal_slope']:.2f}"
    mac[f"CITL{n}"] = f"{pj['calibration'][m]['citl']:+.3f}"
    cv = next(r for r in pj["cv_summary"] if r["model"] == m)
    mac[f"CVAUC{n}"] = f3(cv["cv_auc_mean"]); mac[f"CVBrier{n}"] = f4(cv["cv_brier_mean"])
    rr = rj["per_model"][m]
    mac[f"RepAUC{n}"] = f3(rr["auc"]["mean"]); mac[f"RepAUCsd{n}"] = f3(rr["auc"]["sd"])
    mac[f"RepBrier{n}"] = f4(rr["brier"]["mean"])
for key, n in [("l1_logreg-xgboost", "LRXGB"), ("l1_logreg-cart", "LRCART"), ("cart-xgboost", "CARTXGB")]:
    a_, b_ = key.split("-")
    mac[f"dAUC{n}"] = f3(pj["test"][a_]["auc"] - pj["test"][b_]["auc"])
    mac[f"dAUCci{n}"] = f"[{f3(pj['bootstrap'][key + '_auc']['lo'])}, {f3(pj['bootstrap'][key + '_auc']['hi'])}]"
    mac[f"dBrier{n}"] = f4(pj["test"][a_]["brier"] - pj["test"][b_]["brier"])
    mac[f"dBrierci{n}"] = f"[{f4(pj['bootstrap'][key + '_brier']['lo'])}, {f4(pj['bootstrap'][key + '_brier']['hi'])}]"
pdiff = rj["paired_differences"]
for key, n in [("xgboost-l1_logreg", "XGBLR"), ("xgboost-cart", "XGBCART"), ("l1_logreg-cart", "LRCART")]:
    for metric in ("auc", "brier"):
        d = pdiff[f"{key}_{metric}"]
        fmt = f4
        mac[f"Rep{metric.capitalize()}Diff{n}"] = fmt(d["mean"])
        mac[f"Rep{metric.capitalize()}DiffMin{n}"] = fmt(d["min"])
        mac[f"Rep{metric.capitalize()}DiffMax{n}"] = fmt(d["max"])
        mac[f"Rep{metric.capitalize()}DiffPos{n}"] = str(d["n_positive"])
mac["NRepeats"] = str(rj["n_repeats"])
ag = pj["decision_agreement_at_0.5"]; ad = pj["abs_prob_difference"]
mac["AgreeLRXGB"] = pct(ag["l1_logreg|xgboost"]); mac["AgreeLRCART"] = pct(ag["l1_logreg|cart"])
mac["AgreeCARTXGB"] = pct(ag["cart|xgboost"])
mac["GtTenLRXGB"] = pct(ad["l1_logreg|xgboost"]["share_gt_0.10"]); mac["MeanAbsLRXGB"] = f3(ad["l1_logreg|xgboost"]["mean"])
cx = pj["complexity"]
mac.update({"LRnnz": str(cx["l1_logreg"]["nonzero_coefficients"]), "LRvars": str(cx["l1_logreg"]["original_variables_used"]),
            "CARTleaves": str(cx["cart"]["leaves"]), "CARTdepth": str(cx["cart"]["depth"]),
            "CARTvars": str(cx["cart"]["original_variables_used"]),
            "CARTpath": f"{cx['cart']['mean_path_length']:.1f}", "XGBtrees": str(cx["xgboost"]["trees"]),
            "XGBleaves": f"{cx['xgboost']['total_leaves']:,}", "XGBvars": str(cx["xgboost"]["original_variables_used"])})
sel = {r["model"]: r for r in pj["cv_summary"]}
mac["LRC"] = f"{sel['l1_logreg']['params']['C']:.3f}"
p = sel["xgboost"]["params"]
mac["XGBdepth"] = str(p["max_depth"]); mac["XGBeta"] = f"{p['learning_rate']:.3f}"
mac["XGBntrees"] = str(sel["xgboost"]["n_estimators"])
pc_ = sel["cart"]["params"]
mac["CARTminleaf"] = str(pc_["min_samples_leaf"]); mac["CARTccp"] = f"{pc_['ccp_alpha']:g}"

def g(arm, metric, stat="mean"):
    return S[arm][metric][stat]

mac["NEval"] = str(sj["n_eval_cases"]); mac["NRuns"] = str(sj["n_runs"])
for arm, n in [("retrain|l1_logreg", "RtLR"), ("retrain|xgboost", "RtXGB"), ("seed_only|xgboost", "SdXGB"),
               ("background_50|xgboost", "BgFiftyXGB"), ("background_200|xgboost", "BgTwoXGB"),
               ("background_50|l1_logreg", "BgFiftyLR"), ("background_200|l1_logreg", "BgTwoLR")]:
    for metric, short in [("top3_overlap", "TopThree"), ("top5_overlap", "TopFive"), ("sign_agreement", "Sign"),
                          ("rel_l1", "RelL")]:
        if metric in S[arm]:
            mac[f"{short}{n}"] = f"{g(arm, metric):.2f}"
    if "pred_sd" in S[arm]:
        mac[f"PredSD{n}"] = f3(g(arm, "pred_sd")); mac[f"PredSDninety{n}"] = f3(g(arm, "pred_sd", "p90"))
        mac[f"Flip{n}"] = pct(g(arm, "label_not_unanimous"))
mac["PredSDRtCART"] = f3(g("retrain|cart", "pred_sd")); mac["FlipRtCART"] = pct(g("retrain|cart", "label_not_unanimous"))
mac["PredSDninetyRtCART"] = f3(g("retrain|cart", "pred_sd", "p90"))
mac["PathJacRtCART"] = f"{g('retrain|cart', 'path_var_jaccard'):.2f}"
lr = sj["lr_retrain"]
mac.update({"LRnnzMin": str(lr["nonzero_count_min"]), "LRnnzMax": str(lr["nonzero_count_max"]),
            "LRalways": str(lr["columns_always_selected"]), "LRsometimes": str(lr["columns_sometimes_selected"]),
            "LRsignflip": str(lr["sign_flips_columns"])})
cr = sj["cart_retrain"]
mac["CARTrootTop"] = max(cr["root_variable_counts"], key=cr["root_variable_counts"].get)
mac["CARTrootTopN"] = str(max(cr["root_variable_counts"].values()))
mac["CARTleavesMin"] = str(cr["leaves_min"]); mac["CARTleavesMax"] = str(cr["leaves_max"])
bgb = sj["background_base_values"]
mac["BgBaseSDFiftyXGB"] = f3(bgb["50"]["xgb_base_sd"]); mac["BgBaseSDTwoXGB"] = f3(bgb["200"]["xgb_base_sd"])

for arm, n in [("retrain|l1_logreg", "RtLR"), ("retrain|xgboost", "RtXGB"), ("seed_only|xgboost", "SdXGB"),
               ("background_50|xgboost", "BgFiftyXGB"), ("background_50|l1_logreg", "BgFiftyLR"),
               ("background_200|xgboost", "BgTwoXGB")]:
    mac[f"Conflict{n}"] = pct(g(arm, "sign_conflict"))

# Facts quoted in the text about specific coefficients, cases and references.
coef = pd.read_csv(TABLES / "lr_coefficients.csv").set_index("column")
lrs = pd.read_csv(TABLES / "stability_lr_selection.csv").set_index("column")
for col, n in [("MaxDelqEver=delq_120plus", "MDEOneTwenty"), ("MaxDelq2PublicRecLast12M=delq_120plus", "MDTwelveOneTwenty"),
               ("MSinceMostRecentInqexcl7days==-7", "InqSevenOR"), ("MSinceMostRecentInqexcl7days==-8", "InqEightOR")]:
    mac[f"OR{n}"] = f"{coef.loc[col, 'odds_ratio_per_unit']:.2f}"
    mac[f"Sel{n}"] = str(int(round(lrs.loc[col, "selected_frac"] * sj["n_runs"])))
    mac[f"Neg{n}"] = str(int(round(lrs.loc[col, "selected_frac"] * (1 - lrs.loc[col, "positive_frac_when_selected"]) * sj["n_runs"])))
cat = pd.read_csv(TABLES / "audit_categorical_codes.csv")
br = lambda v, c: cat[(cat.variable == v) & (cat.code == c)].bad_rate.iloc[0]
mac["BadRateMDEOneTwenty"] = pct(br("MaxDelqEver", 3)); mac["BadRateMDENever"] = pct(br("MaxDelqEver", 8))
mac["AllNineBadRate"] = pct(raw["all_minus9_label_counts"]["Bad"] / raw["all_minus9_rows"])
sc = byvar[byvar.case == "special_codes"].set_index("variable")
mac["CaseSAvgMInFile"] = str(sc.loc["AverageMInFile", "raw_value"])
mac["CaseSLRAvg"] = f"{sc.loc['AverageMInFile', 'lr_contribution']:.2f}"
mac["CaseSXGBAvg"] = f"{sc.loc['AverageMInFile', 'xgb_shap_sum']:.2f}"
mac["CaseSNSpecial"] = str(int(cases.loc["special_codes", "n_special_states"]))
for cn, n in [("special_codes", "S"), ("disagreement", "D"), ("near_threshold", "T")]:
    c = cases.loc[cn]
    mac[f"Case{n}LR"] = f"{c.p_l1_logreg:.2f}"; mac[f"Case{n}CART"] = f"{c.p_cart:.2f}"
    mac[f"Case{n}XGB"] = f"{c.p_xgboost:.2f}"; mac[f"Case{n}Leaf"] = str(int(c.cart_leaf_train_n))
pathS = ej["cart_paths"]["special_codes"]["steps"]
mac["CaseSMissingSplits"] = str(sum(s["routed"] == "missing" for s in pathS))
mac["CaseSPathLen"] = str(len(pathS))
swap = pd.read_csv(TABLES / "stability_reference_swap.csv")
for ref, n in [("dev_all", "All"), ("dev_good_only", "Good"), ("dev_bad_only", "Bad")]:
    r = swap[(swap.case == "near_threshold") & (swap.reference == ref)].iloc[0]
    mac[f"SwapT{n}"] = f"{r.sum_attr:+.2f}"; mac[f"SwapBase{n}"] = f"{r.base_logit:+.2f}"
mac["CVAUCCARTdepthTwo"] = f3(pd.read_csv(TABLES / "cart_depth_profile_cv.csv").set_index("param_max_depth").loc[2, "auc"])

with open(RESULTS / "report_numbers.tex", "w") as f:
    f.write("% Auto-generated by scripts/05_figures_tables.py from results/*.json. Do not edit.\n")
    for k, v in sorted(mac.items()):
        # Typographic minus for negative numbers (text-mode hyphens otherwise).
        if any(ch.isdigit() for ch in v):
            v = v.replace("-", "$-$")
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")

# ---------------------------------------------------------------- Table 1 (LaTeX)
rows = []
for m in M:
    n = names[m]
    c = cx[m]
    if m == "l1_logreg":
        struct = f"{c['nonzero_coefficients']} non-zero coef.; {c['original_variables_used']} variables"
    elif m == "cart":
        struct = f"depth {c['depth']}, {c['leaves']} leaves; {c['original_variables_used']} variables"
    else:
        struct = f"{c['trees']} trees, {c['total_leaves']:,} leaves; {c['original_variables_used']} variables"
    rows.append(f"{LAB[m]} & \\CVAUC{n} & \\RepAUC{n}\\,$\\pm$\\,\\RepAUCsd{n} & \\TestAUC{n} \\newline {{\\scriptsize \\AUCci{n}}} "
                f"& \\TestBrier{n} \\newline {{\\scriptsize \\Brierci{n}}} & \\CalSlope{n} & {struct} \\\\")
head = ("\\begin{tabular}{@{}p{1.9cm}ccp{2.0cm}p{2.15cm}cp{3.4cm}@{}}\n\\toprule\n"
        "Model & CV AUC & Repeated AUC & Test AUC & Test Brier & Slope & Structure \\\\\n\\midrule\n")
with open(TABLES / "table1_predictive.tex", "w") as f:
    f.write(head + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

log.note(f"wrote {len(mac)} macros")
log.save()
