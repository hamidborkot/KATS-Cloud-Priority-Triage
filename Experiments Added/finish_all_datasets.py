# -*- coding: utf-8 -*-
"""
finish_all_datasets.py  --  ONE Kaggle-notebook cell (no command-line arguments; safe inside Jupyter).
Reads the cluster-A result files from the all-in-one dataset and produces:
  * ROC-AUC (macro one-vs-rest) per model/seed with 95% CI, from clusterA_predictions.csv (+ cross-check of PR-AUC/FP/FN)
  * LaTeX tables: precision/PR-AUC/Brier/ECE/FP/FN (+ROC-AUC, always-High), confusion matrices, McNemar with direction,
    ablation (one source), GoogleCluster robustness, NetBenefit with false-positive cost, IR-sweep (fixed test set)
  * calibration: per-seed ECE-High from bins with CI, pooled reliability table with Wilson intervals
  * a coverage report (what exists / what is still missing) printed at the end
Set ROOT below if your dataset folder differs.
"""
import os, re, glob, json, warnings
import numpy as np, pandas as pd
from scipy import stats
warnings.filterwarnings("ignore")
ROOT = os.environ.get("ROOT", "/kaggle/input/datasets/mdhamidborkottulla/all-in-one")
OUT = os.environ.get("OUT_DIR", "/kaggle/working/final_tables"); os.makedirs(OUT, exist_ok=True)

# ---------------- inventory ----------------
files = {}
for f in glob.glob(os.path.join(ROOT, "**", "*"), recursive=True):
    if os.path.isfile(f) and f.lower().endswith((".csv", ".txt", ".json", ".parquet")):
        files.setdefault(os.path.splitext(os.path.basename(f))[0], f)
print(f"{len(files)} files under {ROOT}")
for k in sorted(files): print(f"  {k:60s} {os.path.getsize(files[k])/1e6:8.2f} MB")
def load(stem, **kw):
    f = files.get(stem)
    if f is None: return None
    try: return pd.read_parquet(f) if f.endswith(".parquet") else pd.read_csv(f, sep=None, engine="python", **kw) if os.path.getsize(f) < 5e7 else pd.read_csv(f, **kw)
    except Exception as e: print("  !! could not read", stem, e); return None
def first(*stems):
    for s in stems:
        d = load(s)
        if d is not None: return d
ms  = first("clusterA_main_metrics_per_seed", "FINAL_appendix_per_seed_metrics")
cm  = load("clusterA_confusion_matrices"); cal = load("clusterA_calibration_bins")
ah  = load("FINAL_always_high_summary"); mc = load("FINAL_table12_mcnemar_seed2026")
ab  = load("FINAL_consistent_ablation_summary"); n2 = load("N2_googlecluster_frequency_binning_summary"); n1 = load("N1b_googlecluster_grouped_split_summary")
irs = load("clusterA_ir_fixed_test_sensitivity"); mop = load("clusterA_matched_operating_points"); tmp = load("clusterA_temporal_sweep")
DS = ["GoogleCluster", "ITIncident", "MultiCloud", "CICIDS2017"]
ORDER = ["KATS", "LightGBM", "XGBoost", "RandomForest", "BalancedRF", "MLP", "LogReg", "NaiveBayes"]
def tci(x): x = np.asarray(x, float); return stats.t.ppf(.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan
report = {}

# ---------------- ROC-AUC from predictions ----------------
auc = None
try:
    from sklearn.metrics import roc_auc_score, average_precision_score
    pf = files.get("clusterA_predictions")
    if pf is None: raise FileNotFoundError("clusterA_predictions.csv not found under ROOT")
    pr = pd.read_csv(pf); print("\nprediction columns:", list(pr.columns)); print(pr.head(2).to_string())
    low = {c.lower(): c for c in pr.columns}; pick = lambda cands: next((low[c] for c in cands if c in low), None)
    YT = pick(["y_true", "true_class", "true_label", "true", "label", "y"]); YP = pick(["y_pred", "pred_class", "predicted_class", "pred", "prediction"])
    D_, M_, S_ = pick(["dataset"]), pick(["model"]), pick(["seed"])
    PC = [c for c in pr.columns if re.match(r"(p_|proba_|prob_|probability_)", c, re.I)]
    if not all([YT, D_, M_, S_]) or len(PC) != 3: raise ValueError(f"cannot auto-detect columns: y_true={YT} dataset={D_} model={M_} seed={S_} prob={PC}. Edit the pick() lists.")
    classes = sorted(pr[YT].astype(str).unique()); strip = lambda c: re.sub(r"^(p_|proba_|prob_|probability_)", "", c, flags=re.I).lower()
    PC = sorted(PC, key=lambda c: [k.lower() for k in classes].index(strip(c)) if strip(c) in [k.lower() for k in classes] else 99)
    hi = classes.index("High"); rows = []
    for (d, m, s), g in pr.groupby([D_, M_, S_]):
        y = g[YT].astype(str).values; Y = np.stack([(y == c).astype(int) for c in classes], 1); P = g[PC].values
        r = dict(dataset=d, model=m, seed=s, n=len(g), AUC_macro_ovr=roc_auc_score(Y, P, average="macro"), AUC_High_ovr=roc_auc_score(Y[:, hi], P[:, hi]),
                 PRAUC_High_recomputed=average_precision_score(Y[:, hi], P[:, hi]))
        if YP:
            yp = g[YP].astype(str).values; r["FP_High_recomputed"] = int(((yp == "High") & (y != "High")).sum()); r["FN_High_recomputed"] = int(((yp != "High") & (y == "High")).sum())
        rows.append(r)
    auc = pd.DataFrame(rows); auc.to_csv(f"{OUT}/roc_auc_per_seed.csv", index=False)
    aucS = auc.groupby(["dataset", "model"]).AUC_macro_ovr.agg(["mean", "std", tci]).reset_index().rename(columns={"tci": "ci95"}); aucS.to_csv(f"{OUT}/roc_auc_mean_sd_ci95.csv", index=False)
    print(aucS.round(4).to_string())
    if ms is not None:
        j = auc.merge(ms, on=["dataset", "model", "seed"], suffixes=("", "_rep"))
        for a, b in [("PRAUC_High_recomputed", "PRAUC_High"), ("FP_High_recomputed", "FP_High"), ("FN_High_recomputed", "FN_High")]:
            if a in j and b in j: print(f"  cross-check max|{a} - reported {b}| = {np.abs(j[a] - j[b]).max():.6g}")
    report["ROC-AUC"] = f"OK ({len(auc)} dataset/model/seed rows; expected 160)"
except Exception as e:
    report["ROC-AUC"] = f"MISSING: {e}"; print("!! ROC-AUC step:", e)

# ---------------- metrics table ----------------
if ms is not None:
    keys = ["RecallH", "PrecH", "MacroF1", "Kappa", "PRAUC_High", "Brier", "ECE", "FP_High", "FN_High"]
    S = pd.DataFrame([dict(dataset=d, model=m, n_seeds=len(g), **{f"{k}_{a}": v for k in keys for a, v in (("mean", g[k].mean()), ("sd", g[k].std(ddof=1)), ("ci95", tci(g[k])))})
                      for (d, m), g in ms.groupby(["dataset", "model"])]); S.to_csv(f"{OUT}/metrics_mean_sd_ci95.csv", index=False)
    aucm = {} if auc is None else auc.groupby(["dataset", "model"]).AUC_macro_ovr.agg(["mean", "std"]).to_dict("index")
    L = [r"\begin{table*}[t]", r"\caption{High-class precision, PR-AUC, macro one-vs-rest ROC-AUC, Brier (mean of the three one-vs-rest Brier scores), class-wise ECE (10 equal-width bins) and High-class FP/FN on the original test partitions; five-seed mean $\pm$ SD (test $n$: 12{,}000 GoogleCluster and CICIDS2017; 4{,}984 ITIncident; 200 MultiCloud). Always-High is the trivial reference (seed 2026). Per-seed values and 95\% intervals: reproduction package.}",
         r"\label{tab:cm-prauc}", r"\scriptsize", r"\begin{tabular}{@{}llrrrrrrr@{}}", r"\toprule", r"Dataset & Model & Prec.$_H$ & PR-AUC$_H$ & AUC & Brier & ECE & FP$_H$ & FN$_H$ \\ \midrule"]
    for d in DS:
        for i, m in enumerate(ORDER):
            g = ms[(ms.dataset == d) & (ms.model == m)]
            if g.empty: continue
            f = lambda k, n=4: f"${g[k].mean():.{n}f}\\pm{g[k].std(ddof=1):.{n}f}$"
            a = aucm.get((d, m)); at = f"${a['mean']:.4f}\\pm{a['std']:.4f}$" if a else "--"
            L.append(f"{d if i == 0 else ''} & {m} & {f('PrecH')} & {f('PRAUC_High')} & {at} & {f('Brier')} & {f('ECE')} & ${g.FP_High.mean():.1f}$ & ${g.FN_High.mean():.1f}$ \\\\")
        if ah is not None and d in set(ah.dataset):
            a = ah[ah.dataset == d].iloc[0]; L.append(f" & AlwaysHigh & ${a.PrecH_mean:.4f}$ & -- & $0.5$ & -- & -- & ${a.FP_High_mean:.0f}$ & $0$ \\\\")
        L.append(r"\midrule")
    L[-1] = r"\bottomrule"; L += [r"\end{tabular}", r"\end{table*}"]; open(f"{OUT}/table_cm_prauc.tex", "w").write("\n".join(L))
    # NetBenefit
    ix = S.set_index(["dataset", "model"]); nb = []
    for d, b in {"GoogleCluster": "RandomForest", "ITIncident": "LightGBM", "MultiCloud": "NaiveBayes", "CICIDS2017": "LightGBM"}.items():
        k, o = ix.loc[(d, "KATS")], ix.loc[(d, b)]; sav = (o.FN_High_mean - k.FN_High_mean) * 50; dfp = k.FP_High_mean - o.FP_High_mean
        row = dict(dataset=d, competitor=b, FN_KATS=k.FN_High_mean, FN_comp=o.FN_High_mean, FP_KATS=k.FP_High_mean, FP_comp=o.FP_High_mean, net_usd_FPcost0=sav, break_even_cost_per_FP=(sav / dfp if sav > 0 and dfp > 0 else np.nan))
        for c in [0.1, 0.5, 1.0]: row[f"net_usd_FPcost{c}"] = sav - c * dfp
        nb.append(row)
    pd.DataFrame(nb).to_csv(f"{OUT}/netbenefit_recomputed_with_fp_cost.csv", index=False)
    report["metrics/PR-AUC/Brier/ECE/FP/FN"] = f"OK ({len(ms)} rows; expected 160)"
else: report["metrics/PR-AUC/Brier/ECE/FP/FN"] = "MISSING clusterA_main_metrics_per_seed"

# ---------------- confusion matrices ----------------
if cm is not None:
    c26 = cm[cm.seed == 2026].pivot_table(index=["dataset", "model", "true_class"], columns="predicted_class", values="count", aggfunc="sum").reset_index()[["dataset", "model", "true_class", "High", "Medium", "Low"]]
    c26.to_csv(f"{OUT}/confusion_matrices_seed2026_all_models.csv", index=False)
    cm.groupby(["dataset", "model", "true_class", "predicted_class"])["count"].mean().reset_index().to_csv(f"{OUT}/confusion_matrices_mean_over_seeds.csv", index=False)
    T = [r"\begin{table*}[t]", r"\caption{Confusion counts (seed 2026, original test split); columns: predicted High/Medium/Low. All models, datasets and seeds: reproduction package.}", r"\label{tab:cm}", r"\scriptsize",
         r"\begin{tabular}{@{}llrrrrrr@{}}", r"\toprule", r" & & \multicolumn{3}{c}{True High} & \multicolumn{3}{c}{True Medium}\\ \midrule"]
    get = lambda d, m, tc: c26[(c26.dataset == d) & (c26.model == m) & (c26.true_class == tc)].iloc[0][["High", "Medium", "Low"]].astype(int).tolist()
    for d in ["ITIncident", "CICIDS2017", "GoogleCluster", "MultiCloud"]:
        for m in ["KATS", "LightGBM", "XGBoost", "LogReg"]:
            if c26[(c26.dataset == d) & (c26.model == m)].empty: continue
            T.append(f"{d} & {m} & " + " & ".join(map(str, get(d, m, "High") + get(d, m, "Medium"))) + r" \\")
        T.append(r"\midrule")
    T[-1] = r"\bottomrule"; T += [r"\end{tabular}", r"\end{table*}"]; open(f"{OUT}/table_confusion.tex", "w").write("\n".join(T))
    report["confusion matrices"] = f"OK ({cm.groupby(['dataset','model','seed']).ngroups} matrices; expected 160)"
else: report["confusion matrices"] = "MISSING"

# ---------------- calibration ----------------
if cal is not None:
    hi_ = cal[cal.class_index == 0].dropna(subset=["count"]).copy()       # class 0 = High
    def wil(k, n, z=1.96):
        p = k / n; den = 1 + z * z / n; c = (p + z * z / (2 * n)) / den; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den; return c - h, c + h
    E = pd.DataFrame([dict(dataset=d, model=m, seed=s, ECE_High=np.nansum(g["count"] / g["count"].sum() * (g.empirical_frequency - g.mean_confidence).abs())) for (d, m, s), g in hi_.groupby(["dataset", "model", "seed"])])
    E.groupby(["dataset", "model"]).ECE_High.agg(["mean", "std", tci]).reset_index().rename(columns={"tci": "ci95"}).to_csv(f"{OUT}/ece_high_from_bins_mean_sd_ci95.csv", index=False)
    pool = []
    for (d, m, b), g in hi_.groupby(["dataset", "model", "bin"]):
        n = g["count"].sum()
        if n > 0:
            k = (g["count"] * g.empirical_frequency).sum(); lo, up = wil(k, n)
            pool.append(dict(dataset=d, model=m, bin=b, count_total_over_seeds=int(n), mean_confidence=(g["count"] * g.mean_confidence).sum() / n, empirical_frequency=k / n, wilson_lo=lo, wilson_hi=up))
    pd.DataFrame(pool).to_csv(f"{OUT}/reliability_High_pooled_wilson.csv", index=False); report["calibration bins + CIs"] = "OK"
else: report["calibration bins + CIs"] = "MISSING"

# ---------------- McNemar / ablation / GC / IR sweep ----------------
if mc is not None:
    X = [r"\begin{table*}[t]", r"\caption{Exact McNemar tests (seed 2026; overall three-class correctness, not Recall$_H$/Macro-F1/$\kappa$), Holm-corrected over the 28 live-dataset tests. $b_{10}$: KATS correct, baseline wrong; $b_{01}$: reverse. Non-significance is not equivalence.}",
         r"\label{tab:mcnemar-fw}", r"\small", r"\begin{tabular}{@{}llrrrlrc@{}}", r"\toprule", r"Dataset & Baseline & $b_{10}$ & $b_{01}$ & $\Delta$acc & Direction & $p_{\mathrm{Holm}}$ & Sig. \\ \midrule"]
    for _, r in mc.iterrows():
        ph = r["p_holm_all_28_live_tests"]; X.append(f"{r.dataset} & {r.baseline} & {int(r.b10_KATS_correct_baseline_wrong)} & {int(r.b01_KATS_wrong_baseline_correct)} & {r.overall_accuracy_difference_KATS_minus_baseline:+.4f} & {str(r.direction).replace('_',' ').lower()} & {'<0.0001' if ph < 1e-4 else f'{ph:.4f}'} & {chr(92)+'checkmark' if str(r.significant_holm_0_05).upper()=='TRUE' else ''} \\\\")
    X += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]; open(f"{OUT}/table_mcnemar.tex", "w").write("\n".join(X)); report["McNemar + direction"] = f"OK ({len(mc)} tests; expected 28)"
else: report["McNemar + direction"] = "MISSING"
if ab is not None:
    ab = ab[~((ab.variant == "T_NoResampling") & (ab.resampling_method == "none"))]; ab.to_csv(f"{OUT}/ablation_single_source.csv", index=False)
    A = [r"\begin{table*}[t]", r"\caption{Ablation, five-seed mean $\pm$ SD from one results file; $\Delta$ from unrounded means. T\_NoResampling omitted where the IR$>3$ gate is closed (identical to T\_Full by construction).}", r"\label{tab:ablation}", r"\scriptsize",
         r"\begin{tabular}{@{}llrrrrr@{}}", r"\toprule", r"Dataset & Variant & Recall$_H$ & Macro-F1 & $\kappa$ & $\Delta$Recall$_H$ & $\Delta$Macro-F1 \\ \midrule"]
    for d in DS:
        for i, (_, r) in enumerate(ab[ab.dataset == d].iterrows()):
            A.append(f"{d if i == 0 else ''} & {r.variant.replace('_', chr(92)+'_')} & ${r.RecallH_mean:.4f}\\pm{r.RecallH_sd:.4f}$ & ${r.MacroF1_mean:.4f}\\pm{r.MacroF1_sd:.4f}$ & ${r.Kappa_mean:.4f}\\pm{r.Kappa_sd:.4f}$ & ${r.Delta_RecallH:+.4f}$ & ${r.Delta_MacroF1:+.4f}$ \\\\")
        A.append(r"\midrule")
    A[-1] = r"\bottomrule"; A += [r"\end{tabular}", r"\end{table*}"]; open(f"{OUT}/table_ablation.tex", "w").write("\n".join(A)); report["ablation"] = "OK"
else: report["ablation"] = "MISSING"
if n1 is not None or n2 is not None:
    G = [r"\begin{table}[t]", r"\caption{GoogleCluster robustness, five-seed mean $\pm$ SD: (a) frequency-balanced tercile binning; (b) job-grouped split by \texttt{collection\_id}.}", r"\label{tab:gc-robust}", r"\scriptsize", r"\begin{tabular}{@{}llrrr@{}}", r"\toprule", r"Setting & Model & Recall$_H$ & Macro-F1 & $\kappa$ \\ \midrule"]
    for lab, t in [("(a) tercile", n2), ("(b) grouped", n1)]:
        if t is None: continue
        for i, (_, r) in enumerate(t.iterrows()): G.append(f"{lab if i == 0 else ''} & {r.model} & ${r.RecallH_mean:.4f}\\pm{r.RecallH_sd:.4f}$ & ${r.MacroF1_mean:.4f}\\pm{r.MacroF1_sd:.4f}$ & ${r.Kappa_mean:.4f}\\pm{r.Kappa_sd:.4f}$ \\\\")
        G.append(r"\midrule")
    G[-1] = r"\bottomrule"; G += [r"\end{tabular}", r"\end{table}"]; open(f"{OUT}/table_gc_robust.tex", "w").write("\n".join(G)); report["GoogleCluster alt-binning + grouped split"] = "OK"
else: report["GoogleCluster alt-binning + grouped split"] = "MISSING"
if irs is not None:
    S2 = irs.groupby(["training_target_ir", "model"]).agg(seeds=("seed", "nunique"), Recall=("RecallH", "mean"), Recall_sd=("RecallH", "std"), MacroF1=("MacroF1", "mean"), MacroF1_sd=("MacroF1", "std"), Kappa=("Kappa", "mean"), Kappa_sd=("Kappa", "std"),
                                                          PRAUC=("PRAUC_High", "mean"), FP=("FP_High", "mean")).reset_index(); S2.to_csv(f"{OUT}/ir_sweep_fixed_test_summary.csv", index=False)
    report["IR sweep (fixed test)"] = f"{'OK' if len(irs) >= 100 else 'INCOMPLETE'}: {len(irs)} rows, {irs.seed.nunique()} seed(s) (expected 100 rows = 5 seeds x 5 IR x 4 models)"
else: report["IR sweep (fixed test)"] = "MISSING"
report["matched-recall comparison"] = "OK" if mop is not None and len(mop) >= 40 else "MISSING/INCOMPLETE"
report["temporal chronological"] = f"OK ({len(tmp)} rows; expected 36)" if tmp is not None else "MISSING"
print("\n================ COVERAGE REPORT ================")
for k, v in report.items(): print(f"{k:45s} {v}")
json.dump(report, open(f"{OUT}/coverage_report.json", "w"), indent=2); print("\nOutputs in", OUT, sorted(os.listdir(OUT)))
