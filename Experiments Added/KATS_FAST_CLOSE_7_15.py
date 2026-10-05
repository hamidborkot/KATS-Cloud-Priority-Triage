"""Read-only, no-training checker for KATS repair results. Sources are never modified.
Output tables are DRAFT numerical exports, not publication/acceptance certification.
"""
from pathlib import Path
from itertools import product
import io, json, re, zipfile, hashlib, datetime
import pandas as pd
import numpy as np

SOURCE = globals().get("SOURCE", "/kaggle/working/KATS_REVIEWER_REPAIR_R1_RESULTS.zip")
MAIN_FILE = globals().get("MAIN_FILE", "")
ABLATION_FILE = globals().get("ABLATION_FILE", "")
OUT_ROOT = globals().get("OUT_ROOT", "/kaggle/working/KATS_FAST_CHECKS")
DATASETS = ["GoogleCluster", "ITIncident", "MultiCloud", "CICIDS2017"]
MODELS = ["KATS", "LightGBM", "XGBoost", "RandomForest", "BalancedRF", "MLP", "LogReg", "NaiveBayes"]
SEEDS = [42, 7, 13, 99, 2026]
VARIANTS = ["T_NoResampling", "T_NoAsymLoss", "T_NoCalibNB", "T_NoStacking"]
METRICS = ["RecallH", "PrecH", "F1H", "MacroF1", "Kappa", "PRAUC_High", "Brier", "ECE", "FP_High", "FN_High"]
ALIASES = {
    "dataset": ["dataset", "Dataset"], "model": ["model", "Model"],
    "seed": ["seed", "Seed", "random_seed"], "variant": ["variant", "Variant"],
    "experiment": ["experiment", "Experiment"],
    "RecallH": ["RecallH", "Recall_H", "RecallHigh"],
    "PrecH": ["PrecH", "Precision_H", "PrecHigh", "PrecisionHigh"],
    "F1H": ["F1H", "F1_H", "F1High"], "MacroF1": ["MacroF1", "Macro_F1", "macro_f1"],
    "Kappa": ["Kappa", "kappa"], "PRAUC_High": ["PRAUC_High", "PRAUC_H", "PR_AUC_H"],
    "Brier": ["Brier", "brier"], "ECE": ["ECE", "ece"],
    "FP_High": ["FP_High", "FP_H"], "FN_High": ["FN_High", "FN_H"]
}

def normalize(frame):
    d = frame.copy()
    d.columns = [str(c).strip() for c in d.columns]
    for target, options in ALIASES.items():
        hits = [c for c in options if c in d.columns]
        if len(hits) > 1:
            raise ValueError(f"Ambiguous aliases for {target}: {hits}")
        if hits:
            d = d.rename(columns={hits[0]: target})
    for c in ["dataset", "model", "variant", "experiment"]:
        if c in d:
            d[c] = d[c].astype(str).str.strip()
    if "model" in d:
        d["model"] = d["model"].replace({"Log. Reg.": "LogReg", "LogisticRegression": "LogReg", "Random Forest": "RandomForest", "BalancedRandomForest": "BalancedRF", "Naive Bayes": "NaiveBayes"})
    if "variant" in d:
        d["variant"] = d["variant"].replace({"T_NoSMOTE": "T_NoResampling"})
    if "seed" in d:
        d["seed"] = pd.to_numeric(d["seed"], errors="raise")
        if d["seed"].isna().any() or not np.all(d["seed"] == np.floor(d["seed"])):
            raise ValueError("Missing/noninteger seed")
        d["seed"] = d["seed"].astype(int)
    for c in METRICS:
        if c in d:
            d[c] = pd.to_numeric(d[c], errors="raise")
    return d

def main_slice(d, name):
    if not {"dataset", "model", "seed", "RecallH", "MacroF1", "Kappa"}.issubset(d):
        return None
    if "experiment" in d:
        x = d[d["experiment"].eq("main")].copy()
    elif "main" in name.lower() or set(MODELS).issubset(set(d["model"])):
        x = d.copy()
    else:
        return None
    if "variant" in x:
        x = x[x["variant"].eq("T_Full")]
    return x if len(x) else None

def ablation_slice(d, name):
    if not {"dataset", "seed", "variant", "RecallH", "MacroF1", "Kappa"}.issubset(d):
        return None
    if "experiment" in d:
        x = d[d["experiment"].str.startswith("ablation")].copy()
    elif "ablation" in name.lower():
        x = d.copy()
    else:
        return None
    if "model" in x:
        x = x[x["model"].eq("KATS")]
    x = x[x["variant"].isin(["T_Full"] + VARIANTS)]
    return x if len(x) else None

def compare_full(main, ablation, variant="T_Full"):
    a = main[main["model"].eq("KATS")].copy()
    b = ablation[ablation["variant"].eq(variant)].copy()
    keys = ["dataset", "seed"]
    if not len(b):
        return pd.DataFrame()
    if a.duplicated(keys).any() or b.duplicated(keys).any():
        raise ValueError(f"Duplicate pairing keys for {variant}")
    metrics = [c for c in METRICS if c in a and c in b]
    paired = a[keys + metrics].merge(b[keys + metrics], on=keys, how="outer", suffixes=("_main", "_variant"), indicator=True, validate="one_to_one")
    rows = []
    for _, r in paired.iterrows():
        for m in metrics:
            p, q = r[m + "_main"], r[m + "_variant"]
            ok = r["_merge"] == "both" and pd.notna(p) and pd.notna(q) and np.isclose(p, q, atol=1e-8, rtol=0)
            rows.append({"dataset": r.dataset, "seed": int(r.seed), "variant": variant, "metric": m, "main": p, "comparison": q, "difference": q - p, "match": bool(ok)})
    return pd.DataFrame(rows)

def fingerprint(d, keys):
    cols = keys + sorted(c for c in METRICS if c in d)
    return hashlib.sha256(d[cols].sort_values(keys).to_csv(index=False).encode()).hexdigest()

def choose(candidates, override, keys):
    if override:
        if override not in candidates:
            raise ValueError(f"Selected file not a usable candidate: {override}. Candidates: {list(candidates)}")
        return override, candidates[override]
    if not candidates:
        raise ValueError("No per-seed candidate found. Summary-only rows cannot certify coverage; export existing per-seed results, not a new model run.")
    groups = {}
    for name, d in candidates.items():
        groups.setdefault(fingerprint(d, keys), []).append(name)
    if len(groups) != 1:
        raise ValueError(f"Different candidate tables found: {list(candidates)}. Set MAIN_FILE/ABLATION_FILE to an exact member name; no merging or best-score selection performed.")
    name = sorted(candidates)[0]
    return name, candidates[name]

def summarize(d, keys):
    ms = [m for m in METRICS if m in d]
    out = d.groupby(keys, dropna=False)[ms].agg(["mean", "std"])
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    out = out.reset_index()
    counts = d.groupby(keys, dropna=False).agg(n_rows=("seed", "size"), n_seeds=("seed", "nunique")).reset_index()
    return out.merge(counts, on=keys, validate="one_to_one")

def inspect(source=SOURCE, out_root=OUT_ROOT, main_file=MAIN_FILE, ablation_file=ABLATION_FILE):
    p = Path(source)
    if not p.exists():
        raise FileNotFoundError(f"SOURCE does not exist: {source}. Point SOURCE at the EXISTING repair ZIP or its result directory. No retraining needed.")
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out = Path(out_root) / stamp
    out.mkdir(parents=True, exist_ok=False)
    z = zipfile.ZipFile(p) if p.is_file() and zipfile.is_zipfile(p) else None
    if z:
        inventory = [{"name": i.filename, "bytes": i.file_size} for i in z.infolist() if not i.is_dir()]
        read = z.read
    elif p.is_dir():
        inventory = [{"name": f.relative_to(p).as_posix(), "bytes": f.stat().st_size} for f in p.rglob("*") if f.is_file() and str(out_root) not in str(f)]
        read = lambda n: (p / n).read_bytes()
    else:
        raise ValueError("SOURCE must be a ZIP or directory")
    pd.DataFrame(inventory).to_csv(out / "inventory.csv", index=False)
    checks, frames, ledger, metadata = [], {}, [], []
    def check(name, status, detail):
        checks.append({"check": name, "status": status, "detail": str(detail)})
        print(f"[{status}] {name}: {detail}", flush=True)
    print(f"SOURCE: {p}\nOUTPUT: {out}\nNo model fitting or source modification.\n", flush=True)
    for item in inventory:
        name, size = item["name"], item["bytes"]
        low = name.lower()
        if size > 50 * 1024**2:
            check("large_file_skipped", "INFO", f"{name}: {size} bytes; prediction arrays are not validated by this fast checker")
            continue
        if low.endswith(".csv"):
            try:
                head = pd.read_csv(io.BytesIO(read(name)), nrows=3)
                h = normalize(head)
                if {"dataset", "seed", "MacroF1", "Kappa"}.issubset(h):
                    frames[name] = normalize(pd.read_csv(io.BytesIO(read(name))))
                elif {"stratum", "median_rho", "top1_match"}.issubset(head):
                    f = pd.read_csv(io.BytesIO(read(name)))
                    f.to_csv(out / ("fidelity_" + hashlib.sha256(name.encode()).hexdigest()[:8] + ".csv"), index=False)
                    print(f"\nFOUND FIDELITY SUMMARY: {name}\n{f.to_string(index=False)}\n", flush=True)
                    check("fidelity_summary", "FOUND_NOT_IDENTITY_PROOF", name)
                if any(t in low for t in ["fail", "error", "status"]):
                    ledger.append({"source": name, "csv": read(name).decode("utf-8", errors="replace")})
            except Exception as exc:
                check("csv_read", "NEEDS_CHECK", f"{name}: {type(exc).__name__}: {exc}")
        elif low.endswith((".json", ".jsonl", ".log", ".txt")) and any(t in low for t in ["fail", "error", "status", "manifest", "valid", "fidelity"]):
            body = read(name).decode("utf-8", errors="replace")
            metadata.append({"source": name, "text": body})
            if any(t in low for t in ["fail", "error", "status"]) or re.search(r'"(?:failed|failures|errors)"', body):
                ledger.append({"source": name, "text": body})
    (out / "failure_and_status_evidence.json").write_text(json.dumps(ledger, indent=2), encoding="utf-8")
    (out / "metadata_evidence.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if ledger:
        print("\nEXACT FAILURE/STATUS CONTENTS (not guessed unresolved counts):", flush=True)
        for record in ledger:
            body = record.get("text", record.get("csv", ""))
            print(f"\n--- {record['source']} ---\n{body[:20000]}", flush=True)
            if len(body) > 20000:
                print("Display truncated; full text preserved in failure_and_status_evidence.json", flush=True)
        check("failure_disposition", "REVIEW_RECORDS", "Do not erase historical failures or assume failed=7 means seven unresolved model fits.")
    else:
        check("failure_disposition", "NOT_FOUND", "Failure counter alone does not identify tasks. Supply original error/status ledger or notebook traceback; no failure cause inferred.")
    pd.DataFrame([{"file": n, "rows": len(d), "columns": ', '.join(d.columns)} for n, d in frames.items()]).to_csv(out / "metric_candidates.csv", index=False)
    main = abl = None
    try:
        mc = {n: x for n, d in frames.items() if (x := main_slice(d, n)) is not None}
        mn, main = choose(mc, main_file, ["dataset", "model", "seed"])
        check("main_source", "SELECTED", mn)
        main.to_csv(out / "main_per_seed_copy.csv", index=False)
        expected = set(product(DATASETS, MODELS, SEEDS))
        actual = set(map(tuple, main[["dataset", "model", "seed"]].to_numpy()))
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        duplicates = main.duplicated(["dataset", "model", "seed"], keep=False)
        check("main_160_unique_keys", "PASS" if not missing and not extra and not duplicates.any() and len(main) == 160 else "FAIL", f"rows={len(main)}; missing={missing}; extra={extra}; duplicate_rows={int(duplicates.sum())}")
        usable = [m for m in METRICS if m in main]
        finite = np.isfinite(main[usable].to_numpy(dtype=float)).all()
        check("main_metrics_finite", "PASS" if finite else "FAIL", usable)
        summary = summarize(main, ["dataset", "model"])
        summary.to_csv(out / "DRAFT_main_mean_sd.csv", index=False)
        print("\nREGENERATED MAIN SUMMARY:\n" + summary.to_string(index=False), flush=True)
        ac = {n: x for n, d in frames.items() if (x := ablation_slice(d, n)) is not None}
        an, abl = choose(ac, ablation_file, ["dataset", "variant", "seed"])
        check("ablation_source", "SELECTED", an)
        abl.to_csv(out / "ablation_per_seed_copy.csv", index=False)
        explicit = compare_full(main, abl)
        if len(explicit):
            explicit.to_csv(out / "explicit_full_vs_main.csv", index=False)
            matched = explicit["match"].all()
            check("explicit_ablation_full_vs_main", "PASS" if matched else "FAIL", f"{int((~explicit['match']).sum())} differing/missing metric comparisons. Never overwrite these with main values.")
            if not matched:
                print(explicit[~explicit['match']].to_string(index=False), flush=True)
                raise ValueError("Independent ablation T_Full differs; no combined ablation table generated. Reconcile identity or explicitly analyze that separate experiment.")
        else:
            check("full_reference", "INFO", "No independently fitted ablation T_Full rows. Draft deltas use ORIGINAL selected main KATS as reference; this is not a claim that an independent baseline was reproduced.")
        variants = abl[abl["variant"].isin(VARIANTS)].copy()
        expected_v = set(product(DATASETS, VARIANTS, SEEDS))
        actual_v = set(map(tuple, variants[["dataset", "variant", "seed"]].to_numpy()))
        missing_v = sorted(expected_v - actual_v)
        dup_v = variants.duplicated(["dataset", "variant", "seed"], keep=False)
        check("ablation_80_variant_keys", "PASS" if not missing_v and not dup_v.any() and actual_v == expected_v else "FAIL", f"missing={missing_v}; duplicate_rows={int(dup_v.sum())}")
        for ds in ["GoogleCluster", "MultiCloud"]:
            a = main[main["dataset"].eq(ds)]
            b = variants[variants["dataset"].eq(ds)]
            equiv = compare_full(a, b, "T_NoResampling")
            if len(equiv):
                equiv.to_csv(out / f"{ds}_equivalent_check.csv", index=False)
                check(f"{ds}_no_resampling_per_seed", "PASS" if equiv["match"].all() else "FAIL", f"{int((~equiv['match']).sum())} mismatches; treatment equivalence still requires generating-code identity.")
            else:
                check(f"{ds}_no_resampling_per_seed", "NOT_FOUND", "No per-seed equivalent rows; do not manufacture them from means.")
        keys = ["dataset", "seed"]
        ref = main[main["model"].eq("KATS")].copy()
        common = [m for m in METRICS if m in ref and m in variants]
        if ref.duplicated(keys).any() or dup_v.any():
            raise ValueError("Duplicate keys; cannot pair deltas")
        paired = variants.merge(ref[keys + common], on=keys, how="left", suffixes=("", "_full"), validate="many_to_one")
        for m in common:
            paired[f"Delta_{m}"] = paired[m] - paired[f"{m}_full"]
        paired.to_csv(out / "DRAFT_paired_ablation_deltas.csv", index=False)
        r = ref.copy()
        r["variant"] = "T_Full"
        combined = pd.concat([r, variants], ignore_index=True)
        summary_a = summarize(combined, ["dataset", "variant"])
        summary_a.to_csv(out / "DRAFT_ablation_mean_sd.csv", index=False)
        print("\nREGENERATED ABLATION SUMMARY:\n" + summary_a.to_string(index=False), flush=True)
        check("paired_test_ids", "NOT_CHECKED", "This fast script verifies metric keys, not individual test IDs/probabilities/model object identity. Do not turn numerical PASS into all-reviewer sign-off.")
    except Exception as exc:
        check("numeric_table_generation", "STOPPED_NOT_REPAIRED", f"{type(exc).__name__}: {exc}")
    check("comment15_experiment", "DO_NOT_REPEAT_AUTOMATICALLY", "Actual-fitted stack/B1 audit was reported: median rho .921296, Top1 65%, Top3 75%. This script does not rerun SHAP or invent global agreement/model identity.")
    result = pd.DataFrame(checks)
    result.to_csv(out / "checks.csv", index=False)
    report = "# Fast checks for comments 7 and 15\n\nNo training. Sources unchanged.\n\n" + result.to_string(index=False) + "\n\nDraft tables are NOT a publication certificate. Resolve exact FAIL/STOPPED records, inspect failure disposition, and preserve distinct run versions.\n"
    (out / "REPORT.md").write_text(report, encoding="utf-8")
    if z:
        z.close()
    print(f"\nDONE: {out}\nReturn REPORT.md and failure_and_status_evidence.json.\nCompleted models and SHAP were NOT retrained.", flush=True)
    return result

if __name__ == "__main__":
    inspect()
