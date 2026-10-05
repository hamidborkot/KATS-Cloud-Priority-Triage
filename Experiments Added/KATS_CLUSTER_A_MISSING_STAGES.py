# ============================================================
# KATS CLUSTER A — MISSING STAGES ONLY
# Kaggle continuation from the already-uploaded run-results CSVs.
# Set STAGE to exactly one of: temporal, matched, ir, validate
# This script NEVER runs main benchmark or ablation.
# It writes the stage CSV after EVERY completed model result.
# ============================================================

import os, re, ast, json, shutil, warnings
from pathlib import Path
from datetime import datetime
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, cohen_kappa_score, brier_score_loss,
    average_precision_score, confusion_matrix, recall_score,
)
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

# ---------------- CHANGE ONLY THIS LINE ----------------
STAGE = "temporal"       # temporal, matched, ir, or validate
# ---------------------------------------------------------

SEEDS = [42, 7, 13, 99, 2026]
TEMPORAL_SEEDS = [42, 7, 13]
IR_LEVELS = [2, 5, 10, 20, 30]
IR_THRESHOLD = 3.0
N_JOBS = 1

# Read-only dataset containing the eleven files you uploaded.
SAVED = Path("/kaggle/input/datasets/mdhamidborkottulla/run-results")
OUT = Path("/kaggle/working/KATS_CLUSTER_A_FINAL")
OUT.mkdir(parents=True, exist_ok=True)

IT_PATH = "/kaggle/input/datasets/shamiulislamshifat/it-incident-log-dataset/incident_event_log.csv"
CIC_PATH = "/kaggle/input/datasets/ericanacletoribeiro/cicids2017-cleaned-and-preprocessed/cicids2017_cleaned.csv"

BASE_FILES = [
    "clusterA_ablation_per_seed.csv",
    "clusterA_ablation_summary.csv",
    "clusterA_calibration_bins.csv",
    "clusterA_confusion_matrices.csv",
    "clusterA_main_metrics_per_seed.csv",
    "clusterA_metrics_mean_sd.csv",
    "clusterA_predictions.csv",
    "features_CICIDS2017.csv",
    "features_GoogleCluster.csv",
    "features_ITIncident.csv",
    "features_MultiCloud.csv",
]


def log(message):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def atomic_csv(df, filename):
    target = OUT / filename
    temporary = OUT / f".{filename}.tmp"
    df.to_csv(temporary, index=False)
    os.replace(temporary, target)


def copy_saved_results_once():
    for name in BASE_FILES:
        source = SAVED / name
        target = OUT / name
        if not source.exists():
            raise FileNotFoundError(f"Missing uploaded source file: {source}")
        if not target.exists() or target.stat().st_size == 0:
            shutil.copy2(source, target)
            log(f"Copied saved file: {name}")


def norm(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def find_col(df, *aliases, required=True):
    normalized = {norm(c): c for c in df.columns}
    for alias in aliases:
        if norm(alias) in normalized:
            return normalized[norm(alias)]
    for alias in aliases:
        key = norm(alias)
        for normalized_name, original_name in normalized.items():
            if key in normalized_name or normalized_name in key:
                return original_name
    if required:
        raise KeyError(f"Cannot find {aliases}; available={list(df.columns)}")
    return None


def encode_labels(series):
    encoder = LabelEncoder()
    y = encoder.fit_transform(series.astype(str))
    high = int(np.where(encoder.classes_ == "High")[0][0])
    return y, encoder, high


def matrix(df, features):
    return df[features].replace([np.inf, -np.inf], np.nan).fillna(0).astype(float)


def class_weights(y, high_index, alpha=5.0):
    classes, counts = np.unique(y, return_counts=True)
    result = {
        int(c): float(len(y) / (len(classes) * n))
        for c, n in zip(classes, counts)
    }
    if counts.max() / counts.min() > IR_THRESHOLD:
        result[int(high_index)] *= alpha
    return result


def imbalance_ratio(y):
    _, counts = np.unique(y, return_counts=True)
    return float(counts.max() / counts.min())


def binary_ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for b in range(bins):
        if b == bins - 1:
            mask = (p >= edges[b]) & (p <= edges[b + 1])
        else:
            mask = (p >= edges[b]) & (p < edges[b + 1])
        n = int(mask.sum())
        if n:
            total += n / len(y) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(total)


def metric_row(y, pred, proba, encoder, high_index):
    report = classification_report(
        y, pred,
        target_names=encoder.classes_.tolist(),
        output_dict=True,
        zero_division=0,
    )
    briers = []
    eces = []
    for c in range(len(encoder.classes_)):
        y_binary = (y == c).astype(int)
        briers.append(brier_score_loss(y_binary, proba[:, c]))
        eces.append(binary_ece(y_binary, proba[:, c]))
    y_high = (y == high_index).astype(int)
    p_high = (pred == high_index).astype(int)
    return {
        "RecallH": float(report["High"]["recall"]),
        "PrecH": float(report["High"]["precision"]),
        "F1H": float(report["High"]["f1-score"]),
        "MacroF1": float(report["macro avg"]["f1-score"]),
        "Kappa": float(cohen_kappa_score(y, pred)),
        "Brier": float(np.mean(briers)),
        "ECE": float(np.mean(eces)),
        "PRAUC_High": float(average_precision_score(y_high, proba[:, high_index])),
        "FP_High": int(((p_high == 1) & (y_high == 0)).sum()),
        "FN_High": int(((p_high == 0) & (y_high == 1)).sum()),
    }


def make_resampler(dataset, features, categorical, seed, enabled=True):
    if not enabled:
        return "passthrough"
    if dataset == "ITIncident":
        indices = [features.index(x) for x in categorical if x in features]
        return SMOTENC(
            categorical_features=indices,
            random_state=seed,
            k_neighbors=5,
            sampling_strategy="not majority",
        )
    if dataset == "CICIDS2017":
        return SMOTE(
            random_state=seed,
            k_neighbors=5,
            sampling_strategy="not majority",
        )
    return "passthrough"


def kats(dataset, features, categorical, weights, seed):
    def wrapped(estimator, offset=0):
        return ImbPipeline([
            ("resample", make_resampler(dataset, features, categorical, seed + offset, True)),
            ("model", estimator),
        ])

    lightgbm = wrapped(lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        class_weight=weights,
        random_state=seed,
        n_jobs=N_JOBS,
        verbose=-1,
    ))

    forest = wrapped(RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=seed,
        n_jobs=N_JOBS,
    ), 1000)

    bayes = CalibratedClassifierCV(GaussianNB(), cv=3, method="isotonic")

    return StackingClassifier(
        estimators=[("lgb", lightgbm), ("rf", forest), ("nb", bayes)],
        final_estimator=LogisticRegression(
            C=1.0,
            max_iter=2000,
            class_weight=weights,
            random_state=seed,
        ),
        stack_method="predict_proba",
        passthrough=True,
        cv=3,
        n_jobs=N_JOBS,
    )


def baselines(weights, seed):
    return {
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            class_weight=weights,
            random_state=seed,
            n_jobs=N_JOBS,
            verbose=-1,
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            eval_metric="mlogloss",
            random_state=seed,
            n_jobs=N_JOBS,
            verbosity=0,
        ),
        "LogReg": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(
                C=1.0,
                max_iter=2000,
                class_weight="balanced",
                random_state=seed,
            )),
        ]),
    }


def select_kats_threshold(model, X_train, y_train, high_index, seed):
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train,
        test_size=0.15,
        random_state=seed,
        stratify=y_train,
    )
    model.fit(X_fit, y_fit)
    p = model.predict_proba(X_val)[:, high_index]
    y_binary = (y_val == high_index).astype(int)
    precision_floor = max(0.30, 1.5 * y_binary.mean())
    best_threshold = 0.50
    best_score = -np.inf
    for threshold in np.arange(0.15, 0.86, 0.05):
        z = (p >= threshold).astype(int)
        tp = int(((z == 1) & (y_binary == 1)).sum())
        fp = int(((z == 1) & (y_binary == 0)).sum())
        fn = int(((z == 0) & (y_binary == 1)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        score = (precision + recall) / 2.0
        if precision >= precision_floor and score > best_score:
            best_score = score
            best_threshold = float(threshold)
    model.fit(X_train, y_train)
    return model, best_threshold


def high_threshold_predictions(proba, high_index, threshold):
    pred = np.argmax(proba, axis=1)
    pred[proba[:, high_index] >= threshold] = high_index
    return pred


def load_itincident():
    raw = pd.read_csv(IT_PATH, low_memory=False)
    number = find_col(raw, "number")
    modification = find_col(raw, "sys_mod_count")
    opened = find_col(raw, "opened_at")
    priority = find_col(raw, "priority")

    df = raw.sort_values([number, modification]).groupby(
        number, group_keys=False
    ).tail(1).copy()

    df["opened_at_final"] = pd.to_datetime(df[opened], errors="coerce")
    df["priority_label"] = df[priority].map({
        "1 - Critical": "High",
        "2 - High": "High",
        "3 - Moderate": "Medium",
        "4 - Low": "Low",
    })
    df = df.dropna(subset=["opened_at_final", "priority_label"]).copy()

    mapping = {
        "category": "category_enc",
        "location": "location_enc",
        "contact_type": "contact_type_enc",
        "assignment_group": "assignment_group_enc",
        "cmdb_ci": "cmdb_ci_enc",
        "subcategory": "subcategory_enc",
        "knowledge": "knowledge_enc",
    }
    categorical = []
    for source, output in mapping.items():
        found = find_col(df, source, required=False)
        if found is not None:
            df[output] = LabelEncoder().fit_transform(df[found].astype(str))
            categorical.append(output)

    reopened = find_col(df, "reopen_count")
    df["reopen_flag"] = (df[reopened] > 0).astype(int)
    categorical.append("reopen_flag")

    features = [
        find_col(df, "reassignment_count"),
        find_col(df, "reopen_count"),
        find_col(df, "sys_mod_count"),
    ] + categorical
    assert len(features) == 11
    return df, features, categorical


def load_cicids():
    raw = pd.read_csv(CIC_PATH, low_memory=False)
    raw.columns = [c.strip().lower().replace(" ", "_") for c in raw.columns]
    label = find_col(raw, "attack_type", "label")

    def map_priority(value):
        value = str(value).lower()
        if "benign" in value or "normal" in value:
            return "Low"
        if "scan" in value or "patator" in value or "brute" in value:
            return "Medium"
        return "High"

    raw["priority_label"] = raw[label].apply(map_priority)
    df = pd.concat([
        x.sample(frac=0.05, random_state=42)
        for _, x in raw.groupby("priority_label")
    ]).reset_index(drop=True)

    features = [
        c for c in df.columns
        if c not in {label, "priority_label"}
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    df[features] = df[features].replace([np.inf, -np.inf], np.nan).fillna(0)

    if len(df) > 60000:
        fraction = 60000 / len(df)
        df = pd.concat([
            x.sample(frac=fraction, random_state=42)
            for _, x in df.groupby("priority_label")
        ]).reset_index(drop=True)

    return df, features


def read_or_empty(filename, columns):
    path = OUT / filename
    if path.exists() and path.stat().st_size > 0:
        return pd.read_csv(path)
    return pd.DataFrame(columns=columns)


def run_temporal():
    filename = "clusterA_temporal_sweep.csv"
    columns = [
        "train_fraction", "seed", "model", "threshold",
        "RecallH", "PrecH", "F1H", "MacroF1", "Kappa",
        "Brier", "ECE", "PRAUC_High", "FP_High", "FN_High",
    ]
    results = read_or_empty(filename, columns)
    done = set(zip(results.get("train_fraction", []), results.get("seed", []), results.get("model", [])))

    df, features, categorical = load_itincident()
    df = df.sort_values("opened_at_final").reset_index(drop=True)
    X = matrix(df, features)
    y, encoder, high = encode_labels(df["priority_label"])

    for fraction in [0.60, 0.70, 0.80, 0.90]:
        cut = int(len(df) * fraction)
        X_train, X_test = X.iloc[:cut], X.iloc[cut:]
        y_train, y_test = y[:cut], y[cut:]
        assert df["opened_at_final"].iloc[:cut].max() <= df["opened_at_final"].iloc[cut:].min()

        for seed in TEMPORAL_SEEDS:
            weights = class_weights(y_train, high)
            specs = {
                "KATS": lambda: kats("ITIncident", features, categorical, weights, seed),
                "LightGBM": lambda: baselines(weights, seed)["LightGBM"],
                "LogReg": lambda: baselines(weights, seed)["LogReg"],
            }
            for model_name, maker in specs.items():
                key = (fraction, seed, model_name)
                if key in done:
                    continue
                log(f"Temporal: fraction={fraction}, seed={seed}, model={model_name}")
                model = maker()
                if model_name == "KATS":
                    model, threshold = select_kats_threshold(model, X_train, y_train, high, seed)
                else:
                    model.fit(X_train, y_train)
                    threshold = 0.50
                proba = model.predict_proba(X_test)
                pred = high_threshold_predictions(proba, high, threshold) if model_name == "KATS" else model.predict(X_test)
                row = {
                    "train_fraction": fraction,
                    "seed": seed,
                    "model": model_name,
                    "threshold": threshold,
                    **metric_row(y_test, pred, proba, encoder, high),
                }
                results = pd.concat([results, pd.DataFrame([row])], ignore_index=True)
                atomic_csv(results, filename)
                done.add(key)

    atomic_csv(results.sort_values(["train_fraction", "seed", "model"]), filename)
    log(f"Saved {filename}: {len(results)} rows; expected 36")


def nearest_threshold(y_validation, p_high, target_recall):
    candidates = np.arange(0.05, 0.96, 0.01)
    return float(min(
        candidates,
        key=lambda threshold: abs(
            recall_score(y_validation, p_high >= threshold, zero_division=0) - target_recall
        ),
    ))


def run_matched():
    filename = "clusterA_matched_operating_points.csv"
    columns = [
        "dataset", "seed", "model", "target_recall", "threshold",
        "RecallH", "PrecH", "F1H", "MacroF1", "Kappa",
        "Brier", "ECE", "PRAUC_High", "FP_High", "FN_High",
    ]
    results = read_or_empty(filename, columns)
    done = set(zip(results.get("dataset", []), results.get("seed", []), results.get("model", [])))

    it_df, it_features, it_categorical = load_itincident()
    cic_df, cic_features = load_cicids()
    datasets = {
        "ITIncident": (it_df, it_features, it_categorical),
        "CICIDS2017": (cic_df, cic_features, []),
    }
    target_recall = 0.90

    for dataset, (df, features, categorical) in datasets.items():
        X = matrix(df, features)
        y, encoder, high = encode_labels(df["priority_label"])

        for seed in SEEDS:
            X_outer, X_test, y_outer, y_test = train_test_split(
                X, y, test_size=0.20, random_state=seed, stratify=y
            )
            X_fit, X_validation, y_fit, y_validation = train_test_split(
                X_outer, y_outer, test_size=0.15, random_state=seed, stratify=y_outer
            )
            weights = class_weights(y_fit, high)
            specifications = {
                "KATS": lambda: kats(dataset, features, categorical, weights, seed),
                "LightGBM": lambda: baselines(weights, seed)["LightGBM"],
                "XGBoost": lambda: baselines(weights, seed)["XGBoost"],
                "LogReg": lambda: baselines(weights, seed)["LogReg"],
            }

            for model_name, maker in specifications.items():
                key = (dataset, seed, model_name)
                if key in done:
                    continue
                log(f"Matched recall: dataset={dataset}, seed={seed}, model={model_name}")
                model = maker()
                model.fit(X_fit, y_fit)
                validation_proba = model.predict_proba(X_validation)
                threshold = nearest_threshold(
                    (y_validation == high).astype(int),
                    validation_proba[:, high],
                    target_recall,
                )
                model.fit(X_outer, y_outer)
                test_proba = model.predict_proba(X_test)
                test_pred = high_threshold_predictions(test_proba, high, threshold)
                row = {
                    "dataset": dataset,
                    "seed": seed,
                    "model": model_name,
                    "target_recall": target_recall,
                    "threshold": threshold,
                    **metric_row(y_test, test_pred, test_proba, encoder, high),
                }
                results = pd.concat([results, pd.DataFrame([row])], ignore_index=True)
                atomic_csv(results, filename)
                done.add(key)

    atomic_csv(results.sort_values(["dataset", "seed", "model"]), filename)
    log(f"Saved {filename}: {len(results)} rows; expected 40")


def run_ir():
    filename = "clusterA_ir_fixed_test_sensitivity.csv"
    columns = [
        "seed", "training_target_ir", "training_achieved_ir", "training_n",
        "fixed_test_n", "model", "threshold", "RecallH", "PrecH", "F1H",
        "MacroF1", "Kappa", "Brier", "ECE", "PRAUC_High", "FP_High", "FN_High",
    ]
    results = read_or_empty(filename, columns)
    done = set(zip(results.get("seed", []), results.get("training_target_ir", []), results.get("model", [])))

    df, features = load_cicids()
    X = matrix(df, features)
    y, encoder, high = encode_labels(df["priority_label"])

    for seed in SEEDS:
        X_pool, X_test, y_pool, y_test = train_test_split(
            X, y, test_size=0.20, random_state=seed, stratify=y
        )
        indices = {c: np.where(y_pool == c)[0] for c in np.unique(y_pool)}
        counts = {c: len(v) for c, v in indices.items()}
        majority = max(counts, key=counts.get)
        minority = min(counts, key=counts.get)
        middle = [c for c in counts if c not in {majority, minority}][0]
        rng = np.random.RandomState(seed)

        for level in IR_LEVELS:
            majority_n = 3000
            minority_n = int(round(majority_n / level))
            middle_n = 6000 - majority_n - minority_n
            chosen = np.concatenate([
                rng.choice(indices[majority], majority_n, replace=False),
                rng.choice(indices[middle], middle_n, replace=False),
                rng.choice(indices[minority], minority_n, replace=False),
            ])
            rng.shuffle(chosen)
            X_train = X_pool.iloc[chosen]
            y_train = y_pool[chosen]
            weights = class_weights(y_train, high)
            specifications = {
                "KATS": lambda: kats("CICIDS2017", features, [], weights, seed),
                "LightGBM": lambda: baselines(weights, seed)["LightGBM"],
                "XGBoost": lambda: baselines(weights, seed)["XGBoost"],
                "LogReg": lambda: baselines(weights, seed)["LogReg"],
            }

            for model_name, maker in specifications.items():
                key = (seed, level, model_name)
                if key in done:
                    continue
                log(f"IR sensitivity: seed={seed}, IR={level}, model={model_name}")
                model = maker()
                if model_name == "KATS":
                    model, threshold = select_kats_threshold(model, X_train, y_train, high, seed)
                else:
                    model.fit(X_train, y_train)
                    threshold = 0.50
                proba = model.predict_proba(X_test)
                pred = high_threshold_predictions(proba, high, threshold) if model_name == "KATS" else model.predict(X_test)
                row = {
                    "seed": seed,
                    "training_target_ir": level,
                    "training_achieved_ir": imbalance_ratio(y_train),
                    "training_n": len(y_train),
                    "fixed_test_n": len(y_test),
                    "model": model_name,
                    "threshold": threshold,
                    **metric_row(y_test, pred, proba, encoder, high),
                }
                results = pd.concat([results, pd.DataFrame([row])], ignore_index=True)
                atomic_csv(results, filename)
                done.add(key)

    atomic_csv(results.sort_values(["seed", "training_target_ir", "model"]), filename)
    log(f"Saved {filename}: {len(results)} rows; expected 100")


def validate():
    required = BASE_FILES + [
        "clusterA_temporal_sweep.csv",
        "clusterA_matched_operating_points.csv",
        "clusterA_ir_fixed_test_sensitivity.csv",
    ]
    def present(name):
        path = OUT / name
        return path.exists() and path.stat().st_size > 0

    def rows(name):
        return len(pd.read_csv(OUT / name)) if present(name) else 0

    checks = {
        "itincident_feature_count_11": present("features_ITIncident.csv") and len(pd.read_csv(OUT / "features_ITIncident.csv")) == 11,
        "itincident_categorical_count_8": present("features_ITIncident.csv") and int((pd.read_csv(OUT / "features_ITIncident.csv")["type"] == "categorical").sum()) == 8,
        "main_per_seed_rows_160": rows("clusterA_main_metrics_per_seed.csv") == 160,
        "ablation_per_seed_rows_100": rows("clusterA_ablation_per_seed.csv") == 100,
        "temporal_rows_36": rows("clusterA_temporal_sweep.csv") == 36,
        "matched_operating_rows_40": rows("clusterA_matched_operating_points.csv") == 40,
        "fixed_test_ir_rows_100": rows("clusterA_ir_fixed_test_sensitivity.csv") == 100,
        "probability_rows_saved": rows("clusterA_predictions.csv") > 0,
        "all_required_files_present": all(present(name) for name in required),
    }
    with open(OUT / "clusterA_validation_checks.json", "w") as f:
        json.dump(checks, f, indent=2)
    log(json.dumps(checks, indent=2))


if STAGE not in {"temporal", "matched", "ir", "validate"}:
    raise ValueError("STAGE must be temporal, matched, ir, or validate")

copy_saved_results_once()

if STAGE == "temporal":
    run_temporal()
elif STAGE == "matched":
    run_matched()
elif STAGE == "ir":
    run_ir()
else:
    validate()

log(f"STAGE COMPLETE: {STAGE}")
log(f"Output folder: {OUT}")
