from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, cross_val_predict, cross_validate
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

FEATURES_CSV = Path("outputs/features/delta_features_uid.csv")
OUTCOMES_CSV = Path("outputs/review/classification_outcomes.csv")
ICC_KEEP = True
KEEP_CSV = Path("outputs/features/icc_features_keep.csv")
OUT_ROOT = Path("outputs/models/mlp")

RANDOM_STATE = 42
N_SPLITS = 5
N_HOLDOUT_SPLITS = 5
TEST_SIZE = 0.2
ICC_MIN = 0.9
THRESHOLD_SELECTION_METRIC = "balanced_accuracy"

MLP_PARAMS = {
    "hidden_layer_sizes": (64, 32),
    "alpha": 0.0001,
    "max_iter": 1000,
    "early_stopping": True,
    "n_iter_no_change": 50,
    "random_state": RANDOM_STATE,
    "verbose": False,
}


def _build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", MinMaxScaler()),
            ("mlp", MLPClassifier(**MLP_PARAMS)),
        ]
    )


def _aggregate_metric_dicts(metric_dicts: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    if not metric_dicts:
        return {}
    keys = list(metric_dicts[0].keys())
    out: dict[str, dict[str, float]] = {}
    for key in keys:
        sample_value = metric_dicts[0][key]
        if not isinstance(sample_value, (int, float, np.integer, np.floating)):
            continue
        values = np.asarray([row[key] for row in metric_dicts], dtype=float)
        out[key] = {"mean": float(np.nanmean(values)), "std": float(np.nanstd(values))}
    return out


def _metrics_from_threshold(y_true: np.ndarray, y_proba: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = (np.asarray(y_proba, dtype=float) >= float(threshold)).astype(int)
    return {
        "n": int(len(y_true)),
        "pos_rate": float(np.mean(y_true)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)) if len(np.unique(y_true)) > 1 else float("nan"),
    }


def _select_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    *,
    metric_name: str = THRESHOLD_SELECTION_METRIC,
) -> tuple[float, dict[str, float]]:
    candidate_thresholds = np.unique(np.concatenate(([0.0], np.asarray(y_proba, dtype=float), [1.0])))
    best_threshold = 0.5
    best_metrics: dict[str, float] | None = None
    best_score: tuple[float, float, float, float] | None = None

    for threshold in candidate_thresholds:
        metrics = _metrics_from_threshold(y_true, y_proba, float(threshold))
        score = (
            float(metrics[metric_name]),
            float(metrics["recall"]),
            float(metrics["f1"]),
            -abs(float(threshold) - 0.5),
        )
        if best_score is None or score > best_score:
            best_score = score
            best_threshold = float(threshold)
            best_metrics = metrics

    if best_metrics is None:
        raise ValueError("No se ha podido seleccionar un threshold.")

    return best_threshold, best_metrics


def train_model(
    *,
    features_csv: Path = FEATURES_CSV,
    outcomes_csv: Path = OUTCOMES_CSV,
    icc_keep: bool = ICC_KEEP,
    keep_csv: Path = KEEP_CSV,
    out_root: Path = OUT_ROOT,
) -> Path:
    np.random.seed(RANDOM_STATE)

    features = pd.read_csv(features_csv)
    outcomes = pd.read_csv(outcomes_csv)[["uid", "outcome"]]

    before_f = len(features)
    before_o = len(outcomes)
    df = features.merge(outcomes, on="uid", how="inner")
    df = df.sort_values(by="uid").reset_index(drop=True)

    if df.empty:
        raise ValueError("Merge vacio (no hay uids en comun).")

    uids = df["uid"].astype(str).reset_index(drop=True)
    y = pd.to_numeric(df["outcome"], errors="raise").astype(int).to_numpy()
    observed_labels = sorted(int(label) for label in np.unique(y))
    if observed_labels != [0, 1]:
        raise ValueError(f"Outcome debe ser binario 0/1; valores encontrados: {observed_labels}")

    if bool(icc_keep):
        keep_df = pd.read_csv(keep_csv)
        if "icc" in keep_df.columns:
            keep_df = keep_df[pd.to_numeric(keep_df["icc"], errors="coerce") > float(ICC_MIN)]
        keep = keep_df["feature"].astype(str).tolist()
        feature_cols = [f"delta_{f}" for f in keep if f"delta_{f}" in df.columns]
        missing = [f for f in keep if f"delta_{f}" not in df.columns]
    else:
        feature_cols = [c for c in df.columns if c not in {"uid", "outcome"}]
        missing = []

    if not feature_cols:
        raise ValueError("No hay columnas de features seleccionadas.")

    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").to_numpy()

    splitter = StratifiedShuffleSplit(
        n_splits=N_HOLDOUT_SPLITS,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scoring = ["accuracy", "f1", "precision", "recall", "roc_auc", "balanced_accuracy"]

    cv_metric_rows_default: list[dict[str, float]] = []
    cv_metric_rows_threshold: list[dict[str, float]] = []
    test_metric_rows_threshold: list[dict[str, float]] = []
    split_summaries: list[dict[str, float | int]] = []
    threshold_rows: list[dict[str, float]] = []

    for split_idx, (train_idx, test_idx) in enumerate(splitter.split(X, y)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        pipe = _build_pipeline()
        scores = cross_validate(
            pipe,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            return_train_score=False,
            error_score=np.nan,
        )

        cv_row = {
            metric: float(np.nanmean(scores[f"test_{metric}"]))
            for metric in scoring
        }
        cv_metric_rows_default.append(cv_row)

        oof_proba = cross_val_predict(pipe, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
        selected_threshold, cv_threshold_row = _select_threshold(
            y_train,
            oof_proba,
            metric_name=THRESHOLD_SELECTION_METRIC,
        )
        cv_metric_rows_threshold.append(cv_threshold_row)
        threshold_rows.append(
            {
                "threshold": float(selected_threshold),
                "threshold_score": float(cv_threshold_row[THRESHOLD_SELECTION_METRIC]),
            }
        )

        pipe.fit(X_train, y_train)
        test_proba = pipe.predict_proba(X_test)[:, 1]
        test_row = _metrics_from_threshold(y_test, test_proba, selected_threshold)
        test_metric_rows_threshold.append(test_row)
        split_summaries.append(
            {
                "split_index": int(split_idx),
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "train_pos_rate": float(np.mean(y_train)),
                "test_pos_rate": float(np.mean(y_test)),
                "cv_roc_auc": float(cv_row["roc_auc"]),
                "test_roc_auc": float(test_row["roc_auc"]),
                "threshold": float(selected_threshold),
                "threshold_metric": THRESHOLD_SELECTION_METRIC,
                "threshold_score": float(cv_threshold_row[THRESHOLD_SELECTION_METRIC]),
            }
        )

    cv_metrics_default = _aggregate_metric_dicts(cv_metric_rows_default)
    cv_metrics_threshold_optimized_apparent = _aggregate_metric_dicts(cv_metric_rows_threshold)
    test_metrics = _aggregate_metric_dicts(test_metric_rows_threshold)
    threshold_mean = float(np.nanmean([row["threshold"] for row in threshold_rows]))
    threshold_std = float(np.nanstd([row["threshold"] for row in threshold_rows]))

    final_pipe = _build_pipeline().fit(X, y)

    run_dir = out_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    model_artifact = {
        "pipeline": final_pipe,
        "threshold": threshold_mean,
        "threshold_std": threshold_std,
        "threshold_metric": THRESHOLD_SELECTION_METRIC,
        "feature_cols": feature_cols,
        "positive_class": "improved",
        "label_mapping": {
            "original_outcome_1": "improved",
            "original_outcome_0": "not_improved",
            "model_positive_class_1": "improved",
        },
        "label_transform": "y = outcome",
    }
    dump(model_artifact, run_dir / "model.joblib")
    pd.DataFrame({"uid": uids}).to_csv(run_dir / "train_uids.csv", index=False)
    pd.DataFrame({"uid": []}).to_csv(run_dir / "test_uids.csv", index=False)
    metrics_payload = {
        "data": {
            "features_csv": str(features_csv),
            "outcomes_csv": str(outcomes_csv),
            "n_feature_rows": int(before_f),
            "n_outcome_rows": int(before_o),
            "n_merged_rows": int(len(df)),
            "n_features": int(X.shape[1]),
            "selected_feature_columns": int(len(feature_cols)),
            "missing_icc_features": int(len(missing)),
            "label_mapping": model_artifact["label_mapping"],
            "label_transform": model_artifact["label_transform"],
        },
        "training": {
            "icc_keep": bool(icc_keep),
            "icc_min": float(ICC_MIN) if icc_keep else None,
            "keep_csv": str(keep_csv),
            "n_splits": int(N_SPLITS),
            "n_holdout_splits": int(N_HOLDOUT_SPLITS),
            "test_size": float(TEST_SIZE),
            "random_state": int(RANDOM_STATE),
            "threshold_selection_metric": THRESHOLD_SELECTION_METRIC,
            "threshold_aggregation": "mean",
            "delta": {"mode": "relative", "formula": "(fu - bl) / bl"},
            "mlp_params": MLP_PARAMS,
            "early_stopping_note": (
                "MLPClassifier(early_stopping=True) reserves an internal validation "
                "fraction inside each training fold."
            ),
        },
        "split": {
            "mode": "stratified_shuffle_multi_holdout_mean_refit",
            "n_train": int(len(y)),
            "n_test": 0,
            "train_pos_rate": float(np.mean(y)),
            "test_pos_rate": None,
            "threshold": threshold_mean,
            "threshold_std": threshold_std,
            "threshold_metric": THRESHOLD_SELECTION_METRIC,
            "train_uids_csv": "train_uids.csv",
            "test_uids_csv": "test_uids.csv",
        },
        "cv_metrics_default": cv_metrics_default,
        "cv_metrics_threshold_optimized_apparent": cv_metrics_threshold_optimized_apparent,
        "test_metrics": test_metrics,
        "metric_interpretation": {
            "primary_metric": "test_metrics",
            "cv_metrics_default": "Cross-validation metrics on training folds using the default 0.5 threshold.",
            "cv_metrics_threshold_optimized_apparent": (
                "OOF training metrics after selecting the threshold on the same OOF "
                "predictions; useful diagnostically but optimistic/apparent."
            ),
            "test_metrics": (
                "Multi-holdout metrics where the threshold is selected on each train "
                "split and evaluated on its held-out test split."
            ),
        },
        "holdout_thresholds": {
            "mean": threshold_mean,
            "std": threshold_std,
            "per_split": threshold_rows,
        },
        "holdout_summaries": split_summaries,
        "timestamp": datetime.now().isoformat(),
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        f"[train_mlp] rows={len(df)} features={len(feature_cols)} "
        f"threshold_mean={threshold_mean:.3f} "
        f"cv_apparent_roc_auc={cv_metrics_threshold_optimized_apparent['roc_auc']['mean']:.3f} "
        f"test_roc_auc={test_metrics['roc_auc']['mean']:.3f}"
    )
    print(f"[train_mlp] wrote={run_dir / 'model.joblib'}")
    print(f"[train_mlp] wrote={run_dir / 'metrics.json'}")
    print(f"[train_mlp] wrote={run_dir / 'train_uids.csv'}")
    print(f"[train_mlp] wrote={run_dir / 'test_uids.csv'}")
    return run_dir


def main() -> None:
    train_model()


if __name__ == "__main__":
    main()
