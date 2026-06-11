from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, cross_val_predict

def _save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_icc_selected_features(icc_csv: Path, *, icc_min: float | None) -> list[str]:
    icc_df = pd.read_csv(icc_csv)
    if "feature" not in icc_df.columns:
        raise ValueError(f"El CSV de ICC no contiene la columna 'feature': {icc_csv}")
    if "icc" in icc_df.columns and icc_min is not None:
        icc_df = icc_df[pd.to_numeric(icc_df["icc"], errors="coerce") > icc_min]
    return icc_df["feature"].astype(str).tolist()


def _summarize_metrics(metric_dicts: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    keys = list(metric_dicts[0].keys())
    out: dict[str, dict[str, float]] = {}
    for key in keys:
        sample_value = metric_dicts[0][key]
        if not isinstance(sample_value, (int, float, np.integer, np.floating)):
            continue
        values = np.asarray([row[key] for row in metric_dicts], dtype=float)
        out[key] = {"mean": float(np.nanmean(values)), "std": float(np.nanstd(values))}
    return out


def _evaluate_threshold(y_true: np.ndarray, y_proba: np.ndarray, threshold: float) -> dict[str, float]:
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
    metric_name: str,
) -> tuple[float, dict[str, float]]:
    candidate_thresholds = np.unique(
        np.concatenate(([0.0], np.asarray(y_proba, dtype=float), [1.0]))
    )
    best_threshold = 0.5
    best_metrics: dict[str, float] | None = None
    best_score: tuple[float, float, float, float] | None = None

    for threshold in candidate_thresholds:
        metrics = _evaluate_threshold(y_true, y_proba, float(threshold))
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
        raise ValueError("No se ha podido seleccionar un threshold")

    return best_threshold, best_metrics


def train_model(
    *,
    build_pipeline_fn: Callable[[dict], object],
    config: dict,
) -> Path:
    cfg = dict(config)

    model_name = str(cfg["model_name"])
    model_params = dict(cfg["model_params"])
    out_root = Path(cfg["out_root"])
    features_csv = Path(cfg["features_csv"])
    outcomes_csv = Path(cfg["outcomes_csv"])
    icc_keep = bool(cfg["icc_keep"])
    keep_csv = Path(cfg["keep_csv"])
    icc_min = None if cfg["icc_min"] is None else float(cfg["icc_min"])
    random_state = int(cfg["random_state"])
    n_splits = int(cfg.get("n_splits", 5))
    n_holdout_splits = int(cfg["n_holdout_splits"])
    test_size = float(cfg["test_size"])
    metric_name = str(cfg.get("metric_name", "balanced_accuracy"))
    extra_training_metadata = dict(cfg.get("extra_training_metadata", {}))

    random.seed(random_state)
    np.random.seed(random_state)

    features = pd.read_csv(features_csv)
    outcomes = pd.read_csv(outcomes_csv)[["uid", "outcome"]]

    df = features.merge(outcomes, on="uid", how="inner")
    df = df.sort_values(by="uid").reset_index(drop=True)

    y = pd.to_numeric(df["outcome"], errors="raise").astype(int).to_numpy()

    if icc_keep:
        keep = _load_icc_selected_features(keep_csv, icc_min=icc_min)
        feature_cols = [f"delta_{f}" for f in keep if f"delta_{f}" in df.columns]
    else:
        feature_cols = [c for c in df.columns if c not in {"uid", "outcome"}]

    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").to_numpy()

    splitter = StratifiedShuffleSplit(
        n_splits=n_holdout_splits,
        test_size=test_size,
        random_state=random_state,
    )
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    test_metric_rows_threshold: list[dict[str, float]] = []
    selected_thresholds: list[float] = []
    prediction_rows: list[dict[str, object]] = []

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X, y), start=1):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        pipe = build_pipeline_fn(model_params)
        oof_proba = cross_val_predict(pipe, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
        selected_threshold, _ = _select_threshold(y_train, oof_proba, metric_name=metric_name)
        selected_thresholds.append(float(selected_threshold))

        pipe.fit(X_train, y_train)
        test_proba = pipe.predict_proba(X_test)[:, 1]
        test_pred = (np.asarray(test_proba, dtype=float) >= float(selected_threshold)).astype(int)
        test_row = _evaluate_threshold(y_test, test_proba, selected_threshold)
        test_metric_rows_threshold.append(test_row)

        for row_idx, y_true, y_proba, y_pred in zip(test_idx, y_test, test_proba, test_pred):
            prediction_rows.append(
                {
                    "model": model_name,
                    "fold": fold_idx,
                    "uid": str(df.iloc[row_idx]["uid"]),
                    "y_true": int(y_true),
                    "y_proba": float(y_proba),
                    "y_pred": int(y_pred),
                    "threshold": float(selected_threshold),
                }
            )

    test_metrics = _summarize_metrics(test_metric_rows_threshold)
    final_threshold = float(np.nanmean(selected_thresholds))

    final_pipe = build_pipeline_fn(model_params)
    final_pipe.fit(X, y)

    run_dir = out_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    model_artifact = {
        "pipeline": final_pipe,
        "threshold": final_threshold,
        "feature_cols": feature_cols,
    }
    dump(model_artifact, run_dir / "model.joblib")

    training_payload = {
        "model_type": model_name,
        "model_params": model_params,
        "icc_keep": icc_keep,
        "icc_min": icc_min,
        "keep_csv": str(keep_csv),
        "n_holdout_splits": n_holdout_splits,
        "n_splits": n_splits,
        "test_size": test_size,
        "random_state": random_state,
        "metric_name": metric_name,
        "threshold_mean": final_threshold,
        "threshold_std": float(np.nanstd(selected_thresholds)),
    }
    training_payload.update(extra_training_metadata)

    metrics_payload = {
        "data": {
            "features_csv": str(features_csv),
            "outcomes_csv": str(outcomes_csv),
            "n_merged_rows": int(len(df)),
            "n_features": int(X.shape[1]),
        },
        "training": training_payload,
        "split": {
            "mode": "stratified_shuffle_multi_holdout_recalculated_threshold_refit",
            "n_samples": int(len(y)),
            "pos_rate": float(np.mean(y)),
            "threshold": final_threshold,
        },
        "test_metrics": test_metrics,
        "timestamp": datetime.now().isoformat(),
    }
    _save_json(run_dir / "metrics.json", metrics_payload)
    pd.DataFrame(prediction_rows).to_csv(run_dir / "test_predictions.csv", index=False)

    print(
        f"[train_{model_name}] rows={len(df)} features={len(feature_cols)} "
        f"threshold={final_threshold:.3f} "
        f"test_roc_auc={test_metrics['roc_auc']['mean']:.3f}"
    )
    print(f"[train_{model_name}] wrote={run_dir / 'model.joblib'}")
    print(f"[train_{model_name}] wrote={run_dir / 'metrics.json'}")
    print(f"[train_{model_name}] wrote={run_dir / 'test_predictions.csv'}")
    return run_dir
