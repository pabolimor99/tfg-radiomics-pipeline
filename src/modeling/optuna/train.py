from __future__ import annotations

import random
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, cross_val_predict

from src.modeling.train import (
    _evaluate_threshold,
    _load_icc_selected_features,
    _save_json,
    _select_threshold,
    _summarize_metrics,
)

DEFAULT_OPTUNA_CONFIG = {
    "study_name": "model_search",
    "n_trials": 30,
    "n_jobs": 1,
    "metric_name": "balanced_accuracy",
    "random_state": 42,
    "n_splits": 5,
    "n_holdout_splits": 5,
    "test_size": 0.2,
    "features_csv": Path("outputs/features/delta_features_uid.csv"),
    "outcomes_csv": Path("outputs/review/classification_outcomes.csv"),
    "keep_csv": Path("outputs/features/icc_results.csv"),
    "out_root": Path("outputs/models/optuna_model"),
    "icc_grid": [0.7, 0.8, 0.9, 0.95, 0.99],
    "base_model_params": {},
    "catch_exceptions": (ValueError, np.linalg.LinAlgError),
}


def load_dataset(
    config: dict,
    *,
    icc_keep: bool,
    icc_min: float | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    cfg = dict(DEFAULT_OPTUNA_CONFIG)
    cfg.update(config)

    features = pd.read_csv(Path(cfg["features_csv"]))
    outcomes = pd.read_csv(Path(cfg["outcomes_csv"]))[["uid", "outcome"]]
    df = features.merge(outcomes, on="uid", how="inner").sort_values("uid").reset_index(drop=True)

    y = pd.to_numeric(df["outcome"], errors="raise").astype(int).to_numpy()

    keep_csv = Path(cfg["keep_csv"])
    if icc_keep:
        keep = _load_icc_selected_features(keep_csv, icc_min=icc_min)
        feature_cols = [f"delta_{feature}" for feature in keep if f"delta_{feature}" in df.columns]
    else:
        feature_cols = [col for col in df.columns if col not in {"uid", "outcome"}]

    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").to_numpy()
    info = {"n_rows": int(len(df)), "n_features": int(X.shape[1])}
    return X, y, info


def select_icc_config(
    trial: optuna.Trial,
    icc_grid: list[float],
    *,
    force_icc: bool = False,
    fixed_icc_keep: bool | None = None,
    fixed_icc_min: float | None = None,
) -> tuple[bool, float | None]:
    if fixed_icc_keep is not None:
        icc_min = None if fixed_icc_min is None else float(fixed_icc_min)
        return bool(fixed_icc_keep), icc_min if fixed_icc_keep else None

    if force_icc:
        return True, float(trial.suggest_categorical("icc_min", icc_grid))

    icc_mode = trial.suggest_categorical("icc_mode", ["no_icc", "icc"])
    if icc_mode == "no_icc":
        return False, None
    if not icc_grid:
        raise ValueError("icc_grid no puede estar vacío")
    icc_min = float(trial.suggest_categorical("icc_min", icc_grid))
    return True, icc_min


def evaluate_trial(
    *,
    X: np.ndarray,
    y: np.ndarray,
    build_pipeline_fn: Callable[[dict], Any],
    model_params: dict,
    random_state: int,
    n_splits: int,
    n_holdout_splits: int,
    test_size: float,
    metric_name: str,
) -> tuple[float, dict]:
    splitter = StratifiedShuffleSplit(
        n_splits=n_holdout_splits,
        test_size=test_size,
        random_state=random_state,
    )
    
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    test_metric_rows_threshold: list[dict[str, float]] = []
    thresholds: list[float] = []

    for train_idx, test_idx in splitter.split(X, y):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        pipe = build_pipeline_fn(model_params)
        oof_proba = cross_val_predict(pipe, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
        selected_threshold, _ = _select_threshold(y_train, oof_proba, metric_name=metric_name)
        thresholds.append(float(selected_threshold))

        pipe.fit(X_train, y_train)
        test_proba = pipe.predict_proba(X_test)[:, 1]
        test_row = _evaluate_threshold(y_test, test_proba, selected_threshold)
        test_metric_rows_threshold.append(test_row)

    test_scores = np.asarray(
        [row[metric_name] for row in test_metric_rows_threshold],
        dtype=float,
    )
    summary = {
        "objective_mean": float(np.nanmean(test_scores)),
        "objective_std": float(np.nanstd(test_scores)),
        "test_metrics": _summarize_metrics(test_metric_rows_threshold),
        "threshold_mean": float(np.nanmean(thresholds)),
        "threshold_std": float(np.nanstd(thresholds)),
    }
    return float(summary["objective_mean"]), summary


def run_optuna_search(
    *,
    build_pipeline_fn: Callable[[dict], Any],
    sample_params_fn: Callable[[optuna.Trial, dict, int], dict],
    config: dict,
) -> Path:
    cfg = dict(DEFAULT_OPTUNA_CONFIG)
    cfg.update(config)

    random_state = int(cfg["random_state"])
    random.seed(random_state)
    np.random.seed(random_state)

    metric_name = str(cfg["metric_name"])
    out_root = Path(cfg["out_root"])
    out_root.mkdir(parents=True, exist_ok=True)

    trial_summaries: list[dict] = []

    def objective(trial: optuna.Trial) -> float:
        icc_keep, icc_min = select_icc_config(
            trial,
            list(cfg["icc_grid"]),
            force_icc=bool(cfg.get("force_icc", False)),
            fixed_icc_keep=cfg.get("icc_keep"),
            fixed_icc_min=cfg.get("icc_min"),
        )
        X, y, data_info = load_dataset(
            cfg,
            icc_keep=icc_keep,
            icc_min=icc_min,
        )
        model_params = sample_params_fn(trial, dict(cfg["base_model_params"]), random_state)
        score, summary = evaluate_trial(
            X=X,
            y=y,
            build_pipeline_fn=build_pipeline_fn,
            model_params=model_params,
            random_state=random_state,
            n_splits=int(cfg["n_splits"]),
            n_holdout_splits=int(cfg["n_holdout_splits"]),
            test_size=float(cfg["test_size"]),
            metric_name=metric_name,
        )
        trial.set_user_attr("icc_keep", icc_keep)
        trial.set_user_attr("icc_min", icc_min)
        trial.set_user_attr("n_features", data_info["n_features"])
        trial.set_user_attr("threshold_mean", summary["threshold_mean"])
        trial.set_user_attr("threshold_std", summary["threshold_std"])
        trial.set_user_attr("objective_std", summary["objective_std"])
        trial_summaries.append(
            {
                "number": int(trial.number),
                "value": float(score),
                "icc_keep": bool(icc_keep),
                "icc_min": None if icc_min is None else float(icc_min),
                "n_features": int(data_info["n_features"]),
                "params": model_params,
                "threshold_mean": float(summary["threshold_mean"]),
                "threshold_std": float(summary["threshold_std"]),
                "objective_std": float(summary["objective_std"]),
            }
        )
        return score

    study = optuna.create_study(
        direction="maximize",
        study_name=str(cfg["study_name"]),
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    study.optimize(
        objective,
        n_trials=int(cfg["n_trials"]),
        n_jobs=int(cfg["n_jobs"]),
        catch=tuple(cfg.get("catch_exceptions", (ValueError, np.linalg.LinAlgError))),
    )

    best_trial = study.best_trial
    best_icc_keep = bool(best_trial.user_attrs["icc_keep"])
    best_icc_min = best_trial.user_attrs["icc_min"]
    X, y, data_info = load_dataset(
        cfg,
        icc_keep=best_icc_keep,
        icc_min=best_icc_min,
    )
    best_params = sample_params_fn(best_trial, dict(cfg["base_model_params"]), random_state)
    final_run_dir = out_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    final_run_dir.mkdir(parents=True, exist_ok=True)

    best_summary = evaluate_trial(
        X=X,
        y=y,
        build_pipeline_fn=build_pipeline_fn,
        model_params=best_params,
        random_state=random_state,
        n_splits=int(cfg["n_splits"]),
        n_holdout_splits=int(cfg["n_holdout_splits"]),
        test_size=float(cfg["test_size"]),
        metric_name=metric_name,
    )[1]
    payload = {
        "study_name": study.study_name,
        "metric_name": metric_name,
        "n_trials": int(cfg["n_trials"]),
        "best_trial_number": int(best_trial.number),
        "best_value": float(best_trial.value),
        "best_icc_keep": best_icc_keep,
        "best_icc_min": best_icc_min,
        "best_params": best_params,
        "best_summary": best_summary,
        "data": {
            "n_rows": data_info["n_rows"],
            "n_features": data_info["n_features"],
        },
        "config": {
            "test_size": float(cfg["test_size"]),
            "n_splits": int(cfg["n_splits"]),
            "n_holdout_splits": int(cfg["n_holdout_splits"]),
            "icc_grid": list(cfg["icc_grid"]),
            "icc_keep": cfg.get("icc_keep"),
            "icc_min": cfg.get("icc_min"),
            "random_state": random_state,
        },
        "trials": trial_summaries,
        "timestamp": datetime.now().isoformat(),
    }
    _save_json(final_run_dir / "study_results.json", payload)
    pd.DataFrame(
        [
            {
                "number": row["number"],
                "value": row["value"],
                "icc_keep": row["icc_keep"],
                "icc_min": row["icc_min"],
                "n_features": row["n_features"],
                "threshold_mean": row["threshold_mean"],
                "threshold_std": row["threshold_std"],
                "objective_std": row["objective_std"],
                **row["params"],
            }
            for row in trial_summaries
        ]
    ).to_csv(final_run_dir / "trials.csv", index=False)

    best_icc_label = "no_icc" if not best_icc_keep else f"icc>{best_icc_min}"
    print(
        f"[optuna_search] rows={data_info['n_rows']} features={data_info['n_features']} "
        f"metric={metric_name} best_value={best_trial.value:.4f} "
        f"best_trial={best_trial.number} best_icc={best_icc_label}"
    )
    print(f"[optuna_search] wrote={final_run_dir / 'study_results.json'}")
    print(f"[optuna_search] wrote={final_run_dir / 'trials.csv'}")
    return final_run_dir
