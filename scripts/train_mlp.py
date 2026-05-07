from __future__ import annotations

import json
import random
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
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, cross_validate
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
TEST_SIZE = 0.2
ICC_MIN = 0.7

MLP_PARAMS = {
    "hidden_layer_sizes": (256, 128, 64, 32),
    "alpha": 0.12,
    "max_iter": 1000,
    "early_stopping": True,
    "n_iter_no_change": 50,
    "random_state": RANDOM_STATE,
    "verbose": False,
}


def _save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

def train_model(
    *,
    features_csv: Path = FEATURES_CSV,
    outcomes_csv: Path = OUTCOMES_CSV,
    icc_keep: bool = ICC_KEEP,
    keep_csv: Path = KEEP_CSV,
    out_root: Path = OUT_ROOT,
) -> Path:
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    features = pd.read_csv(features_csv)
    outcomes = pd.read_csv(outcomes_csv)[["uid", "outcome"]]

    before_f = len(features)
    before_o = len(outcomes)
    df = features.merge(outcomes, on="uid", how="inner")
    # df = df.sort_values(by="uid").reset_index(drop=True)

    if df.empty:
        raise ValueError("Merge vacio (no hay uids en comun).")

    uids = df["uid"].astype(str).reset_index(drop=True)
    y = pd.to_numeric(df["outcome"], errors="raise").astype(int).to_numpy()
    y = 1 - y

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

    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", MinMaxScaler()),
            ("mlp", MLPClassifier(**MLP_PARAMS)),
        ]
    )

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(X, y))
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    train_uids = uids.iloc[train_idx].reset_index(drop=True)
    test_uids = uids.iloc[test_idx].reset_index(drop=True)

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scoring = ["accuracy", "f1", "precision", "recall", "roc_auc", "balanced_accuracy"]
    scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, return_train_score=False, error_score=np.nan)

    cv_metrics: dict[str, dict[str, float]] = {}
    for metric in scoring:
        key = f"test_{metric}"
        mean = float(np.nanmean(scores[key]))
        std = float(np.nanstd(scores[key]))
        cv_metrics[metric] = {"mean": mean, "std": std}

    pipe.fit(X_train, y_train)

    test_pred = pipe.predict(X_test)
    try:
        test_proba = pipe.predict_proba(X_test)[:, 1]
    except Exception:
        test_proba = None

    test_metrics = {
        "n": int(len(y_test)),
        "pos_rate": float(np.mean(y_test)),
        "accuracy": float(accuracy_score(y_test, test_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, test_pred)),
        "f1": float(f1_score(y_test, test_pred, zero_division=0)),
        "precision": float(precision_score(y_test, test_pred, zero_division=0)),
        "recall": float(recall_score(y_test, test_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, test_proba)) if (test_proba is not None and len(np.unique(y_test)) > 1) else float("nan"),
    }
    run_dir = out_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    dump(pipe, run_dir / "model.joblib")
    pd.DataFrame({"uid": train_uids}).to_csv(run_dir / "train_uids.csv", index=False)
    pd.DataFrame({"uid": test_uids}).to_csv(run_dir / "test_uids.csv", index=False)
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
        },
        "training": {
            "icc_keep": bool(icc_keep),
            "icc_min": float(ICC_MIN) if icc_keep else None,
            "keep_csv": str(keep_csv),
            "n_splits": int(N_SPLITS),
            "test_size": float(TEST_SIZE),
            "random_state": int(RANDOM_STATE),
            "delta": {"mode": "relative", "formula": "(fu - bl) / bl"},
            "mlp_params": MLP_PARAMS,
        },
        "split": {
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "train_pos_rate": float(np.mean(y_train)),
            "test_pos_rate": float(np.mean(y_test)),
            "train_uids_csv": "train_uids.csv",
            "test_uids_csv": "test_uids.csv",
        },
        "cv_metrics": cv_metrics,
        "test_metrics": test_metrics,
        "timestamp": datetime.now().isoformat(),
    }
    _save_json(run_dir / "metrics.json", metrics_payload)

    print(
        f"[train_mlp] rows={len(df)} features={len(feature_cols)} "
        f"cv_roc_auc={cv_metrics['roc_auc']['mean']:.3f} "
        f"test_roc_auc={test_metrics['roc_auc']:.3f}"
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
