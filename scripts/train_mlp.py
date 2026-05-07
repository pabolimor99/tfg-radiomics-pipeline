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
from sklearn.preprocessing import MinMaxScaler, StandardScaler

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

def main() -> None:
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    features = pd.read_csv(FEATURES_CSV)
    outcomes = pd.read_csv(OUTCOMES_CSV)[["uid", "outcome"]]

    before_f = len(features)
    before_o = len(outcomes)
    df = features.merge(outcomes, on="uid", how="inner")

    print(f"[train_mlp] features_rows={before_f} outcomes_rows={before_o} merged_rows={len(df)}")
    if df.empty:
        raise ValueError("Merge vacio (no hay uids en comun).")

    y = pd.to_numeric(df["outcome"], errors="raise").astype(int).to_numpy()

    y = 1 - y

    if bool(ICC_KEEP):
        print(f"Filtrando features por ICC (min={ICC_MIN})")
        keep_df = pd.read_csv(KEEP_CSV)
        if "icc" in keep_df.columns:
            keep_df = keep_df[pd.to_numeric(keep_df["icc"], errors="coerce") > float(ICC_MIN)]
        keep = keep_df["feature"].astype(str).tolist()
        feature_cols = [f"delta_{f}" for f in keep if f"delta_{f}" in df.columns]
        missing = [f for f in keep if f"delta_{f}" not in df.columns]
        print(
            f"[train_mlp] icc_keep=1 icc_min>{ICC_MIN} selected_cols={len(feature_cols)} missing={len(missing)}"
        )
    else:
        feature_cols = [c for c in df.columns if c not in {"uid", "outcome"}]
        print(f"[train_mlp] icc_keep=0 selected_cols={len(feature_cols)}")

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

    print(
        f"[train_mlp] split train={len(train_idx)} (pos_rate={float(np.mean(y_train)):.3f}) "
        f"test={len(test_idx)} (pos_rate={float(np.mean(y_test)):.3f})"
    )

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scoring = ["accuracy", "f1", "precision", "recall", "roc_auc", "balanced_accuracy"]
    scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, return_train_score=False, error_score=np.nan)

    summary: dict[str, dict[str, float]] = {}
    print("[MLP] Metricas de validacion cruzada (media +/- std):")
    for metric in scoring:
        key = f"test_{metric}"
        mean = float(np.nanmean(scores[key]))
        std = float(np.nanstd(scores[key]))
        summary[metric] = {"mean": mean, "std": std}
        print(f"   {metric}: {mean:.3f} +/- {std:.3f}")

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
    print(
        "[MLP] Test (holdout 20%): "
        f"roc_auc={test_metrics['roc_auc']:.3f} "
        f"balanced_accuracy={test_metrics['balanced_accuracy']:.3f} "
        f"f1={test_metrics['f1']:.3f}"
    )

    run_dir = OUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    dump(pipe, run_dir / "model.joblib")
    (run_dir / "cv_metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "meta.json").write_text(
        json.dumps(
            {
                "features_csv": str(FEATURES_CSV),
                "outcomes_csv": str(OUTCOMES_CSV),
                "keep_csv": str(KEEP_CSV),
                "icc_keep": bool(ICC_KEEP),
                "icc_min": float(ICC_MIN) if ICC_KEEP else None,
                "out_root": str(OUT_ROOT),
                "n_rows": int(len(df)),
                "n_features": int(X.shape[1]),
                "n_splits": int(N_SPLITS),
                "test_size": float(TEST_SIZE),
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "random_state": int(RANDOM_STATE),
                "delta": {"mode": "relative", "formula": "(fu - bl) / bl"},
                "mlp_params": MLP_PARAMS,
                "timestamp": datetime.now().isoformat(),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"[train_mlp] wrote={run_dir}")

if __name__ == "__main__":
    main()
