from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import confusion_matrix

from src.reporting.final_model_results.config import MODEL_RUN_ROOTS


def latest_run_dir(root: Path) -> Path:
    run_dirs = [
        path
        for path in root.iterdir()
        if path.is_dir()
        and (path / "metrics.json").is_file()
        and (path / "test_predictions.csv").is_file()
    ]
    if not run_dirs:
        raise FileNotFoundError(f"No valid run directories found in {root}")
    return max(run_dirs, key=lambda path: path.stat().st_mtime)


def load_result(model_name: str, root: Path) -> dict:
    run_dir = latest_run_dir(root)
    preds = pd.read_csv(run_dir / "test_predictions.csv")
    with (run_dir / "metrics.json").open("r", encoding="utf-8") as f:
        metrics = json.load(f)
    artifact = load(run_dir / "model.joblib")

    y_true = preds["y_true"].astype(int).to_numpy()
    y_pred = preds["y_pred"].astype(int).to_numpy()
    y_proba = preds["y_proba"].astype(float).to_numpy()
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    return {
        "model_name": model_name,
        "slug": model_name.lower().replace(" ", "_"),
        "run_dir": run_dir,
        "artifact": artifact,
        "predictions": preds,
        "metrics": metrics,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "cm": cm,
    }


def load_results() -> list[dict]:
    return [load_result(model_name, root) for model_name, root in MODEL_RUN_ROOTS.items()]


def metric(result: dict, name: str, stat: str = "mean") -> float:
    return float(result["metrics"]["test_metrics"][name][stat])


def short_feature_name(name: str, *, max_len: int = 78) -> str:
    text = name.replace("delta_", "", 1)
    if len(text) <= max_len:
        return text
    return f"{text[: max_len - 3]}..."


def selected_feature_names(result: dict) -> np.ndarray:
    artifact = result["artifact"]
    pipe = artifact["pipeline"]
    feature_names = np.asarray(artifact["feature_cols"], dtype=object)

    wilcoxon = pipe.named_steps.get("wilcoxon")
    if wilcoxon is None or not hasattr(wilcoxon, "get_support"):
        return feature_names

    return feature_names[np.asarray(wilcoxon.get_support(), dtype=int)]
