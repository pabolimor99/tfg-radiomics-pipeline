from __future__ import annotations

import pandas as pd
from sklearn.metrics import auc, roc_curve

from src.reporting.final_model_results.config import SUMMARY_CSV
from src.reporting.final_model_results.data import metric


def write_summary(results: list[dict]) -> None:
    rows = []
    for result in results:
        cm = result["cm"]
        y_true = result["y_true"]
        y_proba = result["y_proba"]
        pooled_fpr, pooled_tpr, _ = roc_curve(y_true, y_proba)

        rows.append(
            {
                "model": result["model_name"],
                "run_dir": str(result["run_dir"]),
                "n_test_predictions": int(len(y_true)),
                "tn": int(cm[0, 0]),
                "fp": int(cm[0, 1]),
                "fn": int(cm[1, 0]),
                "tp": int(cm[1, 1]),
                "balanced_accuracy_mean": metric(result, "balanced_accuracy"),
                "balanced_accuracy_std": metric(result, "balanced_accuracy", stat="std"),
                "roc_auc_mean": metric(result, "roc_auc"),
                "roc_auc_std": metric(result, "roc_auc", stat="std"),
                "pooled_roc_auc": float(auc(pooled_fpr, pooled_tpr)),
                "threshold_mean": float(result["metrics"]["training"]["threshold_mean"]),
                "threshold_std": float(result["metrics"]["training"]["threshold_std"]),
            }
        )

    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(SUMMARY_CSV, index=False)
