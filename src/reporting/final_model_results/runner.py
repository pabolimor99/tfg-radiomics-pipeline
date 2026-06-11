from __future__ import annotations

from src.reporting.final_model_results.config import FEATURE_IMPORTANCE_CSV, OUT_DIR, SUMMARY_CSV
from src.reporting.final_model_results.data import load_results
from src.reporting.final_model_results.feature_importance import plot_feature_importances
from src.reporting.final_model_results.icc_wilcoxon import plot_icc_wilcoxon_heatmaps
from src.reporting.final_model_results.plots import (
    plot_global_metric_bars,
    plot_individual_confusion_matrices,
    plot_individual_roc_curves,
    plot_roc_curves,
)
from src.reporting.final_model_results.summary import write_summary


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = load_results()

    combined_cm_path = OUT_DIR / "confusion_matrices_final_models.png"
    if combined_cm_path.exists():
        combined_cm_path.unlink()

    plot_individual_confusion_matrices(results)
    plot_roc_curves(results)
    plot_individual_roc_curves(results)
    plot_global_metric_bars(results)
    plot_feature_importances(results)
    plot_icc_wilcoxon_heatmaps(results)
    write_summary(results)

    print(f"[plot_final_model_results] wrote={OUT_DIR}")
    print(f"[plot_final_model_results] wrote={SUMMARY_CSV}")
    print(f"[plot_final_model_results] wrote={FEATURE_IMPORTANCE_CSV}")
