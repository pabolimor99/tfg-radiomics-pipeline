from __future__ import annotations

from pathlib import Path

MODEL_RUN_ROOTS = {
    "LDA": Path("outputs/models/lda"),
    "MLP": Path("outputs/models/mlp"),
    "Random Forest": Path("outputs/models/random_forest"),
    "LightGBM": Path("outputs/models/lightgbm"),
    "XGBoost": Path("outputs/models/xgboost"),
}

OUT_DIR = Path("figures/model_results")
SUMMARY_CSV = Path("outputs/models/final_model_plot_summary.csv")
FEATURE_IMPORTANCE_CSV = Path("outputs/models/final_model_feature_importance.csv")
OPTUNA_SUMMARY_CSV = Path("outputs/models/summary_optuna_bacc.csv")
CLASS_NAMES = ["No mejora", "Mejora"]
TOP_N_FEATURES = 20
