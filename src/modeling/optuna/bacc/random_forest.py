from pathlib import Path

from src.modeling.models.random_forest import build_random_forest_pipeline
from src.modeling.optuna.search_spaces.random_forest import sample_rf_params
from src.modeling.optuna.train import run_optuna_search

OUT_ROOT = Path("outputs/models/optuna_random_forest_bacc")

OPTUNA_CONFIG = {
    "study_name": "70_0",
    "n_trials": 100,
    "metric_name": "balanced_accuracy",
    "random_state": 42,
    "n_splits": 5,
    "n_holdout_splits": 5,
    "test_size": 0.2,
    "features_csv": Path("outputs/features/delta_features_uid.csv"),
    "outcomes_csv": Path("outputs/review/classification_outcomes.csv"),
    "keep_csv": Path("outputs/features/icc_results.csv"),
    "out_root": OUT_ROOT,
    "icc_grid": [],
    "force_icc": True,
    "icc_keep": True,
    "icc_min": 0.7,
}


def main() -> None:
    run_optuna_search(
        config=OPTUNA_CONFIG,
        build_pipeline_fn=build_random_forest_pipeline,
        sample_params_fn=sample_rf_params,
    )


if __name__ == "__main__":
    main()
