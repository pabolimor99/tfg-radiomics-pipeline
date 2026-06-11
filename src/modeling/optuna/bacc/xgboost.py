from pathlib import Path

from src.modeling.models.xgboost import build_xgboost_pipeline
from src.modeling.optuna.search_spaces.xgboost import sample_xgboost_params
from src.modeling.optuna.train import run_optuna_search

OUT_ROOT = Path("outputs/models/optuna_xgboost_bacc")

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
    "icc_keep": True,
    "icc_min": 0.7,
    "base_model_params": {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": 0,
    },
}


def main() -> None:
    run_optuna_search(
        config=OPTUNA_CONFIG,
        build_pipeline_fn=build_xgboost_pipeline,
        sample_params_fn=sample_xgboost_params,
    )


if __name__ == "__main__":
    main()
