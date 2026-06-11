from pathlib import Path

from src.modeling.models.lda import build_lda_pipeline
from src.modeling.optuna.search_spaces.lda import sample_lda_params
from src.modeling.optuna.train import run_optuna_search

OUT_ROOT = Path("outputs/models/optuna_lda_bacc")

OPTUNA_CONFIG = {
    "study_name": "0_50",
    "n_trials": 50,
    "n_jobs": 1,
    "metric_name": "balanced_accuracy",
    "random_state": 42,
    "n_splits": 5,
    "n_holdout_splits": 5,
    "test_size": 0.2,
    "features_csv": Path("outputs/features/delta_features_uid.csv"),
    "outcomes_csv": Path("outputs/review/classification_outcomes.csv"),
    "keep_csv": Path("outputs/features/icc_results.csv"),
    "out_root": Path("outputs/models/optuna_lda_bacc"),
    "icc_grid": [],
    "icc_keep": False,
    "force_icc": False,
    "icc_min": 0.7,
}


def main() -> None:
    run_optuna_search(
        config=OPTUNA_CONFIG,
        build_pipeline_fn=build_lda_pipeline,
        sample_params_fn=sample_lda_params,
    )


if __name__ == "__main__":
    main()
