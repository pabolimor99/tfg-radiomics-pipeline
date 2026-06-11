from pathlib import Path

import optuna

from src.modeling.models.mlp import build_mlp_pipeline
from src.modeling.optuna.train import run_optuna_search

OUT_ROOT = Path("outputs/models/optuna_mlp_bacc")

OPTUNA_CONFIG = {
    "study_name": "90_0",
    "n_trials": 100,
    "n_jobs": 1,
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
    "icc_min": 0.9,
}


def sample_mlp_params(trial: optuna.Trial, base_model_params: dict, random_state: int) -> dict:
    params = dict(base_model_params)
    layer_options = {
        "16": (16,),
        "32": (32,),
        "64": (64,),
        "32_16": (32, 16),
        "64_32": (64, 32),
    }
    layer_key = trial.suggest_categorical("hidden_layer_sizes", list(layer_options.keys()))
    params["hidden_layer_sizes"] = layer_options[layer_key]
    params["alpha"] = trial.suggest_float("alpha", 7e-3, 2e-2, log=True)
    params["learning_rate_init"] = trial.suggest_float("learning_rate_init", 6e-4, 1.1e-3, log=True)
    params["activation"] = "relu"
    params["scaler_range"] = "zero_one"
    params["wilcoxon_top_k"] = trial.suggest_categorical("wilcoxon_top_k", [None])
    params["max_iter"] = trial.suggest_categorical("max_iter", [300, 600, 1000])
    params["early_stopping"] = True
    params["n_iter_no_change"] = trial.suggest_categorical("n_iter_no_change", [20])
    params["random_state"] = random_state
    params["verbose"] = False
    return params


def main() -> None:
    run_optuna_search(
        config=OPTUNA_CONFIG,
        build_pipeline_fn=build_mlp_pipeline,
        sample_params_fn=sample_mlp_params,
    )


if __name__ == "__main__":
    main()
