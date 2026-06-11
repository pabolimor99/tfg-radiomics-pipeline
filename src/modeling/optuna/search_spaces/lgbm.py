import optuna


def sample_lgbm_params(trial: optuna.Trial, base_model_params: dict, random_state: int) -> dict:
    params = dict(base_model_params)
    params.update(
        {
            "n_estimators": trial.suggest_int(
                "n_estimators", 120, 320, step=20
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.006, 0.04, log=True
            ),
            "num_leaves": trial.suggest_int(
                "num_leaves", 15, 63, step=4
            ),
            "max_depth": trial.suggest_int(
                "max_depth", 2, 5
            ),
            "min_child_samples": trial.suggest_int(
                "min_child_samples", 20, 45
            ),
            "subsample": trial.suggest_float(
                "subsample", 0.65, 1.0
            ),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.65, 1.0
            ),
            "reg_alpha": trial.suggest_float(
                "reg_alpha", 1e-4, 5.0, log=True
            ),
            "reg_lambda": trial.suggest_float(
                "reg_lambda", 1e-5, 5.0, log=True
            ),
            "min_split_gain": trial.suggest_float(
                "min_split_gain", 0.2, 0.8
            ),
            "wilcoxon_top_k": None,
            "random_state": random_state,
        }
    )
    return params
