import optuna


def sample_xgboost_params(
    trial: optuna.Trial, base_model_params: dict, random_state: int
) -> dict:
    params = dict(base_model_params)
    params.update(
        {
            "n_estimators": trial.suggest_int(
                "n_estimators", 80, 260, step=20
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.08, log=True
            ),
            "max_depth": trial.suggest_int(
                "max_depth", 2, 4
            ),
            "min_child_weight": trial.suggest_int(
                "min_child_weight", 1, 12
            ),
            "subsample": trial.suggest_float(
                "subsample", 0.65, 1.0
            ),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.65, 1.0
            ),
            "gamma": trial.suggest_float(
                "gamma", 0.0, 1.0
            ),
            "reg_alpha": trial.suggest_float(
                "reg_alpha", 1e-5, 5.0, log=True
            ),
            "reg_lambda": trial.suggest_float(
                "reg_lambda", 1e-3, 10.0, log=True
            ),
            "wilcoxon_top_k": None,
            # "wilcoxon_top_k": 15,
            # "wilcoxon_top_k": 50,
            "random_state": random_state,
        }
    )
    return params
