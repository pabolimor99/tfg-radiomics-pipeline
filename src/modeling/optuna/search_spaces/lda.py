import optuna

def sample_lda_params(trial: optuna.Trial, base_model_params: dict, random_state: int) -> dict:
    params = dict(base_model_params)
    solver = trial.suggest_categorical("solver", ["svd", "lsqr", "eigen"])
    # solver = trial.suggest_categorical("solver", ["eigen"])
    params["solver"] = solver

    params["n_components"] = 1

    if solver == "lsqr":
        params["shrinkage"] = trial.suggest_categorical("shrinkage", ["auto", None])
    elif solver == "eigen":
        params["shrinkage"] = "auto"
    elif solver == "svd":
        params["shrinkage"] = None

    params["scaler_type"] = trial.suggest_categorical(
        # "scaler_type", ["minmax", "standard", "none"]
        "scaler_type", ["standard", "minmax"]
    )
    params["scaler_range"] = trial.suggest_categorical(
        # "scaler_range", ["zero_one", "minus_one_one"]
        "scaler_range", ["zero_one", "minus_one_one"]
    )
    params["imputer_strategy"] = trial.suggest_categorical(
        # "imputer_strategy", ["median", "mean"]
        "imputer_strategy", ["median", "mean"]
    )
    params["wilcoxon_top_k"] = trial.suggest_categorical(
        "wilcoxon_top_k", [50]
        # "wilcoxon_top_k", [75]
    )
    return params
