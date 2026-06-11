import optuna

def sample_rf_params(trial, base_model_params: dict, random_state: int) -> dict:
    params = dict(base_model_params)

    params["n_estimators"] = trial.suggest_categorical(
        "n_estimators",
        [50, 100, 150],
    )

    params["max_depth"] = trial.suggest_categorical(
        "max_depth",
        [2, 3, 4, 5],
    )

    params["min_samples_split"] = trial.suggest_int(
        "min_samples_split",
        4,
        10,
    )

    params["min_samples_leaf"] = trial.suggest_int(
        "min_samples_leaf",
        2,
        8,
    )

    params["max_features"] = trial.suggest_categorical(
        "max_features",
        ["sqrt", "log2"],
    )

    params["class_weight"] = trial.suggest_categorical(
        "class_weight",
        [None, "balanced"],
    )

    # params["wilcoxon_top_k"] = None
    # params["wilcoxon_top_k"] = 15
    params["wilcoxon_top_k"] = None

    params["random_state"] = random_state
    params["n_jobs"] = -1

    return params
