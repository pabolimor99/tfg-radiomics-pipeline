import optuna

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
