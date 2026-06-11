from pathlib import Path

from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from src.modeling.feature_selection import WilcoxonFeatureSelector
from src.modeling.optuna.config import load_best_optuna_config
from src.modeling.train import train_model

FEATURES_CSV = Path("outputs/features/delta_features_uid.csv")
OUTCOMES_CSV = Path("outputs/review/classification_outcomes.csv")
KEEP_CSV = Path("outputs/features/icc_results.csv")
OUT_ROOT = Path("outputs/models/mlp")
OPTUNA_OUT_ROOT = Path("outputs/models/optuna_mlp_bacc")

def build_mlp_pipeline(model_params: dict) -> Pipeline:
    scaler_range_name = model_params.get("scaler_range", "zero_one")
    scaler_range = (-1, 1) if scaler_range_name == "minus_one_one" else (0, 1)
    wilcoxon_top_k = model_params.get("wilcoxon_top_k")
    mlp_params = {
        key: value
        for key, value in model_params.items()
        if key not in {"scaler_range", "wilcoxon_top_k"}
    }

    return Pipeline(
        [
            ("wilcoxon", WilcoxonFeatureSelector(top_k=wilcoxon_top_k)),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", MinMaxScaler(feature_range=scaler_range)),
            ("mlp", MLPClassifier(**mlp_params)),
        ]
    )


def _build_mlp_config() -> dict:
    best_config = load_best_optuna_config(OPTUNA_OUT_ROOT)
    best_params = dict(best_config["best_params"])
    best_icc_keep = bool(best_config["best_icc_keep"])
    best_icc_min = best_config["best_icc_min"] if best_icc_keep else None

    model_params = {
        key: value
        for key, value in best_params.items()
        if key
        in {
            "hidden_layer_sizes",
            "alpha",
            "learning_rate_init",
            "activation",
            "max_iter",
            "early_stopping",
            "n_iter_no_change",
            "scaler_range",
            "wilcoxon_top_k",
            "random_state",
            "verbose",
        }
        and value is not None
    }

    return {
        "model_name": "mlp",
        "model_params": model_params,
        "out_root": OUT_ROOT,
        "features_csv": FEATURES_CSV,
        "outcomes_csv": OUTCOMES_CSV,
        "icc_keep": best_icc_keep,
        "icc_min": best_icc_min,
        "keep_csv": KEEP_CSV,
        "n_holdout_splits": 5,
        "test_size": 0.2,
        "metric_name": best_config["metric_name"],
        "random_state": 42,
    }


def main() -> None:
    train_model(
        build_pipeline_fn=build_mlp_pipeline,
        config=_build_mlp_config(),
    )

if __name__ == "__main__":
    main()
