from pathlib import Path

from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.modeling.feature_selection import WilcoxonFeatureSelector
from src.modeling.optuna.config import load_best_optuna_config
from src.modeling.train import train_model

FEATURES_CSV = Path("outputs/features/delta_features_uid.csv")
OUTCOMES_CSV = Path("outputs/review/classification_outcomes.csv")
KEEP_CSV = Path("outputs/features/icc_results.csv")
OUT_ROOT = Path("outputs/models/lightgbm")
OPTUNA_OUT_ROOT = Path("outputs/models/optuna_lightgbm_bacc")
RANDOM_STATE = 42


def build_lightgbm_pipeline(model_params: dict) -> Pipeline:
    wilcoxon_top_k = model_params.get("wilcoxon_top_k")
    lgbm_params = {
        key: value for key, value in model_params.items() if key != "wilcoxon_top_k"
    }

    return Pipeline(
        [
            ("wilcoxon", WilcoxonFeatureSelector(top_k=wilcoxon_top_k)),
            ("imputer", SimpleImputer(strategy="median")),
            ("lgbm", LGBMClassifier(**lgbm_params)),
        ]
    )


def _build_lightgbm_config() -> dict:
    best_config = load_best_optuna_config(OPTUNA_OUT_ROOT)
    best_params = dict(best_config["best_params"])
    best_icc_keep = bool(best_config["best_icc_keep"])
    best_icc_min = best_config["best_icc_min"] if best_icc_keep else None

    model_params = {
        key: value
        for key, value in best_params.items()
        if key
        in {
            "n_estimators",
            "learning_rate",
            "num_leaves",
            "max_depth",
            "min_child_samples",
            "subsample",
            "colsample_bytree",
            "reg_alpha",
            "reg_lambda",
            "min_split_gain",
            "wilcoxon_top_k",
            "random_state",
            "n_jobs",
            "verbosity",
        }
        and value is not None
    }

    return {
        "model_name": "lightgbm",
        "model_params": model_params,
        "out_root": OUT_ROOT,
        "features_csv": FEATURES_CSV,
        "outcomes_csv": OUTCOMES_CSV,
        "icc_keep": best_icc_keep,
        "icc_min": best_icc_min,
        "keep_csv": KEEP_CSV,
        "random_state": RANDOM_STATE,
        "n_holdout_splits": 5,
        "test_size": 0.2,
        "metric_name": best_config["metric_name"],
        "extra_training_metadata": {
            "notes": "LightGBM baseline with moderate regularization",
        },
    }


def main() -> None:
    train_model(
        build_pipeline_fn=build_lightgbm_pipeline,
        config=_build_lightgbm_config(),
    )


if __name__ == "__main__":
    main()
