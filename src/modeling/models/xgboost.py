from pathlib import Path

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.modeling.feature_selection import WilcoxonFeatureSelector
from src.modeling.optuna.config import load_best_optuna_config
from src.modeling.train import train_model

FEATURES_CSV = Path("outputs/features/delta_features_uid.csv")
OUTCOMES_CSV = Path("outputs/review/classification_outcomes.csv")
KEEP_CSV = Path("outputs/features/icc_results.csv")
OUT_ROOT = Path("outputs/models/xgboost")
OPTUNA_OUT_ROOT = Path("outputs/models/optuna_xgboost_bacc")
RANDOM_STATE = 42


def build_xgboost_pipeline(model_params: dict) -> Pipeline:
    wilcoxon_top_k = model_params.get("wilcoxon_top_k")
    xgb_params = {
        key: value for key, value in model_params.items() if key != "wilcoxon_top_k"
    }

    return Pipeline(
        [
            ("wilcoxon", WilcoxonFeatureSelector(top_k=wilcoxon_top_k)),
            ("imputer", SimpleImputer(strategy="median")),
            ("xgboost", XGBClassifier(**xgb_params)),
        ]
    )


def _build_xgboost_config() -> dict:
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
            "max_depth",
            "min_child_weight",
            "subsample",
            "colsample_bytree",
            "gamma",
            "reg_alpha",
            "reg_lambda",
            "wilcoxon_top_k",
            "objective",
            "eval_metric",
            "tree_method",
            "random_state",
            "n_jobs",
            "verbosity",
        }
        and value is not None
    }

    return {
        "model_name": "xgboost",
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
            "notes": "XGBoost binary classifier optimized with Optuna",
        },
    }


def main() -> None:
    train_model(
        build_pipeline_fn=build_xgboost_pipeline,
        config=_build_xgboost_config(),
    )


if __name__ == "__main__":
    main()
