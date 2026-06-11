from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.reporting.final_model_results.config import FEATURE_IMPORTANCE_CSV, OUT_DIR, TOP_N_FEATURES
from src.reporting.final_model_results.data import selected_feature_names, short_feature_name


def _load_full_modeling_data(result: dict) -> tuple[np.ndarray, np.ndarray]:
    artifact = result["artifact"]
    feature_cols = list(artifact["feature_cols"])
    data_info = result["metrics"]["data"]

    features = pd.read_csv(Path(data_info["features_csv"]))
    outcomes = pd.read_csv(Path(data_info["outcomes_csv"]))[["uid", "outcome"]]
    df = features.merge(outcomes, on="uid", how="inner").sort_values(by="uid").reset_index(drop=True)

    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").to_numpy()
    y = pd.to_numeric(df["outcome"], errors="raise").astype(int).to_numpy()
    return X, y


def _feature_importance_for_result(result: dict) -> pd.DataFrame:
    pipe = result["artifact"]["pipeline"]
    estimator = list(pipe.named_steps.values())[-1]
    selected_names = selected_feature_names(result)

    if hasattr(estimator, "coef_"):
        values = np.ravel(estimator.coef_).astype(float)
        return pd.DataFrame(
            {
                "model": result["model_name"],
                "feature": selected_names,
                "value": values,
                "importance": np.abs(values),
                "method": "lda_coefficient",
            }
        )

    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=float)
        return pd.DataFrame(
            {
                "model": result["model_name"],
                "feature": selected_names,
                "value": values,
                "importance": values,
                "method": "model_feature_importance",
            }
        )

    X, y = _load_full_modeling_data(result)
    permutation = permutation_importance(
        pipe,
        X,
        y,
        scoring="balanced_accuracy",
        n_repeats=20,
        random_state=42,
        n_jobs=1,
    )
    values = np.asarray(permutation.importances_mean, dtype=float)
    return pd.DataFrame(
        {
            "model": result["model_name"],
            "feature": np.asarray(result["artifact"]["feature_cols"], dtype=object),
            "value": values,
            "importance": values,
            "importance_std": np.asarray(permutation.importances_std, dtype=float),
            "method": "permutation_balanced_accuracy",
        }
    )


def plot_feature_importances(results: list[dict]) -> None:
    all_rows = []

    for result in results:
        importances = _feature_importance_for_result(result)
        all_rows.append(importances)

        top = (
            importances.sort_values("importance", ascending=False)
            .head(TOP_N_FEATURES)
            .sort_values("importance", ascending=True)
        )

        fig_height = max(5.0, 0.32 * len(top) + 1.6)
        fig, ax = plt.subplots(figsize=(9, fig_height))

        labels = [short_feature_name(name) for name in top["feature"]]
        y_pos = np.arange(len(top))

        if top["method"].iloc[0] == "lda_coefficient":
            colors = np.where(top["value"].to_numpy() >= 0, "#4c78a8", "#e45756")
            ax.barh(y_pos, top["value"], color=colors)
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_xlabel("Coeficiente LDA")
            title = "Principales coeficientes - LDA"
        elif top["method"].iloc[0] == "permutation_balanced_accuracy":
            xerr = top.get("importance_std")
            ax.barh(y_pos, top["importance"], xerr=xerr, color="#72b7b2")
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_xlabel("Descenso medio de balanced accuracy")
            title = f"Importancia por permutacion - {result['model_name']}"
        else:
            ax.barh(y_pos, top["importance"], color="#59a14f")
            ax.set_xlabel("Importancia del modelo")
            title = f"Importancia de variables - {result['model_name']}"

        ax.set_yticks(y_pos, labels)
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"feature_importance_{result['slug']}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    FEATURE_IMPORTANCE_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(all_rows, ignore_index=True).to_csv(FEATURE_IMPORTANCE_CSV, index=False)
