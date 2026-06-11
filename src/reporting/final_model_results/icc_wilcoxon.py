from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.reporting.final_model_results.config import OPTUNA_SUMMARY_CSV, OUT_DIR


def _icc_label(row: pd.Series) -> str:
    if not bool(row["icc_keep"]):
        return "Sin ICC"
    return f"ICC > {float(row['icc_min']):.2f}"


def _wilcoxon_label(value: object) -> str:
    if pd.isna(value):
        return "Sin Wilcoxon"
    return str(int(float(value)))


def _plot_icc_wilcoxon_grid(
    *,
    results: list[dict],
    summary: pd.DataFrame,
    row_order: list[str],
    col_order: list[str],
    value_col: str,
    title_prefix: str,
    colorbar_label: str,
    output_suffix: str,
    cmap: str,
    mark: str,
) -> None:
    vmin = float(summary[value_col].min())
    vmax = float(summary[value_col].max())
    threshold_color = vmin + 0.55 * (vmax - vmin)

    for result in results:
        model_key = result["model_name"].lower().replace(" ", "_")
        model_df = summary[summary["model"] == model_key]
        matrix = model_df.pivot(
            index="icc_label",
            columns="wilcoxon_label",
            values=value_col,
        )
        matrix = matrix.reindex(
            index=[row for row in row_order if row in matrix.index],
            columns=[col for col in col_order if col in matrix.columns],
        )

        values = matrix.to_numpy(dtype=float)
        marker_value = np.nanmax(values) if mark == "max" else np.nanmin(values)

        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        im = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(np.arange(matrix.shape[1]), matrix.columns)
        ax.set_yticks(np.arange(matrix.shape[0]), matrix.index)
        ax.set_xlabel("Variables seleccionadas por Wilcoxon")
        ax.set_ylabel("Filtro ICC")
        ax.set_title(f"{title_prefix} segun ICC y Wilcoxon - {result['model_name']}")

        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                value = values[i, j]
                if np.isnan(value):
                    text = "-"
                    color = "black"
                else:
                    marker = "*" if np.isclose(value, marker_value) else ""
                    text = f"{value:.3f}{marker}"
                    color = "white" if value >= threshold_color else "black"
                ax.text(j, i, text, ha="center", va="center", color=color, fontsize=9)

        fig.colorbar(im, ax=ax, label=colorbar_label)
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"heatmap_icc_wilcoxon_{result['slug']}{output_suffix}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_icc_wilcoxon_heatmaps(results: list[dict]) -> None:
    summary = pd.read_csv(OPTUNA_SUMMARY_CSV)
    summary["icc_label"] = summary.apply(_icc_label, axis=1)
    summary["wilcoxon_label"] = summary["wilcoxon_top_k"].apply(_wilcoxon_label)

    row_order = ["Sin ICC", "ICC > 0.70", "ICC > 0.80", "ICC > 0.90", "ICC > 0.99"]
    col_order = ["Sin Wilcoxon", "15", "50", "75"]

    _plot_icc_wilcoxon_grid(
        results=results,
        summary=summary,
        row_order=row_order,
        col_order=col_order,
        value_col="balanced_accuracy_mean",
        title_prefix="Balanced accuracy media",
        colorbar_label="Balanced accuracy media",
        output_suffix="",
        cmap="viridis",
        mark="max",
    )
    _plot_icc_wilcoxon_grid(
        results=results,
        summary=summary,
        row_order=row_order,
        col_order=col_order,
        value_col="balanced_accuracy_std",
        title_prefix="Desviacion estandar de balanced accuracy",
        colorbar_label="Desviacion estandar",
        output_suffix="_std",
        cmap="magma",
        mark="min",
    )
