from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.modeling.analysis.robustness_discrimination import FEATURE_FAMILIES


def short_feature_name(name: str, *, max_len: int = 82) -> str:
    text = name.replace("delta_", "", 1)
    if len(text) <= max_len:
        return text
    return f"{text[: max_len - 3]}..."


def plot_scatter(table: pd.DataFrame, *, out_dir: Path) -> None:
    valid = table.dropna(subset=["icc", "wilcoxon_score"]).copy()

    fig, ax = plt.subplots(figsize=(9, 6))
    cmap = plt.get_cmap("tab10")

    for idx, family in enumerate(FEATURE_FAMILIES):
        group = valid[valid["family"] == family]
        if group.empty:
            continue
        ax.scatter(
            group["icc"],
            group["wilcoxon_score"],
            s=22,
            alpha=0.55,
            color=cmap(idx),
            label=family,
        )

    intersection = valid[valid["in_top_intersection"]]
    ax.scatter(
        intersection["icc"],
        intersection["wilcoxon_score"],
        s=70,
        facecolors="none",
        edgecolors="black",
        linewidths=1.2,
        label="Top 10% ICC y Wilcoxon",
    )

    ax.axvline(0.70, color="gray", linestyle="--", linewidth=1)
    ax.axvline(0.90, color="gray", linestyle=":", linewidth=1)
    ax.axhline(-np.log10(0.05), color="gray", linestyle="--", linewidth=1)
    ax.text(0.705, ax.get_ylim()[1] * 0.96, "ICC 0.70", color="gray", fontsize=8)
    ax.text(0.905, ax.get_ylim()[1] * 0.90, "ICC 0.90", color="gray", fontsize=8)
    ax.text(
        ax.get_xlim()[0] + 0.01,
        -np.log10(0.05) + 0.04,
        "p = 0.05",
        color="gray",
        fontsize=8,
    )

    ax.set_xlabel("ICC")
    ax.set_ylabel(r"Discriminacion Wilcoxon, $-\log_{10}(p)$")
    ax.set_title("Relacion entre robustez ICC y discriminacion Wilcoxon")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
    fig.tight_layout()
    fig.savefig(out_dir / "feature_icc_wilcoxon_scatter.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_top_intersection(table: pd.DataFrame, *, out_dir: Path, top_n_plot: int) -> None:
    top = (
        table[table["in_top_intersection"]]
        .sort_values("combined_rank", ascending=True)
        .head(top_n_plot)
        .sort_values("combined_rank", ascending=False)
    )
    if top.empty:
        return

    labels = [short_feature_name(name) for name in top["delta_feature"]]
    y_pos = np.arange(len(top))

    fig, (ax_icc, ax_wilcoxon) = plt.subplots(
        1,
        2,
        figsize=(13, max(5.5, 0.33 * len(top) + 1.4)),
        sharey=True,
        gridspec_kw={"width_ratios": [1.0, 1.25]},
    )

    ax_icc.barh(y_pos, top["icc"], color="#4c78a8")
    ax_icc.set_xlim(0, 1)
    ax_icc.set_xlabel("ICC")
    ax_icc.set_yticks(y_pos, labels)
    ax_icc.axvline(0.70, color="gray", linestyle="--", linewidth=1)
    ax_icc.axvline(0.90, color="gray", linestyle=":", linewidth=1)
    ax_icc.grid(axis="x", alpha=0.25)

    ax_wilcoxon.barh(y_pos, top["wilcoxon_score"], color="#f28e2b")
    ax_wilcoxon.set_xlabel(r"$-\log_{10}(p)$ Wilcoxon")
    ax_wilcoxon.axvline(-np.log10(0.05), color="gray", linestyle="--", linewidth=1)
    ax_wilcoxon.grid(axis="x", alpha=0.25)

    fig.suptitle("Features en la interseccion de mayor ICC y mayor Wilcoxon", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "feature_icc_wilcoxon_top_intersection.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_family_counts(table: pd.DataFrame, *, out_dir: Path) -> None:
    counts = (
        table[table["in_top_intersection"]]["family"]
        .value_counts()
        .reindex(FEATURE_FAMILIES, fill_value=0)
    )
    counts = counts[counts > 0].sort_values(ascending=True)
    if counts.empty:
        return

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.barh(counts.index, counts.values, color="#59a14f")
    ax.set_xlabel("Numero de features")
    ax.set_title("Familias presentes en la interseccion ICC-Wilcoxon")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "feature_icc_wilcoxon_family_counts.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_all_robustness_discrimination(
    table: pd.DataFrame,
    *,
    out_dir: Path,
    top_n_plot: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_scatter(table, out_dir=out_dir)
    plot_top_intersection(table, out_dir=out_dir, top_n_plot=top_n_plot)
    plot_family_counts(table, out_dir=out_dir)
