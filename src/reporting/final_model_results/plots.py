from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

from src.reporting.final_model_results.config import CLASS_NAMES, OUT_DIR
from src.reporting.final_model_results.data import metric


def _plot_cm(ax: plt.Axes, result: dict, *, title: str | None = None) -> None:
    cm = result["cm"]
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)

    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks([0, 1], CLASS_NAMES)
    ax.set_yticks([0, 1], CLASS_NAMES)
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Clase real")

    for i in range(2):
        for j in range(2):
            color = "white" if cm_norm[i, j] > 0.55 else "black"
            ax.text(
                j,
                i,
                f"{cm[i, j]}\n{cm_norm[i, j] * 100:.1f}%",
                ha="center",
                va="center",
                color=color,
                fontsize=9,
            )

    if title is None:
        title = result["model_name"]
    ax.set_title(title)
    return im


def plot_confusion_matrices(results: list[dict]) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(9, 12), constrained_layout=True)
    axes_flat = axes.ravel()
    last_im = None

    for ax, result in zip(axes_flat, results):
        bacc = metric(result, "balanced_accuracy")
        roc_auc = metric(result, "roc_auc")
        title = f"{result['model_name']}\nBAcc={bacc:.3f}, AUC={roc_auc:.3f}"
        last_im = _plot_cm(ax, result, title=title)

    axes_flat[-1].axis("off")
    fig.colorbar(
        last_im,
        ax=axes_flat.tolist(),
        fraction=0.03,
        pad=0.02,
        shrink=0.72,
        label="Porcentaje por clase real",
    )
    fig.suptitle("Matrices de confusion agregadas en los holdouts de test", fontsize=14, fontweight="bold")
    fig.savefig(OUT_DIR / "confusion_matrices_final_models.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_individual_confusion_matrices(results: list[dict]) -> None:
    for result in results:
        fig, ax = plt.subplots(figsize=(4.8, 4.2))
        _plot_cm(ax, result)
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"confusion_matrix_{result['slug']}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def _mean_roc_by_fold(result: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    preds = result["predictions"]
    mean_fpr = np.linspace(0, 1, 200)
    tprs = []

    for _, fold_df in preds.groupby("fold"):
        y_true = fold_df["y_true"].astype(int).to_numpy()
        y_proba = fold_df["y_proba"].astype(float).to_numpy()
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

    mean_tpr = np.mean(tprs, axis=0)
    std_tpr = np.std(tprs, axis=0)
    mean_tpr[-1] = 1.0
    return mean_fpr, mean_tpr, std_tpr


def plot_roc_curves(results: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))

    for result in results:
        fpr, mean_tpr, _ = _mean_roc_by_fold(result)
        auc_mean = metric(result, "roc_auc")
        auc_std = metric(result, "roc_auc", stat="std")
        ax.plot(fpr, mean_tpr, linewidth=2, label=f"{result['model_name']} ({auc_mean:.3f} +/- {auc_std:.3f})")

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("Tasa de falsos positivos")
    ax.set_ylabel("Tasa de verdaderos positivos")
    ax.set_title("Curvas ROC medias en los holdouts de test")
    ax.legend(title="ROC AUC medio", loc="lower right", fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "roc_curves_final_models.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_individual_roc_curves(results: list[dict]) -> None:
    for result in results:
        fpr, mean_tpr, std_tpr = _mean_roc_by_fold(result)
        auc_mean = metric(result, "roc_auc")
        auc_std = metric(result, "roc_auc", stat="std")

        fig, ax = plt.subplots(figsize=(5.2, 4.6))
        ax.plot(fpr, mean_tpr, linewidth=2, label=f"AUC={auc_mean:.3f} +/- {auc_std:.3f}")
        ax.fill_between(
            fpr,
            np.maximum(mean_tpr - std_tpr, 0),
            np.minimum(mean_tpr + std_tpr, 1),
            alpha=0.18,
        )
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
        ax.set_xlabel("Tasa de falsos positivos")
        ax.set_ylabel("Tasa de verdaderos positivos")
        ax.set_title(f"Curva ROC - {result['model_name']}")
        ax.legend(loc="lower right")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"roc_curve_{result['slug']}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_global_metric_bars(results: list[dict]) -> None:
    model_names = [result["model_name"] for result in results]
    x = np.arange(len(model_names))
    width = 0.36

    bacc_mean = np.asarray([metric(result, "balanced_accuracy") for result in results])
    bacc_std = np.asarray([metric(result, "balanced_accuracy", stat="std") for result in results])
    auc_mean = np.asarray([metric(result, "roc_auc") for result in results])
    auc_std = np.asarray([metric(result, "roc_auc", stat="std") for result in results])

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(
        x - width / 2,
        bacc_mean,
        width,
        yerr=bacc_std,
        capsize=4,
        label="Balanced accuracy",
        color="#4c78a8",
    )
    ax.bar(
        x + width / 2,
        auc_mean,
        width,
        yerr=auc_std,
        capsize=4,
        label="ROC AUC",
        color="#f28e2b",
    )

    lower = min(np.min(bacc_mean - bacc_std), np.min(auc_mean - auc_std))
    upper = max(np.max(bacc_mean + bacc_std), np.max(auc_mean + auc_std))
    y_min = max(0.0, np.floor((lower - 0.005) * 100) / 100)
    y_max = min(1.0, np.ceil((upper + 0.005) * 100) / 100)

    ax.set_ylabel("Valor medio")
    ax.set_title("Comparacion global de rendimiento por modelo")
    ax.set_xticks(x, model_names, rotation=18, ha="right")
    ax.set_ylim(y_min, y_max)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "global_model_performance_bacc_auc.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    mean_lower = min(float(np.min(bacc_mean)), float(np.min(auc_mean)))
    mean_upper = max(float(np.max(bacc_mean)), float(np.max(auc_mean)))
    zoom_y_min = max(0.0, mean_lower - 0.008)
    zoom_y_max = min(1.0, mean_upper + 0.014)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(
        x - width / 2,
        bacc_mean,
        width,
        label="Balanced accuracy",
        color="#4c78a8",
    )
    ax.bar(
        x + width / 2,
        auc_mean,
        width,
        label="ROC AUC",
        color="#f28e2b",
    )

    for xpos, value, std in zip(x - width / 2, bacc_mean, bacc_std):
        ax.text(
            xpos,
            value + 0.001,
            f"{value:.3f}\n+/-{std:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    for xpos, value, std in zip(x + width / 2, auc_mean, auc_std):
        ax.text(
            xpos,
            value + 0.001,
            f"{value:.3f}\n+/-{std:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_ylabel("Valor medio")
    ax.set_title("Comparacion global de rendimiento por modelo")
    ax.set_xticks(x, model_names, rotation=18, ha="right")
    ax.set_ylim(zoom_y_min, zoom_y_max)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "global_model_performance_bacc_auc_zoom.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
