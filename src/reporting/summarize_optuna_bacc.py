from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import pandas as pd


MODEL_ROOTS = {
    "lda": Path("outputs/models/optuna_lda_bacc"),
    "mlp": Path("outputs/models/optuna_mlp_bacc"),
    "random_forest": Path("outputs/models/optuna_random_forest_bacc"),
    "lightgbm": Path("outputs/models/optuna_lightgbm_bacc"),
    "xgboost": Path("outputs/models/optuna_xgboost_bacc"),
}


def _get_nested(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _metric(summary: dict[str, Any], name: str, stat: str) -> Any:
    return _get_nested(summary, "test_metrics", name, stat)


def _as_filter_label(icc_keep: Any, icc_min: Any, wilcoxon_top_k: Any) -> str:
    icc_label = "0"
    if icc_keep and icc_min is not None:
        icc_label = str(int(round(float(icc_min) * 100)))

    wx_label = "0" if wilcoxon_top_k is None else str(wilcoxon_top_k)
    return f"{icc_label}_{wx_label}"


def _is_timestamp_dir(name: str) -> bool:
    return len(name) == 15 and name[8] == "_" and name.replace("_", "").isdigit()


def _read_study(model: str, path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        study = json.load(f)

    summary = study.get("best_summary", {})
    best_params = study.get("best_params", {})
    parent_name = path.parent.name
    study_name = study.get("study_name")
    config_name = study_name if _is_timestamp_dir(parent_name) else parent_name
    icc_keep = study.get("best_icc_keep")
    icc_min = study.get("best_icc_min")
    wilcoxon_top_k = best_params.get("wilcoxon_top_k")

    row = {
        "model": model,
        "config": config_name,
        "study_name": study_name,
        "metric_name": study.get("metric_name"),
        "n_trials": study.get("n_trials"),
        "best_trial_number": study.get("best_trial_number"),
        "best_value": study.get("best_value"),
        "icc_keep": icc_keep,
        "icc_min": icc_min,
        "wilcoxon_top_k": wilcoxon_top_k,
        "filter_label": _as_filter_label(icc_keep, icc_min, wilcoxon_top_k),
        "n_rows": _get_nested(study, "data", "n_rows"),
        "n_features": _get_nested(study, "data", "n_features"),
        "threshold_mean": summary.get("threshold_mean"),
        "threshold_std": summary.get("threshold_std"),
        "accuracy_mean": _metric(summary, "accuracy", "mean"),
        "accuracy_std": _metric(summary, "accuracy", "std"),
        "balanced_accuracy_mean": _metric(summary, "balanced_accuracy", "mean"),
        "balanced_accuracy_std": _metric(summary, "balanced_accuracy", "std"),
        "f1_mean": _metric(summary, "f1", "mean"),
        "f1_std": _metric(summary, "f1", "std"),
        "precision_mean": _metric(summary, "precision", "mean"),
        "precision_std": _metric(summary, "precision", "std"),
        "recall_mean": _metric(summary, "recall", "mean"),
        "recall_std": _metric(summary, "recall", "std"),
        "roc_auc_mean": _metric(summary, "roc_auc", "mean"),
        "roc_auc_std": _metric(summary, "roc_auc", "std"),
        "best_params_json": json.dumps(best_params, sort_keys=True),
        "path": str(path),
    }

    for key, value in best_params.items():
        row[f"param_{key}"] = value

    return row


def collect_results(
    model_roots: dict[str, Path],
    include_old_method: bool = False,
    include_best_dirs: bool = False,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model, root in model_roots.items():
        if not root.exists():
            continue

        for path in root.rglob("study_results.json"):
            parts = set(path.parts)
            if not include_old_method and "old_method" in parts:
                continue
            if not include_best_dirs and path.parent.name.lower().startswith("best"):
                continue
            rows.append(_read_study(model, path))

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values(
        ["model", "best_value"], ascending=[True, False]
    )


def build_best_by_model(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    return (
        df.sort_values(["model", "best_value"], ascending=[True, False])
        .groupby("model", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )


def _safe_stdev(values: list[float]) -> float | None:
    values = [float(value) for value in values if pd.notna(value)]
    if len(values) < 2:
        return None
    return stdev(values)


def build_by_filter(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    rows: list[dict[str, Any]] = []
    for filter_label, group in df.groupby("filter_label"):
        values = group["best_value"].dropna().astype(float).tolist()
        rows.append(
            {
                "filter_label": filter_label,
                "n_models": int(group["model"].nunique()),
                "models": ",".join(sorted(group["model"].unique())),
                "mean_best_value": mean(values) if values else None,
                "std_best_value": _safe_stdev(values),
                "mean_balanced_accuracy_std": group[
                    "balanced_accuracy_std"
                ].mean(),
                "mean_f1": group["f1_mean"].mean(),
                "mean_precision": group["precision_mean"].mean(),
                "mean_recall": group["recall_mean"].mean(),
                "mean_roc_auc": group["roc_auc_mean"].mean(),
                "mean_n_features": group["n_features"].mean(),
            }
        )

    return pd.DataFrame(rows).sort_values("mean_best_value", ascending=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize Optuna balanced-accuracy studies into CSV files."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/models"),
        help="Directory where summary CSV files will be written.",
    )
    parser.add_argument(
        "--include-old-method",
        action="store_true",
        help="Include runs stored below old_method directories.",
    )
    parser.add_argument(
        "--include-best-dirs",
        action="store_true",
        help="Include ad-hoc best*/refined directories in the full summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = collect_results(
        MODEL_ROOTS,
        include_old_method=args.include_old_method,
        include_best_dirs=args.include_best_dirs,
    )
    best_by_model = build_best_by_model(df)
    by_filter = build_by_filter(df)

    summary_csv = args.out_dir / "summary_optuna_bacc.csv"
    best_csv = args.out_dir / "summary_optuna_bacc_best_by_model.csv"
    filter_csv = args.out_dir / "summary_optuna_bacc_by_filter.csv"

    df.to_csv(summary_csv, index=False)
    best_by_model.to_csv(best_csv, index=False)
    by_filter.to_csv(filter_csv, index=False)

    print(f"[summarize_optuna_bacc] wrote={summary_csv} rows={len(df)}")
    print(f"[summarize_optuna_bacc] wrote={best_csv} rows={len(best_by_model)}")
    print(f"[summarize_optuna_bacc] wrote={filter_csv} rows={len(by_filter)}")


if __name__ == "__main__":
    main()
