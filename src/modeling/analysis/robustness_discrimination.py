from __future__ import annotations

from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ranksums


FEATURE_FAMILIES = ["firstorder", "shape", "glcm", "glrlm", "glszm", "gldm", "ngtdm"]


def family_from_feature(feature: str) -> str:
    for family in FEATURE_FAMILIES:
        if f"_{family}_" in feature:
            return family
    return "other"


def compute_wilcoxon(features_csv: Path, outcomes_csv: Path) -> pd.DataFrame:
    features = pd.read_csv(features_csv)
    outcomes = pd.read_csv(outcomes_csv)[["uid", "outcome"]]
    df = features.merge(outcomes, on="uid", how="inner").sort_values("uid").reset_index(drop=True)

    y = pd.to_numeric(df["outcome"], errors="raise").astype(int)
    feature_cols = [col for col in df.columns if col not in {"uid", "outcome"}]

    rows = []
    for col in feature_cols:
        values = pd.to_numeric(df[col], errors="coerce")
        group_0 = values[(y == 0) & values.notna()].to_numpy(dtype=float)
        group_1 = values[(y == 1) & values.notna()].to_numpy(dtype=float)

        stat = np.nan
        p_value = np.nan
        if len(group_0) > 0 and len(group_1) > 0:
            try:
                stat, p_value = ranksums(group_0, group_1)
            except ValueError:
                stat, p_value = np.nan, np.nan

        rows.append(
            {
                "delta_feature": col,
                "feature": col.replace("delta_", "", 1),
                "family": family_from_feature(col),
                "n_non_missing": int(values.notna().sum()),
                "missing_rate": float(values.isna().mean()),
                "median_outcome_0": float(np.nanmedian(group_0)) if len(group_0) else np.nan,
                "median_outcome_1": float(np.nanmedian(group_1)) if len(group_1) else np.nan,
                "median_diff_1_minus_0": (
                    float(np.nanmedian(group_1) - np.nanmedian(group_0))
                    if len(group_0) and len(group_1)
                    else np.nan
                ),
                "wilcoxon_stat": float(stat) if np.isfinite(stat) else np.nan,
                "wilcoxon_p": float(p_value) if np.isfinite(p_value) else np.nan,
            }
        )

    return pd.DataFrame(rows)


def build_feature_table(
    *,
    features_csv: Path,
    outcomes_csv: Path,
    icc_csv: Path,
    top_fraction: float,
) -> pd.DataFrame:
    wilcoxon = compute_wilcoxon(features_csv, outcomes_csv)
    icc = pd.read_csv(icc_csv)
    icc = icc[["feature", "icc"]].copy()
    icc["feature"] = icc["feature"].astype(str)
    icc["icc"] = pd.to_numeric(icc["icc"], errors="coerce")

    table = wilcoxon.merge(icc, on="feature", how="left")

    p_values = pd.to_numeric(table["wilcoxon_p"], errors="coerce").to_numpy(dtype=float)
    p_values = np.clip(p_values, np.nextafter(0, 1), 1.0)
    table["wilcoxon_score"] = -np.log10(p_values)

    valid = table["icc"].notna() & table["wilcoxon_p"].notna()
    top_n = int(ceil(valid.sum() * top_fraction))

    table["icc_rank"] = np.nan
    table["wilcoxon_rank"] = np.nan
    table.loc[valid, "icc_rank"] = table.loc[valid, "icc"].rank(ascending=False, method="min")
    table.loc[valid, "wilcoxon_rank"] = table.loc[valid, "wilcoxon_score"].rank(
        ascending=False,
        method="min",
    )
    table["combined_rank"] = table[["icc_rank", "wilcoxon_rank"]].mean(axis=1)
    table["top_n_cutoff"] = top_n
    table["in_top_icc"] = table["icc_rank"] <= top_n
    table["in_top_wilcoxon"] = table["wilcoxon_rank"] <= top_n
    table["in_top_intersection"] = table["in_top_icc"] & table["in_top_wilcoxon"]

    return table.sort_values(["in_top_intersection", "combined_rank"], ascending=[False, True])


def write_feature_tables(
    table: pd.DataFrame,
    *,
    out_csv: Path,
    out_intersection_csv: Path,
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_csv, index=False)
    table[table["in_top_intersection"]].to_csv(out_intersection_csv, index=False)
