from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

INPUT_CSV = Path("outputs/features/longitudinal_features_long.csv")
OUTPUT_CSV = Path("outputs/features/delta_features_uid.csv")


def build_delta_features(
    input_csv: Path = INPUT_CSV,
    output_csv: Path = OUTPUT_CSV,
) -> Path:
    df = pd.read_csv(input_csv)

    required = {"pair_uid", "timepoint"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {input_csv}: {sorted(missing)}")

    baseline = df[df["timepoint"].astype(str).str.lower().eq("baseline")].copy()
    followup = df[df["timepoint"].astype(str).str.lower().eq("followup")].copy()

    feature_cols = [c for c in df.columns if str(c).startswith("original_")]

    baseline = baseline[["pair_uid"] + feature_cols].rename(
        columns={col: f"baseline_{col}" for col in feature_cols}
    )
    followup = followup[["pair_uid"] + feature_cols].rename(
        columns={col: f"followup_{col}" for col in feature_cols}
    )

    merged = baseline.merge(followup, on="pair_uid", how="inner")

    bases: list[str] = []
    for col in feature_cols:
        if f"baseline_{col}" in merged.columns and f"followup_{col}" in merged.columns:
            bases.append(col)

    out_dict: dict[str, pd.Series] = {"uid": merged["pair_uid"]}
    for base in bases:
        bl = pd.to_numeric(merged[f"baseline_{base}"], errors="coerce")
        fu = pd.to_numeric(merged[f"followup_{base}"], errors="coerce")
        denom = bl.replace(0, np.nan)
        out_dict[f"delta_{base}"] = (fu - bl) / denom

    out = pd.DataFrame(out_dict)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    print(f"[prepare_delta_features] input_rows={len(df)} input_cols={df.shape[1]}")
    print(f"[prepare_delta_features] paired_rows={len(merged)}")
    print(f"[prepare_delta_features] delta_cols={len(bases)}")
    print(f"[prepare_delta_features] wrote={output_csv}")
    return output_csv


def main() -> None:
    build_delta_features(INPUT_CSV, OUTPUT_CSV)


if __name__ == "__main__":
    main()
