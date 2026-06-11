from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

INPUT_CSV = Path("outputs/features/features_uid.csv")
OUTPUT_CSV = Path("outputs/features/delta_features_uid.csv")
MAX_NAN_RATIO = 0.9


def build_delta_features(
    input_csv: Path = INPUT_CSV,
    output_csv: Path = OUTPUT_CSV,
) -> Path:
    df = pd.read_csv(input_csv)
    if "uid" not in df.columns:
        raise ValueError(f"Missing 'uid' column in {input_csv}")

    baseline_cols = [c for c in df.columns if str(c).startswith("baseline_")]
    followup_cols = {c for c in df.columns if str(c).startswith("followup_")}

    bases: list[str] = []
    for bl in baseline_cols:
        base = str(bl).removeprefix("baseline_")
        if f"followup_{base}" in followup_cols:
            bases.append(base)

    out_dict: dict[str, pd.Series] = {"uid": df["uid"]}
    for base in bases:
        bl = pd.to_numeric(df[f"baseline_{base}"], errors="coerce")
        fu = pd.to_numeric(df[f"followup_{base}"], errors="coerce")
        denom = bl.replace(0, np.nan)
        out_dict[f"delta_{base}"] = (fu - bl) / denom

    out = pd.DataFrame(out_dict)
    feature_cols = [c for c in out.columns if c != "uid"]
    nan_ratio = out[feature_cols].isna().mean(axis=0)
    kept_feature_cols = nan_ratio[nan_ratio <= MAX_NAN_RATIO].index.tolist()
    dropped_feature_cols = nan_ratio[nan_ratio > MAX_NAN_RATIO].index.tolist()
    out = out.loc[:, ["uid", *kept_feature_cols]]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    return output_csv


def main() -> None:
    build_delta_features(INPUT_CSV, OUTPUT_CSV)


if __name__ == "__main__":
    main()
