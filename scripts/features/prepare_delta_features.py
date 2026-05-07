from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

INPUT_CSV = Path("outputs/features/features_uid.csv")
OUTPUT_CSV = Path("outputs/features/delta_features_uid.csv")


def build_delta_features(
    input_csv: Path = INPUT_CSV,
    output_csv: Path = OUTPUT_CSV,
) -> Path:
    df = pd.read_csv(input_csv)
    if "uid" not in df.columns:
        raise ValueError(f"Missing 'uid' column in {input_csv}")

    baseline_cols = [c for c in df.columns if str(c).startswith("baseline_original_")]
    followup_cols = {c for c in df.columns if str(c).startswith("followup_original_")}

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

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    print(f"[prepare_delta_features] input_rows={len(df)} input_cols={df.shape[1]}")
    print(f"[prepare_delta_features] delta_cols={len(bases)}")
    print(f"[prepare_delta_features] wrote={output_csv}")
    return output_csv


def main() -> None:
    build_delta_features(INPUT_CSV, OUTPUT_CSV)


if __name__ == "__main__":
    main()
