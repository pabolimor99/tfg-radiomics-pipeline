from __future__ import annotations

from pathlib import Path

import pandas as pd

INPUT_CSV = Path("outputs/features/longitudinal_features.csv")
OUTPUT_CSV = Path("outputs/features/features_uid.csv")


def main() -> None:
    df = pd.read_csv(INPUT_CSV)

    if "uid" not in df.columns:
        raise ValueError(f"Falta columna 'uid' en {INPUT_CSV}")

    baseline_cols = [c for c in df.columns if str(c).startswith("baseline_original_")]
    followup_cols = [c for c in df.columns if str(c).startswith("followup_original_")]
    out_cols = ["uid"] + baseline_cols + followup_cols

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df[out_cols].to_csv(OUTPUT_CSV, index=False)

    print(f"[prepare_mlp_features] input_rows={len(df)} input_cols={df.shape[1]}")
    print(
        f"[prepare_mlp_features] selected_cols={len(out_cols)} (baseline={len(baseline_cols)} followup={len(followup_cols)})"
    )
    print(f"[prepare_mlp_features] wrote={OUTPUT_CSV}")


if __name__ == "__main__":
    main()
