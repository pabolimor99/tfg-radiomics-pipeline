from __future__ import annotations

from pathlib import Path

import pandas as pd

INPUT_CSV = Path("outputs/features/longitudinal_features_long.csv")
OUTPUT_CSV = Path("outputs/features/features_uid.csv")
NON_FEATURE_COLUMNS = {
    "pair_uid",
    "uid",
    "patient_id",
    "pair_index",
    "timepoint",
    "image_path",
    "mask_path",
    "is_structurally_valid",
    "is_radiomics_valid",
    "invalid_reason",
    "mask_empty",
}


def _is_modeling_feature(column: str) -> bool:
    column = str(column)
    return column not in NON_FEATURE_COLUMNS and not column.startswith("diagnostics_")


def build_features_uid(
    input_csv: Path = INPUT_CSV,
    output_csv: Path = OUTPUT_CSV,
) -> Path:
    df = pd.read_csv(input_csv)

    required = {"pair_uid", "patient_id", "pair_index", "timepoint"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas obligatorias en {input_csv}: {sorted(missing)}")

    feature_cols = [c for c in df.columns if _is_modeling_feature(c)]

    pair_order = (
        df.loc[:, ["pair_uid", "patient_id", "pair_index"]]
        .drop_duplicates(subset=["pair_uid"])
        .sort_values(by=["patient_id", "pair_index", "pair_uid"])
        .reset_index(drop=True)
    )
    pair_order["_pair_order"] = range(len(pair_order))

    baseline = (
        df[df["timepoint"] == "baseline"]
        .loc[:, ["pair_uid", *feature_cols]]
        .rename(columns={col: f"baseline_{col}" for col in feature_cols})
    )
    followup = (
        df[df["timepoint"] == "followup"]
        .loc[:, ["pair_uid", *feature_cols]]
        .rename(columns={col: f"followup_{col}" for col in feature_cols})
    )

    out = pair_order.merge(baseline, on="pair_uid", how="left", validate="one_to_one")
    out = out.merge(followup, on="pair_uid", how="left", validate="one_to_one")
    out = out.rename(columns={"pair_uid": "uid"})
    out = out.sort_values("_pair_order").drop(columns=["patient_id", "pair_index", "_pair_order"]).reset_index(drop=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    return output_csv


def main() -> None:
    build_features_uid(INPUT_CSV, OUTPUT_CSV)


if __name__ == "__main__":
    main()
