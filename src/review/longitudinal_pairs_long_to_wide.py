from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_INPUT_CSV = PROJECT_ROOT / "outputs" / "pairs" / "longitudinal_pairs.csv"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "outputs" / "review" / "longitudinal_pairs_review_wide.csv"


def longitudinal_pairs_long_to_wide(df: pd.DataFrame) -> pd.DataFrame:
    required = {
        "pair_uid",
        "patient_id",
        "pair_index",
        "timepoint",
        "image_path",
        "mask_path",
        "is_structurally_valid",
        "is_radiomics_valid",
        "invalid_reason",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en longitudinal_pairs.csv: {sorted(missing)}")

    base = df[df["timepoint"].astype(str).str.lower().eq("baseline")].copy()
    fu = df[df["timepoint"].astype(str).str.lower().eq("followup")].copy()

    base = base.rename(
        columns={
            "pair_uid": "uid",
            "image_path": "baseline_image_path",
            "mask_path": "baseline_mask_path",
            "is_structurally_valid": "baseline_is_structurally_valid",
            "is_radiomics_valid": "baseline_is_radiomics_valid",
            "invalid_reason": "baseline_invalid_reason",
        }
    )
    fu = fu.rename(
        columns={
            "pair_uid": "uid",
            "image_path": "followup_image_path",
            "mask_path": "followup_mask_path",
            "is_structurally_valid": "followup_is_structurally_valid",
            "is_radiomics_valid": "followup_is_radiomics_valid",
            "invalid_reason": "followup_invalid_reason",
        }
    )

    keep_base = [
        "uid",
        "patient_id",
        "pair_index",
        "baseline_image_path",
        "baseline_mask_path",
        "baseline_is_structurally_valid",
        "baseline_is_radiomics_valid",
        "baseline_invalid_reason",
    ]
    keep_fu = [
        "uid",
        "followup_image_path",
        "followup_mask_path",
        "followup_is_structurally_valid",
        "followup_is_radiomics_valid",
        "followup_invalid_reason",
    ]

    out = base[keep_base].merge(fu[keep_fu], on="uid", how="inner")
    out = out.sort_values(by=["patient_id", "pair_index"]).reset_index(drop=True)
    return out


def convert_longitudinal_pairs_long_to_wide(
    input_csv: Path = DEFAULT_INPUT_CSV,
    output_csv: Path = DEFAULT_OUTPUT_CSV,
) -> Path:
    df = pd.read_csv(input_csv)
    wide_df = longitudinal_pairs_long_to_wide(df)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    wide_df.to_csv(output_csv, index=False)
    logging.info("OK: guardado %d pares wide en %s", len(wide_df), output_csv)
    return output_csv


def main() -> None:
    convert_longitudinal_pairs_long_to_wide()


if __name__ == "__main__":
    main()
