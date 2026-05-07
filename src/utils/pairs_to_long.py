from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_CSV = PROJECT_ROOT / "outputs" / "pairs" / "pairs.csv"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "outputs" / "pairs" / "pairs_long.csv"


def pairs_to_long(df: pd.DataFrame) -> pd.DataFrame:
    required = {
        "uid",
        "patient_id",
        "baseline_pair_index",
        "followup_pair_index",
        "baseline_image_path",
        "baseline_mask_path",
        "followup_image_path",
        "followup_mask_path",
        "baseline_is_structurally_valid",
        "baseline_is_radiomics_valid",
        "followup_is_structurally_valid",
        "followup_is_radiomics_valid",
        "source",
        "followup_invalid_reason",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas obligatorias en pairs.csv: {sorted(missing)}")

    base = df[
        [
            "uid",
            "patient_id",
            "baseline_pair_index",
            "baseline_image_path",
            "baseline_mask_path",
            "baseline_is_structurally_valid",
            "baseline_is_radiomics_valid",
            "source",
        ]
    ].copy()
    base = base.rename(
        columns={
            "uid": "pair_uid",
            "baseline_pair_index": "pair_index",
            "baseline_image_path": "image_path",
            "baseline_mask_path": "mask_path",
            "baseline_is_structurally_valid": "is_structurally_valid",
            "baseline_is_radiomics_valid": "is_radiomics_valid",
        }
    )
    base["timepoint"] = "baseline"
    base["invalid_reason"] = ""
    base["uid"] = base["patient_id"].astype(str) + "__bl" + base["pair_index"].astype(str)

    fu = df[
        [
            "uid",
            "patient_id",
            "followup_pair_index",
            "followup_image_path",
            "followup_mask_path",
            "followup_is_structurally_valid",
            "followup_is_radiomics_valid",
            "source",
            "followup_invalid_reason",
        ]
    ].copy()
    fu = fu.rename(
        columns={
            "uid": "pair_uid",
            "followup_pair_index": "pair_index",
            "followup_image_path": "image_path",
            "followup_mask_path": "mask_path",
            "followup_is_structurally_valid": "is_structurally_valid",
            "followup_is_radiomics_valid": "is_radiomics_valid",
            "followup_invalid_reason": "invalid_reason",
        }
    )
    fu["timepoint"] = "followup"
    fu["uid"] = fu["patient_id"].astype(str) + "__fu" + fu["pair_index"].astype(str)

    out = pd.concat([base, fu], ignore_index=True)
    out["timepoint"] = pd.Categorical(out["timepoint"], categories=["baseline", "followup"], ordered=True)
    out = out.sort_values(by=["patient_id", "pair_index", "timepoint"]).reset_index(drop=True)
    cols = [
        "uid",
        "pair_uid",
        "patient_id",
        "pair_index",
        "timepoint",
        "image_path",
        "mask_path",
        "is_structurally_valid",
        "is_radiomics_valid",
        "source",
        "invalid_reason",
    ]
    return out[cols]


def main() -> None:
    p = argparse.ArgumentParser(description="Convierte outputs/pairs/pairs.csv a formato long (2 filas por par).")
    p.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    p.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    args = p.parse_args()

    df = pd.read_csv(args.input_csv)
    long_df = pairs_to_long(df)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(args.output_csv, index=False)
    logging.info("OK: guardado %d filas en %s", len(long_df), args.output_csv)


if __name__ == "__main__":
    main()
