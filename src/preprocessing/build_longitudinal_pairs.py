from pathlib import Path
import logging

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

INPUT_CSV = Path("outputs/indexing/validated_index.csv")
LONGITUDINAL_PAIRS_CSV = Path("outputs/pairs/longitudinal_pairs.csv")

REQUIRE_RADIOMICS_VALID = True


def _build_pair_uid(patient_id: str, pair_index: int) -> str:
    return f"{patient_id}__bl{pair_index}__fu{pair_index}"


def _build_row_uid(pair_uid: str, timepoint: str) -> str:
    suffix = "bl" if str(timepoint) == "baseline" else "fu"
    return f"{pair_uid}__{suffix}"


def _select_longitudinal_rows(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["pair_index"] = pd.to_numeric(df["pair_index"], errors="raise").astype(int)
    df["timepoint"] = df["timepoint"].astype(str).str.lower()

    baseline = df[df["timepoint"] == "baseline"].copy()
    followup = df[df["timepoint"] == "followup"].copy()

    if REQUIRE_RADIOMICS_VALID:
        baseline = baseline[baseline["is_radiomics_valid"] == True].copy()
    else:
        baseline = baseline[baseline["is_structurally_valid"] == True].copy()

    followup = followup[followup["is_structurally_valid"] == True].copy()

    valid_pairs = baseline[["patient_id", "pair_index"]].drop_duplicates()
    followup = followup.merge(valid_pairs, on=["patient_id", "pair_index"], how="inner")
    baseline = baseline.merge(valid_pairs, on=["patient_id", "pair_index"], how="inner")

    result_df = pd.concat([baseline, followup], ignore_index=True)

    timepoint_counts = (
        result_df.groupby(["patient_id", "pair_index"])["timepoint"]
        .nunique()
        .reset_index(name="n_timepoints")
    )
    valid_pairs = timepoint_counts[timepoint_counts["n_timepoints"] == 2][["patient_id", "pair_index"]]
    return result_df.merge(valid_pairs, on=["patient_id", "pair_index"], how="inner").copy()


def build_longitudinal_pairs(input_csv: Path, output_csv: Path) -> None:
    df = pd.read_csv(input_csv)

    required_cols = {
        "patient_id",
        "timepoint",
        "pair_index",
        "image_path",
        "mask_path",
        "is_structurally_valid",
        "is_radiomics_valid",
        "invalid_reason",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas: {missing}")

    result_df = _select_longitudinal_rows(df)
    result_df["pair_uid"] = result_df.apply(
        lambda row: _build_pair_uid(str(row["patient_id"]), int(row["pair_index"])),
        axis=1,
    )
    result_df["uid"] = result_df.apply(
        lambda row: _build_row_uid(str(row["pair_uid"]), str(row["timepoint"])),
        axis=1,
    )

    result_df = result_df[
        [
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
        ]
    ].sort_values(by=["patient_id", "pair_index", "timepoint"]).reset_index(drop=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_csv, index=False)

    logging.info("Longitudinal pairs guardado en %s", output_csv)
    logging.info("Numero de filas: %d", len(result_df))
    logging.info("Numero de pares: %d", int(result_df["pair_uid"].nunique()))


if __name__ == "__main__":
    build_longitudinal_pairs(INPUT_CSV, LONGITUDINAL_PAIRS_CSV)
