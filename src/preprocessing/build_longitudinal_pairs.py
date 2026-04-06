from pathlib import Path
import pandas as pd
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

INPUT_CSV = Path("data/validated_index.csv")
LONGITUDINAL_PAIRS_CSV = Path("data/longitudinal_pairs.csv")

REQUIRE_RADIOMICS_VALID = True
REQUIRE_COMPLETE_PAIR = True


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
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas: {missing}")

    if REQUIRE_RADIOMICS_VALID:
        df = df[df["is_radiomics_valid"] == True].copy()
    else:
        df = df[df["is_structurally_valid"] == True].copy()

    rows = []

    for patient_id, patient_df in tqdm(
        df.groupby("patient_id"),
        desc="Construyendo pares longitudinales"
    ):
        baseline_df = patient_df[patient_df["timepoint"] == "baseline"].copy()
        followup_df = patient_df[patient_df["timepoint"] == "followup"].copy()

        if baseline_df.empty or followup_df.empty:
            continue

        baseline_by_idx = {}
        for _, row in baseline_df.iterrows():
            idx = int(row["pair_index"])
            baseline_by_idx[idx] = row

        followup_by_idx = {}
        for _, row in followup_df.iterrows():
            idx = int(row["pair_index"])
            followup_by_idx[idx] = row

        common_indexes = sorted(set(baseline_by_idx.keys()) & set(followup_by_idx.keys()))

        for pair_index in common_indexes:
            bl_row = baseline_by_idx[pair_index]
            fu_row = followup_by_idx[pair_index]

            baseline_image_path = bl_row["image_path"]
            baseline_mask_path = bl_row["mask_path"]
            followup_image_path = fu_row["image_path"]
            followup_mask_path = fu_row["mask_path"]

            if REQUIRE_COMPLETE_PAIR:
                if pd.isna(baseline_image_path) or pd.isna(baseline_mask_path):
                    continue
                if pd.isna(followup_image_path) or pd.isna(followup_mask_path):
                    continue

            rows.append({
                "uid": f"{patient_id}__bl{pair_index}__fu{pair_index}",
                "patient_id": patient_id,
                "pair_index": pair_index,
                "baseline_pair_index": int(bl_row["pair_index"]),
                "followup_pair_index": int(fu_row["pair_index"]),
                "baseline_image_path": bl_row["image_path"],
                "baseline_mask_path": bl_row["mask_path"],
                "followup_image_path": fu_row["image_path"],
                "followup_mask_path": fu_row["mask_path"],
                "baseline_is_structurally_valid": bl_row["is_structurally_valid"],
                "baseline_is_radiomics_valid": bl_row["is_radiomics_valid"],
                "followup_is_structurally_valid": fu_row["is_structurally_valid"],
                "followup_is_radiomics_valid": fu_row["is_radiomics_valid"],
            })

    result_df = pd.DataFrame(rows)

    if not result_df.empty:
        result_df = result_df.sort_values(
            by=["patient_id", "pair_index"]
        ).reset_index(drop=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_csv, index=False)

    logging.info("Longitudinal pairs guardado en %s", output_csv)
    logging.info("Número de pares longitudinales: %d", len(result_df))


if __name__ == "__main__":
    build_longitudinal_pairs(INPUT_CSV, LONGITUDINAL_PAIRS_CSV)
