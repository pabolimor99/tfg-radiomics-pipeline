from pathlib import Path
import logging

import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def clean_str(value) -> str | None:
    if pd.isna(value):
        return None
    value = str(value).strip()
    if value == "":
        return None
    return value


def read_image(path: str):
    try:
        image = sitk.ReadImage(path)
        return image, None
    except Exception as exc:
        return None, str(exc)


def get_pairs_from_row(row: pd.Series) -> list[dict]:
    patient_id = row["patient_id"]
    pairs = []

    for timepoint in ["baseline", "followup"]:
        for i in range(3):
            image_col = f"{timepoint}_image_{i}"
            mask_col = f"{timepoint}_mask_{i}"

            image_path = clean_str(row.get(image_col))
            mask_path = clean_str(row.get(mask_col))

            if image_path is None and mask_path is None:
                continue

            pairs.append(
                {
                    "patient_id": patient_id,
                    "timepoint": timepoint,
                    "pair_index": i,
                    "image_path": image_path,
                    "mask_path": mask_path,
                }
            )

    return pairs


def validate_pair(pair: dict) -> dict:
    image_path = pair["image_path"]
    mask_path = pair["mask_path"]

    image_exists = image_path is not None and Path(image_path).is_file()
    mask_exists = mask_path is not None and Path(mask_path).is_file()

    image_readable = False
    mask_readable = False
    same_shape = False
    same_spacing = False
    mask_not_empty = False
    image = None
    mask = None
    invalid_reasons = []

    if not image_exists:
        invalid_reasons.append("image_not_found")
    if not mask_exists:
        invalid_reasons.append("mask_not_found")

    if image_exists:
        image, _ = read_image(image_path)
        image_readable = image is not None
        if not image_readable:
            invalid_reasons.append("image_not_readable")

    if mask_exists:
        mask, _ = read_image(mask_path)
        mask_readable = mask is not None
        if not mask_readable:
            invalid_reasons.append("mask_not_readable")

    if mask_readable:
        mask_array = sitk.GetArrayViewFromImage(mask)
        unique_labels = np.unique(mask_array).tolist()
        non_zero_labels = [label for label in unique_labels if label != 0]
        mask_not_empty = len(non_zero_labels) > 0

        if not mask_not_empty:
            invalid_reasons.append("empty_mask")

    if image_readable and mask_readable:
        image_size = image.GetSize()
        mask_size = mask.GetSize()
        image_spacing = image.GetSpacing()
        mask_spacing = mask.GetSpacing()
        same_shape = image_size == mask_size
        same_spacing = image_spacing == mask_spacing

        if not same_shape:
            invalid_reasons.append("different_shape")
        if not same_spacing:
            invalid_reasons.append("different_spacing")

    is_valid = (
        image_exists
        and mask_exists
        and image_readable
        and mask_readable
        and same_shape
        and same_spacing
        and mask_not_empty
    )

    is_structurally_valid = (
        image_exists
        and mask_exists
        and image_readable
        and mask_readable
        and same_shape
        and same_spacing
    )

    is_radiomics_valid = is_structurally_valid and mask_not_empty

    return {
        **pair,
        "is_structurally_valid": is_structurally_valid,
        "is_radiomics_valid": is_radiomics_valid,
        "_is_valid": is_valid,
        "invalid_reason": ";".join(invalid_reasons),
    }


def validate_index(index_csv: Path, output_csv: Path, invalid_output_csv: Path) -> None:
    df = pd.read_csv(index_csv)
    pairs = []
    results = []

    logging.info("Validando %d pacientes", len(df))
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generando pares imagen-mascara"):
        pairs.extend(get_pairs_from_row(row))

    logging.info("Se han generado %d pares imagen-mascara", len(pairs))
    for pair in tqdm(pairs, total=len(pairs), desc="Validando pares imagen-mascara"):
        results.append(validate_pair(pair))

    validated_df = pd.DataFrame(results)
    valid_count = int(validated_df["_is_valid"].sum())
    invalid_df = validated_df[validated_df["_is_valid"] == False].drop(columns=["_is_valid"])
    validated_df = validated_df.drop(columns=["_is_valid"])

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    validated_df.to_csv(output_csv, index=False)
    invalid_df.to_csv(invalid_output_csv, index=False)

    logging.info("Pares validos: %d", valid_count)
    logging.info("Pares invalidos: %d", len(invalid_df))


if __name__ == "__main__":
    index_csv = Path("data/index.csv")
    output_csv = Path("data/validated_index.csv")
    invalid_output_csv = Path("data/invalid_pairs.csv")
    validate_index(index_csv, output_csv, invalid_output_csv)
