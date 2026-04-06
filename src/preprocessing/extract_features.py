from pathlib import Path
import sys
import pandas as pd
from radiomics import featureextractor
import SimpleITK as sitk
import numpy as np
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger("radiomics").setLevel(logging.WARNING)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

INPUT_CSV = PROJECT_ROOT / "data" / "longitudinal_pairs.csv"
PAIRS_CSV = PROJECT_ROOT / "outputs" / "pairs" / "pairs.csv"
OUTPUT_CSV = PROJECT_ROOT / "data" / "longitudinal_features.csv"
ERRORS_CSV = PROJECT_ROOT / "data" / "longitudinal_features_errors.csv"
PYRADIOMICS_PARAMS = Path(__file__).resolve().parent / "pyradiomics_ct.yaml"

INCLUDE_DIAGNOSTICS = False


def resolve_path(path_str: str) -> Path | None:
    if pd.isna(path_str) or str(path_str).strip() == "":
        return None

    path = Path(path_str)

    if path.is_absolute():
        return path

    return (PROJECT_ROOT / path).resolve()


def build_extractor():
    if not PYRADIOMICS_PARAMS.is_file():
        raise FileNotFoundError(f"No se encontró el fichero de parámetros de PyRadiomics: {PYRADIOMICS_PARAMS}")
    return featureextractor.RadiomicsFeatureExtractor(str(PYRADIOMICS_PARAMS))


def clean_result(result: dict) -> dict:
    cleaned = {}
    for key, value in result.items():
        if not INCLUDE_DIAGNOSTICS and str(key).startswith("diagnostics_"):
            continue
        cleaned[key] = value
    return cleaned


def build_binary_mask(mask_path: Path) -> sitk.Image:
    mask = sitk.ReadImage(str(mask_path))
    mask_array = sitk.GetArrayFromImage(mask)
    binary_array = (mask_array > 0).astype(np.uint8)
    binary_mask = sitk.GetImageFromArray(binary_array)
    binary_mask.CopyInformation(mask)
    return binary_mask


def mask_is_empty(mask_path: Path) -> bool:
    mask = sitk.ReadImage(str(mask_path))
    mask_array = sitk.GetArrayViewFromImage(mask)
    return not bool(np.any(mask_array > 0))


def extract_single_timepoint_features(
    extractor,
    image_path: Path,
    mask_path: Path,
    prefix: str
) -> dict:
    if image_path is None or not image_path.exists():
        raise FileNotFoundError(f"Imagen no encontrada: {image_path}")

    if mask_path is None or not mask_path.exists():
        raise FileNotFoundError(f"MÃ¡scara no encontrada: {mask_path}")

    # Tolerate empty masks (e.g., followup empty_mask): skip radiomics and leave features as NaN.
    if mask_is_empty(mask_path):
        return {f"{prefix}_mask_empty": 1}

    binary_mask = build_binary_mask(mask_path)
    result = extractor.execute(str(image_path), binary_mask)
    result = clean_result(result)

    prefixed_result = {}
    prefixed_result[f"{prefix}_mask_empty"] = 0
    for key, value in result.items():
        prefixed_result[f"{prefix}_{key}"] = value

    return prefixed_result


def extract_features(input_csv: Path = INPUT_CSV, output_csv: Path = OUTPUT_CSV, errors_csv: Path = ERRORS_CSV):
    logging.info("Input CSV: %s", input_csv)

    if not input_csv.is_file():
        logging.error("Input CSV file not found: %s", input_csv)
        return

    df = pd.read_csv(input_csv)

    required_cols = {
        "uid",
        "patient_id",
        "baseline_pair_index",
        "followup_pair_index",
        "baseline_image_path",
        "baseline_mask_path",
        "followup_image_path",
        "followup_mask_path",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas obligatorias: {missing}")

    extractor = build_extractor()
    rows = []
    errors = []

    total = len(df)
    logging.info("Extrayendo caracterÃ­sticas longitudinales para %d casos", total)

    for _, row in tqdm(df.iterrows(), total=total, desc="Extrayendo features longitudinales"):
        uid = row["uid"]
        patient_id = row["patient_id"]

        baseline_image_path = resolve_path(row["baseline_image_path"])
        baseline_mask_path = resolve_path(row["baseline_mask_path"])
        followup_image_path = resolve_path(row["followup_image_path"])
        followup_mask_path = resolve_path(row["followup_mask_path"])

        row_content = {
            "uid": uid,
            "patient_id": patient_id,
            "baseline_pair_index": row["baseline_pair_index"],
            "followup_pair_index": row["followup_pair_index"],
            "baseline_image_path": baseline_image_path,
            "baseline_mask_path": baseline_mask_path,
            "followup_image_path": followup_image_path,
            "followup_mask_path": followup_mask_path,
        }

        try:
            baseline_features = extract_single_timepoint_features(
                extractor=extractor,
                image_path=baseline_image_path,
                mask_path=baseline_mask_path,
                prefix="baseline"
            )

            followup_features = extract_single_timepoint_features(
                extractor=extractor,
                image_path=followup_image_path,
                mask_path=followup_mask_path,
                prefix="followup"
            )

            output_row = dict(row_content)
            output_row.update(baseline_features)
            output_row.update(followup_features)
            rows.append(output_row)

        except Exception as e:
            logging.error("Error processing uid %s: %s", uid, e)
            error_row = dict(row_content)
            error_row["error"] = str(e)
            errors.append(error_row)
            continue

    if rows:
        features_df = pd.DataFrame(rows)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        features_df.to_csv(output_csv, index=False)

        logging.info("CaracterÃ­sticas guardadas en %s", output_csv)
        logging.info("Total de casos procesados correctamente: %d", len(rows))
        logging.info("NÃºmero de columnas: %d", len(features_df.columns))
    else:
        logging.warning("No se extrajeron caracterÃ­sticas de ningÃºn caso longitudinal")

    if errors:
        errors_df = pd.DataFrame(errors)
        errors_csv.parent.mkdir(parents=True, exist_ok=True)
        errors_df.to_csv(errors_csv, index=False)
        logging.info("Errores guardados en %s", errors_csv)
        logging.info("Total de casos con error: %d", len(errors))


if __name__ == "__main__":
    try:
        input_csv = INPUT_CSV if INPUT_CSV.is_file() else PAIRS_CSV
        extract_features(input_csv, OUTPUT_CSV, ERRORS_CSV)
    except Exception as e:
        logging.error("An error occurred: %s", e)
        sys.exit(1)
