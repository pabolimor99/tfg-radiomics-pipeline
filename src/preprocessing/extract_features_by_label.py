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

# Input pairs should have the same columns as outputs/longitudinal_pairs.csv (uid, *_image_path, *_mask_path, etc.)
INPUT_CSV = PROJECT_ROOT / "outputs" / "pairs" / "pairs.csv"
OUTPUT_CSV = PROJECT_ROOT / "outputs" / "features" / "radiomics_by_label.csv"
ERRORS_CSV = PROJECT_ROOT / "outputs" / "logs" / "radiomics_by_label_errors.csv"

INCLUDE_DIAGNOSTICS = False
PYRADIOMICS_PARAMS = Path(__file__).resolve().parent / "pyradiomics_ct.yaml"

# Which labels to extract for each pair:
# - "baseline": labels from baseline mask only
# - "followup": labels from followup mask only
# - "union": labels from baseline ∪ followup (captures new lesions in FU)
# - "intersection": labels from baseline ∩ followup (only track lesions present in both)
LABEL_SET_POLICY = "baseline"

# When a label does not exist in followup mask:
# - radiomics columns are left as NaN
# - followup_present is set to False


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


def read_mask(mask_path: Path) -> sitk.Image | None:
    if mask_path is None or not mask_path.exists():
        return None
    return sitk.ReadImage(str(mask_path))


def mask_labels(mask: sitk.Image) -> set[int]:
    arr = sitk.GetArrayViewFromImage(mask)
    labels = np.unique(arr).tolist()
    return set(int(v) for v in labels if int(v) != 0)


def build_binary_mask_for_label(mask: sitk.Image, label: int) -> sitk.Image:
    arr = sitk.GetArrayViewFromImage(mask)
    binary = (arr == int(label)).astype(np.uint8)
    out = sitk.GetImageFromArray(binary)
    out.CopyInformation(mask)
    return out


def extract_radiomics_for_label(
    extractor,
    image_path: Path,
    mask: sitk.Image,
    label: int,
) -> dict:
    binary_mask = build_binary_mask_for_label(mask, label)
    result = extractor.execute(str(image_path), binary_mask)
    return clean_result(result)


def select_labels(bl_mask: sitk.Image | None, fu_mask: sitk.Image | None) -> list[int]:
    bl = mask_labels(bl_mask) if bl_mask is not None else set()
    fu = mask_labels(fu_mask) if fu_mask is not None else set()

    policy = str(LABEL_SET_POLICY).strip().lower()
    if policy == "baseline":
        chosen = bl
    elif policy == "followup":
        chosen = fu
    elif policy == "intersection":
        chosen = bl & fu
    elif policy == "union":
        chosen = bl | fu
    else:
        raise ValueError(f"Unknown LABEL_SET_POLICY: {LABEL_SET_POLICY}")

    return sorted(int(x) for x in chosen)


def prefix_keys(d: dict, prefix: str) -> dict:
    return {f"{prefix}_{k}": v for k, v in d.items()}


def extract_features_by_label(
    input_csv: Path = INPUT_CSV,
    output_csv: Path = OUTPUT_CSV,
    errors_csv: Path = ERRORS_CSV,
):
    logging.info("Input CSV: %s", input_csv)

    if not input_csv.is_file():
        logging.error("Input CSV file not found: %s", input_csv)
        return

    df = pd.read_csv(input_csv)

    required_cols = {
        "uid",
        "patient_id",
        "pair_index",
        "baseline_image_path",
        "baseline_mask_path",
        "followup_image_path",
        "followup_mask_path",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas obligatorias: {missing}")

    extractor = build_extractor()
    rows: list[dict] = []
    errors: list[dict] = []

    total = len(df)
    logging.info("Extrayendo radiomics por label para %d pares", total)

    for _, row in tqdm(df.iterrows(), total=total, desc="Extrayendo features por label"):
        uid = row["uid"]
        patient_id = row["patient_id"]
        pair_index = row["pair_index"]

        baseline_image_path = resolve_path(row["baseline_image_path"])
        baseline_mask_path = resolve_path(row["baseline_mask_path"])
        followup_image_path = resolve_path(row["followup_image_path"])
        followup_mask_path = resolve_path(row["followup_mask_path"])

        if baseline_image_path is None or not baseline_image_path.exists():
            errors.append({
                "uid": uid,
                "patient_id": patient_id,
                "pair_index": pair_index,
                "timepoint": "baseline",
                "lesion_id": None,
                "error": f"Imagen no encontrada: {baseline_image_path}",
            })
            continue

        if baseline_mask_path is None or not baseline_mask_path.exists():
            errors.append({
                "uid": uid,
                "patient_id": patient_id,
                "pair_index": pair_index,
                "timepoint": "baseline",
                "lesion_id": None,
                "error": f"Máscara no encontrada: {baseline_mask_path}",
            })
            continue

        bl_mask = read_mask(baseline_mask_path)
        fu_mask = read_mask(followup_mask_path)

        labels = select_labels(bl_mask, fu_mask)
        if not labels:
            errors.append({
                "uid": uid,
                "patient_id": patient_id,
                "pair_index": pair_index,
                "timepoint": "both",
                "lesion_id": None,
                "error": "No se encontraron labels (no-cero) según LABEL_SET_POLICY",
            })
            continue

        bl_labels = mask_labels(bl_mask) if bl_mask is not None else set()
        fu_labels = mask_labels(fu_mask) if fu_mask is not None else set()

        for lesion_id in labels:
            row_content = {
                "uid": uid,
                "patient_id": patient_id,
                "pair_index": pair_index,
                "lesion_id": int(lesion_id),
                "baseline_image_path": baseline_image_path,
                "baseline_mask_path": baseline_mask_path,
                "followup_image_path": followup_image_path,
                "followup_mask_path": followup_mask_path,
                "baseline_present": int(lesion_id in bl_labels),
                "followup_present": int(lesion_id in fu_labels),
            }

            try:
                baseline_features = {}
                if lesion_id in bl_labels:
                    baseline_features = extract_radiomics_for_label(
                        extractor=extractor,
                        image_path=baseline_image_path,
                        mask=bl_mask,
                        label=lesion_id,
                    )
                baseline_features = prefix_keys(baseline_features, "baseline")

                followup_features = {}
                if followup_image_path is not None and followup_image_path.exists() and fu_mask is not None:
                    if lesion_id in fu_labels:
                        followup_features = extract_radiomics_for_label(
                            extractor=extractor,
                            image_path=followup_image_path,
                            mask=fu_mask,
                            label=lesion_id,
                        )
                followup_features = prefix_keys(followup_features, "followup")

                output_row = dict(row_content)
                output_row.update(baseline_features)
                output_row.update(followup_features)
                rows.append(output_row)

            except Exception as e:
                errors.append({
                    **row_content,
                    "error": str(e),
                })
                continue

    if rows:
        features_df = pd.DataFrame(rows)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        features_df.to_csv(output_csv, index=False)

        logging.info("Features por label guardadas en %s", output_csv)
        logging.info("Filas: %d", len(features_df))
        logging.info("Columnas: %d", len(features_df.columns))
    else:
        logging.warning("No se extrajeron features de ningún par/label")

    if errors:
        errors_df = pd.DataFrame(errors)
        errors_csv.parent.mkdir(parents=True, exist_ok=True)
        errors_df.to_csv(errors_csv, index=False)
        logging.info("Errores guardados en %s", errors_csv)
        logging.info("Total errores: %d", len(errors))


if __name__ == "__main__":
    try:
        extract_features_by_label(INPUT_CSV, OUTPUT_CSV, ERRORS_CSV)
    except Exception as e:
        logging.error("An error occurred: %s", e)
        sys.exit(1)
