from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from radiomics import featureextractor
import SimpleITK as sitk
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger("radiomics").setLevel(logging.ERROR)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

INPUT_CSV = PROJECT_ROOT / "outputs" / "pairs" / "longitudinal_pairs.csv"
OUTPUT_CSV = PROJECT_ROOT / "outputs" / "features" / "longitudinal_features_long.csv"
ERRORS_CSV = PROJECT_ROOT / "outputs" / "logs" / "longitudinal_features_errors.csv"
PYRADIOMICS_PARAMS = Path(__file__).resolve().parent / "pyradiomics_ct.yaml"

INCLUDE_DIAGNOSTICS = False
DEFAULT_WORKERS = max(1, min(16, (os.cpu_count() or 4) - 1))

_WORKER_EXTRACTOR = None


def resolve_path(path_str: str) -> Path | None:
    if pd.isna(path_str) or str(path_str).strip() == "":
        return None

    path = Path(path_str)
    if path.is_absolute():
        return path

    return (PROJECT_ROOT / path).resolve()


def build_extractor():
    if not PYRADIOMICS_PARAMS.is_file():
        raise FileNotFoundError(f"No se encontro el fichero de parametros de PyRadiomics: {PYRADIOMICS_PARAMS}")
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
) -> dict:
    if image_path is None or not image_path.exists():
        raise FileNotFoundError(f"Imagen no encontrada: {image_path}")

    if mask_path is None or not mask_path.exists():
        raise FileNotFoundError(f"Mascara no encontrada: {mask_path}")

    if mask_is_empty(mask_path):
        return {"mask_empty": 1}

    binary_mask = build_binary_mask(mask_path)
    result = extractor.execute(str(image_path), binary_mask)
    result = clean_result(result)

    output = {"mask_empty": 0}
    output.update(result)
    return output


def _init_worker(params_path: str, include_diagnostics: bool) -> None:
    global _WORKER_EXTRACTOR, INCLUDE_DIAGNOSTICS, PYRADIOMICS_PARAMS
    INCLUDE_DIAGNOSTICS = bool(include_diagnostics)
    PYRADIOMICS_PARAMS = Path(params_path)
    _WORKER_EXTRACTOR = build_extractor()


def _extract_for_row(row: dict, extractor) -> tuple[dict | None, dict | None]:
    image_path = resolve_path(row["image_path"])
    mask_path = resolve_path(row["mask_path"])

    row_content = {
        "pair_uid": row["pair_uid"],
        "uid": row["uid"],
        "patient_id": row["patient_id"],
        "pair_index": row["pair_index"],
        "timepoint": row["timepoint"],
        "image_path": image_path,
        "mask_path": mask_path,
    }
    for optional_col in ("is_structurally_valid", "is_radiomics_valid", "invalid_reason"):
        if optional_col in row:
            row_content[optional_col] = row[optional_col]

    try:
        features = extract_single_timepoint_features(
            extractor=extractor,
            image_path=image_path,
            mask_path=mask_path,
        )

        output_row = dict(row_content)
        output_row.update(features)
        return output_row, None

    except Exception as e:
        error_row = dict(row_content)
        error_row["error"] = str(e)
        return None, error_row


def _process_row_worker(row: dict) -> tuple[dict | None, dict | None]:
    global _WORKER_EXTRACTOR
    if _WORKER_EXTRACTOR is None:
        _WORKER_EXTRACTOR = build_extractor()
    return _extract_for_row(row=row, extractor=_WORKER_EXTRACTOR)


def extract_features(
    input_csv: Path = INPUT_CSV,
    output_csv: Path = OUTPUT_CSV,
    errors_csv: Path = ERRORS_CSV,
    workers: int = DEFAULT_WORKERS,
) -> None:
    logging.info("Input CSV: %s", input_csv)

    if not input_csv.is_file():
        logging.error("Input CSV file not found: %s", input_csv)
        return

    df = pd.read_csv(input_csv)

    required_cols = {
        "pair_uid",
        "uid",
        "patient_id",
        "pair_index",
        "timepoint",
        "image_path",
        "mask_path",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas obligatorias: {missing}")

    rows: list[dict] = []
    errors: list[dict] = []

    total = len(df)
    logging.info("Extrayendo caracteristicas longitudinales en formato long para %d filas", total)

    workers = int(workers) if workers is not None else 0
    if workers <= 1:
        extractor = build_extractor()
        for _, row in tqdm(df.iterrows(), total=total, desc="Extrayendo features longitudinales"):
            out_row, err_row = _extract_for_row(row=row.to_dict(), extractor=extractor)
            if out_row is not None:
                rows.append(out_row)
            if err_row is not None:
                errors.append(err_row)
    else:
        records = df.to_dict(orient="records")
        logging.info("Procesando en paralelo con %d workers", workers)
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(str(PYRADIOMICS_PARAMS), bool(INCLUDE_DIAGNOSTICS)),
        ) as ex:
            futures = [ex.submit(_process_row_worker, r) for r in records]
            for fut in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Extrayendo features longitudinales (parallel)",
            ):
                out_row, err_row = fut.result()
                if out_row is not None:
                    rows.append(out_row)
                if err_row is not None:
                    errors.append(err_row)

    if rows:
        features_df = pd.DataFrame(rows)
        features_df = features_df.sort_values(by=["patient_id", "pair_index", "timepoint"]).reset_index(drop=True)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        features_df.to_csv(output_csv, index=False)

        logging.info("Caracteristicas guardadas en %s", output_csv)
        logging.info("Total de filas procesadas correctamente: %d", len(rows))
        logging.info("Numero de columnas: %d", len(features_df.columns))
    else:
        logging.warning("No se extrajeron caracteristicas de ninguna fila longitudinal")

    if errors:
        errors_df = pd.DataFrame(errors)
        errors_df = errors_df.sort_values(by=["patient_id", "pair_index", "timepoint"]).reset_index(drop=True)
        errors_csv.parent.mkdir(parents=True, exist_ok=True)
        errors_df.to_csv(errors_csv, index=False)
        logging.info("Errores guardados en %s", errors_csv)
        logging.info("Total de filas con error: %d", len(errors))


if __name__ == "__main__":
    try:
        workers = int(os.environ.get("RADIOMICS_WORKERS", str(DEFAULT_WORKERS)))
        extract_features(INPUT_CSV, OUTPUT_CSV, ERRORS_CSV, workers=workers)
    except Exception as e:
        logging.error("An error occurred: %s", e)
        sys.exit(1)
