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
PROGRESS_CSV = PROJECT_ROOT / "outputs" / "features" / "longitudinal_features_long.partial.csv"
ERRORS_PROGRESS_CSV = PROJECT_ROOT / "outputs" / "logs" / "longitudinal_features_errors.partial.csv"
PYRADIOMICS_PARAMS = Path(__file__).resolve().parent / "pyradiomics_ct.yaml"

INCLUDE_DIAGNOSTICS = False
DEFAULT_WORKERS = max(1, min(2, (os.cpu_count() or 4) - 1))
MAX_PENDING_FACTOR = 2
CHECKPOINT_EVERY = int(os.environ.get("RADIOMICS_CHECKPOINT_EVERY", "25"))

_WORKER_EXTRACTORS = None


def is_allowed_feature(key: str) -> bool:
    key = str(key)
    return key.startswith(("original_", "log-sigma-")) or (
        key.startswith("wavelet-") and "_firstorder_" in key
    )


def resolve_path(path_str: str) -> Path | None:
    if pd.isna(path_str) or str(path_str).strip() == "":
        return None

    path = Path(path_str)
    if path.is_absolute():
        return path

    return (PROJECT_ROOT / path).resolve()


def build_extractor(*, wavelet_firstorder: bool = False):
    if not PYRADIOMICS_PARAMS.is_file():
        raise FileNotFoundError(f"No se encontro el fichero de parametros de PyRadiomics: {PYRADIOMICS_PARAMS}")
    extractor = featureextractor.RadiomicsFeatureExtractor(str(PYRADIOMICS_PARAMS))
    if wavelet_firstorder:
        extractor.disableAllImageTypes()
        extractor.enableImageTypeByName("Wavelet")
        extractor.disableAllFeatures()
        extractor.enableFeatureClassByName("firstorder")
    else:
        extractor.enableImageTypeByName("Wavelet", enabled=False)
    return extractor


def build_extractors():
    return (build_extractor(), build_extractor(wavelet_firstorder=True))


def clean_result(result: dict) -> dict:
    cleaned = {}
    for key, value in result.items():
        if not INCLUDE_DIAGNOSTICS and str(key).startswith("diagnostics_"):
            continue
        if not is_allowed_feature(key):
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
    extractors,
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
    result = {}
    for extractor in extractors:
        result.update(clean_result(extractor.execute(str(image_path), binary_mask)))

    output = {"mask_empty": 0}
    output.update(result)
    return output


def _init_worker(params_path: str, include_diagnostics: bool) -> None:
    global _WORKER_EXTRACTORS, INCLUDE_DIAGNOSTICS, PYRADIOMICS_PARAMS
    INCLUDE_DIAGNOSTICS = bool(include_diagnostics)
    PYRADIOMICS_PARAMS = Path(params_path)
    _WORKER_EXTRACTORS = build_extractors()


def _extract_for_row(row: dict, extractors) -> tuple[dict | None, dict | None]:
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
            extractors=extractors,
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
    global _WORKER_EXTRACTORS
    if _WORKER_EXTRACTORS is None:
        _WORKER_EXTRACTORS = build_extractors()
    return _extract_for_row(row=row, extractors=_WORKER_EXTRACTORS)


def row_key(row: dict) -> tuple[str, str]:
    return (str(row["pair_uid"]), str(row["timepoint"]))


def load_partial_csv(path: Path, key_columns: tuple[str, ...]) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame(columns=list(key_columns))

    partial = pd.read_csv(path)
    if partial.empty:
        return pd.DataFrame(columns=list(key_columns))
    if not set(key_columns).issubset(set(partial.columns)):
        logging.warning("Checkpoint parcial invalido, faltan columnas %s en %s", key_columns, path)
        return pd.DataFrame(columns=list(key_columns))
    for col in key_columns:
        partial[col] = partial[col].astype(str)
    return partial


def persist_partial_csv(path: Path, rows: list[dict], key_columns: tuple[str, ...]) -> pd.DataFrame:
    current_df = pd.DataFrame(rows)
    if not current_df.empty:
        current_df = current_df.copy()
        for col in key_columns:
            current_df[col] = current_df[col].astype(str)
        current_df = current_df.drop_duplicates(subset=list(key_columns), keep="last").reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    current_df.to_csv(path, index=False)
    return current_df


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

    partial_rows_df = load_partial_csv(PROGRESS_CSV, ("pair_uid", "timepoint"))
    partial_errors_df = load_partial_csv(ERRORS_PROGRESS_CSV, ("pair_uid", "timepoint"))
    completed_row_keys = set(zip(partial_rows_df["pair_uid"], partial_rows_df["timepoint"])) if not partial_rows_df.empty else set()
    completed_error_keys = set(zip(partial_errors_df["pair_uid"], partial_errors_df["timepoint"])) if not partial_errors_df.empty else set()
    completed_keys = completed_row_keys | completed_error_keys
    if completed_keys:
        logging.info("Reanudando desde checkpoints: %d filas ya resueltas", len(completed_keys))

    records = df.to_dict(orient="records")
    pending_records = [r for r in records if row_key(r) not in completed_keys]
    rows: list[dict] = partial_rows_df.to_dict(orient="records") if not partial_rows_df.empty else []
    errors: list[dict] = partial_errors_df.to_dict(orient="records") if not partial_errors_df.empty else []
    since_checkpoint = 0

    total = len(df)
    logging.info("Extrayendo caracteristicas longitudinales en formato long para %d filas", total)
    if not pending_records:
        logging.info("No quedan filas pendientes; consolidando salidas desde checkpoints.")

    workers = int(workers) if workers is not None else 0
    if workers <= 1:
        extractors = build_extractors()
        for row in tqdm(pending_records, total=len(pending_records), desc="Extrayendo features longitudinales"):
            out_row, err_row = _extract_for_row(row=row, extractors=extractors)
            if out_row is not None:
                rows.append(out_row)
                since_checkpoint += 1
            if err_row is not None:
                errors.append(err_row)
                since_checkpoint += 1
            if since_checkpoint >= CHECKPOINT_EVERY:
                rows = persist_partial_csv(PROGRESS_CSV, rows, ("pair_uid", "timepoint")).to_dict(orient="records")
                errors = persist_partial_csv(ERRORS_PROGRESS_CSV, errors, ("pair_uid", "timepoint")).to_dict(orient="records")
                since_checkpoint = 0
    else:
        logging.info("Procesando en paralelo con %d workers", workers)
        max_pending = max(workers, workers * int(os.environ.get("RADIOMICS_MAX_PENDING_FACTOR", str(MAX_PENDING_FACTOR))))
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(str(PYRADIOMICS_PARAMS), bool(INCLUDE_DIAGNOSTICS)),
        ) as ex:
            pending = {}
            record_iter = iter(pending_records)

            for _ in range(min(max_pending, len(pending_records))):
                record = next(record_iter, None)
                if record is None:
                    break
                pending[ex.submit(_process_row_worker, record)] = record

            progress = tqdm(total=len(pending_records), desc="Extrayendo features longitudinales (parallel)")
            while pending:
                for fut in as_completed(tuple(pending)):
                    record = pending.pop(fut)
                    progress.update(1)
                    try:
                        out_row, err_row = fut.result()
                    except Exception as e:
                        out_row = None
                        err_row = {
                            "pair_uid": record["pair_uid"],
                            "uid": record["uid"],
                            "patient_id": record["patient_id"],
                            "pair_index": record["pair_index"],
                            "timepoint": record["timepoint"],
                            "image_path": resolve_path(record["image_path"]),
                            "mask_path": resolve_path(record["mask_path"]),
                            "error": str(e),
                        }
                    if out_row is not None:
                        rows.append(out_row)
                        since_checkpoint += 1
                    if err_row is not None:
                        errors.append(err_row)
                        since_checkpoint += 1
                    if since_checkpoint >= CHECKPOINT_EVERY:
                        rows = persist_partial_csv(PROGRESS_CSV, rows, ("pair_uid", "timepoint")).to_dict(orient="records")
                        errors = persist_partial_csv(ERRORS_PROGRESS_CSV, errors, ("pair_uid", "timepoint")).to_dict(orient="records")
                        since_checkpoint = 0

                    next_record = next(record_iter, None)
                    if next_record is not None:
                        pending[ex.submit(_process_row_worker, next_record)] = next_record
                    break
            progress.close()

    rows = persist_partial_csv(PROGRESS_CSV, rows, ("pair_uid", "timepoint")).to_dict(orient="records")
    errors = persist_partial_csv(ERRORS_PROGRESS_CSV, errors, ("pair_uid", "timepoint")).to_dict(orient="records")

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

    if PROGRESS_CSV.is_file():
        PROGRESS_CSV.unlink()
    if ERRORS_PROGRESS_CSV.is_file():
        ERRORS_PROGRESS_CSV.unlink()


if __name__ == "__main__":
    try:
        workers = int(os.environ.get("RADIOMICS_WORKERS", str(DEFAULT_WORKERS)))
        extract_features(INPUT_CSV, OUTPUT_CSV, ERRORS_CSV, workers=workers)
    except Exception as e:
        logging.error("An error occurred: %s", e)
        sys.exit(1)
