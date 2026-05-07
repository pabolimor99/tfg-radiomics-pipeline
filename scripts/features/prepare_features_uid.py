from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
import os
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.radiomics.extract_features import (
    INCLUDE_DIAGNOSTICS,
    PYRADIOMICS_PARAMS,
    build_extractor,
    clean_result,
    resolve_path,
)
import SimpleITK as sitk
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

INPUT_CSV = Path("outputs/pairs/pairs.csv")
OUTPUT_CSV = Path("outputs/features/features_uid.csv")
ERRORS_CSV = Path("outputs/logs/features_uid_errors.csv")
DEFAULT_WORKERS = max(1, min(16, (os.cpu_count() or 4) - 1))

_WORKER_EXTRACTOR = None


def build_binary_mask(mask_path: Path) -> sitk.Image:
    mask = sitk.ReadImage(str(mask_path))
    mask_array = sitk.GetArrayFromImage(mask)
    binary_array = (mask_array > 0).astype("uint8")
    binary_mask = sitk.GetImageFromArray(binary_array)
    binary_mask.CopyInformation(mask)
    return binary_mask


def mask_is_empty(mask_path: Path) -> bool:
    mask = sitk.ReadImage(str(mask_path))
    mask_array = sitk.GetArrayViewFromImage(mask)
    return not bool((mask_array > 0).any())


def extract_single_timepoint_features(
    extractor,
    image_path: Path | None,
    mask_path: Path | None,
    prefix: str,
) -> dict:
    if image_path is None or not image_path.exists():
        raise FileNotFoundError(f"Imagen no encontrada: {image_path}")
    if mask_path is None or not mask_path.exists():
        raise FileNotFoundError(f"Mascara no encontrada: {mask_path}")
    if mask_is_empty(mask_path):
        return {}

    binary_mask = build_binary_mask(mask_path)
    result = extractor.execute(str(image_path), binary_mask)
    result = clean_result(result)
    return {f"{prefix}_{key}": value for key, value in result.items() if str(key).startswith("original_")}


def _init_worker(params_path: str, include_diagnostics: bool) -> None:
    global _WORKER_EXTRACTOR
    from src.radiomics import extract_features as extract_features_module

    extract_features_module.INCLUDE_DIAGNOSTICS = bool(include_diagnostics)
    extract_features_module.PYRADIOMICS_PARAMS = Path(params_path)
    _WORKER_EXTRACTOR = build_extractor()


def _extract_for_row(row: dict, extractor) -> tuple[dict | None, dict | None]:
    uid = str(row["uid"])
    row_content = {"uid": uid}

    baseline_image_path = resolve_path(row.get("baseline_image_path"))
    baseline_mask_path = resolve_path(row.get("baseline_mask_path"))
    followup_image_path = resolve_path(row.get("followup_image_path"))
    followup_mask_path = resolve_path(row.get("followup_mask_path"))

    try:
        baseline_features = extract_single_timepoint_features(
            extractor=extractor,
            image_path=baseline_image_path,
            mask_path=baseline_mask_path,
            prefix="baseline",
        )
        followup_features = extract_single_timepoint_features(
            extractor=extractor,
            image_path=followup_image_path,
            mask_path=followup_mask_path,
            prefix="followup",
        )

        output_row = dict(row_content)
        output_row.update(baseline_features)
        output_row.update(followup_features)
        return output_row, None
    except Exception as exc:
        error_row = dict(row_content)
        error_row["error"] = str(exc)
        return None, error_row


def _process_row_worker(row: dict) -> tuple[dict | None, dict | None]:
    global _WORKER_EXTRACTOR
    if _WORKER_EXTRACTOR is None:
        _WORKER_EXTRACTOR = build_extractor()
    return _extract_for_row(row=row, extractor=_WORKER_EXTRACTOR)


def build_features_uid(
    input_csv: Path = INPUT_CSV,
    output_csv: Path = OUTPUT_CSV,
    errors_csv: Path = ERRORS_CSV,
    workers: int = DEFAULT_WORKERS,
) -> Path:
    df = pd.read_csv(input_csv)

    required = {
        "uid",
        "baseline_image_path",
        "baseline_mask_path",
        "followup_image_path",
        "followup_mask_path",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas obligatorias en {input_csv}: {sorted(missing)}")

    rows: list[dict] = []
    errors: list[dict] = []

    total = len(df)
    logging.info("Preparando features longitudinales por par para %d filas", total)

    workers = int(workers) if workers is not None else 0
    if workers <= 1:
        extractor = build_extractor()
        for _, row in tqdm(df.iterrows(), total=total, desc="Extrayendo features por par"):
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
            futures = [ex.submit(_process_row_worker, record) for record in records]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Extrayendo features por par (parallel)"):
                out_row, err_row = fut.result()
                if out_row is not None:
                    rows.append(out_row)
                if err_row is not None:
                    errors.append(err_row)

    if not rows:
        raise ValueError("No se extrajeron features por par.")

    out = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    if errors:
        err_df = pd.DataFrame(errors)
        errors_csv.parent.mkdir(parents=True, exist_ok=True)
        err_df.to_csv(errors_csv, index=False)
        logging.info("Errores guardados en %s", errors_csv)

    print(f"[prepare_features_uid] input_rows={len(df)}")
    print(f"[prepare_features_uid] output_rows={len(out)} output_cols={out.shape[1]}")
    print(f"[prepare_features_uid] wrote={output_csv}")
    return output_csv


def main() -> None:
    workers = int(os.environ.get("PAIRWISE_FEATURE_WORKERS", str(DEFAULT_WORKERS)))
    build_features_uid(INPUT_CSV, OUTPUT_CSV, ERRORS_CSV, workers=workers)


if __name__ == "__main__":
    main()
