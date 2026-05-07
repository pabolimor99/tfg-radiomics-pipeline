from pathlib import Path
import sys
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
from radiomics import featureextractor
import SimpleITK as sitk
import numpy as np
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger("radiomics").setLevel(logging.WARNING)


class _SuppressPyRadiomicsNoisyWarnings(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if "GLCM is symmetrical, therefore Sum Average = 2 * Joint Average" in msg:
            return False
        return True


logging.getLogger("radiomics").addFilter(_SuppressPyRadiomicsNoisyWarnings())

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

INPUT_CSV = PROJECT_ROOT / "outputs" / "pairs" / "pairs.csv"
OUTPUT_CSV = PROJECT_ROOT / "outputs" / "features" / "radiomics_by_label.csv"
ERRORS_CSV = PROJECT_ROOT / "outputs" / "logs" / "radiomics_by_label_errors.csv"
METADATA_ROOT = PROJECT_ROOT / "data" / "raw"

INCLUDE_DIAGNOSTICS = False
PYRADIOMICS_PARAMS = Path(__file__).resolve().parent / "pyradiomics_ct.yaml"

LABEL_SET_POLICY = "baseline"
MAX_COG_DISTANCE = None

DEFAULT_WORKERS = max(1, min(16, (os.cpu_count() or 4) - 1))

_WORKER_EXTRACTOR = None
_WORKER_METADATA_CACHE: dict[str, pd.DataFrame | None] | None = None



def resolve_path(path_str: str) -> Path | None:
    if pd.isna(path_str) or str(path_str).strip() == "":
        return None

    path = Path(path_str)
    if path.is_absolute():
        return path

    return (PROJECT_ROOT / path).resolve()


def load_patient_metadata(patient_id: str) -> pd.DataFrame | None:
    meta_path = (METADATA_ROOT / str(patient_id) / "metadata" / f"{patient_id}.csv").resolve()
    if not meta_path.exists():
        return None
    try:
        return pd.read_csv(meta_path)
    except Exception:
        logging.exception("No se pudo leer el CSV de metadatos: %s", meta_path)
        return None


def _to_int_or_none(v) -> int | None:
    if v is None or pd.isna(v):
        return None
    try:
        return int(v)
    except Exception:
        try:
            return int(str(v).strip())
        except Exception:
            return None


def _parse_xyz(s) -> tuple[float, float, float] | None:
    if s is None or pd.isna(s):
        return None
    txt = str(s).strip()
    if txt == "" or txt.lower() in {"none", "nan", "null"}:
        return None
    parts = [p for p in txt.replace(",", " ").split() if p.strip() != ""]
    if len(parts) < 3:
        return None
    try:
        x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
        return x, y, z
    except Exception:
        return None


def _metadata_candidates_with_cog(
    metadata: pd.DataFrame | None,
    timepoint: str,
    image_id: int | None,
) -> pd.DataFrame:
    if metadata is None or image_id is None:
        return pd.DataFrame()

    tp = str(timepoint).strip().lower()
    if tp == "baseline":
        img_col = "img_id_bl"
        cog_col = "cog_bl"
    elif tp == "followup":
        img_col = "img_id_fu"
        cog_col = "cog_fu"
    else:
        return pd.DataFrame()

    needed = {"lesion_id", img_col, cog_col}
    if not needed.issubset(set(metadata.columns)):
        return pd.DataFrame()

    m = metadata.copy()
    m["_lesion_id_int"] = pd.to_numeric(m["lesion_id"], errors="coerce")
    m["_img_id_int"] = pd.to_numeric(m[img_col], errors="coerce")
    m = m.dropna(subset=["_lesion_id_int", "_img_id_int"])
    m["_lesion_id_int"] = m["_lesion_id_int"].astype(int)
    m["_img_id_int"] = m["_img_id_int"].astype(int)
    m = m[m["_img_id_int"] == int(image_id)]

    m["_cog_xyz"] = m[cog_col].map(_parse_xyz)
    m = m.dropna(subset=["_cog_xyz"])

    m["_cog_x"] = m["_cog_xyz"].map(lambda t: float(t[0]))
    m["_cog_y"] = m["_cog_xyz"].map(lambda t: float(t[1]))
    m["_cog_z"] = m["_cog_xyz"].map(lambda t: float(t[2]))

    keep_cols = ["_lesion_id_int", "_cog_x", "_cog_y", "_cog_z"]
    if "lesion_type" in m.columns:
        keep_cols.append("lesion_type")
    return m[keep_cols].copy()


def match_metadata_by_spatial_proximity(
    metadata: pd.DataFrame | None,
    timepoint: str,
    image_id: int | None,
    centroid_xyz: tuple[float, float, float] | None,
    max_distance: float | None = MAX_COG_DISTANCE,
) -> tuple[int | None, str | None, float | None]:
    if centroid_xyz is None:
        return None, None, None

    candidates = _metadata_candidates_with_cog(metadata, timepoint, image_id)
    if candidates.empty:
        return None, None, None

    cx, cy, cz = centroid_xyz
    dx = candidates["_cog_x"].to_numpy(dtype=float) - float(cx)
    dy = candidates["_cog_y"].to_numpy(dtype=float) - float(cy)
    dz = candidates["_cog_z"].to_numpy(dtype=float) - float(cz)
    dist = np.sqrt(dx * dx + dy * dy + dz * dz)

    best_idx = int(dist.argmin())
    best_dist = float(dist[best_idx])

    if max_distance is not None and best_dist > float(max_distance):
        return None, None, None

    best_row = candidates.iloc[best_idx]
    matched_lesion_id = int(best_row["_lesion_id_int"])
    matched_type = None
    if "lesion_type" in best_row.index and not pd.isna(best_row["lesion_type"]):
        t = str(best_row["lesion_type"]).strip()
        matched_type = t if t != "" else None

    return matched_lesion_id, matched_type, best_dist


def _label_geometries(mask: sitk.Image | None) -> dict[int, dict]:
    if mask is None:
        return {}

    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(mask)

    out: dict[int, dict] = {}
    for label in stats.GetLabels():
        lab = int(label)
        if lab == 0:
            continue

        centroid_phys = stats.GetCentroid(lab)
        centroid_idx = mask.TransformPhysicalPointToContinuousIndex(centroid_phys)
        bbox = stats.GetBoundingBox(lab)

        out[lab] = {
            "centroid_x": float(centroid_idx[0]),
            "centroid_y": float(centroid_idx[1]),
            "centroid_z": float(centroid_idx[2]),
            "volume_vox": int(stats.GetNumberOfPixels(lab)),
            "volume_mm3": float(stats.GetPhysicalSize(lab)),
            "bbox_x": int(bbox[0]),
            "bbox_y": int(bbox[1]),
            "bbox_z": int(bbox[2]),
            "bbox_sx": int(bbox[3]),
            "bbox_sy": int(bbox[4]),
            "bbox_sz": int(bbox[5]),
        }

    return out


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


def _init_worker(params_path: str, include_diagnostics: bool) -> None:
    global _WORKER_EXTRACTOR, _WORKER_METADATA_CACHE, INCLUDE_DIAGNOSTICS, PYRADIOMICS_PARAMS
    INCLUDE_DIAGNOSTICS = bool(include_diagnostics)
    PYRADIOMICS_PARAMS = Path(params_path)
    _WORKER_EXTRACTOR = build_extractor()
    _WORKER_METADATA_CACHE = {}


def _extract_for_pair_row(
    row: dict,
    extractor,
    metadata_cache: dict[str, pd.DataFrame | None],
) -> tuple[list[dict], list[dict]]:
    uid = row["uid"]
    patient_id = row["patient_id"]
    pair_index = row["pair_index"]
    patient_id_str = str(patient_id)

    baseline_image_id = _to_int_or_none(row.get("baseline_pair_index"))
    followup_image_id = _to_int_or_none(row.get("followup_pair_index"))
    if baseline_image_id is None:
        baseline_image_id = _to_int_or_none(pair_index)
    if followup_image_id is None:
        followup_image_id = _to_int_or_none(pair_index)

    if patient_id_str not in metadata_cache:
        metadata_cache[patient_id_str] = load_patient_metadata(patient_id_str)
    patient_metadata = metadata_cache[patient_id_str]

    baseline_image_path = resolve_path(row["baseline_image_path"])
    baseline_mask_path = resolve_path(row["baseline_mask_path"])
    followup_image_path = resolve_path(row["followup_image_path"])
    followup_mask_path = resolve_path(row["followup_mask_path"])

    out_rows: list[dict] = []
    out_errors: list[dict] = []

    if baseline_image_path is None or not baseline_image_path.exists():
        out_errors.append(
            {
                "uid": uid,
                "patient_id": patient_id,
                "pair_index": pair_index,
                "timepoint": "baseline",
                "lesion_id": None,
                "error": f"Imagen no encontrada: {baseline_image_path}",
            }
        )
        return out_rows, out_errors

    if baseline_mask_path is None or not baseline_mask_path.exists():
        out_errors.append(
            {
                "uid": uid,
                "patient_id": patient_id,
                "pair_index": pair_index,
                "timepoint": "baseline",
                "lesion_id": None,
                "error": f"Máscara no encontrada: {baseline_mask_path}",
            }
        )
        return out_rows, out_errors

    bl_mask = read_mask(baseline_mask_path)
    fu_mask = read_mask(followup_mask_path)

    bl_geom = _label_geometries(bl_mask)
    fu_geom = _label_geometries(fu_mask)

    labels = select_labels(bl_mask, fu_mask)
    if not labels:
        out_errors.append(
            {
                "uid": uid,
                "patient_id": patient_id,
                "pair_index": pair_index,
                "timepoint": "both",
                "lesion_id": None,
                "error": "No se encontraron labels (no-cero) según LABEL_SET_POLICY",
            }
        )
        return out_rows, out_errors

    bl_labels = mask_labels(bl_mask) if bl_mask is not None else set()
    fu_labels = mask_labels(fu_mask) if fu_mask is not None else set()

    for lesion_id in labels:
        baseline_matched_lesion_id = None
        baseline_match_distance = None
        baseline_match_method = None
        baseline_lesion_type = None

        followup_matched_lesion_id = None
        followup_match_distance = None
        followup_match_method = None
        followup_lesion_type = None

        if int(lesion_id) in bl_geom:
            g = bl_geom[int(lesion_id)]
            centroid = (g["centroid_x"], g["centroid_y"], g["centroid_z"])
            mlid, mtype, mdist = match_metadata_by_spatial_proximity(
                metadata=patient_metadata,
                timepoint="baseline",
                image_id=baseline_image_id,
                centroid_xyz=centroid,
                max_distance=MAX_COG_DISTANCE,
            )
            if mlid is not None:
                baseline_matched_lesion_id = int(mlid)
                baseline_lesion_type = mtype
                baseline_match_method = "spatial_cog"
                baseline_match_distance = mdist

        if int(lesion_id) in fu_geom:
            g = fu_geom[int(lesion_id)]
            centroid = (g["centroid_x"], g["centroid_y"], g["centroid_z"])
            mlid, mtype, mdist = match_metadata_by_spatial_proximity(
                metadata=patient_metadata,
                timepoint="followup",
                image_id=followup_image_id,
                centroid_xyz=centroid,
                max_distance=MAX_COG_DISTANCE,
            )
            if mlid is not None:
                followup_matched_lesion_id = int(mlid)
                followup_lesion_type = mtype
                followup_match_method = "spatial_cog"
                followup_match_distance = mdist

        matched_lesion_type = baseline_lesion_type or followup_lesion_type
        matched_lesion_id = baseline_matched_lesion_id or followup_matched_lesion_id
        match_method = baseline_match_method or followup_match_method
        match_distance = baseline_match_distance if baseline_match_distance is not None else followup_match_distance

        row_content = {
            "uid": uid,
            "patient_id": patient_id,
            "pair_index": pair_index,
            "lesion_id": int(lesion_id),
            "lesion_type": matched_lesion_type,
            "baseline_lesion_type": baseline_lesion_type,
            "followup_lesion_type": followup_lesion_type,
            "matched_lesion_id": matched_lesion_id,
            "match_method": match_method,
            "match_distance": match_distance,
            "baseline_matched_lesion_id": baseline_matched_lesion_id,
            "baseline_match_method": baseline_match_method,
            "baseline_match_distance": baseline_match_distance,
            "followup_matched_lesion_id": followup_matched_lesion_id,
            "followup_match_method": followup_match_method,
            "followup_match_distance": followup_match_distance,
            "baseline_image_id": baseline_image_id,
            "followup_image_id": followup_image_id,
            "baseline_image_path": baseline_image_path,
            "baseline_mask_path": baseline_mask_path,
            "followup_image_path": followup_image_path,
            "followup_mask_path": followup_mask_path,
            "baseline_present": int(lesion_id in bl_labels),
            "followup_present": int(lesion_id in fu_labels),
        }

        if int(lesion_id) in bl_geom:
            row_content.update({f"baseline_{k}": v for k, v in bl_geom[int(lesion_id)].items()})
        if int(lesion_id) in fu_geom:
            row_content.update({f"followup_{k}": v for k, v in fu_geom[int(lesion_id)].items()})

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
            out_rows.append(output_row)

        except Exception as e:
            out_errors.append({**row_content, "error": str(e)})
            continue

    return out_rows, out_errors


def _process_pair_row_worker(row: dict) -> tuple[list[dict], list[dict]]:
    global _WORKER_EXTRACTOR, _WORKER_METADATA_CACHE
    if _WORKER_EXTRACTOR is None:
        _WORKER_EXTRACTOR = build_extractor()
    if _WORKER_METADATA_CACHE is None:
        _WORKER_METADATA_CACHE = {}
    return _extract_for_pair_row(row=row, extractor=_WORKER_EXTRACTOR, metadata_cache=_WORKER_METADATA_CACHE)


def extract_features_by_label(
    input_csv: Path = INPUT_CSV,
    output_csv: Path = OUTPUT_CSV,
    errors_csv: Path = ERRORS_CSV,
    workers: int = DEFAULT_WORKERS,
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

    workers = int(workers) if workers is not None else 0
    if workers > 1:
        rows: list[dict] = []
        errors: list[dict] = []

        records = df.to_dict(orient="records")
        logging.info("Extrayendo radiomics por label para %d pares (parallel, workers=%d)", len(records), workers)

        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(str(PYRADIOMICS_PARAMS), bool(INCLUDE_DIAGNOSTICS)),
        ) as ex:
            futures = [ex.submit(_process_pair_row_worker, r) for r in records]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Extrayendo features por label (parallel)"):
                pair_rows, pair_errors = fut.result()
                rows.extend(pair_rows)
                errors.extend(pair_errors)

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

        return

    extractor = build_extractor()
    rows: list[dict] = []
    errors: list[dict] = []
    metadata_cache: dict[str, pd.DataFrame | None] = {}

    total = len(df)
    logging.info("Extrayendo radiomics por label para %d pares", total)

    for _, row in tqdm(df.iterrows(), total=total, desc="Extrayendo features por label"):
        uid = row["uid"]
        patient_id = row["patient_id"]
        pair_index = row["pair_index"]
        patient_id_str = str(patient_id)

        baseline_image_id = _to_int_or_none(row.get("baseline_pair_index"))
        followup_image_id = _to_int_or_none(row.get("followup_pair_index"))
        if baseline_image_id is None:
            baseline_image_id = _to_int_or_none(pair_index)
        if followup_image_id is None:
            followup_image_id = _to_int_or_none(pair_index)

        if patient_id_str not in metadata_cache:
            metadata_cache[patient_id_str] = load_patient_metadata(patient_id_str)
        patient_metadata = metadata_cache[patient_id_str]

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

        bl_geom = _label_geometries(bl_mask)
        fu_geom = _label_geometries(fu_mask)

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
            baseline_matched_lesion_id = None
            baseline_match_distance = None
            baseline_match_method = None
            baseline_lesion_type = None

            followup_matched_lesion_id = None
            followup_match_distance = None
            followup_match_method = None
            followup_lesion_type = None

            if int(lesion_id) in bl_geom:
                g = bl_geom[int(lesion_id)]
                centroid = (g["centroid_x"], g["centroid_y"], g["centroid_z"])
                mlid, mtype, mdist = match_metadata_by_spatial_proximity(
                    metadata=patient_metadata,
                    timepoint="baseline",
                    image_id=baseline_image_id,
                    centroid_xyz=centroid,
                    max_distance=MAX_COG_DISTANCE,
                )
                if mlid is not None:
                    baseline_matched_lesion_id = int(mlid)
                    baseline_lesion_type = mtype
                    baseline_match_method = "spatial_cog"
                    baseline_match_distance = mdist

            if int(lesion_id) in fu_geom:
                g = fu_geom[int(lesion_id)]
                centroid = (g["centroid_x"], g["centroid_y"], g["centroid_z"])
                mlid, mtype, mdist = match_metadata_by_spatial_proximity(
                    metadata=patient_metadata,
                    timepoint="followup",
                    image_id=followup_image_id,
                    centroid_xyz=centroid,
                    max_distance=MAX_COG_DISTANCE,
                )
                if mlid is not None:
                    followup_matched_lesion_id = int(mlid)
                    followup_lesion_type = mtype
                    followup_match_method = "spatial_cog"
                    followup_match_distance = mdist

            matched_lesion_type = baseline_lesion_type or followup_lesion_type
            matched_lesion_id = baseline_matched_lesion_id or followup_matched_lesion_id
            match_method = baseline_match_method or followup_match_method
            match_distance = baseline_match_distance if baseline_match_distance is not None else followup_match_distance

            row_content = {
                "uid": uid,
                "patient_id": patient_id,
                "pair_index": pair_index,
                "lesion_id": int(lesion_id),
                "lesion_type": matched_lesion_type,
                "baseline_lesion_type": baseline_lesion_type,
                "followup_lesion_type": followup_lesion_type,
                "matched_lesion_id": matched_lesion_id,
                "match_method": match_method,
                "match_distance": match_distance,
                "baseline_matched_lesion_id": baseline_matched_lesion_id,
                "baseline_match_method": baseline_match_method,
                "baseline_match_distance": baseline_match_distance,
                "followup_matched_lesion_id": followup_matched_lesion_id,
                "followup_match_method": followup_match_method,
                "followup_match_distance": followup_match_distance,
                "baseline_image_id": baseline_image_id,
                "followup_image_id": followup_image_id,
                "baseline_image_path": baseline_image_path,
                "baseline_mask_path": baseline_mask_path,
                "followup_image_path": followup_image_path,
                "followup_mask_path": followup_mask_path,
                "baseline_present": int(lesion_id in bl_labels),
                "followup_present": int(lesion_id in fu_labels),
            }

            if int(lesion_id) in bl_geom:
                row_content.update({f"baseline_{k}": v for k, v in bl_geom[int(lesion_id)].items()})
            if int(lesion_id) in fu_geom:
                row_content.update({f"followup_{k}": v for k, v in fu_geom[int(lesion_id)].items()})

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
        workers = int(os.environ.get("RADIOMICS_WORKERS", str(DEFAULT_WORKERS)))
        extract_features_by_label(INPUT_CSV, OUTPUT_CSV, ERRORS_CSV, workers=workers)
    except Exception as e:
        logging.error("An error occurred: %s", e)
        sys.exit(1)
