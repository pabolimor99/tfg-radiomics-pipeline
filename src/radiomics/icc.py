import gc
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pingouin as pg
import SimpleITK as sitk
from radiomics import featureextractor
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger("radiomics").setLevel(logging.ERROR)
logging.getLogger("pykwalify").setLevel(logging.ERROR)
logging.getLogger("pykwalify.core").setLevel(logging.ERROR)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

INPUT_CSV = PROJECT_ROOT / "outputs" / "pairs" / "longitudinal_pairs.csv"
OUTPUT_CSV = PROJECT_ROOT / "outputs" / "features" / "icc_results.csv"
FEATURES_CACHE_CSV = PROJECT_ROOT / "outputs" / "features" / "icc_perturbed_features.csv"
FEATURES_PROGRESS_CSV = PROJECT_ROOT / "outputs" / "features" / "icc_perturbed_features.partial.csv"
PYRADIOMICS_PARAMS = Path(__file__).resolve().parent / "pyradiomics_ct.yaml"

ICC_THRESHOLD = 0.70
DEFAULT_WORKERS = max(1, min(4, (os.cpu_count() or 4) - 1))
ICC_SUBSET_SIZE = int(os.environ.get("ICC_SUBSET_SIZE", "150"))
MAX_PENDING_FACTOR = 2
CHECKPOINT_EVERY = int(os.environ.get("ICC_CHECKPOINT_EVERY", "25"))

_WORKER_EXTRACTORS = None


def is_allowed_feature(key: str) -> bool:
    key = str(key)
    return not key.startswith("diagnostics_")


def resolve_path(path_str):
    if pd.isna(path_str) or str(path_str).strip() == "":
        return None

    path = Path(path_str)
    if path.is_absolute():
        return path

    return (PROJECT_ROOT / path).resolve()


def binary_mask(mask_path):
    mask = sitk.ReadImage(str(mask_path))
    mask_array = sitk.GetArrayFromImage(mask)
    binary_array = (mask_array > 0).astype(np.uint8)
    out = sitk.GetImageFromArray(binary_array)
    out.CopyInformation(mask)
    return out


def mask_is_empty(mask_img):
    arr = sitk.GetArrayViewFromImage(mask_img)
    return not bool(np.any(arr > 0))


def physical_center(img):
    size = np.array(list(img.GetSize()), dtype=float)
    center_index = (size - 1.0) / 2.0
    return img.TransformContinuousIndexToPhysicalPoint(tuple(center_index.tolist()))


def resample_mask(mask, reference_mask, transform=None):
    transform = transform if transform is not None else sitk.Transform()
    return sitk.Resample(mask, reference_mask, transform, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)


def rotate_mask(mask_img, degrees):
    angle = float(np.deg2rad(float(degrees)))
    tx = sitk.Euler3DTransform()
    tx.SetCenter(physical_center(mask_img))
    tx.SetRotation(0.0, 0.0, angle)
    return resample_mask(mask_img, mask_img, tx)


def translate_mask_vox(mask_img, shift_vox_xyz):
    sx, sy, sz = mask_img.GetSpacing()
    dx, dy, dz = shift_vox_xyz
    tx = sitk.TranslationTransform(3)
    tx.SetOffset((float(dx) * float(sx), float(dy) * float(sy), float(dz) * float(sz)))
    return resample_mask(mask_img, mask_img, tx)


def erode_mask(mask_img, radius=1):
    r = int(radius)
    return sitk.BinaryErode(mask_img, [r, r, r])


def dilate_mask(mask_img, radius=1):
    r = int(radius)
    return sitk.BinaryDilate(mask_img, [r, r, r])


def load_data(input_csv):
    return pd.read_csv(input_csv)


def build_extractor():
    if Path(PYRADIOMICS_PARAMS).is_file():
        extractor = featureextractor.RadiomicsFeatureExtractor(str(PYRADIOMICS_PARAMS))
    else:
        extractor = featureextractor.RadiomicsFeatureExtractor()
    return extractor


def build_extractors():
    return (build_extractor(),)


def extract_baseline_subset(data):
    rows = data[(data["timepoint"] == "baseline") & (data["is_radiomics_valid"] == True)].copy()
    rows["patient_id"] = rows["patient_id"].astype(str)
    rows["pair_index"] = pd.to_numeric(rows["pair_index"], errors="raise").astype(int)
    rows["timepoint"] = rows["timepoint"].astype(str)
    rows = rows.sort_values(by=["patient_id", "pair_index", "uid"]).reset_index(drop=True)
    rows = rows.head(int(ICC_SUBSET_SIZE)).reset_index(drop=True)
    logging.info(
        "Usando subset para ICC: %d sujetos (limite=%d)",
        int(len(rows)),
        int(ICC_SUBSET_SIZE),
    )
    return rows


def generate_permutations(subset):
    result = []

    rotations_deg = [-3.0, 3.0]
    shifts_vox = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0)]

    for _, row in tqdm(subset.iterrows(), total=len(subset), desc="Generando perturbaciones"):
        uid = row.get("uid")
        image_path = resolve_path(row.get("image_path"))
        mask_path = resolve_path(row.get("mask_path"))
        if image_path is None or mask_path is None:
            continue
        if not image_path.is_file() or not mask_path.is_file():
            continue

        image_path_s = str(image_path.resolve())
        mask_path_s = str(mask_path.resolve())

        result.append({"uid": uid, "variant": "orig", "image_path": image_path_s, "mask_path": mask_path_s, "op": "orig"})

        for deg in rotations_deg:
            result.append(
                {
                    "uid": uid,
                    "variant": f"rot_{deg:+.0f}",
                    "image_path": image_path_s,
                    "mask_path": mask_path_s,
                    "op": "rot",
                    "deg": float(deg),
                }
            )

        for dx, dy, dz in shifts_vox:
            result.append(
                {
                    "uid": uid,
                    "variant": f"tx_{dx}_{dy}_{dz}",
                    "image_path": image_path_s,
                    "mask_path": mask_path_s,
                    "op": "shift_vox",
                    "dx": int(dx),
                    "dy": int(dy),
                    "dz": int(dz),
                }
            )

        result.append({"uid": uid, "variant": "erode_r1", "image_path": image_path_s, "mask_path": mask_path_s, "op": "erode", "radius": 1})
        result.append({"uid": uid, "variant": "dilate_r1", "image_path": image_path_s, "mask_path": mask_path_s, "op": "dilate", "radius": 1})

    return result


def _init_worker(params_path: str) -> None:
    global _WORKER_EXTRACTORS, PYRADIOMICS_PARAMS
    PYRADIOMICS_PARAMS = Path(params_path)
    _WORKER_EXTRACTORS = build_extractors()


def _extract_one(recipe: dict):
    global _WORKER_EXTRACTORS
    if _WORKER_EXTRACTORS is None:
        _init_worker(str(PYRADIOMICS_PARAMS))

    uid = recipe["uid"]
    variant = recipe["variant"]
    image_path = Path(recipe["image_path"])
    mask_path = Path(recipe["mask_path"])

    mask0 = binary_mask(mask_path)
    if mask_is_empty(mask0):
        return None

    op = recipe.get("op", "orig")
    if op == "orig":
        mask_img = mask0
    elif op == "rot":
        mask_img = rotate_mask(mask0, float(recipe["deg"]))
    elif op == "shift_vox":
        mask_img = translate_mask_vox(mask0, (int(recipe["dx"]), int(recipe["dy"]), int(recipe["dz"])))
    elif op == "erode":
        mask_img = erode_mask(mask0, radius=int(recipe.get("radius", 1)))
    elif op == "dilate":
        mask_img = dilate_mask(mask0, radius=int(recipe.get("radius", 1)))
    else:
        raise ValueError(f"Operacion desconocida: {op!r}")

    if mask_is_empty(mask_img):
        return None

    if int(mask_img.GetDimension()) < 2:
        return None
    if any(int(s) < 2 for s in mask_img.GetSize()):
        return None

    image_img = sitk.ReadImage(str(image_path))
    features = {}
    for extractor in _WORKER_EXTRACTORS:
        features.update(extractor.execute(image_img, mask_img))
    clean = {k: v for k, v in features.items() if is_allowed_feature(k)}
    del image_img, mask0, mask_img, features
    gc.collect()
    return {"uid": uid, "variant": variant, **clean}


def recipe_key(recipe: dict) -> tuple[str, str]:
    return (str(recipe["uid"]), str(recipe["variant"]))


def load_partial_checkpoint(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame(columns=["uid", "variant"])

    partial = pd.read_csv(path)
    if partial.empty:
        return pd.DataFrame(columns=["uid", "variant"])
    if not {"uid", "variant"}.issubset(set(partial.columns)):
        logging.warning("Checkpoint parcial invalido, no contiene uid/variant: %s", path)
        return pd.DataFrame(columns=["uid", "variant"])
    partial["uid"] = partial["uid"].astype(str)
    partial["variant"] = partial["variant"].astype(str)
    return partial


def persist_partial_checkpoint(path: Path, rows: list[dict], *, base_df: pd.DataFrame | None = None) -> pd.DataFrame:
    current_df = pd.DataFrame(rows)
    if base_df is not None and not base_df.empty:
        current_df = pd.concat([base_df, current_df], ignore_index=True)
    if not current_df.empty:
        current_df = current_df.drop_duplicates(subset=["uid", "variant"], keep="last").reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    current_df.to_csv(path, index=False)
    return current_df


def extract_features_for_permutations(permutations, progress_csv: Path | None = None):

    workers = int(os.environ.get("ICC_WORKERS", str(DEFAULT_WORKERS)))
    checkpoint_every = max(1, int(os.environ.get("ICC_CHECKPOINT_EVERY", str(CHECKPOINT_EVERY))))
    progress_path = progress_csv if progress_csv is not None else FEATURES_PROGRESS_CSV
    partial_df = load_partial_checkpoint(progress_path)
    completed_keys = set()
    if not partial_df.empty:
        completed_keys = set(zip(partial_df["uid"].astype(str), partial_df["variant"].astype(str)))
        logging.info("Reanudando desde checkpoint: %d extracciones ya completadas", len(completed_keys))

    pending_permutations = [r for r in permutations if recipe_key(r) not in completed_keys]
    if not pending_permutations:
        return partial_df

    if workers <= 1:
        _init_worker(str(PYRADIOMICS_PARAMS))
        rows = partial_df.to_dict(orient="records") if not partial_df.empty else []
        since_checkpoint = 0
        for r in tqdm(pending_permutations, total=len(pending_permutations), desc="Extrayendo caracteristicas"):
            try:
                out = _extract_one(r)
            except Exception as e:
                logging.warning("Fallo en uid=%s variant=%s: %s", r.get("uid"), r.get("variant"), str(e))
                continue
            if out is not None:
                rows.append(out)
                since_checkpoint += 1
                if since_checkpoint >= checkpoint_every:
                    partial_df = persist_partial_checkpoint(progress_path, rows)
                    rows = partial_df.to_dict(orient="records")
                    since_checkpoint = 0
    else:
        rows = partial_df.to_dict(orient="records") if not partial_df.empty else []
        since_checkpoint = 0
        logging.info("Extraccion con %d workers", workers)
        max_pending = max(workers, workers * int(os.environ.get("ICC_MAX_PENDING_FACTOR", str(MAX_PENDING_FACTOR))))
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(str(PYRADIOMICS_PARAMS),),
        ) as ex:
            pending = {}
            recipe_iter = iter(pending_permutations)

            for _ in range(min(max_pending, len(pending_permutations))):
                recipe = next(recipe_iter, None)
                if recipe is None:
                    break
                pending[ex.submit(_extract_one, recipe)] = recipe

            progress = tqdm(total=len(pending_permutations), desc="Extrayendo caracteristicas", unit="perm")
            while pending:
                for fut in as_completed(tuple(pending)):
                    recipe = pending.pop(fut)
                    progress.update(1)
                    try:
                        out = fut.result()
                    except Exception as e:
                        logging.warning("Fallo en uid=%s variant=%s: %s", recipe.get("uid"), recipe.get("variant"), str(e))
                        out = None
                    if out is not None:
                        rows.append(out)
                        since_checkpoint += 1
                        if since_checkpoint >= checkpoint_every:
                            partial_df = persist_partial_checkpoint(progress_path, rows)
                            rows = partial_df.to_dict(orient="records")
                            since_checkpoint = 0

                    next_recipe = next(recipe_iter, None)
                    if next_recipe is not None:
                        pending[ex.submit(_extract_one, next_recipe)] = next_recipe
                    break
            progress.close()

    if not rows:
        raise ValueError("No se extrajeron caracteristicas.")
    final_df = persist_partial_checkpoint(progress_path, rows)
    return final_df


def compute_icc(features):
    if not isinstance(features, pd.DataFrame):
        features = pd.DataFrame(features)

    feature_cols = [c for c in features.columns if c not in {"uid", "variant"}]

    rows = []
    for col in tqdm(feature_cols, total=len(feature_cols), desc="Calculando ICC"):
        pivot = features.pivot(index="uid", columns="variant", values=col)
        pivot = pivot.apply(pd.to_numeric, errors="coerce")
        pivot = pivot.replace([np.inf, -np.inf], np.nan)
        pivot = pivot.dropna(axis=0, how="any")

        n_subjects = int(pivot.shape[0])
        n_variants = int(pivot.shape[1])
        if n_subjects < 2 or n_variants < 2:
            icc = float("nan")
        else:
            long_balanced = pivot.reset_index().melt(id_vars="uid", var_name="variant", value_name="rating")
            long_balanced["rating"] = pd.to_numeric(long_balanced["rating"], errors="coerce").astype("float64")
            long_balanced = long_balanced.dropna(subset=["rating"])
            try:
                icc_tbl = pg.intraclass_corr(data=long_balanced, targets="uid", raters="variant", ratings="rating")
                icc_row = icc_tbl.loc[icc_tbl["Type"] == "ICC(A,1)"]
                icc = float(icc_row["ICC"].iloc[0]) if not icc_row.empty else float("nan")
            except (AssertionError, ValueError):
                icc = float("nan")

        rows.append({"feature": col, "icc": icc, "n_subjects": n_subjects, "n_variants": n_variants})

    return pd.DataFrame(rows).sort_values(by="icc", ascending=False).reset_index(drop=True)


def identify_features_to_discard(icc_results):
    bad = icc_results[icc_results["icc"].isna() | (icc_results["icc"] < float(ICC_THRESHOLD))]
    return bad["feature"].astype(str).tolist()


def measure_and_discard_features_by_icc(data):
    subset = extract_baseline_subset(data)
    permutations = generate_permutations(subset)
    use_cache = str(os.environ.get("ICC_USE_CACHE", "1")).strip() != "0"
    reextract = str(os.environ.get("ICC_REEXTRACT", "0")).strip() == "1"

    if use_cache and FEATURES_CACHE_CSV.is_file() and not reextract:
        logging.info("Cargando features cache: %s", FEATURES_CACHE_CSV)
        features = pd.read_csv(FEATURES_CACHE_CSV)
    else:
        features = extract_features_for_permutations(permutations, progress_csv=FEATURES_PROGRESS_CSV)
        FEATURES_CACHE_CSV.parent.mkdir(parents=True, exist_ok=True)
        features.to_csv(FEATURES_CACHE_CSV, index=False)
        logging.info("Features guardadas en cache: %s", FEATURES_CACHE_CSV)
        if FEATURES_PROGRESS_CSV.is_file():
            FEATURES_PROGRESS_CSV.unlink()
            logging.info("Checkpoint eliminado: %s", FEATURES_PROGRESS_CSV)
    icc_results = compute_icc(features)
    features_to_discard = identify_features_to_discard(icc_results)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    icc_results.to_csv(OUTPUT_CSV, index=False)

    logging.info(
        "ICC terminado: total_features=%d keep=%d discard=%d (umbral=%.2f)",
        int(len(icc_results)),
        int((~icc_results["feature"].isin(features_to_discard)).sum()),
        int(icc_results["feature"].isin(features_to_discard).sum()),
        float(ICC_THRESHOLD),
    )

    return icc_results


def save_data(data, output_csv):
    pd.DataFrame(data).to_csv(output_csv, index=False)


if __name__ == "__main__":
    data = load_data(INPUT_CSV)
    icc_results = measure_and_discard_features_by_icc(data)
    save_data(icc_results, OUTPUT_CSV)
