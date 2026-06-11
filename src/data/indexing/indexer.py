from pathlib import Path
import pandas as pd
import logging
from tqdm import tqdm


def list_files(folder: Path) -> list[str]:
    if not folder.is_dir():
        return []
    return sorted(str(f) for f in folder.iterdir() if f.is_file())


def pad_list(values: list[str], size: int) -> list[str | None]:
    return values + [None] * (size - len(values))


def create_metadata_csv(raw_dir: Path, output_csv: Path) -> None:
    rows = []

    patient_dirs = [p for p in raw_dir.iterdir() if p.is_dir()]

    max_bl_images = 0
    max_bl_masks = 0
    max_fu_images = 0
    max_fu_masks = 0

    patient_data = []

    logging.info("Indexando metadatos de pacientes")
    for patient_dir in tqdm(patient_dirs, total=len(patient_dirs), desc="Indexando"):
        patient_id = patient_dir.name

        bl_images = list_files(patient_dir / "baseline" / "images")
        bl_masks = list_files(patient_dir / "baseline" / "masks")
        fu_images = list_files(patient_dir / "followup" / "images")
        fu_masks = list_files(patient_dir / "followup" / "masks")

        max_bl_images = max(max_bl_images, len(bl_images))
        max_bl_masks = max(max_bl_masks, len(bl_masks))
        max_fu_images = max(max_fu_images, len(fu_images))
        max_fu_masks = max(max_fu_masks, len(fu_masks))

        patient_data.append({
            "patient_id": patient_id,
            "bl_images": bl_images,
            "bl_masks": bl_masks,
            "fu_images": fu_images,
            "fu_masks": fu_masks,
        })

    for patient in patient_data:
        row = {
            "patient_id": patient["patient_id"],
        }

        padded_bl_images = pad_list(patient["bl_images"], max_bl_images)
        padded_bl_masks = pad_list(patient["bl_masks"], max_bl_masks)
        padded_fu_images = pad_list(patient["fu_images"], max_fu_images)
        padded_fu_masks = pad_list(patient["fu_masks"], max_fu_masks)

        for i, path in enumerate(padded_bl_images):
            row[f"baseline_image_{i}"] = path

        for i, path in enumerate(padded_bl_masks):
            row[f"baseline_mask_{i}"] = path

        for i, path in enumerate(padded_fu_images):
            row[f"followup_image_{i}"] = path

        for i, path in enumerate(padded_fu_masks):
            row[f"followup_mask_{i}"] = path

        rows.append(row)

    df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    logging.info("Resumen del dataset:")
    logging.info(f"Max baseline images: {max_bl_images}")
    logging.info(f"Max baseline masks: {max_bl_masks}")
    logging.info(f"Max followup images: {max_fu_images}")
    logging.info(f"Max followup masks: {max_fu_masks}")

    print("\nDistribucion baseline images:")
    print(pd.Series([len(patient["bl_images"]) for patient in patient_data]).value_counts().sort_index())

    print("\nDistribucion baseline masks:")
    print(pd.Series([len(patient["bl_masks"]) for patient in patient_data]).value_counts().sort_index())

    print("\nDistribucion followup images:")
    print(pd.Series([len(patient["fu_images"]) for patient in patient_data]).value_counts().sort_index())

    print("\nDistribucion followup masks:")
    print(pd.Series([len(patient["fu_masks"]) for patient in patient_data]).value_counts().sort_index())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    raw_dir = Path("data/raw")
    output_csv = Path("data/index.csv")
    create_metadata_csv(raw_dir, output_csv)
