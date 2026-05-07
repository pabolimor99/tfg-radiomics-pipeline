from pathlib import Path
import shutil
from collections import defaultdict
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def extract_patient_id(filename: str) -> str:
    return filename[:10]

def group_by_patient_id(base_dir: Path) -> dict[str, list[Path]]:
    grouped_files = defaultdict(list)

    for file_path in base_dir.iterdir():
        if not file_path.is_file():
            continue
        patient_id = extract_patient_id(file_path.name)
        grouped_files[patient_id].append(file_path)

    logging.info("Se han detectado %d pacientes", len(grouped_files))
    return grouped_files


def reorganize_dataset(base_dir: Path, raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    grouped_files = group_by_patient_id(base_dir)

    total_files = sum(len(files) for files in grouped_files.values())
    copied_files = 0

    logging.info("Se van a copiar %d archivos", total_files)

    for patient_id, files in tqdm(
    grouped_files.items(),
    total=len(grouped_files),
    desc="Reorganizando pacientes"):
        patient_dir = raw_dir / patient_id
        patient_dir.mkdir(parents=True, exist_ok=True)

        for file_path in files:
            dst = patient_dir / file_path.name
            shutil.copy2(file_path, dst)
            copied_files += 1

    logging.info("Reorganización completada: %d archivos copiados", copied_files)

if __name__ == "__main__":
    inputs_dir = Path("data/test/inputsTr")
    targets_dir = Path("data/test/targetsTr")
    raw_dir = Path("data/raw")

    reorganize_dataset(inputs_dir, raw_dir)
    reorganize_dataset(targets_dir, raw_dir)