from pathlib import Path
import logging
from tqdm import tqdm
import shutil

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def organize_files(raw_dir: Path)-> None:
    pacientes = 0

    for patient_dir in tqdm(raw_dir.iterdir(), total=len(list(raw_dir.iterdir())), desc="Organizando pacientes"):
        if not patient_dir.is_dir():
            continue

        (patient_dir / "baseline" / "images").mkdir(parents=True, exist_ok=True)
        (patient_dir / "baseline" / "masks").mkdir(parents=True, exist_ok=True)
        (patient_dir / "baseline" / "metadata").mkdir(parents=True, exist_ok=True)

        (patient_dir / "followup" / "images").mkdir(parents=True, exist_ok=True)
        (patient_dir / "followup" / "masks").mkdir(parents=True, exist_ok=True)
        (patient_dir / "followup" / "metadata").mkdir(parents=True, exist_ok=True)

        (patient_dir / "metadata").mkdir(parents=True, exist_ok=True)

        for file_path in patient_dir.iterdir():
            if not file_path.is_file():
                continue

            name = file_path.name

            if name.endswith(".csv"):
                dst = patient_dir / "metadata" / name

            elif "_BL_mask_" in name:
                dst = patient_dir / "baseline" / "masks" / name

            elif "_BL_img_" in name:
                dst = patient_dir / "baseline" / "images" / name

            elif "_BL_" in name and name.endswith(".json"):
                dst = patient_dir / "baseline" / "metadata" / name

            elif "_FU_mask_" in name:
                dst = patient_dir / "followup" / "masks" / name

            elif "_FU_img_" in name:
                dst = patient_dir / "followup" / "images" / name

            elif "_FU_" in name and name.endswith(".json"):
                dst = patient_dir / "followup" / "metadata" / name

            else:
                continue

            shutil.move(str(file_path), str(dst))

        pacientes += 1
    
    logging.info("Organización completada: se han organizado %d pacientes", pacientes)
            
if __name__ == '__main__':
    raw_dir = Path('data/raw')
    organize_files(raw_dir)