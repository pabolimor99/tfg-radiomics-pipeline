from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

from src.data.ingest.grouper import reorganize_dataset
from src.data.ingest.organizer import organize_files

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DEFAULT_INPUTS_DIR = Path("data/base/inputsTr")
DEFAULT_TARGETS_DIR = Path("data/base/targetsTr")
DEFAULT_RAW_DIR = Path("data/raw")


def prepare_raw(
    *,
    inputs_dir: Path = DEFAULT_INPUTS_DIR,
    targets_dir: Path = DEFAULT_TARGETS_DIR,
    raw_dir: Path = DEFAULT_RAW_DIR,
    clean: bool = False,
) -> Path:
    if not inputs_dir.is_dir():
        raise FileNotFoundError(f"No existe: {inputs_dir}")
    if not targets_dir.is_dir():
        raise FileNotFoundError(f"No existe: {targets_dir}")

    if clean and raw_dir.exists():
        logging.info("Eliminando raw_dir existente: %s", raw_dir)
        shutil.rmtree(raw_dir)

    reorganize_dataset(inputs_dir, raw_dir)
    reorganize_dataset(targets_dir, raw_dir)
    organize_files(raw_dir)

    logging.info("data/raw preparado en %s", raw_dir)
    return raw_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reconstruye data/raw desde data/base/inputsTr y data/base/targetsTr."
    )
    parser.add_argument("--inputs-dir", type=Path, default=DEFAULT_INPUTS_DIR)
    parser.add_argument("--targets-dir", type=Path, default=DEFAULT_TARGETS_DIR)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Borra raw-dir antes de reconstruirlo.",
    )
    args = parser.parse_args()

    prepare_raw(
        inputs_dir=args.inputs_dir,
        targets_dir=args.targets_dir,
        raw_dir=args.raw_dir,
        clean=args.clean,
    )


if __name__ == "__main__":
    main()
