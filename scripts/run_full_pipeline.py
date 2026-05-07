from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reorganize import (
    LONGITUDINAL_FEATURES_LONG_CSV,
    REVIEW_WIDE_CSV,
    run_data_pipeline,
)
from prepare_delta_features import OUTPUT_CSV as DELTA_FEATURES_CSV
from prepare_delta_features import build_delta_features
from train_mlp import (
    ICC_KEEP,
    KEEP_CSV,
    OUTCOMES_CSV,
    train_model,
)
from src.radiomics.icc import INPUT_CSV as ICC_INPUT_CSV
from src.radiomics.icc import OUTPUT_CSV as ICC_RESULTS_CSV
from src.radiomics.icc import load_data, measure_and_discard_features_by_icc, save_data


def _ensure_exists(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Falta {description}: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Orquesta el pipeline completo: organizacion de datos, preprocessing, extraccion, review legacy, delta features, ICC y entrenamiento."
    )
    parser.add_argument(
        "--skip-data",
        action="store_true",
        help="No rehace la organizacion/preprocessing/extraccion; parte de los CSV ya generados.",
    )
    parser.add_argument(
        "--skip-icc",
        action="store_true",
        help="Omite el paso ICC aunque train_mlp tenga ICC_KEEP activado.",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Ejecuta el pipeline de datos pero no entrena el modelo.",
    )
    parser.add_argument(
        "--generate-review-wide",
        action="store_true",
        help="Genera el CSV wide legacy para revision visual.",
    )
    args = parser.parse_args()

    if not args.skip_data:
        print("[pipeline] Step 1/4 data pipeline")
        run_data_pipeline(
            extract_radiomics=True,
            generate_review_wide=bool(args.generate_review_wide),
        )
    else:
        print("[pipeline] Step 1/4 data pipeline skipped")

    _ensure_exists(LONGITUDINAL_FEATURES_LONG_CSV, "CSV de features longitudinales en long")
    if args.generate_review_wide:
        _ensure_exists(REVIEW_WIDE_CSV, "CSV wide legacy para revision visual")

    print("[pipeline] Step 2/4 delta features")
    build_delta_features()
    _ensure_exists(DELTA_FEATURES_CSV, "CSV de delta features")

    if ICC_KEEP and not args.skip_icc:
        print("[pipeline] Step 3/4 ICC")
        icc_data = load_data(ICC_INPUT_CSV)
        icc_results = measure_and_discard_features_by_icc(icc_data)
        save_data(icc_results, ICC_RESULTS_CSV)
        _ensure_exists(KEEP_CSV, "CSV de features keep por ICC")
    else:
        print("[pipeline] Step 3/4 ICC skipped")

    if args.skip_train:
        print("[pipeline] Step 4/4 train skipped")
        print(f"[pipeline] review csv: {REVIEW_WIDE_CSV}")
        print(f"[pipeline] delta csv: {DELTA_FEATURES_CSV}")
        return

    _ensure_exists(OUTCOMES_CSV, "CSV de outcomes clasificados")

    print("[pipeline] Step 4/4 train")
    run_dir = train_model()
    print(f"[pipeline] training run: {run_dir}")


if __name__ == "__main__":
    main()
