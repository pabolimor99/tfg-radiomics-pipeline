from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.ingest.grouper import reorganize_dataset
from src.data.ingest.organizer import organize_files
from src.data.indexing.indexer import create_metadata_csv
from src.data.indexing.validator import validate_index
from src.preprocessing.build_longitudinal_pairs import build_longitudinal_pairs
from src.radiomics.extract_features import extract_features
from src.utils.longitudinal_pairs_long_to_wide import longitudinal_pairs_long_to_wide

import pandas as pd

INPUTS_DIR = Path("data/base/inputsTr")
TARGETS_DIR = Path("data/base/targetsTr")
RAW_DIR = Path("data/raw")
INDEX_CSV = Path("outputs/indexing/index.csv")
VALIDATED_INDEX_CSV = Path("outputs/indexing/validated_index.csv")
INVALID_OUTPUT_CSV = Path("outputs/indexing/invalid_pairs.csv")
LONGITUDINAL_PAIRS_CSV = Path("outputs/pairs/longitudinal_pairs.csv")
REVIEW_WIDE_CSV = Path("outputs/review/longitudinal_pairs_review_wide.csv")
LONGITUDINAL_FEATURES_LONG_CSV = Path("outputs/features/longitudinal_features_long.csv")
ERRORS_CSV = Path("outputs/logs/longitudinal_features_errors.csv")


def run_data_pipeline(
    *,
    extract_radiomics: bool = True,
    generate_review_wide: bool = False,
) -> None:
    reorganize_dataset(INPUTS_DIR, RAW_DIR)
    reorganize_dataset(TARGETS_DIR, RAW_DIR)
    organize_files(RAW_DIR)
    create_metadata_csv(RAW_DIR, INDEX_CSV)
    validate_index(INDEX_CSV, VALIDATED_INDEX_CSV, INVALID_OUTPUT_CSV)
    build_longitudinal_pairs(VALIDATED_INDEX_CSV, LONGITUDINAL_PAIRS_CSV)

    if generate_review_wide:
        review_df = longitudinal_pairs_long_to_wide(pd.read_csv(LONGITUDINAL_PAIRS_CSV))
        REVIEW_WIDE_CSV.parent.mkdir(parents=True, exist_ok=True)
        review_df.to_csv(REVIEW_WIDE_CSV, index=False)

    if extract_radiomics:
        extract_features(LONGITUDINAL_PAIRS_CSV, LONGITUDINAL_FEATURES_LONG_CSV, ERRORS_CSV)


if __name__ == "__main__":
    run_data_pipeline(extract_radiomics=True, generate_review_wide=True)
