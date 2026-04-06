from pathlib import Path

from src.data.grouper import reorganize_dataset
from src.data.organizer import organize_files
from src.data.indexer import create_metadata_csv
from src.data.validator import validate_index
from src.preprocessing.build_longitudinal_pairs import build_longitudinal_pairs
from src.preprocessing.extract_features import extract_features

INPUTS_DIR = Path("data/base/inputsTr")
TARGETS_DIR = Path("data/base/targetsTr")
RAW_DIR = Path("data/raw")
INDEX_CSV = Path("outputs/indexing/index.csv")
VALIDATED_INDEX_CSV = Path("outputs/indexing/validated_index.csv")
INVALID_OUTPUT_CSV = Path("outputs/indexing/invalid_pairs.csv")
LONGITUDINAL_PAIRS_CSV = Path("outputs/pairs/longitudinal_pairs.csv")
LONGITUDINAL_FEATURES_CSV = Path("outputs/features/longitudinal_features.csv")
ERRORS_CSV = Path("outputs/logs/longitudinal_features_errors.csv")

if __name__ == "__main__":

    reorganize_dataset(INPUTS_DIR, RAW_DIR)
    reorganize_dataset(TARGETS_DIR, RAW_DIR)
    organize_files(RAW_DIR)
    create_metadata_csv(RAW_DIR, INDEX_CSV)
    validate_index(INDEX_CSV, VALIDATED_INDEX_CSV, INVALID_OUTPUT_CSV)
    build_longitudinal_pairs(VALIDATED_INDEX_CSV, LONGITUDINAL_PAIRS_CSV)
    # extract_features(LONGITUDINAL_PAIRS_CSV, LONGITUDINAL_FEATURES_CSV, ERRORS_CSV)
    
