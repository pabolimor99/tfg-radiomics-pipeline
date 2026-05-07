from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from features.prepare_features_uid import OUTPUT_CSV as FEATURES_UID_CSV
from features.prepare_features_uid import build_features_uid
from features.prepare_delta_features import OUTPUT_CSV as DELTA_FEATURES_CSV
from features.prepare_delta_features import build_delta_features


def prepare_modeling_features() -> tuple[Path, Path]:
    features_uid_csv = build_features_uid()
    delta_features_csv = build_delta_features()
    return features_uid_csv, delta_features_csv


def main() -> None:
    features_uid_csv, delta_features_csv = prepare_modeling_features()
    print(f"[prepare_modeling_features] features_uid={features_uid_csv}")
    print(f"[prepare_modeling_features] delta_features={delta_features_csv}")


if __name__ == "__main__":
    main()
