from __future__ import annotations

from pathlib import Path

from src.features.prepare_delta_features import build_delta_features
from src.features.prepare_features_uid import build_features_uid


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
