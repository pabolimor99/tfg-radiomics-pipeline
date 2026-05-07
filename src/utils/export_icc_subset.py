from __future__ import annotations

from pathlib import Path

import pandas as pd

INPUT_CSV = Path("outputs-base/pairs/pairs_long.csv")
OUTPUT_CSV = Path("outputs/features/icc_subset_keys.csv")
SUBSET_SIZE = 150


def export_icc_subset(
    input_csv: Path = INPUT_CSV,
    output_csv: Path = OUTPUT_CSV,
    subset_size: int = SUBSET_SIZE,
) -> None:
    if not input_csv.is_file():
        raise FileNotFoundError(
            f"No se encuentra el CSV fuente para exportar el subset ICC: {input_csv}"
        )

    df = pd.read_csv(input_csv)

    required = {"patient_id", "pair_index", "timepoint", "is_radiomics_valid"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas obligatorias: {sorted(missing)}")

    subset = df[(df["timepoint"] == "baseline") & (df["is_radiomics_valid"] == True)].copy()
    subset = subset.head(int(subset_size))
    subset = subset.loc[:, ["patient_id", "pair_index", "timepoint"]].copy()
    subset["patient_id"] = subset["patient_id"].astype(str)
    subset["pair_index"] = pd.to_numeric(subset["pair_index"], errors="raise").astype(int)
    subset["timepoint"] = subset["timepoint"].astype(str)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    subset.to_csv(output_csv, index=False)

    print(f"[export_icc_subset] rows={len(subset)}")
    print(f"[export_icc_subset] wrote={output_csv}")


def main() -> None:
    export_icc_subset()


if __name__ == "__main__":
    main()
