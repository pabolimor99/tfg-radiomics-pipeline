from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DEFAULT_INPUT_CSV = Path("outputs/features/longitudinal_features_long.csv")
DEFAULT_OUTPUT_CSV = Path("outputs/review/stratified_sample_baseline.csv")
DEFAULT_VOLUME_COL = "original_shape_VoxelVolume"


def stratified_sample_baseline(df: pd.DataFrame, *, n_total: int, seed: int, volume_col: str) -> pd.DataFrame:
    required = {"uid", "patient_id", "pair_index", "timepoint", "mask_path", "mask_empty", volume_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en el CSV: {sorted(missing)}")

    n_total = int(n_total)
    if n_total < 3:
        raise ValueError("--n-total debe ser >= 3.")

    baseline = df[df["timepoint"].astype(str).str.lower().eq("baseline")].copy()
    baseline["mask_empty"] = pd.to_numeric(baseline["mask_empty"], errors="coerce")
    baseline[volume_col] = pd.to_numeric(baseline[volume_col], errors="coerce")

    eligible = baseline[(baseline["mask_empty"] == 0) & (baseline[volume_col].notna())].copy()
    if eligible.empty:
        raise ValueError("No hay casos baseline elegibles (mask_empty==0 y volumen no-NA).")

    out = eligible[["uid", "patient_id", "pair_index", "mask_path", volume_col]].rename(
        columns={volume_col: "voxel_volume"}
    )
    out["volume_group"] = pd.qcut(out["voxel_volume"], q=3, labels=["low", "mid", "high"], duplicates="drop")
    out = out.dropna(subset=["volume_group"]).copy()

    per_group = n_total // 3
    counts = out["volume_group"].value_counts()
    lacking = {g: int(per_group - counts.get(g, 0)) for g in ["low", "mid", "high"] if int(counts.get(g, 0)) < per_group}
    if lacking:
        raise ValueError(f"No hay suficientes casos por tercil: faltan {lacking}.")

    sampled = (
        out.groupby("volume_group", observed=True, group_keys=False)
        .sample(n=per_group, random_state=int(seed))
        .sort_values(by=["volume_group", "patient_id", "pair_index", "uid"])
        .reset_index(drop=True)
    )
    sampled.insert(3, "timepoint", "baseline")
    sampled["stratum"] = sampled["volume_group"].astype(str)
    return sampled[
        ["uid", "patient_id", "pair_index", "timepoint", "mask_path", "voxel_volume", "volume_group", "stratum"]
    ]


def main() -> None:
    p = argparse.ArgumentParser(description="Muestreo estratificado de casos baseline por terciles de volumen.")
    p.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    p.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    p.add_argument("--n-total", type=int, default=50, help="Total deseado (se trunca a múltiplos de 3).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--volume-col", type=str, default=DEFAULT_VOLUME_COL)
    args = p.parse_args()

    df = pd.read_csv(args.input_csv)
    sampled = stratified_sample_baseline(df, n_total=args.n_total, seed=args.seed, volume_col=args.volume_col)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    sampled.to_csv(args.output_csv, index=False)

    if int(args.n_total) % 3 != 0:
        logging.info(
            "Nota: n_total=%d no es múltiplo de 3; se muestrean %d (= 3 * (%d // 3)).",
            int(args.n_total),
            int(len(sampled)),
            int(args.n_total),
        )

    logging.info("OK: %d casos muestreados guardados en %s", int(len(sampled)), args.output_csv)


if __name__ == "__main__":
    main()
