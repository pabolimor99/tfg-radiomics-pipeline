from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

BASE_PAIRS_CSV = Path("outputs/pairs/longitudinal_pairs.csv")
VALIDATED_INDEX_CSV = Path("outputs/indexing/validated_index.csv")
INVALID_PAIRS_CSV = Path("outputs/indexing/invalid_pairs.csv")
OUTPUT_PAIRS_CSV = Path("outputs/pairs/pairs.csv")

OUTCOMES_CSV = Path("outputs/review/classification_outcomes.csv")
OUTPUT_WITH_OUTCOME_CSV = Path("outputs/pairs/pairs_with_outcome.csv")
OUTPUT_MISSING_OUTCOME_CSV = Path("outputs/pairs/pairs_missing_outcome.csv")


@dataclass(frozen=True)
class PairKey:
    patient_id: str
    pair_index: int


def _pick_best_row(rows: pd.DataFrame) -> pd.Series:
    if rows.empty:
        raise ValueError("rows is empty")

    if "is_radiomics_valid" in rows.columns:
        r = rows[rows["is_radiomics_valid"] == True]
        if not r.empty:
            return r.iloc[0]

    if "is_structurally_valid" in rows.columns:
        r = rows[rows["is_structurally_valid"] == True]
        if not r.empty:
            return r.iloc[0]

    return rows.iloc[0]


def _build_baseline_lookup(validated_index: pd.DataFrame) -> dict[PairKey, pd.Series]:
    baseline = validated_index[validated_index["timepoint"] == "baseline"].copy()
    baseline["pair_index"] = pd.to_numeric(baseline["pair_index"], errors="coerce").astype("Int64")
    baseline = baseline.dropna(subset=["patient_id", "pair_index"])

    lookup: dict[PairKey, pd.Series] = {}
    for (patient_id, pair_index), group in baseline.groupby(["patient_id", "pair_index"]):
        key = PairKey(str(patient_id), int(pair_index))
        lookup[key] = _pick_best_row(group)
    return lookup


def build_pairs_with_invalid_followup(
    base_pairs_csv: Path,
    validated_index_csv: Path,
    invalid_pairs_csv: Path,
    output_csv: Path,
    outcomes_csv: Path | None = None,
    output_pairs_with_outcome_csv: Path | None = None,
    output_pairs_missing_outcome_csv: Path | None = None,
) -> None:
    base_pairs = pd.read_csv(base_pairs_csv)
    validated_index = pd.read_csv(validated_index_csv)
    invalid_pairs = pd.read_csv(invalid_pairs_csv)

    required_base_cols = {
        "uid",
        "patient_id",
        "pair_index",
        "baseline_pair_index",
        "followup_pair_index",
        "baseline_image_path",
        "baseline_mask_path",
        "followup_image_path",
        "followup_mask_path",
        "baseline_is_structurally_valid",
        "baseline_is_radiomics_valid",
        "followup_is_structurally_valid",
        "followup_is_radiomics_valid",
    }
    missing = required_base_cols - set(base_pairs.columns)
    if missing:
        raise ValueError(f"base_pairs_csv missing columns: {sorted(missing)}")

    baseline_lookup = _build_baseline_lookup(validated_index)

    followup_invalid = invalid_pairs[invalid_pairs["timepoint"] == "followup"].copy()
    followup_invalid["pair_index"] = pd.to_numeric(followup_invalid["pair_index"], errors="coerce").astype("Int64")
    followup_invalid = followup_invalid.dropna(subset=["patient_id", "pair_index"])

    reconstructed_rows: list[dict] = []
    missing_baseline = 0

    for _, fu in followup_invalid.iterrows():
        patient_id = str(fu["patient_id"])
        pair_index = int(fu["pair_index"])
        key = PairKey(patient_id, pair_index)

        bl = baseline_lookup.get(key)
        if bl is None:
            missing_baseline += 1
            continue

        uid = f"{patient_id}__bl{pair_index}__fu{pair_index}"

        reconstructed_rows.append(
            {
                "uid": uid,
                "patient_id": patient_id,
                "pair_index": pair_index,
                "baseline_pair_index": pair_index,
                "followup_pair_index": pair_index,
                "baseline_image_path": bl.get("image_path"),
                "baseline_mask_path": bl.get("mask_path"),
                "followup_image_path": fu.get("image_path"),
                "followup_mask_path": fu.get("mask_path"),
                "baseline_is_structurally_valid": bl.get("is_structurally_valid"),
                "baseline_is_radiomics_valid": bl.get("is_radiomics_valid"),
                "followup_is_structurally_valid": fu.get("is_structurally_valid"),
                "followup_is_radiomics_valid": fu.get("is_radiomics_valid"),
                "followup_invalid_reason": fu.get("invalid_reason"),
                "source": "invalid_followup",
            }
        )

    reconstructed = pd.DataFrame(reconstructed_rows)

    base_pairs = base_pairs.copy()
    base_pairs["source"] = "valid_longitudinal_pairs"
    base_pairs["followup_invalid_reason"] = pd.NA

    combined = pd.concat([base_pairs, reconstructed], ignore_index=True, sort=False)
    combined = combined.drop_duplicates(subset=["uid"], keep="first")

    combined = combined.sort_values(by=["patient_id", "pair_index"]).reset_index(drop=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_csv, index=False)

    print(f"Base pairs: {len(base_pairs)}")
    print(f"Invalid followup rows: {len(followup_invalid)}")
    print(f"Reconstructed pairs added: {len(combined) - len(base_pairs)}")
    print(f"Missing baseline for invalid FU: {missing_baseline}")
    print(f"Output written: {output_csv}")

    if outcomes_csv is None or not outcomes_csv.is_file():
        return

    outcomes = pd.read_csv(outcomes_csv)
    if "uid" not in outcomes.columns or "outcome" not in outcomes.columns:
        return

    outcomes = outcomes[["uid", "outcome"]].copy()
    outcomes = outcomes.dropna(subset=["uid"])
    outcomes["uid"] = outcomes["uid"].astype(str)
    outcomes = outcomes.drop_duplicates(subset=["uid"], keep="last")

    enriched = combined.merge(outcomes, on="uid", how="left")

    if output_pairs_with_outcome_csv is not None:
        output_pairs_with_outcome_csv.parent.mkdir(parents=True, exist_ok=True)
        enriched.to_csv(output_pairs_with_outcome_csv, index=False)
        print(f"Output with outcomes written: {output_pairs_with_outcome_csv}")

    if output_pairs_missing_outcome_csv is not None:
        missing = enriched[enriched["outcome"].isna()].copy()
        output_pairs_missing_outcome_csv.parent.mkdir(parents=True, exist_ok=True)
        missing.to_csv(output_pairs_missing_outcome_csv, index=False)
        print(f"Pairs missing outcome written: {output_pairs_missing_outcome_csv}")


def main() -> None:
    build_pairs_with_invalid_followup(
        base_pairs_csv=BASE_PAIRS_CSV,
        validated_index_csv=VALIDATED_INDEX_CSV,
        invalid_pairs_csv=INVALID_PAIRS_CSV,
        output_csv=OUTPUT_PAIRS_CSV,
        outcomes_csv=OUTCOMES_CSV,
        output_pairs_with_outcome_csv=OUTPUT_WITH_OUTCOME_CSV,
        output_pairs_missing_outcome_csv=OUTPUT_MISSING_OUTCOME_CSV,
    )


if __name__ == "__main__":
    main()
