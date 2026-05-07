from pathlib import Path

import pandas as pd

OUTCOMES_CSV = Path("outputs/review/classification_outcomes.csv")


def main() -> None:
    outcomes = pd.read_csv(OUTCOMES_CSV, usecols=["outcome"])
    outcomes["outcome"] = pd.to_numeric(outcomes["outcome"], errors="raise").astype(int)

    total = len(outcomes)

    counts = outcomes["outcome"].value_counts().sort_index()

    for label in sorted(counts.index.tolist()):
        count = int(counts[label])
        percentage = 100.0 * count / total
        print(f"Clase {label}: {count}/{total} ({percentage:.2f}%)")


if __name__ == "__main__":
    main()
