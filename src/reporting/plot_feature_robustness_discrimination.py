from __future__ import annotations

from pathlib import Path

from src.modeling.analysis.robustness_discrimination import (
    build_feature_table,
    write_feature_tables,
)
from src.modeling.analysis.robustness_discrimination_plots import (
    plot_all_robustness_discrimination,
)


FEATURES_CSV = Path("outputs/features/delta_features_uid.csv")
OUTCOMES_CSV = Path("outputs/review/classification_outcomes.csv")
ICC_CSV = Path("outputs/features/icc_results.csv")

OUT_DIR = Path("figures/model_results")
OUT_CSV = Path("outputs/models/feature_icc_wilcoxon.csv")
OUT_INTERSECTION_CSV = Path("outputs/models/feature_icc_wilcoxon_top_intersection.csv")

TOP_FRACTION = 0.10
TOP_N_PLOT = 20


def main() -> None:
    table = build_feature_table(
        features_csv=FEATURES_CSV,
        outcomes_csv=OUTCOMES_CSV,
        icc_csv=ICC_CSV,
        top_fraction=TOP_FRACTION,
    )

    write_feature_tables(
        table,
        out_csv=OUT_CSV,
        out_intersection_csv=OUT_INTERSECTION_CSV,
    )
    plot_all_robustness_discrimination(
        table,
        out_dir=OUT_DIR,
        top_n_plot=TOP_N_PLOT,
    )

    print(f"[plot_feature_robustness_discrimination] wrote={OUT_CSV}")
    print(f"[plot_feature_robustness_discrimination] wrote={OUT_INTERSECTION_CSV}")
    print(f"[plot_feature_robustness_discrimination] wrote={OUT_DIR}")


if __name__ == "__main__":
    main()
