import json
from pathlib import Path

def load_best_optuna_config(OPTUNA_OUT_ROOT) -> dict:
    best_payload = None
    best_path = None

    for run_dir in OPTUNA_OUT_ROOT.iterdir():
        study_results_path = run_dir / "study_results.json"
        if not study_results_path.is_file():
            continue

        with study_results_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        if best_payload is None or float(payload["best_value"]) > float(best_payload["best_value"]):
            best_payload = payload
            best_path = study_results_path

    if best_payload is None:
        raise FileNotFoundError(f"No study_results.json found in {OPTUNA_OUT_ROOT}")

    best_payload["source_path"] = str(best_path)
    return best_payload
