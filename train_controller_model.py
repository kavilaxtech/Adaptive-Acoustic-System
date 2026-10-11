
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    BASELINE_FILTER_LENGTH,
    BASELINE_MU,
    DATA_ROOT,
    MODEL_DIR,
    MODEL_PATH,
    RANDOM_STATE,
    RESULTS_DIR,
)
from src.acoustic_features import FEATURE_NAMES
from src.dataset import discover_mimii_fan_files, evenly_limit, split_test_files
from src.selector import (
    candidate_configurations,
    evaluate_recording,
    fit_regressor,
    select_configuration,
)

PARAMETER_GRID = [
    {"n_estimators": 120, "max_depth": None, "min_samples_leaf": 2, "max_features": 0.8},
    {"n_estimators": 240, "max_depth": None, "min_samples_leaf": 2, "max_features": 0.8},
    {"n_estimators": 120, "max_depth": 12, "min_samples_leaf": 2, "max_features": 0.8},
    {"n_estimators": 240, "max_depth": 12, "min_samples_leaf": 2, "max_features": 0.8},
    {"n_estimators": 120, "max_depth": 12, "min_samples_leaf": 4, "max_features": 0.8},
    {"n_estimators": 240, "max_depth": 12, "min_samples_leaf": 4, "max_features": 0.8},
]


def json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)}")


def generate_rows(files: list[Path], segment_seconds: float, candidates: list[dict], role: str):
    """Run every candidate on each supplied source recording; skip unusable files."""
    rows_all = []
    per_file = []
    manifest = []
    start = time.time()
    for index, path in enumerate(files, start=1):
        try:
            features, rows = evaluate_recording(path, segment_seconds, candidates)
            for row in rows:
                row["source_path"] = str(path.resolve())
                row["split"] = role
            rows_all.extend(rows)
            per_file.append({"path": path, "features": features, "rows": rows})
            manifest.append({"split": role, "file": path.name, "path": str(path.resolve()), "status": "used"})
            print(f"[{role}] {index}/{len(files)}: {path.name}")
        except Exception as exc:  # skip only the failed recording; log the exact reason
            warnings.warn(f"Skipping {path}: {exc}")
            manifest.append({"split": role, "file": path.name, "path": str(path.resolve()), "status": f"skipped: {exc}"})
    print(f"Completed {role}: {len(per_file)} usable recordings in {time.time() - start:.1f}s")
    return pd.DataFrame(rows_all), per_file, manifest


def validation_regret(model, validation_records, candidates):
    regrets = []
    for record in validation_records:
        chosen, _ = select_configuration(model, record["features"], candidates)
        actual = next(
            row for row in record["rows"]
            if row["mu"] == chosen["mu"] and row["filter_len"] == chosen["filter_len"]
        )
        oracle = max(record["rows"], key=lambda row: float(row["utility"]))
        regrets.append(float(oracle["utility"]) - float(actual["utility"]))
    return float(np.mean(regrets)) if regrets else float("inf")


def evaluate_on_test(model, test_records, candidates):
    results = []
    for record in test_records:
        chosen, predictions = select_configuration(model, record["features"], candidates)
        actual_selected = next(
            row for row in record["rows"]
            if row["mu"] == chosen["mu"] and row["filter_len"] == chosen["filter_len"]
        )
        oracle = max(record["rows"], key=lambda row: float(row["utility"]))
        baseline = next(
            (row for row in record["rows"]
             if row["mu"] == BASELINE_MU and row["filter_len"] == BASELINE_FILTER_LENGTH),
            None,
        )
        if baseline is None:
            raise RuntimeError("The configured baseline must be included in the candidate grid")
        results.append({
            "file": record["path"].name,
            "source_path": str(record["path"].resolve()),
            "ai_selected_mu": float(chosen["mu"]),
            "ai_selected_filter_len": int(chosen["filter_len"]),
            "ai_predicted_utility": float(chosen["predicted_utility"]),
            "ai_selected_actual_utility": float(actual_selected["utility"]),
            "ai_selected_actual_nr_db": float(actual_selected["steady_state_noise_reduction_db"]),
            "baseline_mu": BASELINE_MU,
            "baseline_filter_len": BASELINE_FILTER_LENGTH,
            "baseline_utility": float(baseline["utility"]),
            "baseline_nr_db": float(baseline["steady_state_noise_reduction_db"]),
            "oracle_mu_simulation_only": float(oracle["mu"]),
            "oracle_filter_len_simulation_only": int(oracle["filter_len"]),
            "oracle_utility_simulation_only": float(oracle["utility"]),
            "oracle_nr_db_simulation_only": float(oracle["steady_state_noise_reduction_db"]),
            "selection_regret": float(oracle["utility"]) - float(actual_selected["utility"]),
            "within_0_5_utility_of_oracle": bool(float(oracle["utility"]) - float(actual_selected["utility"]) <= 0.5),
            "recording_name_prefix": "anomaly" if record["path"].name.lower().startswith("anomaly_") else "normal" if record["path"].name.lower().startswith("normal_") else "unknown",
            "evidence_type": "real MIMII audio; controller outputs and utility are from assumed-path ANC simulation",
        })
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=str(DATA_ROOT), help="Directory containing extracted MIMII fan train/ and test/ folders")
    parser.add_argument("--machine-id", default="00", help="MIMII fan machine ID such as 00, or 'all'")
    parser.add_argument("--max-train-files", type=int, default=80, help="Max normal training recordings; use 0 for all")
    parser.add_argument("--max-test-files", type=int, default=30, help="Max official test recordings; use 0 for all")
    parser.add_argument("--segment-seconds", type=float, default=1.0, help="Centered source-audio segment per WAV")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    args = parser.parse_args()

    if args.segment_seconds < 0.5:
        parser.error("--segment-seconds must be at least 0.5 for stable feature analysis")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    train_all, test_all = discover_mimii_fan_files(args.data_root, args.machine_id)
    train_files = evenly_limit(train_all, args.max_train_files if args.max_train_files > 0 else None)
    test_files = split_test_files(test_all, args.max_test_files if args.max_test_files > 0 else None)
    if len(train_files) < 8:
        raise RuntimeError(f"Need at least 8 normal train WAVs for a useful pilot; found {len(train_files)}")
    if len(test_files) < 2:
        raise RuntimeError(f"Need at least 2 test WAVs; found {len(test_files)}")

    fit_files, validation_files = train_test_split(
        train_files, test_size=0.20, random_state=args.seed, shuffle=True
    )
    candidates = candidate_configurations()
    print(f"Selected {len(train_files)} normal train WAVs, split into {len(fit_files)} fitting and {len(validation_files)} validation recordings.")
    print(f"Selected {len(test_files)} held-out MIMII test WAVs. No generated audio will be used.")
    print(f"Testing {len(candidates)} ANC configurations per recording; segment={args.segment_seconds:.2f}s.")

    train_rows, train_records, manifest_train = generate_rows(fit_files, args.segment_seconds, candidates, "train")
    val_rows, val_records, manifest_val = generate_rows(validation_files, args.segment_seconds, candidates, "validation")
    if train_rows.empty or val_rows.empty:
        raise RuntimeError("Could not extract usable training/validation records. Check your WAV files.")

    best_params = None
    best_regret = float("inf")
    tune_rows = []
    for param_index, params in enumerate(PARAMETER_GRID, start=1):
        model = fit_regressor(train_rows, params, random_state=args.seed)
        regret = validation_regret(model, val_records, candidates)
        tune_rows.append({**params, "validation_mean_selection_regret": regret})
        print(f"Tuning {param_index}/{len(PARAMETER_GRID)}: params={params}, mean validation regret={regret:.4f}")
        if regret < best_regret:
            best_regret = regret
            best_params = params

    # Refit on the complete development train split (fitting + validation WAVs).
    development_rows = pd.concat([train_rows, val_rows], ignore_index=True)
    final_model = fit_regressor(development_rows, best_params, random_state=args.seed)

    test_rows, test_records, manifest_test = generate_rows(test_files, args.segment_seconds, candidates, "test")
    if not test_records:
        raise RuntimeError("No usable official test recordings were found")
    test_predictions = evaluate_on_test(final_model, test_records, candidates)
    test_predictions.to_csv(RESULTS_DIR / "controller_selection_test.csv", index=False)
    pd.DataFrame(tune_rows).to_csv(RESULTS_DIR / "hyperparameter_validation.csv", index=False)
    pd.DataFrame(manifest_train + manifest_val + manifest_test).to_csv(RESULTS_DIR / "dataset_manifest.csv", index=False)

    model_bundle = {
        "model": final_model,
        "feature_names": FEATURE_NAMES,
        "candidate_configurations": candidates,
        "best_hyperparameters": best_params,
        "validation_mean_selection_regret": best_regret,
        "machine_id": args.machine_id,
        "segment_seconds": args.segment_seconds,
        "trained_on_real_mimii_wavs": True,
        "training_target": "simulation-derived utility of controller configurations run against real MIMII audio through assumed FIR paths",
        "evidence_limit": "simulation results are not physical ANC measurements; the dataset has no measured ANC reference/error pairs or measured secondary path",
    }
    import joblib
    joblib.dump(model_bundle, MODEL_PATH)

    selection_nr = test_predictions["ai_selected_actual_nr_db"].to_numpy(dtype=float)
    baseline_nr = test_predictions["baseline_nr_db"].to_numpy(dtype=float)
    oracle_nr = test_predictions["oracle_nr_db_simulation_only"].to_numpy(dtype=float)
    regrets = test_predictions["selection_regret"].to_numpy(dtype=float)
    summary = {
        "dataset": "MIMII fan audio",
        "machine_id_filter": args.machine_id,
        "training_recordings_used": int(len(train_records)),
        "validation_recordings_used": int(len(val_records)),
        "test_recordings_used": int(len(test_records)),
        "candidate_configurations": candidates,
        "best_hyperparameters_selected_on_validation": best_params,
        "validation_mean_selection_regret": float(best_regret),
        "mean_test_ai_selected_steady_state_nr_db": float(np.nanmean(selection_nr)),
        "mean_test_fixed_baseline_steady_state_nr_db": float(np.nanmean(baseline_nr)),
        "mean_test_oracle_steady_state_nr_db_simulation_only": float(np.nanmean(oracle_nr)),
        "mean_test_ai_minus_baseline_nr_db": float(np.nanmean(selection_nr - baseline_nr)),
        "mean_test_selection_regret": float(np.nanmean(regrets)),
        "median_test_selection_regret": float(np.nanmedian(regrets)),
        "p90_test_selection_regret": float(np.nanpercentile(regrets, 90)),
        "fraction_within_0_5_utility_of_simulation_oracle": float(test_predictions["within_0_5_utility_of_oracle"].mean()),
        "evidence_type": "MIMII recorded inputs; simulated ANC paths/outputs; no synthetic audio examples generated",
        "caveat": "The simulation oracle and reduction figures are not measurements of physical noise cancellation. Results are valid only for this stated simulation setup until validated using measured reference/error signals and acoustic paths.",
    }
    with open(RESULTS_DIR / "controller_selection_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, default=json_default)

    # This chart uses only test results produced from the selected actual MIMII recordings.
    figure, axis = plt.subplots(figsize=(9, 5))
    labels = ["Fixed FxNLMS baseline", "AI-selected config", "Best config in simulation"]
    means = [float(np.nanmean(baseline_nr)), float(np.nanmean(selection_nr)), float(np.nanmean(oracle_nr))]
    spreads = [float(np.nanstd(baseline_nr)), float(np.nanstd(selection_nr)), float(np.nanstd(oracle_nr))]
    axis.bar(labels, means, yerr=spreads, capsize=5)
    axis.set_ylabel("Steady-state signal-level reduction (dB)")
    axis.set_title("MIMII test recordings: assumed-path ANC simulation comparison")
    axis.tick_params(axis="x", rotation=12)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(RESULTS_DIR / "controller_selection_comparison.png", dpi=180)
    plt.close(figure)

    print("\nTRAINING COMPLETE")
    print(f"Model saved: {MODEL_PATH}")
    print(f"Test predictions: {RESULTS_DIR / 'controller_selection_test.csv'}")
    print(f"Summary: {RESULTS_DIR / 'controller_selection_summary.json'}")
    print(json.dumps(summary, indent=2, default=json_default))


if __name__ == "__main__":
    main()
