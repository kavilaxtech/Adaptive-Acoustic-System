"""Training/evaluation utilities for the ANC-configuration selection model."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from config import CANDIDATE_FILTER_LENGTHS, CANDIDATE_MUS
from src.acoustic_features import FEATURE_NAMES, extract_features
from src.anc_simulation import configuration_utility, run_fxnlms_on_recording
from src.audio_io import load_segment


def candidate_configurations() -> list[dict]:
    return [
        {"mu": float(mu), "filter_len": int(length)}
        for mu in CANDIDATE_MUS
        for length in CANDIDATE_FILTER_LENGTHS
    ]


def evaluate_recording(
    path: str | Path,
    segment_seconds: float,
    candidates: list[dict] | None = None,
) -> tuple[np.ndarray, list[dict]]:
    """Extract features and benchmark all candidates on one real recording.

    Candidate scores are generated from the ANC simulation driven by this real
    recording. They are not measured physical ANC labels.
    """
    audio, sample_rate = load_segment(path, segment_seconds)
    features = extract_features(audio, sample_rate)
    candidates = candidates or candidate_configurations()
    rows = []
    for candidate in candidates:
        metrics = run_fxnlms_on_recording(
            audio,
            sample_rate,
            mu=candidate["mu"],
            filter_len=candidate["filter_len"],
        )
        score = configuration_utility(metrics)
        row = {
            "source_file": Path(path).name,
            "mu": candidate["mu"],
            "filter_len": candidate["filter_len"],
            "utility": float(score),
            "stable": bool(metrics["stable"]),
            "steady_state_noise_reduction_db": float(metrics["steady_state_noise_reduction_db"]),
            "noise_reduction_db": float(metrics["noise_reduction_db"]),
            "control_to_primary_rms_db": float(metrics["control_to_primary_rms_db"]),
            "evidence_type": metrics["evidence_type"],
        }
        for name, value in zip(FEATURE_NAMES, features):
            row[name] = float(value)
        rows.append(row)
    return features, rows


def feature_config_matrix(feature_rows: np.ndarray, configs: list[dict]) -> np.ndarray:
    """Repeat each audio feature vector for all controller candidates."""
    feature_rows = np.asarray(feature_rows, dtype=np.float32)
    if feature_rows.ndim == 1:
        feature_rows = feature_rows.reshape(1, -1)
    matrix = []
    for row in feature_rows:
        for cfg in configs:
            matrix.append(np.concatenate([row, [cfg["mu"], float(cfg["filter_len"])]]))
    return np.asarray(matrix, dtype=np.float32)


def fit_regressor(training_rows: pd.DataFrame, params: dict, random_state: int = 42) -> RandomForestRegressor:
    feature_and_setting_cols = FEATURE_NAMES + ["mu", "filter_len"]
    model = RandomForestRegressor(
        n_estimators=int(params["n_estimators"]),
        max_depth=params["max_depth"],
        min_samples_leaf=int(params["min_samples_leaf"]),
        max_features=params["max_features"],
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(
        training_rows[feature_and_setting_cols].to_numpy(dtype=np.float32),
        training_rows["utility"].to_numpy(dtype=np.float32),
    )
    return model


def select_configuration(model, features: np.ndarray, candidates: list[dict]) -> tuple[dict, list[float]]:
    matrix = feature_config_matrix(features, candidates)
    predictions = np.asarray(model.predict(matrix), dtype=float).reshape(-1)
    best_index = int(np.argmax(predictions))
    choice = dict(candidates[best_index])
    choice["predicted_utility"] = float(predictions[best_index])
    return choice, predictions.tolist()


def choose_oracle(rows: list[dict]) -> dict:
    """Return the best feasible config according to the declared simulation score."""
    if not rows:
        raise ValueError("No ANC candidate results were produced")
    return max(rows, key=lambda row: float(row["utility"]))


def rows_to_frame(per_record_rows: Iterable[list[dict]]) -> pd.DataFrame:
    return pd.DataFrame([row for group in per_record_rows for row in group])
