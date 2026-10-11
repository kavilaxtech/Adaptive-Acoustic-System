"""Locate the official MIMII fan WAV recordings without generating audio."""
from __future__ import annotations

import re
from pathlib import Path


def _machine_id_from_name(path: Path) -> str | None:
    match = re.search(r"id_(\d{2})_", path.name.lower())
    return match.group(1) if match else None


def _path_has_component(path: Path, component: str) -> bool:
    return component.lower() in {part.lower() for part in path.parts}


def discover_mimii_fan_files(data_root: str | Path, machine_id: str = "00") -> tuple[list[Path], list[Path]]:
    """Return (normal training WAVs, official test WAVs) for a fan machine ID.

    Expected official-layout path components include `train` and `test`, e.g.
    `.../fan/train/normal_id_00_....wav` and `.../fan/test/anomaly_id_00_....wav`.
    File label names are used only to find the normal-only training split. The
    anomaly/normal test labels are not used as model targets.
    """
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(
            f"MIMII data folder does not exist: {root}\n"
            "Download and extract the DCASE 2020 MIMII fan development archive first."
        )
    all_wavs = sorted(p for p in root.rglob("*.wav") if p.is_file())
    if not all_wavs:
        raise FileNotFoundError(
            f"No WAV recordings found under {root}. The dataset archive may not have been extracted."
        )
    filtered = [p for p in all_wavs if machine_id == "all" or _machine_id_from_name(p) == str(machine_id).zfill(2)]
    training = [
        p for p in filtered
        if _path_has_component(p, "train") and p.name.lower().startswith("normal_")
    ]
    testing = [p for p in filtered if _path_has_component(p, "test")]

    if not training:
        raise FileNotFoundError(
            "Could not find normal training WAVs for the selected machine ID. "
            "Expected a MIMII DCASE layout containing train/normal_id_XX_*.wav. "
            f"Data root: {root}; machine_id={machine_id!r}."
        )
    if not testing:
        raise FileNotFoundError(
            "Could not find MIMII test WAVs. Use the development archive containing a test/ folder; "
            "test recordings are kept separate from model training."
        )
    return training, testing


def evenly_limit(paths: list[Path], max_count: int | None) -> list[Path]:
    """Choose a deterministic spread of files without creating new audio."""
    paths = sorted(paths)
    if max_count is None or max_count <= 0 or len(paths) <= max_count:
        return paths
    import numpy as np
    indices = np.linspace(0, len(paths) - 1, num=max_count, dtype=int)
    return [paths[int(i)] for i in indices]


def split_test_files(paths: list[Path], max_count: int | None) -> list[Path]:
    """Limit test recordings while preserving normal/anomaly representation where possible."""
    paths = sorted(paths)
    if max_count is None or max_count <= 0 or len(paths) <= max_count:
        return paths
    normal = [p for p in paths if p.name.lower().startswith("normal_")]
    anomaly = [p for p in paths if p.name.lower().startswith("anomaly_")]
    other = [p for p in paths if p not in set(normal) and p not in set(anomaly)]
    if normal and anomaly:
        n_each = max(1, max_count // 2)
        chosen = evenly_limit(normal, min(n_each, len(normal))) + evenly_limit(anomaly, min(n_each, len(anomaly)))
        if len(chosen) < max_count:
            remaining = [p for p in paths if p not in set(chosen)]
            chosen.extend(evenly_limit(remaining, max_count - len(chosen)))
        return sorted(chosen[:max_count])
    return evenly_limit(paths, max_count)
