"""Extract a small, real MIMII fan subset from the official ZIP without unpacking everything.

This script does not generate audio. It copies selected original WAV files from the
DCASE 2020 dev_data_fan.zip archive into the directory structure expected by this project.
Only the ZIP's central directory is scanned; only the selected WAV payloads are decompressed.
"""
from __future__ import annotations

import argparse
import csv
import re
import zipfile
from pathlib import Path, PurePosixPath


def machine_id_for(member_name: str) -> str | None:
    match = re.search(r"id_(\d{2})_", PurePosixPath(member_name.replace("\\", "/")).name.lower())
    return match.group(1) if match else None


def spread_sample(members: list[zipfile.ZipInfo], count: int) -> list[zipfile.ZipInfo]:
    members = sorted(members, key=lambda m: m.filename)
    if count <= 0 or len(members) <= count:
        return members
    if count == 1:
        return [members[len(members) // 2]]
    # Deterministically sample across sorted filename range rather than taking only first files.
    positions = [round(i * (len(members) - 1) / (count - 1)) for i in range(count)]
    return [members[i] for i in positions]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, help="Path to the downloaded official dev_data_fan.zip")
    parser.add_argument("--output-root", default="data/mimii", help="Subset destination root (default: data/mimii)")
    parser.add_argument("--machine-id", default="00", help="Machine ID, e.g. 00, 02, 04, or 06")
    parser.add_argument("--train-count", type=int, default=80, help="Normal training WAVs to extract (default: 80)")
    parser.add_argument("--test-per-class", type=int, default=15, help="Normal and anomaly test WAVs each (default: 15)")
    parser.add_argument("--force", action="store_true", help="Overwrite selected files if they already exist")
    args = parser.parse_args()

    archive = Path(args.archive)
    if not archive.is_file():
        parser.error(f"Archive not found: {archive}")
    if args.train_count < 8:
        parser.error("--train-count must be at least 8 for a useful pilot")
    if args.test_per_class < 1:
        parser.error("--test-per-class must be at least 1")
    machine_id = str(args.machine_id).zfill(2)
    out_root = Path(args.output_root)

    with zipfile.ZipFile(archive, "r") as zf:
        train_pool: list[zipfile.ZipInfo] = []
        test_normal_pool: list[zipfile.ZipInfo] = []
        test_anomaly_pool: list[zipfile.ZipInfo] = []
        for info in zf.infolist():
            if info.is_dir() or not info.filename.lower().endswith(".wav"):
                continue
            parts = {p.lower() for p in PurePosixPath(info.filename.replace("\\", "/")).parts}
            if machine_id_for(info.filename) != machine_id:
                continue
            base = PurePosixPath(info.filename.replace("\\", "/")).name.lower()
            if "train" in parts and base.startswith("normal_id_"):
                train_pool.append(info)
            elif "test" in parts and base.startswith("normal_id_"):
                test_normal_pool.append(info)
            elif "test" in parts and base.startswith("anomaly_id_"):
                test_anomaly_pool.append(info)

        train_chosen = spread_sample(train_pool, args.train_count)
        normal_chosen = spread_sample(test_normal_pool, args.test_per_class)
        anomaly_chosen = spread_sample(test_anomaly_pool, args.test_per_class)
        if len(train_chosen) < 8:
            raise SystemExit(f"Found only {len(train_chosen)} normal training files for machine ID {machine_id}. Check that this is the official DCASE 2020 fan development archive.")
        if not normal_chosen or not anomaly_chosen:
            raise SystemExit(
                f"Could not find both normal and anomalous test WAVs for ID {machine_id}. "
                f"Found normal={len(normal_chosen)}, anomaly={len(anomaly_chosen)}."
            )

        manifest_rows = []
        total_bytes = 0
        selected: list[tuple[str, zipfile.ZipInfo, str]] = []
        for info in train_chosen:
            selected.append(("train", info, "normal_train"))
        for info in normal_chosen:
            selected.append(("test", info, "normal_test"))
        for info in anomaly_chosen:
            selected.append(("test", info, "anomaly_test"))

        for split, info, role in selected:
            filename = PurePosixPath(info.filename.replace("\\", "/")).name
            destination = out_root / "fan" / split / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() and not args.force:
                # Keep existing file; still include it in the manifest.
                if destination.stat().st_size != info.file_size:
                    raise SystemExit(
                        f"Destination already exists with a different size: {destination}. "
                        "Use --force to replace it or choose a clean --output-root."
                    )
            else:
                with zf.open(info, "r") as src, destination.open("wb") as dst:
                    while True:
                        block = src.read(1024 * 1024)
                        if not block:
                            break
                        dst.write(block)
            total_bytes += info.file_size
            manifest_rows.append({
                "split": split,
                "role": role,
                "machine_id": machine_id,
                "archive_member": info.filename,
                "extracted_file": str(destination),
                "uncompressed_bytes": info.file_size,
            })

    manifest_path = out_root / "subset_manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    print("MIMII subset extraction complete (original WAV files only; no generated audio).")
    print(f"Machine ID: {machine_id}")
    print(f"Normal training WAVs: {len(train_chosen)}")
    print(f"Normal test WAVs: {len(normal_chosen)}")
    print(f"Anomaly test WAVs: {len(anomaly_chosen)}")
    print(f"Selected uncompressed audio total: {total_bytes / (1024 * 1024):.1f} MiB")
    print(f"Subset folder: {out_root / 'fan'}")
    print(f"Manifest: {manifest_path}")
    print("Next command:")
    print(f'python train_controller_model.py --data-root "{out_root}" --machine-id {machine_id} --max-train-files {len(train_chosen)} --max-test-files {len(normal_chosen) + len(anomaly_chosen)} --segment-seconds 1.0')


if __name__ == "__main__":
    main()
