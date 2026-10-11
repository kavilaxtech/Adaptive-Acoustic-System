# MIMII experiment guide

## Objective

The AI is not a fan-fault detector. It recommends one initial FxNLMS configuration (step size and filter length) based on acoustic features. We test whether that recommendation improves the simulated outcome relative to a fixed baseline.

## Why the targets are simulation-derived

MIMII provides audio recordings, but not the reference microphone, error microphone and measured secondary path needed to train a model directly on measured ANC outcomes. We therefore run each candidate setting against each real source recording inside a controlled software ANC model. The utility is computed from that simulation. This avoids generated audio but does not eliminate assumptions from the simulated acoustic paths.

## Split protocol

- Training: normal fan recordings under the official MIMII `train/` directory.
- Validation: a file-level split of those training recordings.
- Test: official `test/` recordings held out from model fitting.
- Overlapping windows from a single recording are not created; each source file contributes one centred real-data segment.
- Test `normal_`/`anomaly_` labels are not model targets. They are not used to train fault detection.

## Selection utility

For each candidate:
- mark unstable runs infeasible;
- reject runs where control RMS exceeds the modelled primary-disturbance RMS by more than 6 dB;
- otherwise score the run using steady-state signal-level reduction in dB, with a small penalty if control RMS exceeds primary RMS.

This is a declared engineering objective for the simulator, not a universal ANC objective. Sensitivity-check the constraints and weights before final academic reporting.

## Metrics

This is a configuration-selection task, not a classification task. Do not report “accuracy” as the primary model metric. Report:

- mean and median selection regret versus the best candidate found by the same simulation on each held-out recording;
- fraction of recordings within 0.5 utility units of that simulation-only oracle;
- mean steady-state signal-level reduction for the fixed baseline and AI-selected setting;
- variability across recordings;
- stability failures and controller-to-primary RMS ratios.

The simulation-only oracle is a diagnostic upper bound under the chosen candidate grid; it is not an achievable measured physical result or a fair standalone system.

## Limits

The source waveform is real, but the primary/secondary FIR paths are assumed. A real-world ANC claim requires synchronized reference/error microphones and a measured secondary path, or a benchmark with those signals. A microphone recording alone cannot establish harvested mechanical/electrical energy either.

## Laptop-friendly subset

Use `prepare_mimii_subset.py` to extract only 80 normal training files and 15 normal + 15 anomalous test files for one machine ID from the official fan archive. This keeps the on-disk extracted dataset and model run small. The official `dev_data_fan.zip` archive is still about 1.4 GB and must be downloaded as distributed; delete it after successful extraction if disk space is limited.

```powershell
python prepare_mimii_subset.py --archive data/mimii/dev_data_fan.zip --output-root data/mimii --machine-id 00 --train-count 80 --test-per-class 15
python train_controller_model.py --data-root data/mimii --machine-id 00 --max-train-files 80 --max-test-files 30 --segment-seconds 1.0
```
