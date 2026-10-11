# Start here — AI-Driven Adaptive Acoustic System

This version does **not** create or include synthetic audio samples. It requires actual WAV recordings from the MIMII fan dataset or another real industrial recording supplied by you.

## What this version does

1. Extracts spectral/statistical features from recorded audio using FFT/STFT.
2. Runs a baseline FxNLMS-style ANC algorithm using the recorded audio as the excitation source.
3. Trains a Random Forest regressor to recommend an initial ANC configuration from the acoustic features.
4. Estimates vibration-energy availability from explicit user-entered assumptions.
5. Provides a Streamlit dashboard for uploaded WAV files and plots.

## Important scientific limitation

The MIMII dataset contains machine-audio recordings, not measured ANC reference/error microphone pairs and measured secondary acoustic paths. The trainer therefore measures candidate settings in a clearly labelled software ANC simulation driven by actual MIMII recordings and **assumed FIR paths**. The AI is trained on these simulation-derived outcomes. This is not evidence of physical factory noise cancellation. A hardware or suitable measured ANC dataset is needed to validate physical attenuation.

The energy model is also an estimate based on parameters entered by the user. It is not a measurement of harvested electrical power. No vibration or electrical data are fabricated by this code.

## 1. Use a small subset of real data (recommended)

The official DCASE 2020 fan archive is about 1.4 GB. You do **not** need to extract all WAV files or train on the complete dataset for the first run. The trainer already limits the pilot to at most 80 normal training files and 30 test files. The helper below extracts only those selected WAVs from the archive, which keeps the working dataset much smaller.

Download only the fan archive from the official dataset page:

https://zenodo.org/records/3678171

Keep the archive, for example, at `data/mimii/dev_data_fan.zip`. Then run the following command from the project root:

```powershell
python prepare_mimii_subset.py --archive data/mimii/dev_data_fan.zip --output-root data/mimii --machine-id 00 --train-count 80 --test-per-class 15
```

This extracts only 80 normal training WAVs plus 15 normal and 15 anomalous test WAVs for the chosen machine ID. These are original MIMII recordings; the helper does not generate audio. The subset is written in this format:

```text
data/mimii/fan/
├── train/   (80 normal recordings)
└── test/    (15 normal + 15 anomaly recordings)
```

Then run the trainer:

```powershell
python train_controller_model.py --data-root data/mimii --machine-id 00 --max-train-files 80 --max-test-files 30 --segment-seconds 1.0
```

The ZIP download itself is still about 1.4 GB because the official fan data is distributed as one archive; selective extraction reduces the **disk space used after extraction** and the amount of data processed. Once the subset has been extracted and you have confirmed it works, you can delete the downloaded ZIP to recover disk space. If the archive download itself is too large, ask before downloading it—we can discuss using a compact teaching subset, which is smaller but is not the official DCASE benchmark split.

## 2. Install packages

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell activation is blocked, use Command Prompt:

```bat
.venv\Scripts\activate.bat
```

## 3. Train the ANC-configuration selector

```bash
python train_controller_model.py --data-root data/mimii --machine-id 00
```

The default pilot uses up to 80 normal training WAV recordings and 30 held-out test WAVs. To run over all available recordings (may take much longer):

```bash
python train_controller_model.py --data-root data/mimii --machine-id 00 --max-train-files 0 --max-test-files 0
```

To include all fan machine IDs:

```bash
python train_controller_model.py --data-root data/mimii --machine-id all
```

The model is fitted only on normal recordings from the official training directory. The official test recordings are held out. Their normal/anomaly names are not model targets; they are only source audio conditions for testing controller selection.

## 4. Inspect results

Training saves:

- `model/anc_config_selector.joblib` — trained selector and metadata
- `results/controller_selection_test.csv` — per-recording selected vs baseline results
- `results/controller_selection_summary.json` — aggregate test results and caveats
- `results/controller_selection_comparison.png` — comparison plot using test recordings
- `results/hyperparameter_validation.csv` — validation-only hyperparameter selection
- `results/dataset_manifest.csv` — files used and any files skipped

Do not describe the outputs as measured real-world noise attenuation. They are simulation results using actual recording waveforms and assumed acoustic paths.

## 5. Run the dashboard

```bash
streamlit run app.py
```

Upload a real WAV recording. The dashboard shows the waveform, FFT, features, AI-recommended ANC configuration, simulated baseline/selected residuals and a separate modelled energy budget.

## What the model learns

For each real recording used in the development set, the trainer evaluates a predefined grid of FxNLMS settings under the same assumed FIR paths. The utility score rewards simulated steady-state signal-level reduction, rejects unstable configurations, and penalizes excessive control output. A Random Forest regressor learns to estimate that utility from acoustic features plus candidate settings. For an unseen recording, the candidate with the highest predicted utility is recommended.

This design does not create generated audio files. The controller targets are simulation-derived and must be validated against measured ANC behaviour before a physical-performance claim is made.
