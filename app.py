
from __future__ import annotations

import io
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import streamlit as st

from config import BASELINE_FILTER_LENGTH, BASELINE_MU, MODEL_PATH, RESULTS_DIR
from src.acoustic_features import FEATURE_NAMES, extract_features, get_spectrum
from src.anc_simulation import run_fxnlms_on_recording
from src.audio_io import center_crop
from src.energy_model import estimate_energy_budget
from src.selector import select_configuration

st.set_page_config(page_title="Adaptive Acoustic System", page_icon="🔊", layout="wide")
st.title("AI-Driven Adaptive Acoustic System")
st.caption("Real-recording acoustic analysis · AI-guided ANC configuration · vibration-energy estimation")

# st.info(
#     "Use a real industrial WAV recording, such as a MIMII fan sample. The ANC result is a software "
#     "simulation using assumed FIR acoustic paths; it is not a measured physical noise-cancellation result. "
#     "The energy calculation is a separate estimate based on assumptions you enter."
# )

if not MODEL_PATH.is_file():
    st.warning("The ANC configuration-selection model has not been trained yet.")
    st.code("python train_controller_model.py --data-root data/mimii --machine-id 00")
    st.stop()

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
candidates = bundle["candidate_configurations"]
segment_seconds = float(bundle.get("segment_seconds", 1.0))

uploaded = st.file_uploader("Upload a real WAV recording", type=["wav"])
if uploaded is not None:
    try:
        raw, sr = sf.read(io.BytesIO(uploaded.getvalue()), dtype="float32", always_2d=False)
        if raw.ndim == 2:
            raw = np.mean(raw, axis=1, dtype=np.float64).astype(np.float32)
        if raw.ndim != 1 or raw.size == 0 or not np.all(np.isfinite(raw)):
            raise ValueError("Audio must be a non-empty, finite mono/stereo WAV recording.")
        segment = center_crop(raw, int(sr), segment_seconds)
        st.success(f"Loaded recording: {uploaded.name} | sample rate: {sr} Hz | selected segment: {segment_seconds:.2f}s")

        features = extract_features(segment, int(sr))
        chosen, predicted_scores = select_configuration(model, features, candidates)
        feature_table = {name: float(value) for name, value in zip(FEATURE_NAMES, features)}

        baseline = run_fxnlms_on_recording(
            segment, int(sr), mu=BASELINE_MU, filter_len=BASELINE_FILTER_LENGTH
        )
        selected = run_fxnlms_on_recording(
            segment, int(sr), mu=chosen["mu"], filter_len=chosen["filter_len"]
        )

        st.subheader("1. Acoustic analysis of the uploaded recording")
        c1, c2, c3 = st.columns(3)
        c1.metric("Dominant frequency", f"{feature_table['dominant_frequency_hz']:.1f} Hz")
        c2.metric("Spectral centroid", f"{feature_table['spectral_centroid_hz']:.1f} Hz")
        c3.metric("RMS amplitude", f"{10 ** feature_table['log_rms']:.5f} (feature scale)")

        left, right = st.columns(2)
        with left:
            fig, ax = plt.subplots(figsize=(7, 3.5))
            time_axis = np.arange(segment.size) / int(sr)
            ax.plot(time_axis, segment, linewidth=0.65)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude")
            ax.set_title("Uploaded audio waveform")
            ax.grid(alpha=0.25)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        with right:
            freq, mag = get_spectrum(segment, int(sr))
            fig, ax = plt.subplots(figsize=(7, 3.5))
            ax.plot(freq, mag, linewidth=0.75)
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Magnitude (arbitrary scale)")
            ax.set_title("FFT magnitude spectrum")
            ax.set_xlim(0, min(int(sr) / 2, 8000))
            ax.grid(alpha=0.25)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        st.subheader("2. AI-selected ANC configuration")
        st.write(
            "The Random Forest predicts a simulation-derived utility score for each candidate controller setting. "
            "The highest-scoring setting is selected; the result is a recommendation for this simulation, not a guarantee."
        )
        m1, m2, m3 = st.columns(3)
        m1.metric("Recommended step size (μ)", f"{chosen['mu']:.3f}")
        m2.metric("Filter length", f"{chosen['filter_len']} coefficients")
        m3.metric("Predicted utility", f"{chosen['predicted_utility']:.3f}")

        st.subheader("3. ANC simulation comparison")
        st.caption("Both runs use the same uploaded recording but assumed primary/secondary FIR paths.")
        r1, r2, r3 = st.columns(3)
        r1.metric("Baseline reduction", f"{baseline['steady_state_noise_reduction_db']:.2f} dB")
        r2.metric("AI-selected reduction", f"{selected['steady_state_noise_reduction_db']:.2f} dB")
        r3.metric("Difference vs baseline", f"{selected['steady_state_noise_reduction_db'] - baseline['steady_state_noise_reduction_db']:+.2f} dB")
        st.write(
            f"Baseline: μ={BASELINE_MU}, filter length={BASELINE_FILTER_LENGTH}. "
            f"AI-selected: μ={chosen['mu']}, filter length={chosen['filter_len']}."
        )

        plot_limit = min(len(baseline["primary"]), int(baseline["sample_rate"] * 0.25))
        t = np.arange(plot_limit) / baseline["sample_rate"]
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(t, baseline["primary"][:plot_limit], label="Modelled primary disturbance", linewidth=0.9)
        ax.plot(t, baseline["residual"][:plot_limit], label="Baseline residual", linewidth=0.8)
        ax.plot(t, selected["residual"][:plot_limit], label="AI-selected residual", linewidth=0.8)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Normalised simulation amplitude")
        ax.set_title("First 250 ms: primary and simulated residual signals")
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("4. Extracted features")
        st.dataframe({"Feature": list(feature_table.keys()), "Value": list(feature_table.values())}, use_container_width=True, hide_index=True)

    except Exception as exc:
        st.error(f"Could not process this recording: {exc}")
else:
    st.write("Upload a real WAV file to inspect its frequencies and run the ANC configuration selector. No sample WAVs are generated or bundled.")

st.divider()
st.subheader("5. Vibration-energy estimate")
st.write(
    "Enter assumptions from a suitable vibration source or measurement. The current software estimates energy "
    "from these inputs; it does not infer electrical output from the microphone recording."
)
with st.form("energy_form"):
    left, mid, right = st.columns(3)
    with left:
        amplitude = st.number_input("Normalized vibration amplitude (0–1)", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
        assumed_input_power = st.number_input("Assumed input power at amplitude 1 (mW)", min_value=0.0, value=10.0, step=1.0)
        efficiency = st.number_input("Assumed net conversion efficiency (0–1)", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
    with mid:
        load_power = st.number_input("Assumed load power (mW)", min_value=0.0, value=1.0, step=0.25)
        duration = st.number_input("Observation duration (seconds)", min_value=0.01, value=60.0, step=10.0)
    with right:
        storage_capacity = st.number_input("Virtual storage capacity (mJ)", min_value=0.0, value=100.0, step=10.0)
        initial_storage = st.number_input("Initial stored energy (mJ)", min_value=0.0, value=20.0, step=5.0)
    submitted = st.form_submit_button("Calculate modelled energy budget")

if submitted:
    try:
        energy = estimate_energy_budget(
            normalized_vibration_amplitude=amplitude,
            assumed_input_power_mw_at_unit_amplitude=assumed_input_power,
            net_conversion_efficiency=efficiency,
            load_power_mw=load_power,
            duration_seconds=duration,
            initial_storage_mj=initial_storage,
            storage_capacity_mj=storage_capacity,
        )
        e1, e2, e3 = st.columns(3)
        e1.metric("Modelled harvested power", f"{energy['modelled_harvested_power_mw']:.4f} mW")
        e2.metric("Modelled harvested energy", f"{energy['modelled_harvested_energy_mj']:.4f} mJ")
        e3.metric("Modelled final storage", f"{energy['modelled_final_storage_mj']:.4f} mJ")
        st.json(energy)
    except Exception as exc:
        st.error(str(exc))

st.caption("Evidence boundary: the audio input may be real MIMII/user audio; ANC paths and residual are simulated; energy values are modelled from user-entered assumptions.")
