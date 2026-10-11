# AI-Driven Adaptive Acoustic System

A software-only proof of concept for industrial audio feature extraction, AI-based ANC configuration selection, FxNLMS-style adaptive noise-control simulation and vibration-energy estimation.

- Real WAV inputs only; no synthetic audio generator is included.
- MIMII audio is used for acoustic features and as input to the controller-simulation experiments.
- The ANC simulation uses assumed FIR acoustic paths because MIMII does not include a measured ANC reference/error pair or measured secondary path.
- The energy module is assumption-driven and does not claim physical harvesting measurements.

See `README_FIRST.md` for installation, dataset preparation, training and dashboard instructions.
