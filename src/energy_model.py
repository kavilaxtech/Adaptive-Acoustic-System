"""Transparent energy-budget calculations based on user-supplied assumptions.

This is an engineering sensitivity model, not a measurement of harvested power.
The project does not generate fake vibration or electrical-output samples.
"""
from __future__ import annotations


def estimate_energy_budget(
    normalized_vibration_amplitude: float,
    assumed_input_power_mw_at_unit_amplitude: float,
    net_conversion_efficiency: float,
    load_power_mw: float,
    duration_seconds: float,
    initial_storage_mj: float,
    storage_capacity_mj: float,
) -> dict:
    """Calculate a transparent modelled energy budget.

    Input power and normalized amplitude are supplied as explicit assumptions.
    Assumed input power scales quadratically with normalized amplitude. This is
    a user-selected approximation, not a transducer-calibrated equation.
    Since mW * seconds = mJ, units remain internally consistent.
    """
    amp = float(normalized_vibration_amplitude)
    p_unit = float(assumed_input_power_mw_at_unit_amplitude)
    eta = float(net_conversion_efficiency)
    load = float(load_power_mw)
    duration = float(duration_seconds)
    capacity = float(storage_capacity_mj)
    initial = float(initial_storage_mj)
    if not 0.0 <= amp <= 1.0:
        raise ValueError("Normalized vibration amplitude must be between 0 and 1")
    if p_unit < 0 or load < 0 or duration <= 0 or capacity < 0:
        raise ValueError("Power/capacity must be non-negative and duration must be positive")
    if not 0.0 <= eta <= 1.0:
        raise ValueError("Net conversion efficiency must be between 0 and 1")
    if not 0.0 <= initial <= capacity:
        raise ValueError("Initial storage must be between zero and storage capacity")

    p_in = p_unit * amp**2
    p_harvested = p_in * eta
    harvested_mj = p_harvested * duration
    load_mj = load * duration
    unbounded_final = initial + harvested_mj - load_mj
    final_storage = min(capacity, max(0.0, unbounded_final))
    unmet_load = max(0.0, -unbounded_final)
    spilled_energy = max(0.0, unbounded_final - capacity)

    return {
        "normalized_vibration_amplitude": amp,
        "assumed_input_power_mw_at_unit_amplitude": p_unit,
        "assumed_net_conversion_efficiency": eta,
        "modelled_input_power_mw": p_in,
        "modelled_harvested_power_mw": p_harvested,
        "modelled_harvested_energy_mj": harvested_mj,
        "assumed_load_power_mw": load,
        "assumed_load_energy_mj": load_mj,
        "initial_storage_mj": initial,
        "modelled_final_storage_mj": final_storage,
        "unmet_load_energy_mj": unmet_load,
        "energy_above_storage_capacity_mj": spilled_energy,
        "duration_seconds": duration,
        "evidence_type": "modelled estimate from user-provided assumptions; not physical measurement",
    }
