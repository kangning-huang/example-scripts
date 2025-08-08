"""Psychrometric utilities used throughout the pipeline."""
from __future__ import annotations

import numpy as np

CP_AIR = 1004.0  # J kg-1 K-1
LAMBDA = 2.45e6  # J kg-1


def saturation_vapor_pressure(T_C: float | np.ndarray) -> np.ndarray:
    """Saturation vapour pressure (Pa) using the Tetens formula."""
    T_K = np.asarray(T_C) + 273.15
    return 611.2 * np.exp(17.62 * (T_K - 273.15) / (243.12 + (T_K - 273.15)))


def saturation_specific_humidity(T_C: float | np.ndarray, P_Pa: float | np.ndarray) -> np.ndarray:
    e_s = saturation_vapor_pressure(T_C)
    return 0.622 * e_s / (np.asarray(P_Pa) - 0.378 * e_s)


def specific_humidity(T_C: float | np.ndarray, RH_pct: float | np.ndarray, P_Pa: float | np.ndarray) -> np.ndarray:
    e_s = saturation_vapor_pressure(T_C)
    e = np.asarray(RH_pct) / 100.0 * e_s
    return 0.622 * e / (np.asarray(P_Pa) - 0.378 * e)


def saturation_specific_humidity_Ts(Ts_C: float, P_Pa: float | np.ndarray) -> np.ndarray:
    return saturation_specific_humidity(Ts_C, P_Pa)


__all__ = [
    "CP_AIR",
    "LAMBDA",
    "saturation_vapor_pressure",
    "saturation_specific_humidity",
    "specific_humidity",
]
