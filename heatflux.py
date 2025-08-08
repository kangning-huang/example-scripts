"""Heat-flux helper functions for the human energy-balance model."""
from __future__ import annotations

import numpy as np

from psychro import CP_AIR

SIGMA = 5.670374419e-8  # Stefan-Boltzmann constant


def h_c(U_ms: float | np.ndarray) -> np.ndarray:
    """Convective heat-transfer coefficient (Eq. 6)."""
    U = np.asarray(U_ms)
    return 1.4 * np.sqrt(np.maximum(U, 0))


def lw_in(Ldown: float, Lg: float, eps_s: float, sky_partition: float = 0.5) -> float:
    """Incoming longwave to the body (Eq. 4)."""
    return eps_s * (sky_partition * Lg + (1 - sky_partition) * Ldown)


def lw_out(Ts_C: float, eps_s: float) -> float:
    """Outgoing longwave from the skin (Eq. 5)."""
    T_K = Ts_C + 273.15
    return eps_s * SIGMA * T_K ** 4


def lambda_E_o(h_e: float, q_s_Ts: float, q_a: float) -> float:
    """Potential evaporative heat loss (Eq. 7)."""
    return (h_e / CP_AIR) * (q_s_Ts - q_a)


def lambda_E(lambda_E_o: float, lambdaE_max: float) -> float:
    """Actual evaporative heat loss limited by ``lambdaE_max`` (Eq. 8)."""
    return float(min(lambda_E_o, lambdaE_max))


__all__ = ["h_c", "lw_in", "lw_out", "lambda_E_o", "lambda_E"]
