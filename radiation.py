"""Radiation utilities.

This module implements three small helpers:

* :func:`erbs_split` – partition global shortwave radiation into direct and
  diffuse components using the Erbs et al. (1982) correlation.
* :func:`projected_area_factor` – compute the projected area factor
  :math:`\phi(\mu)` for a standing human, where :math:`\mu` is the cosine of
  the solar zenith angle.
* :func:`shortwave_absorbed` – evaluate Eq. (3) of Fan & McColl (2024) for
  either ``"sun"`` or ``"shade"`` scenarios.
"""
from __future__ import annotations

import math
from datetime import datetime
from typing import Tuple
import pandas as pd


def _extraterrestrial_irradiance(ts: pd.Timestamp) -> float:
    """Return the extraterrestrial irradiance (W m⁻²).

    The expression follows the standard approximation with a varying
    Earth–Sun distance. ``ts`` must be timezone-aware.
    """
    day = ts.timetuple().tm_yday
    return 1367 * (1 + 0.033 * math.cos(2 * math.pi * day / 365))


def erbs_split(Rs: float, sza_deg: float, ts: pd.Timestamp) -> Tuple[float, float]:
    """Split global shortwave ``Rs`` (W m⁻²) into direct and diffuse parts.

    Parameters
    ----------
    Rs: float
        Global horizontal shortwave radiation [W m⁻²].
    sza_deg: float
        Solar zenith angle in degrees.
    ts: :class:`~pandas.Timestamp`
        Timestamp of the observation (UTC). Required for day-of-year.

    Returns
    -------
    tuple
        ``(R_b, R_d)`` – beam and diffuse components (W m⁻²).
    """
    mu = math.cos(math.radians(sza_deg))
    if mu <= 0:
        return 0.0, 0.0

    I0 = _extraterrestrial_irradiance(ts)
    kt = Rs / max(I0 * mu, 1e-6)
    kt = max(0.0, min(kt, 1.0))

    if kt <= 0.22:
        kd = 1 - 0.09 * kt
    elif kt <= 0.8:
        kd = 0.9511 - 0.1604 * kt + 4.388 * kt ** 2 - 16.638 * kt ** 3 + 12.336 * kt ** 4
    else:
        kd = 0.165

    Rd = kd * Rs
    Rb = Rs - Rd
    return Rb, Rd


def projected_area_factor(mu: float) -> float:
    """Projected area factor :math:`\phi(\mu)` for a standing cylinder.

    A simple linear form :math:`\phi = 0.308\mu + 0.478` is used, which
    reproduces values reported in urban climate texts.
    """
    return 0.308 * mu + 0.478


def shortwave_absorbed(Rb: float, Rd: float, Rg: float, alpha: float, phi: float, scenario: str) -> float:
    """Compute the absorbed shortwave radiation (Eq. 3).

    Parameters
    ----------
    Rb, Rd, Rg: float
        Beam, diffuse and ground-reflected shortwave fluxes (W m⁻²).
    alpha: float
        Shortwave body reflectance.
    phi: float
        Projected area factor.
    scenario: {"sun", "shade"}
        Exposure scenario.
    """
    absorp = 0.5 * Rd + 0.5 * Rg
    if scenario.lower() == "sun":
        absorp += Rb * phi
    return (1 - alpha) * absorp


__all__ = ["erbs_split", "projected_area_factor", "shortwave_absorbed"]
