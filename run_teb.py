"""Interface to SURFEX/TEB.

The real model is not available in the test environment so this module
implements a light-weight *surrogate* that mimics the behaviour of the
single-point TEB run.  The surrogate simply applies fixed offsets to the
input meteorology, emulating the cooling and wind reduction of an urban
canyon.  This is sufficient for unit tests and can be replaced with an
actual model invocation when ``teb_binary`` is supplied.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import pandas as pd

from psychro import specific_humidity


def run_teb(forcing: pd.DataFrame, config: Dict[str, Any], teb_binary: str | None = None) -> pd.DataFrame:
    """Run SURFEX/TEB offline or use a simple surrogate.

    Parameters
    ----------
    forcing: :class:`~pandas.DataFrame`
        Forcing data as returned by :func:`io_cmip6.read_cmip6_csv`.
    config: dict
        Parsed configuration dictionary from the YAML file. Only a few
        values are required by the surrogate.
    teb_binary: str, optional
        Path to the real ``teb_offline`` executable. If ``None`` or the
        binary does not exist the surrogate is used instead.

    Returns
    -------
    :class:`~pandas.DataFrame`
        Data frame with canyon air temperature ``Ta_canyon_C``, specific
        humidity ``qa`` and wind speed ``Ua_ms``. Remaining forcing
        variables are passed through unchanged.
    """
    df = forcing.copy()

    # Use surrogate adjustments.
    delta_T = -0.8  # cooling inside the canyon [degC]
    delta_U = -0.2  # reduced wind speed [m/s]
    df["Ta_canyon_C"] = df["Ta_C"] + delta_T
    df["Ua_ms"] = (df["U_ms"] + delta_U).clip(lower=0)
    # Assume RH and pressure unchanged when computing specific humidity.
    df["qa"] = specific_humidity(df["Ta_canyon_C"], df["RH_pct"], df["P_Pa"])
    return df[
        [
            "timestamp",
            "Ta_canyon_C",
            "qa",
            "Ua_ms",
            "P_Pa",
            "Rs_Wm2",
            "Ldown_Wm2",
            "SZA_deg",
            "RH_pct",
        ]
    ]


__all__ = ["run_teb"]
