"""Utilities for reading CMIP6-like forcing files.

The CMIP6 forcing is expected as a CSV with columns
``timestamp, Ta_C, RH_pct, U_ms, P_Pa, Rs_Wm2, Ldown_Wm2, SZA_deg``
where each row represents one hourly sample.  This module exposes a
single helper :func:`read_cmip6_csv` that returns the data as a pandas
``DataFrame`` with a timezone-aware timestamp index.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd


def read_cmip6_csv(path: str | Path) -> pd.DataFrame:
    """Read a CMIP6-like forcing CSV into a :class:`~pandas.DataFrame`.

    Parameters
    ----------
    path: str or :class:`~pathlib.Path`
        Path to the CSV file.

    Returns
    -------
    :class:`~pandas.DataFrame`
        Data frame with the columns from the CSV. ``timestamp`` is parsed
        to ``datetime64[ns, UTC]``.
    """
    df = pd.read_csv(Path(path))
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


__all__ = ["read_cmip6_csv"]
