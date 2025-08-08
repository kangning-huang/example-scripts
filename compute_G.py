"""Computation of the human heat-storage rate ``G``.

The equations implemented here follow Fan & McColl (2024).  The key
relations are reproduced from their Methods section:

* Eq. (1): ``G = R_n - H - lambdaE + M``
* Eq. (2): ``R_n = f_s * (R_in + L_in - L_out)``
* Eq. (3): short-wave absorption handled via :func:`radiation.shortwave_absorbed`.
* Eq. (4)-(5): long-wave terms via :func:`heatflux.lw_in` and :func:`heatflux.lw_out`.
* Eq. (6): convective heat flux ``H``.
* Eq. (7)-(8): evaporative heat loss ``lambdaE`` with a ceiling ``lambdaE_max``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict
import pandas as pd
import numpy as np

from radiation import erbs_split, projected_area_factor, shortwave_absorbed
from psychro import CP_AIR, specific_humidity, saturation_specific_humidity
from heatflux import h_c, lw_in, lw_out, lambda_E_o, lambda_E


@dataclass
class Config:
    human: Dict[str, Any]
    ground: Dict[str, Any]


def compute_G(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Compute all heat-budget terms and ``G``.

    Parameters
    ----------
    df: DataFrame
        Data from :func:`run_teb.run_teb` (one or more hours).
    cfg: dict
        Configuration dictionary (parsed from YAML).
    """
    human = cfg["human"]
    ground = cfg["ground"]

    alpha = human["alpha_shortwave"]
    eps = human["skin_emissivity"]
    fs = human["fs_radiation"]
    Ts = human["Ts_C"]
    lambdaE_max = human["lambdaE_max_Wm2"]
    M = human["metabolic_M_Wm2"]
    scenario = human["scenario"]

    out = []
    for row in df.itertuples(index=False):
        Rb, Rd = erbs_split(row.Rs_Wm2, row.SZA_deg, row.timestamp)
        Rg = ground["albedo"] * row.Rs_Wm2
        mu = max(0.0, np.cos(np.radians(row.SZA_deg)))
        phi = projected_area_factor(mu)
        Rin = shortwave_absorbed(Rb, Rd, Rg, alpha, phi, scenario)
        Lin = lw_in(row.Ldown_Wm2, row.Ldown_Wm2, eps, ground["sky_LW_partition"])
        Lout = lw_out(Ts, eps)
        Rn = fs * (Rin + Lin - Lout)
        qc = row.qa
        qs = saturation_specific_humidity(Ts, row.P_Pa)
        hc = h_c(row.Ua_ms)
        H = hc * (Ts - row.Ta_canyon_C)
        he = hc
        lambdaE0 = lambda_E_o(he, qs, qc)
        LE = lambda_E(lambdaE0, lambdaE_max)
        G = Rn - H - LE + M
        uncomp = G > 0
        out.append(
            {
                "timestamp": row.timestamp,
                "Ta_canyon_C": row.Ta_canyon_C,
                "qa": qc,
                "Ua_ms": row.Ua_ms,
                "Rin": Rin,
                "Lin": Lin,
                "Lout": Lout,
                "Rn": Rn,
                "H": H,
                "lambdaE": LE,
                "M": M,
                "G_Wm2": G,
                "uncompensable": uncomp,
            }
        )
    return pd.DataFrame(out)


__all__ = ["compute_G", "Config"]
