"""
Transformation functions: Hill saturation curve and adstock (carry-over) models.
"""

import numpy as np
import pandas as pd


def hill_transform_vec(x: np.ndarray, n: float, S: float) -> np.ndarray:
    x = np.maximum(x, 0.0)
    xn = x ** n
    return xn / (xn + S ** n + 1e-30)


def hill_transform(x: float, n: float, S: float) -> float:
    if x <= 0:
        return 0.0
    return (x ** n) / (x ** n + S ** n + 1e-30)


def adstock_nerlove_arrow(x: pd.Series, lam: float) -> np.ndarray:
    arr = x.values.astype(float)
    out = np.empty_like(arr)
    out[0] = arr[0] / (1.0 - lam + 1e-8)
    for t in range(1, len(arr)):
        out[t] = arr[t] + lam * out[t - 1]
    return out


def adstock_weibull(x: pd.Series, alpha: float, theta: float) -> np.ndarray:
    L = len(x)
    t_idx = np.arange(1, L + 1, dtype=float)
    weights = (alpha / theta) * ((t_idx / theta) ** (alpha - 1)) * np.exp(-((t_idx / theta) ** alpha))
    weights = weights / (weights.sum() + 1e-12)
    return np.convolve(x.values, weights, mode='full')[:L]


def _adstock(series: pd.Series, adstock_type: str, params_dict: dict, chan_idx: int) -> np.ndarray:
    if adstock_type == "weibull":
        return adstock_weibull(series, params_dict["adstock_alpha"][chan_idx],
                               params_dict["adstock_theta"][chan_idx])
    return adstock_nerlove_arrow(series, params_dict["adstock_lambda"][chan_idx])
