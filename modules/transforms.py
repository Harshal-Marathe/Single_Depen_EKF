"""
Transformation functions: Hill saturation curve, power transformation,
and adstock (carry-over) models including Weibull with per-lag weights.
"""

import numpy as np
import pandas as pd


# ── Hill transformation ─────────────────────────────────────────────────────

def hill_transform_vec(x: np.ndarray, n: float, S: float) -> np.ndarray:
    x = np.maximum(x, 0.0)
    xn = x ** n
    return xn / (xn + S ** n + 1e-30)


def hill_transform(x: float, n: float, S: float) -> float:
    if x <= 0:
        return 0.0
    return (x ** n) / (x ** n + S ** n + 1e-30)


# ── Power transformation ────────────────────────────────────────────────────

def power_transform_vec(x: np.ndarray, n: float) -> np.ndarray:
    """Element-wise power transform: x^n, clamped to non-negative inputs."""
    return np.maximum(x, 0.0) ** n


def power_transform(x: float, n: float) -> float:
    if x <= 0:
        return 0.0
    return x ** n


# ── Adstock: Nerlove-Arrow (geometric / instant) ────────────────────────────

def adstock_nerlove_arrow(x: pd.Series, lam: float) -> np.ndarray:
    arr = x.values.astype(float)
    out = np.empty_like(arr)
    decay_for_seed = min(lam, 0.9)
    out[0] = arr[0] / (1.0 - decay_for_seed + 1e-8)
    for t in range(1, len(arr)):
        out[t] = arr[t] + lam * out[t - 1]
    return out


# ── Adstock: Weibull with per-lag weights (delayed / distributed) ───────────

def weibull_lag_weights(shape_k: float, scale_lam: float, n_lags: int) -> np.ndarray:
    """
    Compute normalised Weibull PDF weights for lag = 0, 1, …, n_lags-1.

    Formula (per spec):
        w_lag = (k / λ) * ((lag + 1) / λ)^(k - 1)
    No exp term — this is the discrete Weibull PDF evaluated at
    each (lag+1), then normalised to sum to 1.
    """
    lags = np.arange(1, n_lags + 1, dtype=float)          # lag+1 for lag=0..n_lags-1
    w = (shape_k / scale_lam) * ((lags / scale_lam) ** (shape_k - 1))
    w = np.maximum(w, 0.0)
    total = w.sum()
    if total < 1e-12:
        w = np.ones(n_lags) / n_lags
    else:
        w /= total
    return w


def adstock_weibull_lagged(x: pd.Series,
                            shape_k: float,
                            scale_lam: float,
                            n_lags: int) -> np.ndarray:
    """
    Weighted sum of lagged raw spend using Weibull-derived weights.

    adstocked[t] = Σ_{l=0}^{n_lags-1}  w_l * x[t - l]
    """
    arr = x.values.astype(float)
    T = len(arr)
    weights = weibull_lag_weights(shape_k, scale_lam, n_lags)
    out = np.zeros(T)
    for l, w in enumerate(weights):
        if w < 1e-15:
            continue
        if l == 0:
            out += w * arr
        else:
            out[l:] += w * arr[:-l]
    return out


# ── Dispatcher ───────────────────────────────────────────────────────────────

def _adstock(series: pd.Series, adstock_type: str, params_dict: dict,
             chan_idx: int) -> np.ndarray:
    if adstock_type == "weibull":
        n_lags = int(params_dict.get("adstock_n_lags", 8))
        return adstock_weibull_lagged(
            series,
            params_dict["adstock_shape"][chan_idx],
            params_dict["adstock_scale"][chan_idx],
            n_lags,
        )
    # instant / Nerlove-Arrow
    return adstock_nerlove_arrow(series, params_dict["adstock_lambda"][chan_idx])


def apply_transformation(x: np.ndarray, transform_type: str,
                          n: float, S: float = 1.0) -> np.ndarray:
    """
    Apply either 'power' or 'hill' transformation to array x.
    For power: returns x^n
    For hill:  returns x^n / (x^n + S^n)
    """
    if transform_type == "hill":
        return hill_transform_vec(x, n, S)
    return power_transform_vec(x, n)
