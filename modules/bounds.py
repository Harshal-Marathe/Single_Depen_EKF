"""
Build the initial theta vector (theta0) and its optimizer bounds.
Per-channel-only bounds — there is intentionally no global bounds panel.
"""

import numpy as np

from modules.params import safe_median


def _build_theta0_and_bounds(df, g):
    N_MEDIA = g["N_MEDIA"]; N_COMP = g["N_COMP"]
    N_OWN_NONMEDIA = g["N_OWN_NONMEDIA"]; N_COMP_NONMEDIA = g["N_COMP_NONMEDIA"]
    N_PRICE = g["N_PRICE"]; N_CROSS = g["N_CROSS"]; N_EFFECTORS = g["N_EFFECTORS"]
    MEDIA_COLS = g["MEDIA_COLS"]; COMP_MEDIA_COLS = g["COMP_MEDIA_COLS"]
    USE_ORGANIC_DRIFT = g["USE_ORGANIC_DRIFT"]; ADSTOCK_TYPE = g["ADSTOCK_TYPE"]
    POSITIVE_BETA_COLS = g.get("POSITIVE_BETA_COLS", [])
    PCB = g.get("PER_CHANNEL_BOUNDS", {})

    def _b(col, param, lo_def, hi_def):
        if col in PCB and param in PCB[col]:
            return PCB[col][param]
        return (lo_def, hi_def)

    ls_bounds   = [_b(c, "ls",     0.5,  1.0)  for c in MEDIA_COLS]
    ls_init     = [np.clip(0.98, lo, hi)        for lo, hi in ls_bounds]
    delta_bounds= [(0.0, None)                  for _ in MEDIA_COLS]
    n_bounds    = [_b(c, "hill_n", 0.3,  5.0)  for c in MEDIA_COLS]
    n_init      = [np.clip(1.5, lo, hi)         for lo, hi in n_bounds]
    s_bounds    = [_b(c, "hill_s", 1e-6, 1e8)  for c in MEDIA_COLS]
    s_init      = [safe_median(df[c])           for c in MEDIA_COLS]

    all_adstock_cols = list(MEDIA_COLS) + list(COMP_MEDIA_COLS)
    if ADSTOCK_TYPE == "weibull":
        a_bounds = [_b(c, "adstock_alpha", 0.01, 3.0) for c in all_adstock_cols]
        t_bounds = [_b(c, "adstock_theta", 0.01, 3.0) for c in all_adstock_cols]
        adstock_init        = [np.clip(0.5, lo, hi) for lo, hi in a_bounds] + \
                              [np.clip(0.5, lo, hi) for lo, hi in t_bounds]
        adstock_bounds_list = a_bounds + t_bounds
    else:
        lam_bounds = [_b(c, "adstock_lambda", 0.0, 0.99) for c in all_adstock_cols]
        adstock_init        = [np.clip(0.5, lo, hi) for lo, hi in lam_bounds]
        adstock_bounds_list = lam_bounds

    ls_comp_bounds = [_b(c, "ls",     0.8,  1.0)  for c in COMP_MEDIA_COLS]
    n_comp_bounds  = [_b(c, "hill_n", 0.3,  5.0)  for c in COMP_MEDIA_COLS]
    s_comp_bounds  = [_b(c, "hill_s", 1e-6, 1e8)  for c in COMP_MEDIA_COLS]
    own_nm_delta_bounds = [(0.0, None) for _ in g["OWN_NONMEDIA_COLS"]]

    theta0 = np.concatenate([
        ls_init, [1.0], np.full(N_MEDIA, 0.05), np.full(N_EFFECTORS, 0.02),
        n_init, s_init, adstock_init,
        np.full(N_OWN_NONMEDIA, 0.98), np.full(N_COMP_NONMEDIA, 0.98),
        np.full(N_OWN_NONMEDIA, 0.01), np.full(N_COMP_NONMEDIA, -0.01),
        np.full(N_COMP, 0.98), np.full(N_COMP, 0.02), np.full(N_COMP, 1.5),
        [safe_median(df[c]) for c in COMP_MEDIA_COLS] if COMP_MEDIA_COLS else [],
        np.full(N_CROSS, 0.02), np.full(N_CROSS, 1.5), np.full(N_CROSS, 1.0),
        np.full(N_PRICE, 0.98), np.full(N_PRICE, -0.01),
        [0.0] if USE_ORGANIC_DRIFT else [],
        [max(df[g["TARGET_COL"]].std() * 0.3, 1e-3)],
    ])

    bounds = (
        ls_bounds + [(0.7, 0.999999)] + delta_bounds +
        [(0, None)] * N_EFFECTORS + n_bounds + s_bounds + adstock_bounds_list +
        [(0.8, 1.0)] * N_OWN_NONMEDIA + [(0.8, 1.0)] * N_COMP_NONMEDIA +
        own_nm_delta_bounds + [(None, 0)] * N_COMP_NONMEDIA +
        ls_comp_bounds + [(None, 0)] * N_COMP +
        n_comp_bounds + s_comp_bounds +
        [(0, None)] * N_CROSS + [(0.3, 5.0)] * N_CROSS + [(1e-6, None)] * N_CROSS +
        [(0.8, 1.0)] * N_PRICE + [(None, 0)] * N_PRICE +
        ([(-1.0, 1.0)] if USE_ORGANIC_DRIFT else []) + [(1e-3, None)]
    )
    return theta0, bounds
