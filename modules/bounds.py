"""
Build the initial theta vector (theta0) and its optimizer bounds.
Supports both adstock types (instant/weibull) and both transformation
types (power/hill) with appropriate parameter bounds per spec.
"""

import numpy as np

from modules.params import safe_median


def _build_theta0_and_bounds(df, g):
    N_MEDIA = g["N_MEDIA"]; N_COMP = g["N_COMP"]
    N_OWN_NONMEDIA = g["N_OWN_NONMEDIA"]; N_COMP_NONMEDIA = g["N_COMP_NONMEDIA"]
    N_PRICE = g["N_PRICE"]; N_CROSS = g["N_CROSS"]; N_EFFECTORS = g["N_EFFECTORS"]
    MEDIA_COLS = g["MEDIA_COLS"]; COMP_MEDIA_COLS = g["COMP_MEDIA_COLS"]
    USE_ORGANIC_DRIFT = g["USE_ORGANIC_DRIFT"]
    ADSTOCK_TYPE  = g["ADSTOCK_TYPE"]
    TRANSFORM_TYPE = g["TRANSFORM_TYPE"]
    POSITIVE_BETA_COLS = g.get("POSITIVE_BETA_COLS", [])
    NEGATIVE_BETA_COLS = g.get("NEGATIVE_BETA_COLS", [])
    PCB = g.get("PER_CHANNEL_BOUNDS", {})

    def _b(col, param, lo_def, hi_def):
        if col in PCB and param in PCB[col]:
            return PCB[col][param]
        return (lo_def, hi_def)

    # ── Beta persistence ──────────────────────────────────────────────
    ls_bounds = [_b(c, "ls", 0.5, 1.0) for c in MEDIA_COLS]
    ls_init   = [np.clip(0.98, lo, hi) for lo, hi in ls_bounds]

    # G0 bound
    G0_bound = (0.7, 0.999999)
    G0_init  = 1.0

    # delta bounds: positive or negative constraint per media col
    def _delta_bound(col):
        if col in POSITIVE_BETA_COLS: return (0.0, None)
        if col in NEGATIVE_BETA_COLS: return (None, 0.0)
        return (None, None)
    delta_bounds = [_delta_bound(c) for c in MEDIA_COLS]
    delta_init   = np.full(N_MEDIA, 0.05)

    # gamma bounds (intercept effectors — always positive).
    # gamma is in TARGET units per period, so it is scaled to the target:
    # a fixed 0.02 is meaningless for a target in the millions and the
    # optimizer can't move it. Start small (so the model starts close to
    # "no boost") and cap the per-period boost at 5% of the target mean.
    y_scale = max(float(np.nanmean(np.abs(df[g["TARGET_COL"]].values))), 1e-6)
    gamma_bounds = [(0.0, 0.05 * y_scale)] * N_EFFECTORS
    gamma_init   = np.full(N_EFFECTORS, 1e-3 * y_scale / max(N_EFFECTORS, 1))

    # ── Transformation parameters ─────────────────────────────────────
    if TRANSFORM_TYPE == "power":
        # n: (0, 1) per spec
        n_bounds = [_b(c, "transform_n", 0.01, 1.0) for c in MEDIA_COLS]
        n_init   = [np.clip(0.5, lo, hi) for lo, hi in n_bounds]
        # S is unused for power but keep a slot so theta layout is consistent
        S_bounds = [(1e-6, 1e8)] * N_MEDIA
        S_init   = [safe_median(df[c]) for c in MEDIA_COLS]
    else:
        # Hill: n: (1, 15), S > 0 per spec
        n_bounds = [_b(c, "hill_n", 1.0, 15.0) for c in MEDIA_COLS]
        n_init   = [np.clip(2.0, lo, hi) for lo, hi in n_bounds]
        S_bounds = [_b(c, "hill_s", 1e-6, 1e8) for c in MEDIA_COLS]
        S_init   = [safe_median(df[c]) for c in MEDIA_COLS]

    # ── Intercept effector exponent ni ───────────────────────────────
    if TRANSFORM_TYPE == "power":
        n_int_bounds = [(0.01, 1.0)] * N_EFFECTORS
        n_int_init   = [0.5] * N_EFFECTORS
    else:
        n_int_bounds = [(1.0, 15.0)] * N_EFFECTORS
        n_int_init   = [2.0] * N_EFFECTORS

    # ── Intercept effector S (half-sat / scale) ──────────────────────
    # Effectors are transformed on x / x_ref (x_ref = max |x| on the train
    # split), so every effector lives on a comparable 0..1 scale and
    # S_intercept is in units of that scale (not raw units). This keeps
    # gamma / n / S well-conditioned no matter how large the raw variable is.
    INTERCEPT_EFFECTORS = g.get("INTERCEPT_EFFECTORS", [])
    g["INTERCEPT_XREF"] = {
        c: max(float(np.nanmax(np.abs(df[c].values.astype(float)))), 1e-12)
        for c in INTERCEPT_EFFECTORS
    }
    S_int_bounds = [(1e-3, 10.0)] * N_EFFECTORS
    S_int_init   = [
        float(np.clip(safe_median(df[c] / g["INTERCEPT_XREF"][c], default=0.5), 1e-2, 5.0))
        for c in INTERCEPT_EFFECTORS
    ]

    # ── Adstock parameters ────────────────────────────────────────────
    all_adstock_cols = list(MEDIA_COLS) + list(COMP_MEDIA_COLS)
    N_ADSTOCK = len(all_adstock_cols)

    if ADSTOCK_TYPE == "weibull":
        # shape k: (0.1, 5.0), scale lambda: (0.1, 5.0) — user can tighten via PCB
        shape_bounds = [_b(c, "adstock_shape", 0.1, 5.0) for c in all_adstock_cols]
        scale_bounds = [_b(c, "adstock_scale", 0.1, 5.0) for c in all_adstock_cols]
        adstock_init   = ([np.clip(1.5, lo, hi) for lo, hi in shape_bounds] +
                          [np.clip(1.0, lo, hi) for lo, hi in scale_bounds])
        adstock_bounds = shape_bounds + scale_bounds
    else:
        # instant (Nerlove-Arrow): lambda: (0, 1)
        lam_bounds = [_b(c, "adstock_lambda", 0.0, 0.99) for c in all_adstock_cols]
        adstock_init   = [np.clip(0.5, lo, hi) for lo, hi in lam_bounds]
        adstock_bounds = lam_bounds

    # ── Non-media / organic ───────────────────────────────────────────
    own_nm_ls_bounds    = [(0.8, 1.0)] * N_OWN_NONMEDIA
    comp_nm_ls_bounds   = [(0.8, 1.0)] * N_COMP_NONMEDIA
    def _nm_delta_bound(col):
        if col in POSITIVE_BETA_COLS: return (0.0, None)
        if col in NEGATIVE_BETA_COLS: return (None, 0.0)
        return (None, None)
    own_nm_delta_bounds = [_nm_delta_bound(c) for c in g["OWN_NONMEDIA_COLS"]]
    comp_nm_delta_bounds = [(None, 0)] * N_COMP_NONMEDIA

    # ── Competitor media ──────────────────────────────────────────────
    ls_comp_bounds  = [_b(c, "ls",     0.8, 1.0)  for c in COMP_MEDIA_COLS]
    n_comp_bounds   = [_b(c, "hill_n", 0.3, 5.0)  for c in COMP_MEDIA_COLS]
    s_comp_bounds   = [_b(c, "hill_s", 1e-6, 1e8) for c in COMP_MEDIA_COLS]

    # ── theta0 assembly ───────────────────────────────────────────────
    theta0 = np.concatenate([
        ls_init,
        [G0_init],
        delta_init,
        gamma_init,
        n_init,
        S_init,
        n_int_init,
        S_int_init,
        adstock_init,
        np.full(N_OWN_NONMEDIA, 0.98),
        np.full(N_COMP_NONMEDIA, 0.98),
        np.full(N_OWN_NONMEDIA, 0.01),
        np.full(N_COMP_NONMEDIA, -0.01),
        np.full(N_COMP, 0.98),
        np.full(N_COMP, 0.02),
        np.full(N_COMP, 1.5),
        [safe_median(df[c]) for c in COMP_MEDIA_COLS] if COMP_MEDIA_COLS else [],
        np.full(N_CROSS, 0.02),
        np.full(N_CROSS, 1.5),
        np.full(N_CROSS, 1.0),
        np.full(N_PRICE, 0.98),
        np.full(N_PRICE, -0.01),
        [0.0] if USE_ORGANIC_DRIFT else [],
        [max(df[g["TARGET_COL"]].std() * 0.3, 1e-3)],
    ])

    bounds = (
        ls_bounds +
        [G0_bound] +
        delta_bounds +
        gamma_bounds +
        n_bounds +
        S_bounds +
        n_int_bounds +
        S_int_bounds +
        adstock_bounds +
        own_nm_ls_bounds +
        comp_nm_ls_bounds +
        own_nm_delta_bounds +
        comp_nm_delta_bounds +
        ls_comp_bounds +
        [(None, 0)] * N_COMP +           # delta_comp
        n_comp_bounds +
        s_comp_bounds +
        [(0, None)] * N_CROSS +          # cross_delta
        [(0.3, 5.0)] * N_CROSS +         # cross_n
        [(1e-6, None)] * N_CROSS +       # cross_S
        [(0.8, 1.0)] * N_PRICE +         # Ls_price
        [(None, 0)] * N_PRICE +          # delta_price
        ([(-1.0, 1.0)] if USE_ORGANIC_DRIFT else []) +
        [(1e-3, None)]                   # sigma_y
    )
    return theta0, bounds


def build_normalized_problem(theta0, bounds, floor=1e-3, unbounded_mult=20.0):
    """
    Rescales theta into a per-parameter-normalized space before handing it
    to a gradient-based optimizer (L-BFGS-B / SLSQP), and returns an
    `unscale` function to map results back.

    WHY THIS EXISTS
    ----------------
    `theta` mixes parameters on wildly different natural scales in one flat
    vector — e.g. Hill's n ~ O(1-15), Ls/delta ~ O(0-1), S (half-saturation)
    ~ O(1e-6 to 1e8), sigma_y ~ O(target scale). scipy's L-BFGS-B/SLSQP
    approximate the gradient with forward differences using a single scalar
    step (`eps`, default ~1.49e-8) applied identically to every dimension.
    For small-range parameters like `n`, that absolute step is often
    smaller than the numerical noise floor of the Kalman filter/smoother
    recursion itself — the optimizer reads the resulting near-zero /
    noise-dominated finite-difference "gradient" as "no signal, stop
    moving this dimension", and the parameter just sits at its init value
    (theta0) forever, even though a real, non-flat optimum exists a bit
    further away. This was the direct cause of Hill's `n` consistently
    landing exactly on its init value (2.0) for every channel.

    HOW IT WORKS
    ------------
    - Fully-bounded dims [lo, hi] -> normalized to exactly [0, 1], so a
      single normalized `eps` corresponds to a step of `eps * (hi - lo)`
      in real units — proportional to that parameter's own natural range,
      instead of an arbitrary absolute number.
    - Partially/fully unbounded dims (e.g. unsigned betas, sigma_y's open
      upper end) don't have a real range to normalize by. Rather than
      inventing one arbitrary huge constant for all of them (that's
      exactly what made Nevergrad's own ±1e6 sentinel bounds misbehave —
      it let a couple of huge-range dimensions dominate the search), each
      one gets a window sized proportionally to its OWN starting value
      (`unbounded_mult * |theta0_i|`, floored so a zero-valued init still
      gets a usable window). Any real one-sided bound that *does* exist
      (e.g. gamma >= 0, sigma_y >= 1e-3) is preserved exactly — only the
      unconstrained side gets the proportional window, purely to condition
      the finite-difference step size, never to actually cap the search
      (its normalized bound stays None on that side, so scipy can still
      walk arbitrarily far from theta0 if the data wants it to).
    """
    n = len(theta0)
    lo_ref = np.empty(n)
    width = np.empty(n)
    norm_bounds = []
    for i, (lo, hi) in enumerate(bounds):
        x0 = float(theta0[i])
        if lo is not None and hi is not None:
            lo_ref[i] = lo
            width[i] = max(hi - lo, floor)
            norm_bounds.append((0.0, 1.0))
        else:
            w = unbounded_mult * max(abs(x0), floor)
            ref_lo = lo if lo is not None else x0 - w
            ref_hi = hi if hi is not None else x0 + w
            lo_ref[i] = ref_lo
            width[i] = max(ref_hi - ref_lo, floor)
            nb_lo = 0.0 if lo is not None else None
            nb_hi = (hi - ref_lo) / width[i] if hi is not None else None
            norm_bounds.append((nb_lo, nb_hi))

    theta0_norm = np.clip((theta0 - lo_ref) / width, 0.0, 1.0)

    def unscale(theta_norm):
        return lo_ref + np.asarray(theta_norm) * width

    return theta0_norm, norm_bounds, unscale
