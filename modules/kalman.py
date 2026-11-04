"""
Extended Kalman Filter core: observation/transition/process-noise matrix
builders, the forward filter, and the RTS (Rauch-Tung-Striebel) smoother.

State equations (all 4 adstock × transform combinations):

Instant + Power:
  β_i,t = λ_i · β_i,t-1  +  δ_i · x_i,t^n_i
           + Σ_j∈synergy  δ_ij · x_j,t^n_j

Instant + Hill:
  β_i,t = λ_i · β_i,t-1  +  δ_i · Hill(x_i,t; n_i, S_i)
           + Σ_j∈synergy  δ_ij · Hill(x_j,t; n_j, S_j)

Weibull + Power:
  β_i,t = Σ_{l=0}^{L-1} w_l · x_i,t-l   (weighted lag sum)
           +  δ_i · x_i,t^n_i
           + Σ_j∈synergy  δ_ij · x_j,t^n_j

Weibull + Hill:
  β_i,t = Σ_{l=0}^{L-1} w_l · x_i,t-l
           +  δ_i · Hill(x_i,t; n_i, S_i)
           + Σ_j∈synergy  δ_ij · Hill(x_j,t; n_j, S_j)

Intercept (all modes):
  I_t = G0 · I_{t-1}  +  Σ_k γ_k · media_k,t^{n_k_intercept}
"""

import numpy as np
import pandas as pd

from modules.transforms import (
    apply_transformation,
    hill_transform_vec,
    adstock_weibull_lagged,
    adstock_nerlove_arrow,
    weibull_lag_weights,
)


# ── Adstock pre-computation ──────────────────────────────────────────────────

def _precompute_adstocked(df, g, params):
    """
    For Weibull adstock: compute the weighted-lag sum for each channel.
    For Instant (Nerlove-Arrow): standard geometric carry-over.
    Returns dict col -> np.ndarray of length T.
    """
    adstock_type = g["ADSTOCK_TYPE"]
    n_lags = int(params.get("adstock_n_lags", 8))
    adstocked = {}

    for i, col in enumerate(g["MEDIA_COLS"]):
        if adstock_type == "weibull":
            adstocked[col] = adstock_weibull_lagged(
                df[col], params["adstock_shape"][i], params["adstock_scale"][i], n_lags)
        else:
            adstocked[col] = adstock_nerlove_arrow(df[col], params["adstock_lambda"][i])

    for j, col in enumerate(g["COMP_MEDIA_COLS"]):
        ci = g["N_MEDIA"] + j
        if adstock_type == "weibull":
            adstocked[col] = adstock_weibull_lagged(
                df[col], params["adstock_shape"][ci], params["adstock_scale"][ci], n_lags)
        else:
            adstocked[col] = adstock_nerlove_arrow(df[col], params["adstock_lambda"][ci])

    return adstocked


# ── Observation matrix ───────────────────────────────────────────────────────

def _build_observation_matrix(df, g, adstocked_media):
    """
    y_t = I_t · 1
          + Σ β_i,t · adstocked_media_i,t       (own media)
          + Σ β_j,t · adstocked_media_j,t       (comp media)
          + Σ β_k  · nonmedia_k,t               (own non-media — raw)
          + Σ β_k  · comp_nonmedia_k,t          (comp non-media — raw)
          + Σ β_p  · price_p,t
          + Σ β_d  · dummy_d,t
    """
    T = len(df)
    cols = [np.ones(T)]
    for c in g["MEDIA_COLS"]:         cols.append(adstocked_media[c])
    for c in g["COMP_MEDIA_COLS"]:    cols.append(adstocked_media[c])
    for c in g["OWN_NONMEDIA_COLS"]:  cols.append(df[c].values.astype(float))
    for c in g["COMP_NONMEDIA_COLS"]: cols.append(df[c].values.astype(float))
    for c in g["PRICE_COLS"]:         cols.append(df[c].values.astype(float))
    for c in g["DUMMY_COLS"]:         cols.append(df[c].values.astype(float))
    return np.column_stack(cols)


# ── Transition matrix ────────────────────────────────────────────────────────

def _build_transition_matrix(g, params):
    N_MEDIA = g["N_MEDIA"]; N_COMP = g["N_COMP"]
    N_OWN_NONMEDIA = g["N_OWN_NONMEDIA"]; N_COMP_NONMEDIA = g["N_COMP_NONMEDIA"]
    N_PRICE = g["N_PRICE"]; N_DUMMIES = g["N_DUMMIES"]; SEASONAL_DIM = g["SEASONAL_DIM"]
    dim = 1 + N_MEDIA + N_COMP + N_OWN_NONMEDIA + N_COMP_NONMEDIA + N_PRICE + N_DUMMIES + SEASONAL_DIM
    Tmat = np.eye(dim)
    Tmat[0, 0] = params["G0"]
    for i in range(N_MEDIA): Tmat[i+1, i+1] = params["Ls"][i]
    for j in range(N_COMP):  Tmat[1+N_MEDIA+j, 1+N_MEDIA+j] = params["Ls_comp"][j]
    for k in range(N_OWN_NONMEDIA):
        idx = 1+N_MEDIA+N_COMP+k; Tmat[idx, idx] = params["Ls_own_nonmedia"][k]
    for k in range(N_COMP_NONMEDIA):
        idx = 1+N_MEDIA+N_COMP+N_OWN_NONMEDIA+k; Tmat[idx, idx] = params["Ls_comp_nonmedia"][k]
    for d in range(N_DUMMIES):
        idx = 1+N_MEDIA+N_COMP+N_OWN_NONMEDIA+N_COMP_NONMEDIA+d; Tmat[idx, idx] = 0.98
    for p in range(N_PRICE):
        idx = 1+N_MEDIA+N_COMP+N_OWN_NONMEDIA+N_COMP_NONMEDIA+p; Tmat[idx, idx] = params["Ls_price"][p]
    return Tmat


# ── Process noise ────────────────────────────────────────────────────────────

def _build_process_noise(g):
    N_MEDIA = g["N_MEDIA"]; N_COMP = g["N_COMP"]
    N_OWN_NONMEDIA = g["N_OWN_NONMEDIA"]; N_COMP_NONMEDIA = g["N_COMP_NONMEDIA"]
    N_PRICE = g["N_PRICE"]; N_DUMMIES = g["N_DUMMIES"]; SEASONAL_DIM = g["SEASONAL_DIM"]
    dim = 1 + N_MEDIA + N_COMP + N_OWN_NONMEDIA + N_COMP_NONMEDIA + N_PRICE + N_DUMMIES + SEASONAL_DIM
    Q = np.eye(dim) * 1e-6; Q[0, 0] = 1e-4
    for d in range(N_DUMMIES):
        idx = 1+N_MEDIA+N_COMP+N_OWN_NONMEDIA+N_COMP_NONMEDIA+d; Q[idx, idx] = 5e-3
    return Q


# ── Transformation helper ────────────────────────────────────────────────────

def _transform_media(x: np.ndarray, transform_type: str,
                     n: float, S: float) -> np.ndarray:
    """Apply configured transformation to a media array."""
    return apply_transformation(x, transform_type, n, S)


# ── Main Kalman filter ───────────────────────────────────────────────────────

def run_kalman_filter(df, params, g):
    TARGET_COL = g["TARGET_COL"]; MEDIA_COLS = g["MEDIA_COLS"]
    COMP_MEDIA_COLS = g["COMP_MEDIA_COLS"]; OWN_NONMEDIA_COLS = g["OWN_NONMEDIA_COLS"]
    COMP_NONMEDIA_COLS = g["COMP_NONMEDIA_COLS"]; PRICE_COLS = g["PRICE_COLS"]
    INTERCEPT_EFFECTORS = g["INTERCEPT_EFFECTORS"]; CROSS_MEDIA_PAIRS = g["CROSS_MEDIA_PAIRS"]
    USE_ORGANIC_DRIFT = g["USE_ORGANIC_DRIFT"]
    N_MEDIA = g["N_MEDIA"]; N_COMP = g["N_COMP"]
    N_OWN_NONMEDIA = g["N_OWN_NONMEDIA"]; N_COMP_NONMEDIA = g["N_COMP_NONMEDIA"]
    N_PRICE = g["N_PRICE"]; N_DUMMIES = g["N_DUMMIES"]
    N_CROSS = g["N_CROSS"]; SEASONAL_DIM = g["SEASONAL_DIM"]
    ADSTOCK_TYPE  = g["ADSTOCK_TYPE"]
    TRANSFORM_TYPE = g["TRANSFORM_TYPE"]
    NEGATIVE_BETA_COLS = set(g.get("NEGATIVE_BETA_COLS", []))

    T_len = len(df)
    dim = 1 + N_MEDIA + N_COMP + N_OWN_NONMEDIA + N_COMP_NONMEDIA + N_PRICE + N_DUMMIES + SEASONAL_DIM

    # ── Pre-compute adstocked series (used in observation matrix) ────
    adstocked_media = _precompute_adstocked(df, g, params)

    # ── Build matrices ────────────────────────────────────────────────
    L_mat = _build_observation_matrix(df, g, adstocked_media)
    Tmat  = _build_transition_matrix(g, params)
    Q     = _build_process_noise(g)
    R     = params["sigma_y"] ** 2

    # ── Pre-compute transformed own-media (for state equation) ───────
    # Applied to RAW spend (not adstocked) — this is the δ_i · f(x_i,t) term
    # in the state transition equation.
    transformed_own = np.stack([
        _transform_media(
            df[c].values.astype(float),
            TRANSFORM_TYPE,
            params["n_params"][i],
            params["S_params"][i],
        )
        for i, c in enumerate(MEDIA_COLS)
    ], axis=1) if MEDIA_COLS else np.zeros((T_len, 0))

    # ── Competitor media: always Hill on adstocked ────────────────────
    transformed_comp = np.stack([
        hill_transform_vec(adstocked_media[c], params["n_comp"][j], params["S_comp"][j])
        for j, c in enumerate(COMP_MEDIA_COLS)
    ], axis=1) if COMP_MEDIA_COLS else np.zeros((T_len, 0))

    # ── Cross-media synergy (Hill on adstocked source) ────────────────
    transformed_cross = np.zeros((T_len, N_CROSS))
    for k, (tgt, src) in enumerate(CROSS_MEDIA_PAIRS):
        transformed_cross[:, k] = hill_transform_vec(
            adstocked_media[src], params["cross_n"][k], params["cross_S"][k])

    # ── Intercept boosts: I_t += Σ γ_k · f(x_k,t; n_k_intercept, S_k_intercept) ──
    # ANY dataset column can be an effector (media, non-media, or other);
    # all get the configured Hill/Power transformation, each with its own
    # fitted n_intercept / S_intercept.
    intercept_boost = np.zeros(T_len)
    for k, col in enumerate(INTERCEPT_EFFECTORS):
        ni_int = params["n_intercept"][k]
        Si_int = params["S_intercept"][k]
        if col in df.columns:
            raw = df[col].values.astype(float)
            intercept_boost += params["gamma"][k] * _transform_media(
                raw, TRANSFORM_TYPE, ni_int, Si_int)

    # ── Weibull weighted-lag arrays for state transition ─────────────
    # When adstock = weibull the β state includes the weighted lag sum;
    # these are the same as adstocked_media (already computed above).
    # For instant adstock this component is zero (λ in Tmat handles persistence).

    # ── Initialise filter arrays ──────────────────────────────────────
    x_filt = np.zeros((T_len, dim)); P_filt = np.zeros((T_len, dim, dim))
    x_pred = np.zeros((T_len, dim)); P_pred = np.zeros((T_len, dim, dim))
    cross_beta_contrib = np.zeros((T_len, N_CROSS))
    yhat = np.zeros(T_len); residuals = np.zeros(T_len); loglik = 0.0

    x0 = np.zeros(dim)
    x0[0] = df[TARGET_COL].mean() * 0.8
    for i, col in enumerate(MEDIA_COLS):
        x0[i+1] = g["INITIAL_MEDIA_BETAS"].get(col, 0.0)
    for j, col in enumerate(COMP_MEDIA_COLS):
        x0[1+N_MEDIA+j] = g["INITIAL_COMP_BETAS"].get(col, -0.0001)
    for k, col in enumerate(OWN_NONMEDIA_COLS):
        x0[1+N_MEDIA+N_COMP+k] = g["INITIAL_OWN_NONMEDIA_BETAS"].get(col, 0.0)
    for k, col in enumerate(COMP_NONMEDIA_COLS):
        x0[1+N_MEDIA+N_COMP+N_OWN_NONMEDIA+k] = g["INITIAL_COMP_NONMEDIA_BETAS"].get(col, -0.01)
    for p, col in enumerate(PRICE_COLS):
        x0[1+N_MEDIA+N_COMP+N_OWN_NONMEDIA+N_COMP_NONMEDIA+p] = g["INITIAL_PRICE_BETA"].get(col, -0.1)

    x_filt[0] = x0
    target_var = float(np.var(df[TARGET_COL].values))
    base_var   = max(target_var, 1e-6)
    P0 = np.eye(dim) * base_var
    P0[0, 0] = base_var * 4.0
    P_filt[0] = P0
    target_vals = df[TARGET_COL].values

    # Positivity / negativity constraint index sets
    positive_cols = set(g.get("POSITIVE_BETA_COLS", []))
    negative_cols = set(g.get("NEGATIVE_BETA_COLS", []))
    positive_state_idx = (
        [i + 1 for i, col in enumerate(MEDIA_COLS) if col in positive_cols] +
        [1 + N_MEDIA + N_COMP + k for k, col in enumerate(OWN_NONMEDIA_COLS)
         if col in positive_cols]
    )
    negative_state_idx = (
        [i + 1 for i, col in enumerate(MEDIA_COLS) if col in negative_cols] +
        [1 + N_MEDIA + N_COMP + k for k, col in enumerate(OWN_NONMEDIA_COLS)
         if col in negative_cols] +
        [1 + N_MEDIA + N_COMP + N_OWN_NONMEDIA + k
         for k, col in enumerate(COMP_NONMEDIA_COLS) if col in negative_cols] +
        [1 + N_MEDIA + N_COMP + N_OWN_NONMEDIA + N_COMP_NONMEDIA + p
         for p, col in enumerate(PRICE_COLS) if col in negative_cols]
    )

    # ── Forward filter loop ───────────────────────────────────────────
    for t in range(T_len):
        if t > 0:
            x_p = Tmat @ x_filt[t-1]

            # ── Own-media state equations ────────────────────────────
            # All 4 cases share the δ_i · f(x_i,t) term here.
            # For Weibull, the Tmat diagonal (Ls) carries the previous
            # β forward while the weibull lag component is in
            # adstocked_media (observation) — but the STATE equation
            # for β still needs to receive the new-period shock.
            for i in range(N_MEDIA):
                # New-period transformed shock (common to all 4 cases)
                shock = params["delta"][i] * transformed_own[t, i]
                x_p[i+1] += shock
                if MEDIA_COLS[i] in positive_cols:
                    x_p[i+1] = max(x_p[i+1], 1e-8)
                elif MEDIA_COLS[i] in negative_cols:
                    x_p[i+1] = min(x_p[i+1], -1e-8)

            # ── Competitor media ─────────────────────────────────────
            for j in range(N_COMP):
                x_p[1+N_MEDIA+j] += params["delta_comp"][j] * transformed_comp[t, j]
                x_p[1+N_MEDIA+j]  = min(x_p[1+N_MEDIA+j], -1e-8)

            # ── Cross-media synergy ───────────────────────────────────
            for k, (tgt, src) in enumerate(CROSS_MEDIA_PAIRS):
                tgt_idx = MEDIA_COLS.index(tgt)
                contrib = params["cross_delta"][k] * transformed_cross[t, k]
                x_p[tgt_idx+1] += contrib
                cross_beta_contrib[t, k] = contrib

            # ── Own non-media ─────────────────────────────────────────
            for k, col in enumerate(OWN_NONMEDIA_COLS):
                si = 1+N_MEDIA+N_COMP+k
                x_p[si] += params["delta_own_nonmedia"][k] * df[col].iloc[t]
                if col in positive_cols:
                    x_p[si] = max(x_p[si], 1e-8)

            # ── Competitor non-media ──────────────────────────────────
            for k, col in enumerate(COMP_NONMEDIA_COLS):
                si = 1+N_MEDIA+N_COMP+N_OWN_NONMEDIA+k
                x_p[si] += params["delta_comp_nonmedia"][k] * df[col].iloc[t]
                x_p[si]  = min(x_p[si], -1e-8)

            # ── Price ─────────────────────────────────────────────────
            for p, col in enumerate(PRICE_COLS):
                si = 1+N_MEDIA+N_COMP+N_OWN_NONMEDIA+N_COMP_NONMEDIA+p
                x_p[si] += params["delta_price"][p] * df[col].iloc[t]
                x_p[si]  = min(x_p[si], -1e-8)

            # ── Intercept boost ───────────────────────────────────────
            x_p[0] += intercept_boost[t]
            if USE_ORGANIC_DRIFT:
                x_p[0] += params["mu"]

            x_pred[t] = x_p
            P_pred[t]  = Tmat @ P_filt[t-1] @ Tmat.T + Q
        else:
            x_pred[0] = x_filt[0].copy(); P_pred[0] = P_filt[0].copy()

        # ── Kalman update ─────────────────────────────────────────────
        L_t = L_mat[t]
        yhat[t]      = L_t @ x_pred[t]
        residuals[t] = target_vals[t] - yhat[t]
        LP  = L_t @ P_pred[t]
        S_t = max(LP @ L_t + R, 1e-8)
        K   = (P_pred[t] @ L_t) / S_t
        x_filt[t] = x_pred[t] + K * residuals[t]
        for idx in positive_state_idx:
            if x_filt[t, idx] < 0:
                x_filt[t, idx] = 1e-8
        for idx in negative_state_idx:
            if x_filt[t, idx] > 0:
                x_filt[t, idx] = -1e-8
        P_filt[t] = (np.eye(dim) - np.outer(K, L_t)) @ P_pred[t]
        loglik += -0.5 * (np.log(2 * np.pi * S_t) + residuals[t]**2 / S_t)

    return yhat, residuals, x_filt, P_filt, x_pred, P_pred, Tmat, cross_beta_contrib, loglik


def rts_smoother(x_filt, P_filt, x_pred, P_pred, Tmat):
    T_len, dim = x_filt.shape
    x_smooth = x_filt.copy(); P_smooth = P_filt.copy()
    for t in range(T_len - 2, -1, -1):
        C = P_filt[t] @ Tmat.T @ np.linalg.inv(P_pred[t+1])
        x_smooth[t] = x_filt[t] + C @ (x_smooth[t+1] - x_pred[t+1])
        P_smooth[t] = P_filt[t] + C @ (P_smooth[t+1] - P_pred[t+1]) @ C.T
    return x_smooth, P_smooth
