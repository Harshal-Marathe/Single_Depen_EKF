"""
Extended Kalman Filter core: observation/transition/process-noise matrix
builders, the forward filter, and the RTS (Rauch-Tung-Striebel) smoother.
"""

import numpy as np
import pandas as pd

from modules.transforms import hill_transform_vec, _adstock


def _precompute_adstocked(df, g, params):
    adstocked_media = {}
    for i, col in enumerate(g["MEDIA_COLS"]):
        adstocked_media[col] = _adstock(df[col], g["ADSTOCK_TYPE"], params, i)
    for j, col in enumerate(g["COMP_MEDIA_COLS"]):
        adstocked_media[col] = _adstock(df[col], g["ADSTOCK_TYPE"], params, g["N_MEDIA"] + j)
    return adstocked_media


def _build_observation_matrix(df, g, adstocked_media):
    T = len(df)
    cols = [np.ones(T)]
    for c in g["MEDIA_COLS"]:         cols.append(adstocked_media[c])
    for c in g["COMP_MEDIA_COLS"]:    cols.append(adstocked_media[c])
    for c in g["OWN_NONMEDIA_COLS"]:  cols.append(df[c].values.astype(float))
    for c in g["COMP_NONMEDIA_COLS"]: cols.append(df[c].values.astype(float))
    for c in g["PRICE_COLS"]:         cols.append(df[c].values.astype(float))
    for c in g["DUMMY_COLS"]:         cols.append(df[c].values.astype(float))
    return np.column_stack(cols)


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


def _build_process_noise(g):
    N_MEDIA = g["N_MEDIA"]; N_COMP = g["N_COMP"]
    N_OWN_NONMEDIA = g["N_OWN_NONMEDIA"]; N_COMP_NONMEDIA = g["N_COMP_NONMEDIA"]
    N_PRICE = g["N_PRICE"]; N_DUMMIES = g["N_DUMMIES"]; SEASONAL_DIM = g["SEASONAL_DIM"]
    dim = 1 + N_MEDIA + N_COMP + N_OWN_NONMEDIA + N_COMP_NONMEDIA + N_PRICE + N_DUMMIES + SEASONAL_DIM
    Q = np.eye(dim) * 1e-6; Q[0, 0] = 1e-4
    for d in range(N_DUMMIES):
        idx = 1+N_MEDIA+N_COMP+N_OWN_NONMEDIA+N_COMP_NONMEDIA+d; Q[idx, idx] = 5e-3
    return Q


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

    T_len = len(df)
    dim = 1 + N_MEDIA + N_COMP + N_OWN_NONMEDIA + N_COMP_NONMEDIA + N_PRICE + N_DUMMIES + SEASONAL_DIM

    adstocked_media = _precompute_adstocked(df, g, params)
    L_mat = _build_observation_matrix(df, g, adstocked_media)
    Tmat  = _build_transition_matrix(g, params)
    Q     = _build_process_noise(g)
    R     = params["sigma_y"] ** 2

    hill_own = np.stack([
        hill_transform_vec(adstocked_media[c], params["n_params"][i], params["S_params"][i])
        for i, c in enumerate(MEDIA_COLS)
    ], axis=1) if MEDIA_COLS else np.zeros((T_len, 0))

    hill_comp = np.stack([
        hill_transform_vec(adstocked_media[c], params["n_comp"][j], params["S_comp"][j])
        for j, c in enumerate(COMP_MEDIA_COLS)
    ], axis=1) if COMP_MEDIA_COLS else np.zeros((T_len, 0))

    hill_cross = np.zeros((T_len, N_CROSS))
    for k, (tgt, src) in enumerate(CROSS_MEDIA_PAIRS):
        hill_cross[:, k] = hill_transform_vec(adstocked_media[src],
                                               params["cross_n"][k], params["cross_S"][k])

    intercept_boost = np.zeros(T_len)
    for k, col in enumerate(INTERCEPT_EFFECTORS):
        i = MEDIA_COLS.index(col)
        intercept_boost += params["gamma"][k] * hill_own[:, i]

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
    P0 = np.eye(dim) * 1.0; P0[0, 0] = 10.0
    P_filt[0] = P0
    target_vals = df[TARGET_COL].values

    for t in range(T_len):
        if t > 0:
            x_p = Tmat @ x_filt[t-1]
            for i in range(N_MEDIA):
                x_p[i+1] += params["delta"][i] * hill_own[t, i]
            for j in range(N_COMP):
                x_p[1+N_MEDIA+j] += params["delta_comp"][j] * hill_comp[t, j]
                x_p[1+N_MEDIA+j]  = min(x_p[1+N_MEDIA+j], -1e-8)
            for k, (tgt, src) in enumerate(CROSS_MEDIA_PAIRS):
                tgt_idx = MEDIA_COLS.index(tgt)
                contrib = params["cross_delta"][k] * hill_cross[t, k]
                x_p[tgt_idx+1] += contrib
                cross_beta_contrib[t, k] = contrib
            for k, col in enumerate(OWN_NONMEDIA_COLS):
                x_p[1+N_MEDIA+N_COMP+k] += params["delta_own_nonmedia"][k] * df[col].iloc[t]
            for k, col in enumerate(COMP_NONMEDIA_COLS):
                si = 1+N_MEDIA+N_COMP+N_OWN_NONMEDIA+k
                x_p[si] += params["delta_comp_nonmedia"][k] * df[col].iloc[t]
                x_p[si]  = min(x_p[si], -1e-8)
            for p, col in enumerate(PRICE_COLS):
                si = 1+N_MEDIA+N_COMP+N_OWN_NONMEDIA+N_COMP_NONMEDIA+p
                x_p[si] += params["delta_price"][p] * df[col].iloc[t]
                x_p[si]  = min(x_p[si], -1e-8)
            x_p[0] += intercept_boost[t]
            if USE_ORGANIC_DRIFT: x_p[0] += params["mu"]
            x_pred[t] = x_p
            P_pred[t]  = Tmat @ P_filt[t-1] @ Tmat.T + Q
        else:
            x_pred[0] = x_filt[0].copy(); P_pred[0] = P_filt[0].copy()

        L_t = L_mat[t]
        yhat[t]      = L_t @ x_pred[t]
        residuals[t] = target_vals[t] - yhat[t]
        LP  = L_t @ P_pred[t]
        S_t = max(LP @ L_t + R, 1e-8)
        K   = (P_pred[t] @ L_t) / S_t
        x_filt[t] = x_pred[t] + K * residuals[t]
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
