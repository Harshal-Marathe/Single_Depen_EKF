"""
Full EKF MMM pipeline: optimize on the train split, run the filter +
smoother on the full dataset, and assemble contributions / ROI / parameter
tables for the Results tab.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from modules.dependencies import NEVERGRAD_AVAILABLE
from modules.params import _make_globals, unpack_theta
from modules.bounds import _build_theta0_and_bounds
from modules.optimizer import run_nevergrad_optimizer
from modules.kalman import (
    run_kalman_filter, rts_smoother, _precompute_adstocked, _build_observation_matrix,
)
from modules.transforms import hill_transform_vec


def run_full_ekf_pipeline(df_full, config, max_iter, method, ng_cfg=None):
    g = _make_globals(config)
    n_train  = config["n_train"]
    df_train = df_full.iloc[:n_train].copy().reset_index(drop=True)
    theta0, bounds = _build_theta0_and_bounds(df_train, g)

    if method == "Nevergrad" and NEVERGRAD_AVAILABLE and ng_cfg:
        best_theta, _ = run_nevergrad_optimizer(df_train, g, theta0, bounds, ng_cfg)
        opt_success = True; opt_nit = ng_cfg.get("budget", 500)
    else:
        def objective(theta):
            p = unpack_theta(theta, g)
            _, _, _, _, _, _, _, _, loglik = run_kalman_filter(df_train, p, g)
            return -loglik
        opt = minimize(objective, theta0, method=method,
                       bounds=bounds, options={"maxiter": max_iter, "ftol": 1e-9})
        best_theta = opt.x; opt_success = opt.success; opt_nit = opt.nit

    params = unpack_theta(best_theta, g)
    yhat, residuals, x_filt, P_filt, x_pred, P_pred, Tmat, cross_beta_contrib, loglik = \
        run_kalman_filter(df_full, params, g)
    x_smooth, P_smooth = rts_smoother(x_filt, P_filt, x_pred, P_pred, Tmat)

    TARGET_COL = g["TARGET_COL"]; MEDIA_COLS = g["MEDIA_COLS"]
    COMP_MEDIA_COLS = g["COMP_MEDIA_COLS"]; PRICE_COLS = g["PRICE_COLS"]
    ADSTOCK_TYPE = g["ADSTOCK_TYPE"]

    adstocked_media = _precompute_adstocked(df_full, g, params)
    L_mat_full  = _build_observation_matrix(df_full, g, adstocked_media)
    yhat_smooth = (L_mat_full * x_smooth).sum(axis=1)

    target_vals  = df_full[TARGET_COL].values
    resid_smooth = target_vals - yhat_smooth
    mape  = float(np.mean(np.abs(resid_smooth / (target_vals + 1e-12))))
    ss_res = np.sum(resid_smooth**2); ss_tot = np.sum((target_vals - target_vals.mean())**2)
    r2    = 1.0 - ss_res / (ss_tot + 1e-12)

    hill_own_full = np.stack([
        hill_transform_vec(adstocked_media[c], params["n_params"][i], params["S_params"][i])
        for i, c in enumerate(MEDIA_COLS)
    ], axis=1) if MEDIA_COLS else np.zeros((len(df_full), 0))

    contrib_df = df_full[[TARGET_COL]].copy()
    contrib_df["ShortTerm_Intercept"] = x_smooth[:, 0]
    contrib_df["LongTerm_Intercept"]  = 0.0
    for i, col in enumerate(MEDIA_COLS):
        contrib_df[f"ShortTerm_{col}"] = x_smooth[:, i+1] * adstocked_media[col]
        contrib_df[f"LongTerm_{col}"]  = params["gamma"][i] * hill_own_full[:, i]
    for j, col in enumerate(COMP_MEDIA_COLS):
        contrib_df[f"ShortTerm_{col}"] = x_smooth[:, 1+g["N_MEDIA"]+j] * adstocked_media[col]
        contrib_df[f"LongTerm_{col}"]  = 0.0
    for p, col in enumerate(PRICE_COLS):
        si = 1+g["N_MEDIA"]+g["N_COMP"]+g["N_OWN_NONMEDIA"]+g["N_COMP_NONMEDIA"]+p
        contrib_df[f"ShortTerm_{col}"] = x_smooth[:, si] * df_full[col].values
        contrib_df[f"LongTerm_{col}"]  = 0.0

    roi_rows = []
    for col in MEDIA_COLS:
        tc = contrib_df[f"ShortTerm_{col}"].sum() + contrib_df[f"LongTerm_{col}"].sum()
        ts = df_full[col].sum()
        roi_rows.append({"Channel": col, "TotalSpend": ts, "TotalContrib": tc,
                         "ROI": tc/ts if ts > 0 else 0})
    roi_df = pd.DataFrame(roi_rows)

    param_rows = []
    for i, col in enumerate(MEDIA_COLS):
        param_rows += [
            {"Category":"Media","Variable":col,"Parameter":"Ls",           "Value":params["Ls"][i]},
            {"Category":"Media","Variable":col,"Parameter":"Delta",         "Value":params["delta"][i]},
            {"Category":"Media","Variable":col,"Parameter":"n (Hill slope)","Value":params["n_params"][i]},
            {"Category":"Media","Variable":col,"Parameter":"S (Half-sat)",  "Value":params["S_params"][i]},
        ]
        if ADSTOCK_TYPE == "weibull":
            param_rows += [
                {"Category":"Media","Variable":col,"Parameter":"Adstock α","Value":params["adstock_alpha"][i]},
                {"Category":"Media","Variable":col,"Parameter":"Adstock θ","Value":params["adstock_theta"][i]},
            ]
        else:
            param_rows.append({"Category":"Media","Variable":col,
                                "Parameter":"Adstock λ","Value":params["adstock_lambda"][i]})
    for j, col in enumerate(COMP_MEDIA_COLS):
        param_rows += [
            {"Category":"CompMedia","Variable":col,"Parameter":"Ls_comp",   "Value":params["Ls_comp"][j]},
            {"Category":"CompMedia","Variable":col,"Parameter":"Delta_comp", "Value":params["delta_comp"][j]},
            {"Category":"CompMedia","Variable":col,"Parameter":"n_comp",     "Value":params["n_comp"][j]},
            {"Category":"CompMedia","Variable":col,"Parameter":"S_comp",     "Value":params["S_comp"][j]},
        ]
        if ADSTOCK_TYPE == "weibull":
            param_rows += [
                {"Category":"CompMedia","Variable":col,"Parameter":"Adstock α","Value":params["adstock_alpha"][g["N_MEDIA"]+j]},
                {"Category":"CompMedia","Variable":col,"Parameter":"Adstock θ","Value":params["adstock_theta"][g["N_MEDIA"]+j]},
            ]
        else:
            param_rows.append({"Category":"CompMedia","Variable":col,
                                "Parameter":"Adstock λ","Value":params["adstock_lambda"][g["N_MEDIA"]+j]})
    for i, col in enumerate(PRICE_COLS):
        param_rows += [
            {"Category":"Price","Variable":col,"Parameter":"Ls_price",  "Value":params["Ls_price"][i]},
            {"Category":"Price","Variable":col,"Parameter":"Delta_price","Value":params["delta_price"][i]},
        ]
    param_rows.append({"Category":"Global","Variable":"Intercept","Parameter":"G0",     "Value":params["G0"]})
    param_rows.append({"Category":"Global","Variable":"Noise",    "Parameter":"sigma_y","Value":params["sigma_y"]})
    if g["USE_ORGANIC_DRIFT"]:
        param_rows.append({"Category":"Global","Variable":"Organic drift","Parameter":"mu","Value":params["mu"]})
    param_df = pd.DataFrame(param_rows)

    return {
        "params":params,"yhat_smooth":yhat_smooth,"residuals":resid_smooth,
        "x_smooth":x_smooth,"P_smooth":P_smooth,"adstocked_media":adstocked_media,
        "contrib_df":contrib_df,"roi_df":roi_df,"param_df":param_df,
        "loglik":loglik,"mape":mape,"r2":r2,
        "success":opt_success,"nit":opt_nit,"g":g,
    }
