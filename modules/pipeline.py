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
    _effector_series,
)
from modules.transforms import apply_transformation, hill_transform_vec


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

        def _fit(x0, bnds):
            return minimize(objective, x0, method=method, bounds=bnds,
                            options={"maxiter": max_iter, "ftol": 1e-9})

        N_EFF = g["N_EFFECTORS"]
        if N_EFF > 0:
            # Two-stage fit. Stage 1: intercept effectors switched off
            # (gamma pinned to 0) so every other parameter settles first.
            # Stage 2: release gamma starting from the stage-1 solution.
            # Keeping the better of the two guarantees adding effectors
            # can never fit worse than the no-effector model.
            gi = g["N_MEDIA"] + 1 + g["N_MEDIA"]          # start of gamma block
            th1 = theta0.copy(); th1[gi:gi+N_EFF] = 0.0
            b1 = list(bounds);   b1[gi:gi+N_EFF] = [(0.0, 0.0)] * N_EFF
            opt1 = _fit(th1, b1)
            th2 = opt1.x.copy(); th2[gi:gi+N_EFF] = theta0[gi:gi+N_EFF]
            opt2 = _fit(th2, bounds)
            opt = opt2 if opt2.fun <= opt1.fun else opt1
        else:
            opt = _fit(theta0, bounds)
        best_theta = opt.x; opt_success = opt.success; opt_nit = opt.nit

    params = unpack_theta(best_theta, g)
    yhat, residuals, x_filt, P_filt, x_pred, P_pred, Tmat, cross_beta_contrib, loglik = \
        run_kalman_filter(df_full, params, g)
    x_smooth, P_smooth = rts_smoother(x_filt, P_filt, x_pred, P_pred, Tmat)

    TARGET_COL = g["TARGET_COL"]; MEDIA_COLS = g["MEDIA_COLS"]
    COMP_MEDIA_COLS = g["COMP_MEDIA_COLS"]; PRICE_COLS = g["PRICE_COLS"]
    ADSTOCK_TYPE   = g["ADSTOCK_TYPE"]
    TRANSFORM_TYPE = g["TRANSFORM_TYPE"]

    # Re-apply positivity / negativity floors after RTS smoothing
    positive_cols = set(g.get("POSITIVE_BETA_COLS", []))
    negative_cols = set(g.get("NEGATIVE_BETA_COLS", []))
    for col in positive_cols:
        if col in MEDIA_COLS:
            idx = MEDIA_COLS.index(col) + 1
            x_smooth[:, idx] = np.maximum(x_smooth[:, idx], 0.0)
        elif col in g["OWN_NONMEDIA_COLS"]:
            idx = 1 + g["N_MEDIA"] + g["N_COMP"] + g["OWN_NONMEDIA_COLS"].index(col)
            x_smooth[:, idx] = np.maximum(x_smooth[:, idx], 0.0)
    for col in negative_cols:
        if col in MEDIA_COLS:
            idx = MEDIA_COLS.index(col) + 1
            x_smooth[:, idx] = np.minimum(x_smooth[:, idx], 0.0)
        elif col in g["OWN_NONMEDIA_COLS"]:
            idx = 1 + g["N_MEDIA"] + g["N_COMP"] + g["OWN_NONMEDIA_COLS"].index(col)
            x_smooth[:, idx] = np.minimum(x_smooth[:, idx], 0.0)
        elif col in g["COMP_NONMEDIA_COLS"]:
            idx = 1 + g["N_MEDIA"] + g["N_COMP"] + g["N_OWN_NONMEDIA"] + g["COMP_NONMEDIA_COLS"].index(col)
            x_smooth[:, idx] = np.minimum(x_smooth[:, idx], 0.0)
        elif col in PRICE_COLS:
            idx = 1 + g["N_MEDIA"] + g["N_COMP"] + g["N_OWN_NONMEDIA"] + g["N_COMP_NONMEDIA"] + PRICE_COLS.index(col)
            x_smooth[:, idx] = np.minimum(x_smooth[:, idx], 0.0)

    adstocked_media = _precompute_adstocked(df_full, g, params)
    L_mat_full  = _build_observation_matrix(df_full, g, adstocked_media)
    yhat_smooth = (L_mat_full * x_smooth).sum(axis=1)

    target_vals  = df_full[TARGET_COL].values
    resid_smooth = target_vals - yhat_smooth
    mape  = float(np.mean(np.abs(resid_smooth / (target_vals + 1e-12))))
    ss_res = np.sum(resid_smooth**2); ss_tot = np.sum((target_vals - target_vals.mean())**2)
    r2    = 1.0 - ss_res / (ss_tot + 1e-12)

    # Transformed own-media (raw spend) — used for LongTerm contribution
    transformed_own_full = np.stack([
        apply_transformation(
            df_full[c].values.astype(float),
            TRANSFORM_TYPE,
            params["n_params"][i],
            params["S_params"][i],
        )
        for i, c in enumerate(MEDIA_COLS)
    ], axis=1) if MEDIA_COLS else np.zeros((len(df_full), 0))

    contrib_df = df_full[[TARGET_COL]].copy()

    G0 = float(params["G0"])
    prev_intercept = np.empty(len(df_full))
    prev_intercept[1:] = x_smooth[:-1, 0]
    prev_intercept[0]  = x_smooth[0, 0]
    long_term_intercept = G0 * prev_intercept
    contrib_df["LongTerm_Intercept"]  = long_term_intercept
    contrib_df["ShortTerm_Intercept"] = x_smooth[:, 0] - long_term_intercept

    for i, col in enumerate(MEDIA_COLS):
        contrib_df[f"ShortTerm_{col}"] = x_smooth[:, i+1] * adstocked_media[col]
        contrib_df[f"LongTerm_{col}"]  = 0.0
    for j, col in enumerate(COMP_MEDIA_COLS):
        contrib_df[f"ShortTerm_{col}"] = x_smooth[:, 1+g["N_MEDIA"]+j] * adstocked_media[col]
        contrib_df[f"LongTerm_{col}"]  = 0.0
    for k, col in enumerate(g["OWN_NONMEDIA_COLS"]):
        si = 1+g["N_MEDIA"]+g["N_COMP"]+k
        contrib_df[f"ShortTerm_{col}"] = x_smooth[:, si] * df_full[col].values
        contrib_df[f"LongTerm_{col}"]  = 0.0
    for k, col in enumerate(g["COMP_NONMEDIA_COLS"]):
        si = 1+g["N_MEDIA"]+g["N_COMP"]+g["N_OWN_NONMEDIA"]+k
        contrib_df[f"ShortTerm_{col}"] = x_smooth[:, si] * df_full[col].values
        contrib_df[f"LongTerm_{col}"]  = 0.0
    for p, col in enumerate(PRICE_COLS):
        si = 1+g["N_MEDIA"]+g["N_COMP"]+g["N_OWN_NONMEDIA"]+g["N_COMP_NONMEDIA"]+p
        contrib_df[f"ShortTerm_{col}"] = x_smooth[:, si] * df_full[col].values
        contrib_df[f"LongTerm_{col}"]  = 0.0

    for k, col in enumerate(g["INTERCEPT_EFFECTORS"]):
        if col in df_full.columns:
            contrib_df[f"LongTerm_{col}"] = params["gamma"][k] * _effector_series(
                df_full, col, g, params, k)

    for k, (tgt, src) in enumerate(g["CROSS_MEDIA_PAIRS"]):
        contrib_df[f"Synergy_{tgt}_from_{src}"] = cross_beta_contrib[:, k]

    spend_map = config.get("media_spend_map", {}) or {}

    def _spend_sum(col):
        """Resolve the raw Spend total for `col` using the user-defined
        spend map from Tab 4 (A2 · Spend Mapping). Falls back to the
        variable's own raw values if no mapping was saved for it."""
        m = spend_map.get(col)
        spend_col = m["spend_col"] if m else col
        if not spend_col or spend_col not in df_full.columns:
            spend_col = col
        return float(df_full[spend_col].sum())

    roi_rows = []
    for col in MEDIA_COLS:
        tc = contrib_df[f"ShortTerm_{col}"].sum() + contrib_df[f"LongTerm_{col}"].sum()
        ts = _spend_sum(col)
        roi_rows.append({"Channel": col, "Type": "Own Media", "TotalSpend": ts,
                          "TotalContrib": tc, "ROI": tc/ts if ts > 0 else 0})
    for col in COMP_MEDIA_COLS:
        tc = contrib_df[f"ShortTerm_{col}"].sum() + contrib_df[f"LongTerm_{col}"].sum()
        ts = _spend_sum(col)
        roi_rows.append({"Channel": col, "Type": "Competitor Media", "TotalSpend": ts,
                          "TotalContrib": tc, "ROI": tc/ts if ts > 0 else 0})
    roi_df = pd.DataFrame(roi_rows)

    synergy_rows = []
    for k, (tgt, src) in enumerate(g["CROSS_MEDIA_PAIRS"]):
        col_name = f"Synergy_{tgt}_from_{src}"
        total_synergy = float(contrib_df[col_name].sum())
        tgt_total = (contrib_df[f"ShortTerm_{tgt}"].sum()
                     + contrib_df[f"LongTerm_{tgt}"].sum()) if tgt in MEDIA_COLS else np.nan
        synergy_rows.append({
            "Source Channel": src,
            "Target Channel": tgt,
            "Total Synergy Contribution": total_synergy,
            "Avg Synergy / Period": float(contrib_df[col_name].mean()),
            "Cross Delta": float(params["cross_delta"][k]),
            "Cross Hill n": float(params["cross_n"][k]),
            "Cross Hill S": float(params["cross_S"][k]),
            "Share of Target's Total Contrib (%)": (
                round(100 * total_synergy / tgt_total, 2)
                if tgt_total and tgt_total != 0 and not np.isnan(tgt_total) else np.nan
            ),
        })
    synergy_df = pd.DataFrame(synergy_rows)

    # ── Parameter table ───────────────────────────────────────────────
    param_rows = []
    for k, col in enumerate(g["INTERCEPT_EFFECTORS"]):
        if col in MEDIA_COLS:                      effector_kind = "Media"
        elif col in g["OWN_NONMEDIA_COLS"]:        effector_kind = "Non-media"
        else:                                      effector_kind = "Other"
        param_rows.append({
            "Category": "Intercept Effector", "Variable": f"{col} ({effector_kind})",
            "Parameter": "Gamma (boost coeff.)", "Value": params["gamma"][k],
        })
        param_rows.append({
            "Category": "Intercept Effector", "Variable": f"{col} ({effector_kind})",
            "Parameter": "n_intercept (exponent)", "Value": params["n_intercept"][k],
        })
        param_rows.append({
            "Category": "Intercept Effector", "Variable": f"{col} ({effector_kind})",
            "Parameter": "S_intercept (half-sat, in units of max(x))", "Value": params["S_intercept"][k],
        })

    for i, col in enumerate(MEDIA_COLS):
        if TRANSFORM_TYPE == "hill":
            transform_rows = [
                {"Category":"Media","Variable":col,"Parameter":"n (Hill slope)","Value":params["n_params"][i]},
                {"Category":"Media","Variable":col,"Parameter":"S (Half-sat)",  "Value":params["S_params"][i]},
            ]
        else:
            transform_rows = [
                {"Category":"Media","Variable":col,"Parameter":"n (Power exponent)","Value":params["n_params"][i]},
            ]
        param_rows += [
            {"Category":"Media","Variable":col,"Parameter":"Ls",    "Value":params["Ls"][i]},
            {"Category":"Media","Variable":col,"Parameter":"Delta", "Value":params["delta"][i]},
        ] + transform_rows

        if ADSTOCK_TYPE == "weibull":
            param_rows += [
                {"Category":"Media","Variable":col,"Parameter":"Adstock shape k","Value":params["adstock_shape"][i]},
                {"Category":"Media","Variable":col,"Parameter":"Adstock scale λ","Value":params["adstock_scale"][i]},
            ]
        else:
            param_rows.append({"Category":"Media","Variable":col,
                                "Parameter":"Adstock λ (instant)","Value":params["adstock_lambda"][i]})

    for j, col in enumerate(COMP_MEDIA_COLS):
        param_rows += [
            {"Category":"CompMedia","Variable":col,"Parameter":"Ls_comp",   "Value":params["Ls_comp"][j]},
            {"Category":"CompMedia","Variable":col,"Parameter":"Delta_comp", "Value":params["delta_comp"][j]},
            {"Category":"CompMedia","Variable":col,"Parameter":"n_comp",     "Value":params["n_comp"][j]},
            {"Category":"CompMedia","Variable":col,"Parameter":"S_comp",     "Value":params["S_comp"][j]},
        ]
        ci = g["N_MEDIA"] + j
        if ADSTOCK_TYPE == "weibull":
            param_rows += [
                {"Category":"CompMedia","Variable":col,"Parameter":"Adstock shape k","Value":params["adstock_shape"][ci]},
                {"Category":"CompMedia","Variable":col,"Parameter":"Adstock scale λ","Value":params["adstock_scale"][ci]},
            ]
        else:
            param_rows.append({"Category":"CompMedia","Variable":col,
                                "Parameter":"Adstock λ","Value":params["adstock_lambda"][ci]})

    for i, col in enumerate(PRICE_COLS):
        param_rows += [
            {"Category":"Price","Variable":col,"Parameter":"Ls_price",  "Value":params["Ls_price"][i]},
            {"Category":"Price","Variable":col,"Parameter":"Delta_price","Value":params["delta_price"][i]},
        ]
    for k, (tgt, src) in enumerate(g["CROSS_MEDIA_PAIRS"]):
        pair_label = f"{src}→{tgt}"
        param_rows += [
            {"Category":"Synergy","Variable":pair_label,"Parameter":"Cross Delta", "Value":params["cross_delta"][k]},
            {"Category":"Synergy","Variable":pair_label,"Parameter":"Cross Hill n","Value":params["cross_n"][k]},
            {"Category":"Synergy","Variable":pair_label,"Parameter":"Cross Hill S","Value":params["cross_S"][k]},
        ]
    param_rows.append({"Category":"Global","Variable":"Intercept","Parameter":"G0",     "Value":params["G0"]})
    param_rows.append({"Category":"Global","Variable":"Noise",    "Parameter":"sigma_y","Value":params["sigma_y"]})
    if g["USE_ORGANIC_DRIFT"]:
        param_rows.append({"Category":"Global","Variable":"Organic drift","Parameter":"mu","Value":params["mu"]})

    if ADSTOCK_TYPE == "weibull":
        param_rows.append({"Category":"Global","Variable":"Weibull adstock",
                            "Parameter":"n_lags", "Value": g["ADSTOCK_N_LAGS"]})

    param_df = pd.DataFrame(param_rows)

    return {
        "params":params,"yhat_smooth":yhat_smooth,"residuals":resid_smooth,
        "x_smooth":x_smooth,"P_smooth":P_smooth,"adstocked_media":adstocked_media,
        "contrib_df":contrib_df,"roi_df":roi_df,"param_df":param_df,"synergy_df":synergy_df,
        "loglik":loglik,"mape":mape,"r2":r2,
        "success":opt_success,"nit":opt_nit,"g":g,
    }
