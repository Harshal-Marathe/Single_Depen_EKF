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
from modules.kalman import run_kalman_filter, rts_smoother, _build_observation_matrix
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

    # FIX: the forward filter pass now floors any POSITIVE_BETA_COLS
    # state at every timestep (see kalman.py), but RTS smoothing is a
    # backward blend with FUTURE smoothed values and is not itself
    # bound-aware — it can still pull a flagged variable's smoothed
    # beta slightly below zero even though the filtered pass never
    # went negative. Re-apply the same floor here, on the actual
    # values used for contributions / ROI / response curves, so what
    # gets displayed always honors the constraint.
    positive_cols = set(g.get("POSITIVE_BETA_COLS", []))
    for col in positive_cols:
        if col in MEDIA_COLS:
            idx = MEDIA_COLS.index(col) + 1
            x_smooth[:, idx] = np.maximum(x_smooth[:, idx], 0.0)
        elif col in g["OWN_NONMEDIA_COLS"]:
            idx = 1 + g["N_MEDIA"] + g["N_COMP"] + g["OWN_NONMEDIA_COLS"].index(col)
            x_smooth[:, idx] = np.maximum(x_smooth[:, idx], 0.0)
    L_mat_full  = _build_observation_matrix(df_full, g)
    yhat_smooth = (L_mat_full * x_smooth).sum(axis=1)

    target_vals  = df_full[TARGET_COL].values
    resid_smooth = target_vals - yhat_smooth
    mape  = float(np.mean(np.abs(resid_smooth / (target_vals + 1e-12))))
    ss_res = np.sum(resid_smooth**2); ss_tot = np.sum((target_vals - target_vals.mean())**2)
    r2    = 1.0 - ss_res / (ss_tot + 1e-12)

    # Mirrors the same change made in kalman.py's run_kalman_filter: Hill
    # is now applied to raw media spend, not adstocked spend, so this
    # must match in order for LongTerm_{col} to reflect what the filter
    # actually used for the gamma boost.
    hill_own_full = np.stack([
        hill_transform_vec(df_full[c].values.astype(float), params["n_params"][i], params["S_params"][i])
        for i, c in enumerate(MEDIA_COLS)
    ], axis=1) if MEDIA_COLS else np.zeros((len(df_full), 0))

    contrib_df = df_full[[TARGET_COL]].copy()
    # FIX: LongTerm_Intercept was hardcoded to 0.0 — never actually
    # computed. The intercept state evolves as
    #   x[0,t] = G0 * x[0,t-1] + (new boosts this period)
    # so G0 * x[0,t-1] IS the "carryover" — the persisted baseline /
    # brand-equity level that rolls forward from history — while the
    # remainder is the new incremental shock added that period. This
    # decomposition sums back to the full intercept value (no double
    # counting against the rest of yhat's additive breakdown), and
    # surfaces the carryover as its own line in the Long-Term table,
    # where it will typically dominate since G0 is usually close to 1.
    G0 = float(params["G0"])
    prev_intercept = np.empty(len(df_full))
    prev_intercept[1:] = x_smooth[:-1, 0]
    prev_intercept[0]  = x_smooth[0, 0]   # no prior period — use current as the seed
    long_term_intercept = G0 * prev_intercept
    contrib_df["LongTerm_Intercept"]  = long_term_intercept
    contrib_df["ShortTerm_Intercept"] = x_smooth[:, 0] - long_term_intercept
    for i, col in enumerate(MEDIA_COLS):
        contrib_df[f"ShortTerm_{col}"] = x_smooth[:, i+1] * df_full[col].values.astype(float)
        contrib_df[f"LongTerm_{col}"]  = 0.0
    for j, col in enumerate(COMP_MEDIA_COLS):
        contrib_df[f"ShortTerm_{col}"] = x_smooth[:, 1+g["N_MEDIA"]+j] * df_full[col].values.astype(float)
        contrib_df[f"LongTerm_{col}"]  = 0.0
    # FIX: the EKF state vector includes a slot for every Non-media /
    # organic column (this is where prophet_trend, prophet_yearly, etc.
    # live) and every Competitor Non-media column — see
    # _build_observation_matrix() / _build_transition_matrix() in
    # kalman.py, which both already account for these slots. The
    # contribution table previously jumped straight from CompMedia to
    # Price and never read these two blocks out of x_smooth, so any
    # variable configured as "non_media" (including merged prophet
    # columns) was estimated by the model but invisible in the Results
    # tab. Restoring them here using the same state-vector layout the
    # filter itself uses (see kalman.py / params.py for the indexing).
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

    # FIX: gamma is indexed by each variable's POSITION WITHIN
    # INTERCEPT_EFFECTORS, not its position within MEDIA_COLS — these
    # only happened to line up before when effectors was an exact,
    # identically-ordered copy of media (the UI default). Any time a
    # user deselected some media channels as effectors, gamma[i] above
    # was silently pulling the WRONG coefficient. This loop uses each
    # effector's own index k, and now also supports non-media/organic
    # columns (e.g. prophet_*) as effectors — boosting the intercept
    # via gamma * raw_value_t, the same convention used everywhere
    # else non-media variables are treated (no Hill/saturation).
    for k, col in enumerate(g["INTERCEPT_EFFECTORS"]):
        if col in MEDIA_COLS:
            i = MEDIA_COLS.index(col)
            contrib_df[f"LongTerm_{col}"] = params["gamma"][k] * hill_own_full[:, i]
        elif col in g["OWN_NONMEDIA_COLS"]:
            contrib_df[f"LongTerm_{col}"] = params["gamma"][k] * df_full[col].values.astype(float)

    # FIX: cross_beta_contrib (the per-period synergy uplift one media
    # channel injects into another, configured in Tab 4 · Section C)
    # was already computed inside run_kalman_filter and captured here
    # into a local variable — but nothing ever did anything with it
    # afterward. It never made it into contrib_df, so it was invisible
    # everywhere in the Results tab and in the CSV export. Surfacing
    # it now as one column per configured cross-media pair.
    for k, (tgt, src) in enumerate(g["CROSS_MEDIA_PAIRS"]):
        contrib_df[f"Synergy_{tgt}_from_{src}"] = cross_beta_contrib[:, k]

    roi_rows = []
    for col in MEDIA_COLS:
        tc = contrib_df[f"ShortTerm_{col}"].sum() + contrib_df[f"LongTerm_{col}"].sum()
        ts = df_full[col].sum()
        roi_rows.append({"Channel": col, "TotalSpend": ts, "TotalContrib": tc,
                         "ROI": tc/ts if ts > 0 else 0})
    roi_df = pd.DataFrame(roi_rows)

    # FIX (continued): per-pair synergy summary table, one row per
    # configured cross-media relationship — this is the "by each
    # variable level" breakdown, as opposed to one aggregated number.
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

    param_rows = []
    for k, col in enumerate(g["INTERCEPT_EFFECTORS"]):
        effector_kind = "Media" if col in MEDIA_COLS else "Non-media"
        param_rows.append({
            "Category": "Intercept Effector", "Variable": f"{col} ({effector_kind})",
            "Parameter": "Gamma (boost coeff.)", "Value": params["gamma"][k],
        })
    for i, col in enumerate(MEDIA_COLS):
        param_rows += [
            {"Category":"Media","Variable":col,"Parameter":"Ls",           "Value":params["Ls"][i]},
            {"Category":"Media","Variable":col,"Parameter":"Delta",         "Value":params["delta"][i]},
            {"Category":"Media","Variable":col,"Parameter":"n (Hill slope)","Value":params["n_params"][i]},
            {"Category":"Media","Variable":col,"Parameter":"S (Half-sat)",  "Value":params["S_params"][i]},
        ]
    for j, col in enumerate(COMP_MEDIA_COLS):
        param_rows += [
            {"Category":"CompMedia","Variable":col,"Parameter":"Ls_comp",   "Value":params["Ls_comp"][j]},
            {"Category":"CompMedia","Variable":col,"Parameter":"Delta_comp", "Value":params["delta_comp"][j]},
            {"Category":"CompMedia","Variable":col,"Parameter":"n_comp",     "Value":params["n_comp"][j]},
            {"Category":"CompMedia","Variable":col,"Parameter":"S_comp",     "Value":params["S_comp"][j]},
        ]
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
    param_df = pd.DataFrame(param_rows)

    return {
        "params":params,"yhat_smooth":yhat_smooth,"residuals":resid_smooth,
        "x_smooth":x_smooth,"P_smooth":P_smooth,
        "contrib_df":contrib_df,"roi_df":roi_df,"param_df":param_df,"synergy_df":synergy_df,
        "loglik":loglik,"mape":mape,"r2":r2,
        "success":opt_success,"nit":opt_nit,"g":g,
    }