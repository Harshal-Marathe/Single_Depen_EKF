"""
Tab 5 — Run EKF Model: optimizer selection (L-BFGS-B / SLSQP / Nevergrad)
and kicking off the full pipeline.
"""

import pandas as pd
import streamlit as st

from modules.ui_helpers import section, info, ng_info, prophet_info, need_data, need_config
from modules.pipeline import run_full_ekf_pipeline


def _show_co_summary(r, label):
    """Cochrane-Orcutt result block (no-op when the fit wasn't run with CO).
    ρ is the AR(1) coefficient of the observation errors; DW "before" is the
    ordinary fit's one-step residuals (iteration 1, ρ = 0), DW "whitened" is
    u_t = e_t − ρ·e_{t-1} for the final fit and should sit near 2 if CO did
    its job."""
    co = r.get("co_info")
    if not co:
        return
    hist = co["history"]
    first, last = hist[0], hist[-1]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric(f"AR(1) ρ · {label}", f"{co['rho']:.3f}")
    k2.metric("CO iterations", f"{co['n_iter']} / {co['max_iter']}",
              delta="converged" if co["converged"] else "not converged",
              delta_color="normal" if co["converged"] else "inverse")
    k3.metric("Durbin-Watson before (ρ=0)", f"{first['dw_structural']:.2f}")
    k4.metric("Durbin-Watson whitened (final)", f"{last['dw_whitened']:.2f}")
    if not co["converged"]:
        st.warning(
            f"Cochrane-Orcutt hit the iteration limit ({co['max_iter']}) before ρ "
            f"settled within ±{co['tol']:g}. The fit above uses ρ = {co['rho']:.3f} "
            f"(the value its parameters were actually fitted with); raise **Max CO "
            f"iterations** in Tab 4 to keep iterating."
        )
    with st.expander(f"🔁 Cochrane-Orcutt iterations · {label}"):
        hdf = pd.DataFrame(hist).rename(columns={
            "iteration": "Iter", "rho_used": "ρ used in fit", "rho_hat": "ρ̂ from residuals",
            "dw_structural": "DW (residuals)", "dw_whitened": "DW (whitened)",
        })[["Iter", "ρ used in fit", "ρ̂ from residuals", "DW (residuals)", "DW (whitened)"]]
        st.dataframe(hdf.style.format({
            "ρ used in fit": "{:.3f}", "ρ̂ from residuals": "{:.3f}",
            "DW (residuals)": "{:.2f}", "DW (whitened)": "{:.2f}"}),
            use_container_width=True, hide_index=True)
        st.caption(
            "Iteration 1 is the ordinary fit (ρ = 0). Each later row refits with the "
            "previous row's ρ̂. Residuals are one-step-ahead structural residuals on "
            "the **train** window. DW ≈ 2 → no first-order autocorrelation; the "
            "residual DW stays low by design (the errors ARE assumed AR(1)) — it is "
            "the whitened column that should approach 2. Fitted values, "
            "contributions and ROI are on the original, non-differenced scale."
        )


def render_tab5(nevergrad_available: bool):
    section("05", "Run EKF Model")
    if st.session_state.df is None:     need_data()
    if st.session_state.config is None: need_config()

    config = st.session_state.config
    if "media" not in config:
        st.error("Configuration is incomplete — please re-save it in **Tab 4**.")
        st.stop()

    df = st.session_state.df
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Media channels", len(config["media"]))
    c2.metric("Price vars",     len(config.get("price", [])))
    combo = f"{config["adstock_type"].title()} × {config.get("transform_type","hill").title()}"
    c3.metric("Adstock×Transform", combo)
    c4.metric("Train / Test",   f"{config['n_train']} / {config['n_test']}")

    pb_cols = config.get("positive_beta_cols", [])
    pcb     = config.get("per_channel_bounds", {})
    if pb_cols: st.info(f"🔒 Positive-beta: {', '.join(pb_cols)}")
    if pcb:
        n_pcb = sum(len(v) for v in pcb.values())
        st.info(f"🎛️ Per-variable bounds: {len(pcb)} channel(s), {n_pcb} params")

    if config.get("use_cochrane_orcutt"):
        st.info(
            f"🔁 **Cochrane-Orcutt AR(1) correction is ON** — the model will be fitted "
            f"up to {config.get('co_max_iter', 5)} times (ρ re-estimated from the "
            f"residuals each round, tolerance {config.get('co_tol', 0.01):g}), so expect "
            f"roughly that multiple of the normal run time. Change it in Tab 4."
        )

    prophet_in_model = [c for c in config.get("non_media", []) if c.startswith("prophet_")]
    if prophet_in_model:
        prophet_info(f"📌 Prophet control variables in model: <b>{', '.join(prophet_in_model)}</b>")

    # Validate all config columns exist in df
    missing_cols = [
        col for col in (
            config.get("media", []) + config.get("non_media", []) +
            config.get("comp_media", []) + config.get("price", []) +
            [config.get("target", "")]
        )
        if col and col not in df.columns
    ]
    if missing_cols:
        st.error(
            f"❌ Columns in saved config are missing from dataset: "
            f"`{'`, `'.join(missing_cols)}`. Re-save configuration in Tab 4."
        )
        st.stop()

    st.divider()
    st.markdown("### Optimizer Selection")
    OPTIMIZER_OPTIONS = ["L-BFGS-B", "SLSQP"]
    if nevergrad_available: OPTIMIZER_OPTIONS.append("Nevergrad")
    method = st.selectbox("Optimizer", OPTIMIZER_OPTIONS,
                           help="L-BFGS-B/SLSQP: gradient-based. "
                                "Nevergrad: derivative-free search on the same −loglik objective.")

    ng_cfg = None
    if method == "Nevergrad":
        if not nevergrad_available:
            st.error("Nevergrad not installed. Run `pip install nevergrad`."); st.stop()
        ng_info(
            "🟣 <b>Nevergrad Optimizer</b><br>"
            "Loss = <code>−loglik</code> — same objective as L-BFGS-B / SLSQP, "
            "just optimised with a derivative-free search strategy instead of a gradient-based one."
        )
        ng_col1, ng_col2 = st.columns(2)
        with ng_col1:
            ng_strategy = st.selectbox("Strategy",
                ["TwoPointsDE","NGOpt","CMA","PSO","DE","OnePlusOne","RandomSearch","MetaModel"])
            ng_budget   = st.number_input("Budget (evaluations)", 100, 10000, 500, 50)
        with ng_col2:
            ng_workers  = st.number_input("Parallel workers", 1, 8, 1, 1)
            max_iter    = ng_budget

        ng_cfg = {
            "strategy": ng_strategy, "budget": int(ng_budget), "num_workers": int(ng_workers),
        }
    else:
        col1, _ = st.columns(2)
        with col1: max_iter = st.number_input("Max iterations", 100, 5000, 800, 100)

    if st.button("🚀 Run EKF MMM", type="primary", use_container_width=True):
        spinner_text = "Running EKF optimisation… (30–300 s depending on optimizer / budget)"
        if config.get("use_cochrane_orcutt"):
            spinner_text += f"  [Cochrane-Orcutt: up to {config.get('co_max_iter', 5)} rounds]"
        with st.spinner(spinner_text):
            try:
                results = run_full_ekf_pipeline(df, config, max_iter, method, ng_cfg=ng_cfg)
                st.session_state.model_results = results
                st.session_state.model_fitted  = True
                st.success("✅ Model fitted!")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("MAPE",      f"{results['mape']:.2%}")
                c2.metric("R²",        f"{results['r2']:.4f}")
                c3.metric("Log-Lik",   f"{results['loglik']:.1f}")
                c4.metric("Converged", "Yes ✅" if results["success"] else "Partial ⚠️")
            except Exception as e:
                st.exception(e)

    if st.session_state.model_fitted:
        res = st.session_state.model_results; st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("MAPE", f"{res['mape']:.2%}")
        c2.metric("R²",   f"{res['r2']:.4f}")
        c3.metric("Log-Lik", f"{res['loglik']:.2f}")
        _show_co_summary(res, config["target"])
        st.caption("Proceed to **Tab 6** for full results.")
