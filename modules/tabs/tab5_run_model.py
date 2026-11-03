"""
Tab 5 — Run EKF Model: optimizer selection (L-BFGS-B / SLSQP / Nevergrad)
and kicking off the full pipeline.
"""

import streamlit as st

from modules.ui_helpers import section, info, ng_info, prophet_info, need_data, need_config
from modules.pipeline import run_full_ekf_pipeline


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
    c3.metric("Competitor media", len(config.get("comp_media", [])))
    c4.metric("Train / Test",   f"{config['n_train']} / {config['n_test']}")

    pb_cols = config.get("positive_beta_cols", [])
    pcb     = config.get("per_channel_bounds", {})
    if pb_cols: st.info(f"🔒 Positive-beta: {', '.join(pb_cols)}")
    if pcb:
        n_pcb = sum(len(v) for v in pcb.values())
        st.info(f"🎛️ Per-variable bounds: {len(pcb)} channel(s), {n_pcb} params")

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
                                "Nevergrad: derivative-free multi-objective.")

    ng_cfg = None
    if method == "Nevergrad":
        if not nevergrad_available:
            st.error("Nevergrad not installed. Run `pip install nevergrad`."); st.stop()
        ng_info(
            "🟣 <b>Nevergrad Multi-Objective Optimizer</b><br>"
            "Loss = <code>−w_loglik·loglik + w_mape·max(0,MAPE−target)² "
            "+ w_r2·max(0,R²_target−R²)² + w_pos·positivity_violations</code>"
        )
        ng_col1, ng_col2 = st.columns(2)
        with ng_col1:
            ng_strategy = st.selectbox("Strategy",
                ["NgIohh","DE","TwoPointsDE","CMA","PSO","OnePlusOne","RandomSearch","MetaModel"])
            ng_budget   = st.number_input("Budget (evaluations)", 100, 10000, 500, 50)
        with ng_col2:
            ng_workers  = st.number_input("Parallel workers", 1, 8, 1, 1)
            max_iter    = ng_budget

        st.markdown("#### Loss Weights")
        wc1, wc2, wc3, wc4 = st.columns(4)
        with wc1: w_loglik = st.number_input("w_loglik", 0.0, 100.0,    1.0, 0.5)
        with wc2: w_mape   = st.number_input("w_mape",   0.0, 1000.0,  10.0, 1.0)
        with wc3: w_r2     = st.number_input("w_r2",     0.0, 1000.0,   5.0, 0.5)
        with wc4: w_pos    = st.number_input("w_pos",    0.0, 10000.0, 100.0, 10.0)
        tc1, tc2 = st.columns(2)
        with tc1: mape_target = st.slider("MAPE target", 0.01, 0.50, 0.10, 0.01, format="%.2f")
        with tc2: r2_target   = st.slider("R² target",   0.50, 0.99, 0.80, 0.01, format="%.2f")
        ng_cfg = {
            "strategy": ng_strategy, "budget": int(ng_budget), "num_workers": int(ng_workers),
            "ng_weights": {
                "w_loglik": w_loglik, "w_mape": w_mape, "w_r2": w_r2, "w_pos": w_pos,
                "mape_target": mape_target, "r2_target": r2_target,
            },
        }
        st.markdown(
            f"> Loss = `−{w_loglik:.1f}·loglik` + "
            f"`{w_mape:.1f}·max(0,MAPE−{mape_target:.2f})²` + "
            f"`{w_r2:.1f}·max(0,{r2_target:.2f}−R²)²` + "
            f"`{w_pos:.1f}·positivity_violations`"
        )
    else:
        col1, _ = st.columns(2)
        with col1: max_iter = st.number_input("Max iterations", 100, 5000, 800, 100)

    if st.button("🚀 Run EKF MMM", type="primary", use_container_width=True):
        with st.spinner("Running EKF optimisation… (30–300 s depending on optimizer / budget)"):
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
        st.caption("Proceed to **Tab 6** for full results.")