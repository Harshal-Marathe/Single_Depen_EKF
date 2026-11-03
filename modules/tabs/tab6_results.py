"""
Tab 6 — Results & ROI Analytics: performance metrics, fit charts,
residual diagnostics, channel contributions, ROI, parameters, and
interactive response curves.
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from modules.ui_helpers import section, need_model
from modules.transforms import hill_transform
from modules.exports import build_betas_df, build_master_workbook_bytes


def render_tab6():
    section("06", "Results & ROI Analytics")
    if not st.session_state.model_fitted: need_model()

    res    = st.session_state.model_results
    config = st.session_state.config
    df     = st.session_state.df
    target = config["target"]; g = res["g"]

    st.markdown("### A · Model Performance")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAPE",         f"{res['mape']:.2%}")
    c2.metric("R²",           f"{res['r2']:.4f}")
    c3.metric("Log-Lik",      f"{res['loglik']:.2f}")
    c4.metric("Observations", len(df))

    st.markdown("### B · Actual vs Predicted")
    x_axis  = np.arange(len(df)); n_train = config["n_train"]
    fig_avp = go.Figure()
    fig_avp.add_vrect(x0=n_train, x1=len(df)-1, fillcolor="#fef3c7", opacity=0.35,
                      layer="below", line_width=0,
                      annotation_text="Test period", annotation_position="top left")
    fig_avp.add_trace(go.Scatter(x=x_axis, y=df[target].values,
                                  name="Actual", line=dict(color="#1e40af", width=2)))
    fig_avp.add_trace(go.Scatter(x=x_axis, y=res["yhat_smooth"],
                                  name="EKF Smoothed", line=dict(color="#f59e0b", width=2, dash="dash")))
    fig_avp.update_layout(height=460, template="plotly_white",
                           title="Actual vs EKF Smoothed",
                           legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig_avp, use_container_width=True)

    st.markdown("### C · Smoothed Intercept State")
    fig_int = go.Figure()
    fig_int.add_trace(go.Scatter(x=x_axis, y=res["x_smooth"][:, 0],
                                  name="Intercept", line=dict(color="#10b981", width=2)))
    fig_int.update_layout(height=340, template="plotly_white",
                           title="Smoothed Intercept (Base Demand) Over Time")
    st.plotly_chart(fig_int, use_container_width=True)

    st.markdown("### D · Residual Diagnostics")
    col_a, col_b = st.columns(2)
    with col_a:
        fig_h = px.histogram(res["residuals"], nbins=40, title="Residual Distribution",
                              color_discrete_sequence=["#3b82f6"])
        fig_h.update_layout(template="plotly_white", height=340)
        st.plotly_chart(fig_h, use_container_width=True)
    with col_b:
        sr = np.sort(res["residuals"])
        tq = np.sort(np.random.normal(0, np.std(res["residuals"]), len(sr)))
        fig_qq = px.scatter(x=tq, y=sr, title="Q-Q Plot",
                             labels={"x": "Theoretical Normal", "y": "Sample Residuals"})
        fig_qq.add_trace(go.Scatter(x=tq, y=tq, mode="lines",
                                     name="Ideal", line=dict(color="red", dash="dash")))
        fig_qq.update_layout(template="plotly_white", height=340)
        st.plotly_chart(fig_qq, use_container_width=True)

    st.markdown("### E · Channel Contributions")
    contrib_df = res["contrib_df"]
    short_cols = [c for c in contrib_df.columns if c.startswith("ShortTerm_")]
    long_cols  = [c for c in contrib_df.columns if c.startswith("LongTerm_")]
    totals_st  = contrib_df[short_cols].sum()
    totals_lt  = contrib_df[long_cols].sum()

    t1, t2 = st.columns(2)
    with t1:
        st.markdown("#### Short-Term Contribution Summary")
        sv = totals_st.values; sv_tot = sv.sum() + 1e-12
        df_st = pd.DataFrame({
            "Channel": [c.replace("ShortTerm_", "") for c in short_cols],
            "Total Contrib": sv.round(2),
            "Share (%)": (sv / sv_tot * 100).round(1),
        }).sort_values("Total Contrib", ascending=False).reset_index(drop=True)
        df_st.insert(0, "Rank", range(1, len(df_st)+1))
        st.dataframe(df_st, use_container_width=True, hide_index=True)
    with t2:
        st.markdown("#### Long-Term Contribution Summary")
        lv = totals_lt.values; lv_abs = np.abs(lv); lv_tot = lv_abs.sum() + 1e-12
        df_lt = pd.DataFrame({
            "Channel": [c.replace("LongTerm_", "") for c in long_cols],
            "Total Contrib": lv.round(2),
            "Share (%)": (lv_abs / lv_tot * 100).round(1),
        }).sort_values("Total Contrib", ascending=False).reset_index(drop=True)
        df_lt.insert(0, "Rank", range(1, len(df_lt)+1))
        st.dataframe(df_lt, use_container_width=True, hide_index=True)

    # FIX: cross-media synergy contributions (configured in Tab 4 ·
    # Section C) were computed by the filter and present in contrib_df
    # (see pipeline.py) but nothing in this tab ever read out the
    # Synergy_ columns — they were invisible in the UI even though the
    # numbers existed. Adding a dedicated summary here.
    synergy_cols = [c for c in contrib_df.columns if c.startswith("Synergy_")]
    if synergy_cols:
        st.markdown("#### 🔗 Cross-Media Synergy Contribution Summary")
        totals_syn = contrib_df[synergy_cols].sum()
        syn_vals = totals_syn.values
        syn_tot  = np.abs(syn_vals).sum() + 1e-12
        df_syn = pd.DataFrame({
            "Synergy Pair": [c.replace("Synergy_", "").replace("_from_", " ← from ")
                              for c in synergy_cols],
            "Total Contrib": syn_vals.round(2),
            "Share (%)": (np.abs(syn_vals) / syn_tot * 100).round(1),
        }).sort_values("Total Contrib", ascending=False).reset_index(drop=True)
        df_syn.insert(0, "Rank", range(1, len(df_syn)+1))
        st.dataframe(df_syn, use_container_width=True, hide_index=True)

        fig_syn = px.bar(
            df_syn.sort_values("Total Contrib"),
            x="Total Contrib", y="Synergy Pair", orientation="h",
            title="Cross-Media Synergy Contribution by Pair",
            color="Total Contrib", color_continuous_scale="Purples",
        )
        fig_syn.update_layout(height=max(300, 45*len(synergy_cols)), template="plotly_white")
        st.plotly_chart(fig_syn, use_container_width=True)
    else:
        st.caption("No cross-media synergy pairs were configured in Tab 4 · Section C.")

    with st.expander("📊 Pie Charts"):
        pc1, pc2 = st.columns(2)
        with pc1:
            fig_pie_st = px.pie(
                values=np.maximum(totals_st.values, 0),
                names=[c.replace("ShortTerm_", "") for c in short_cols],
                title="Short-term share",
                color_discrete_sequence=px.colors.sequential.Blues_r,
            )
            st.plotly_chart(fig_pie_st, use_container_width=True)
        with pc2:
            fig_pie_lt = px.pie(
                values=np.abs(totals_lt.values),
                names=[c.replace("LongTerm_", "") for c in long_cols],
                title="Long-term share",
                color_discrete_sequence=px.colors.sequential.Greens_r,
            )
            st.plotly_chart(fig_pie_lt, use_container_width=True)

    fig_st_area = px.area(
        contrib_df.reset_index(), x=x_axis, y=short_cols,
        title="Short-term Contributions Over Time",
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    fig_st_area.update_layout(height=400, template="plotly_white")
    st.plotly_chart(fig_st_area, use_container_width=True)

    with st.expander("Full contribution table"):
        st.dataframe(contrib_df.style.format("{:,.4f}"), use_container_width=True)
    st.download_button("📥 Download Contribution Table",
                       contrib_df.to_csv().encode(), "contributions.csv", "text/csv")

    st.markdown("### E2 · Synergy / Cross-Media Effects")
    synergy_df = res.get("synergy_df")
    synergy_cols = [c for c in contrib_df.columns if c.startswith("Synergy_")]
    if synergy_df is None or synergy_df.empty or not synergy_cols:
        st.info(
            "No cross-media synergy relationships are configured for this model. "
            "Set them up in **Tab 4 · Section C (Cross-media Learning)** and re-run."
        )
    else:
        st.caption(
            "One row per configured source → target relationship "
            "(per-variable level, not a single aggregated number)."
        )
        st.dataframe(
            synergy_df.style.format({
                "Total Synergy Contribution": "{:,.4f}",
                "Avg Synergy / Period":       "{:,.4f}",
                "Cross Delta":                "{:.4f}",
                "Cross Hill n":                "{:.3f}",
                "Cross Hill S":                "{:.3f}",
                "Share of Target's Total Contrib (%)": "{:.2f}",
            }),
            use_container_width=True, hide_index=True,
        )
        st.download_button("📥 Download Synergy Table",
                           synergy_df.to_csv(index=False).encode(),
                           "synergy_table.csv", "text/csv")

        pair_labels = [f"{row['Source Channel']} → {row['Target Channel']}"
                        for _, row in synergy_df.iterrows()]
        fig_syn_bar = px.bar(
            x=synergy_df["Total Synergy Contribution"].values, y=pair_labels,
            orientation="h", title="Total Synergy Contribution by Pair",
            labels={"x": "Total Contribution", "y": ""},
            color=synergy_df["Total Synergy Contribution"].values,
            color_continuous_scale="Purples",
        )
        fig_syn_bar.update_layout(height=max(300, 50*len(pair_labels)), template="plotly_white")
        st.plotly_chart(fig_syn_bar, use_container_width=True)

        fig_syn_area = px.area(
            contrib_df.reset_index(), x=x_axis, y=synergy_cols,
            title="Synergy Contributions Over Time (per pair)",
            color_discrete_sequence=px.colors.qualitative.Vivid,
        )
        fig_syn_area.update_layout(height=380, template="plotly_white")
        st.plotly_chart(fig_syn_area, use_container_width=True)

    st.markdown("### F · Estimated Parameters")
    param_df = res["param_df"]
    st.dataframe(param_df.style.format({"Value": "{:.6f}"}),
                 use_container_width=True, hide_index=True)
    st.download_button("📥 Download Parameters",
                       param_df.to_csv(index=False).encode(), "parameters.csv", "text/csv")

    st.markdown("### G · ROI Analytics")
    roi_df = res["roi_df"]
    if not roi_df.empty:
        c1, c2 = st.columns([2, 1])
        with c1:
            fig_roi = px.bar(roi_df.sort_values("ROI", ascending=True),
                              x="ROI", y="Channel", orientation="h",
                              title="ROI by Media Channel",
                              color="ROI", color_continuous_scale="RdYlGn")
            fig_roi.add_vline(x=1.0, line_dash="dash", line_color="black",
                               annotation_text="Break-even")
            fig_roi.update_layout(height=360, template="plotly_white")
            st.plotly_chart(fig_roi, use_container_width=True)
        with c2:
            st.dataframe(roi_df.style.format({
                "TotalSpend":   "{:,.2f}",
                "TotalContrib": "{:,.2f}",
                "ROI":          "{:.4f}",
            }), use_container_width=True, hide_index=True)
        best  = roi_df.loc[roi_df["ROI"].idxmax()]
        worst = roi_df.loc[roi_df["ROI"].idxmin()]
        st.success(f"🏆 Best ROI: **{best['Channel']}** ({best['ROI']:.4f})")
        st.warning(f"⚠️ Lowest ROI: **{worst['Channel']}** ({worst['ROI']:.4f})")
        st.download_button("📥 Download ROI Report",
                           roi_df.to_csv(index=False).encode(), "roi_report.csv", "text/csv")

    st.markdown("### H · Response Curves")
    params       = res["params"]
    media_list   = config["media"]

    if not media_list:
        st.info("No media channels configured.")
    else:
        sel = st.selectbox("Select channel", media_list, key="rc_sel")
        idx = media_list.index(sel)

        rc1, rc2, rc3 = st.columns(3)
        with rc1: x_max_pct = st.slider("X-axis max (%)", 50, 300, 150, 10, key="rc_xmax")
        with rc2: n_points  = st.slider("Curve resolution", 50, 500, 200, 50, key="rc_npts")
        with rc3: show_ci   = st.checkbox("Show beta band", value=True, key="rc_ci")

        col_data = df[sel]; x_max = col_data.max() * x_max_pct / 100
        x_range  = np.linspace(0, x_max, n_points)
        beta_med = float(np.median(res["x_smooth"][:, idx+1]))
        beta_p25 = float(np.percentile(res["x_smooth"][:, idx+1], 25))
        beta_p75 = float(np.percentile(res["x_smooth"][:, idx+1], 75))
        n_v = params["n_params"][idx]; S_v = params["S_params"][idx]

        # Hill is applied directly to raw media spend (adstock has been
        # removed from the model entirely), so the response curve's
        # x-axis is raw spend with no carry-over scaling applied.
        def make_resp(b):
            return np.array([b * hill_transform(x, n_v, S_v) for x in x_range])

        resp_med = make_resp(beta_med)
        resp_p25 = make_resp(beta_p25)
        resp_p75 = make_resp(beta_p75)
        marginal = np.gradient(resp_med, x_range)

        pct_10 = np.percentile(col_data[col_data > 0], 10) if (col_data > 0).any() else 0
        pct_50 = np.percentile(col_data[col_data > 0], 50) if (col_data > 0).any() else 0
        pct_90 = np.percentile(col_data[col_data > 0], 90) if (col_data > 0).any() else 0
        mean_v = col_data.mean()
        def resp_at(v): return float(np.interp(v, x_range, resp_med))

        fig_rc = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35],
                                subplot_titles=[
                                    f"Total Response — {sel}  (β median={beta_med:.4f})",
                                    "Marginal Response"],
                                vertical_spacing=0.08)

        if show_ci:
            fig_rc.add_trace(go.Scatter(
                x=np.concatenate([x_range, x_range[::-1]]),
                y=np.concatenate([resp_p75, resp_p25[::-1]]),
                fill="toself", fillcolor="rgba(59,130,246,0.15)",
                line=dict(color="rgba(0,0,0,0)"), name="β 25–75th band",
            ), row=1, col=1)

        fig_rc.add_trace(go.Scatter(x=x_range, y=resp_med, mode="lines",
                                     name="Response (β median)",
                                     line=dict(color="#1e40af", width=3)), row=1, col=1)

        for pv, pn, clr in [
            (pct_10, "P10", "#94a3b8"), (mean_v, "Mean", "#f59e0b"),
            (pct_50, "Median", "#10b981"), (pct_90, "P90", "#ef4444"),
        ]:
            if 0 < pv <= x_max:
                fig_rc.add_vline(x=pv, line_dash="dot", line_color=clr, line_width=1.5,
                                  annotation_text=pn, annotation_font_size=10, row=1, col=1)
                fig_rc.add_trace(go.Scatter(
                    x=[pv], y=[resp_at(pv)], mode="markers",
                    marker=dict(color=clr, size=9),
                    name=f"{pn} → {resp_at(pv):,.2f}"), row=1, col=1)

        fig_rc.add_trace(go.Scatter(x=x_range, y=marginal, mode="lines", name="Marginal",
                                     line=dict(color="#7c3aed", width=2)), row=2, col=1)
        fig_rc.add_hline(y=0, line_dash="dash", line_color="#94a3b8", row=2, col=1)
        fig_rc.update_xaxes(title_text="Spend / Input", row=2, col=1)
        fig_rc.update_yaxes(title_text="KPI Contribution", row=1, col=1)
        fig_rc.update_yaxes(title_text="dKPI / dSpend", row=2, col=1)
        fig_rc.update_layout(height=620, template="plotly_white",
                              legend=dict(orientation="h", y=1.08, x=0))
        st.plotly_chart(fig_rc, use_container_width=True)

        pc = st.columns(6)
        pc[0].metric("β median", f"{beta_med:.4f}")
        pc[1].metric("β P25",    f"{beta_p25:.4f}")
        pc[2].metric("β P75",    f"{beta_p75:.4f}")
        pc[3].metric("Hill n",   f"{n_v:.3f}")
        pc[4].metric("Hill S",   f"{S_v:.3f}")
        if adstock_type == "weibull":
            pc[5].metric("Weibull α/θ",
                         f"{params['adstock_alpha'][idx]:.3f}/{params['adstock_theta'][idx]:.3f}")
        else:
            pc[5].metric("NA λ", f"{params['adstock_lambda'][idx]:.3f}")
        st.caption(f"Adstock: **{adstock_label}** · Steady-state multiplier: **{am:.3f}×**")

        pcb = config.get("per_channel_bounds", {}).get(sel, {})
        if pcb:
            st.markdown("#### Per-Variable Bounds Applied")
            st.dataframe(
                pd.DataFrame([{"Parameter": k,
                               "Min": f"{v[0]:.4g}",
                               "Max": f"{v[1]:.4g}" if v[1] is not None else "∞"}
                              for k, v in pcb.items()]),
                use_container_width=True, hide_index=True,
            )

    st.divider()
    st.markdown("### I · Export Center")
    info_text = (
        "Three ways to take results outside the app: just the betas, just the "
        "hyperparameters, or one structured Excel workbook with everything laid "
        "out as <b>Raw variables → Transformed variables → Betas → "
        "Contributions (Beta × Transformed)</b>, plus hyperparameters / ROI / "
        "synergy on their own sheets."
    )
    st.markdown(f'<div class="info-box">{info_text}</div>', unsafe_allow_html=True)

    ec1, ec2, ec3 = st.columns(3)
    with ec1:
        betas_df = build_betas_df(res, df)
        st.download_button(
            "📥 Download Betas (Time Series)",
            betas_df.to_csv(index=False).encode(),
            "betas_timeseries.csv", "text/csv",
            use_container_width=True,
        )
    with ec2:
        st.download_button(
            "📥 Download Hyperparameters",
            param_df.to_csv(index=False).encode(),
            "hyperparameters.csv", "text/csv",
            use_container_width=True,
        )
    with ec3:
        workbook_bytes = build_master_workbook_bytes(res, config, df)
        st.download_button(
            "📊 Download Full Excel Workbook",
            workbook_bytes,
            "rainbrain_model_export.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )