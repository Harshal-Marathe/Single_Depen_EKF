"""
Tab 4 — Model Configuration: variable roles, positive-beta constraints,
cross-media learning, per-variable hyperparameter bounds, train/test
split, and saving the final config.

Adstock has been removed from the model entirely (own and competitor
media both now contribute via raw spend directly), so there is no
longer an Adstock Function section here.
"""

import numpy as np
import pandas as pd
import streamlit as st

from modules.ui_helpers import (
    section, info, positive_info, per_channel_info, prophet_info,
    need_data, safe_multiselect,
)


def render_tab4():
    section("04", "Model Configuration")
    if st.session_state.df is None: need_data()

    # Always read from session state — captures prophet cols after rerun
    df = st.session_state.df
    num_cols = df.select_dtypes(include=np.number).columns.tolist()

    if st.session_state.prophet_cols_added:
        missing_now = [c for c in st.session_state.prophet_cols_added if c not in num_cols]
        if missing_now:
            st.error(
                f"⚠️ Prophet column(s) were added but are no longer numeric / present in the "
                f"dataset: `{'`, `'.join(missing_now)}`. Re-run the prophet merge in Tab 2."
            )
        prophet_info(
            f"📌 Prophet columns available: "
            f"<b>{', '.join(st.session_state.prophet_cols_added)}</b>. "
            "They are pre-selected in <b>Non-media / organic</b> below — "
            "just save the configuration to include them in the model."
        )

    # ── A. Variable Selection ─────────────────────────────────────────
    st.markdown("### A · Variable Selection")
    col1, col2 = st.columns(2)
    with col1:
        target = st.selectbox("🎯 Dependent variable (KPI)", num_cols, key="cfg_target")
    with col2:
        remaining = [c for c in num_cols if c != target]
        media = safe_multiselect("📺 Media / paid channels", options=remaining,
                                  default=[], key="cfg_media")

    remaining2 = [c for c in remaining if c not in media]
    col3, col4 = st.columns(2)
    with col3:
        # FIX: prophet columns are passed via `require=`, not `default=`.
        # `default` only ever applies the very first time this key is
        # created — Streamlit (and our own sanitizer) treats the key as
        # "owned by the widget" from that point on, which is almost
        # immediately (the key exists with value [] as soon as this
        # widget first renders, before the user clicks anything). Using
        # `require=` re-injects the prophet columns into the stored
        # selection on EVERY render, so they survive no matter what else
        # changes (media picks, target, tab switches, etc).
        prophet_in_scope = [c for c in st.session_state.prophet_cols_added
                             if c in remaining2]
        non_media = safe_multiselect(
            "🗂️ Non-media / organic (incl. prophet columns)",
            options=remaining2,
            require=prophet_in_scope,
            key="cfg_nonmedia",
        )
    with col4:
        remaining3 = [c for c in remaining2 if c not in non_media]
        price_vars = safe_multiselect("💲 Price variables", options=remaining3,
                                       default=[], key="cfg_price")

    remaining4 = [c for c in remaining3 if c not in price_vars]
    col5, col6 = st.columns(2)
    with col5:
        comp_media = safe_multiselect("📉 Competitor media", options=remaining4,
                                       default=[], key="cfg_comp_media")
    with col6:
        remaining5 = [c for c in remaining4 if c not in comp_media]
        comp_nonmedia = safe_multiselect("📉 Competitor non-media", options=remaining5,
                                          default=[], key="cfg_comp_nonmedia")

    st.divider()

    # ── B. Positive Beta Constraints ──────────────────────────────────
    st.markdown("### B · Positive Beta Constraints")
    positive_info(
        "🔒 <b>Positive Beta Enforcement</b> — Select variables that must contribute "
        "<b>positively</b> to the KPI. The optimizer enforces a non-negative lower bound "
        "on their <code>delta</code> parameters."
    )
    all_constrainable = list(media) + list(non_media)
    if all_constrainable:
        positive_beta_cols = safe_multiselect(
            "📈 Variables that must have positive betas",
            options=all_constrainable, default=list(media), key="positive_beta_cols")
    else:
        positive_beta_cols = []
        st.info("Select media or non-media channels in Section A first.")

    st.divider()

    # ── C. Cross-media Learning ───────────────────────────────────────
    st.markdown("### C · Cross-media Learning")
    info("Define which channels learn from each other.")
    cross_map = {}
    if media:
        for tgt in media:
            sources = safe_multiselect(
                f"Channels that influence **{tgt}**",
                options=[m for m in media if m != tgt], default=[], key=f"cross_{tgt}")
            if sources:
                cross_map[tgt] = set(sources)

    st.divider()

    # ── D. Per-Variable Hyperparameter Bounds ────────────────────────
    st.markdown("### D · Per-Variable Hyperparameter Bounds")
    per_channel_info(
        "🎛️ <b>All hyperparameter bounds are set here, per variable.</b> "
        "Defaults are derived automatically from each channel's data distribution. "
        "Expand a channel to customise its bounds."
    )

    all_channel_cols = list(media) + list(comp_media)
    per_channel_bounds: dict = {}

    if all_channel_cols:
        for col in all_channel_cols:
            is_comp = col in comp_media
            col_label = f"📉 {col}  *(competitor)*" if is_comp else f"📺 {col}"
            vals    = df[col][df[col] > 0] if col in df.columns else pd.Series([1.0])
            col_med = float(vals.median()) if len(vals) > 0 else 1.0
            col_p90 = float(vals.quantile(0.90)) if len(vals) > 0 else 1.0
            col_max = float(vals.max())    if len(vals) > 0 else 1.0
            def_s_lo = max(1e-6, col_med * 0.01)
            def_s_hi = col_max * 5.0

            with st.expander(col_label, expanded=False):
                st.caption(f"📊 median: **{col_med:,.2f}** · P90: **{col_p90:,.2f}** · "
                           f"max: **{col_max:,.2f}**")
                per_channel_bounds[col] = {}

                st.markdown("**Beta Persistence (Ls)**")
                lsc1, lsc2 = st.columns(2)
                with lsc1:
                    ls_lo = st.number_input("Ls min", 0.0, 1.0,
                                             0.8 if is_comp else 0.5, 0.01, key=f"ls_lo_{col}")
                with lsc2:
                    ls_hi = st.number_input("Ls max", 0.0, 1.0, 1.0, 0.01, key=f"ls_hi_{col}")
                per_channel_bounds[col]["ls"] = (ls_lo, ls_hi)
                st.divider()

                st.markdown("**Hill Slope (n)**")
                hnc1, hnc2 = st.columns(2)
                with hnc1:
                    hn_lo = st.number_input("Hill n min", 0.01, 20.0, 0.3, 0.1, key=f"hn_lo_{col}")
                with hnc2:
                    hn_hi = st.number_input("Hill n max", 0.01, 20.0, 5.0, 0.1, key=f"hn_hi_{col}")
                per_channel_bounds[col]["hill_n"] = (hn_lo, hn_hi)
                st.divider()

                st.markdown(f"**Hill Half-Saturation (S)** — data range: 0 – {col_max:,.0f}")
                hsc1, hsc2 = st.columns(2)
                with hsc1:
                    hs_lo = st.number_input("Hill S min", 1e-9, 1e12,
                                             float(f"{def_s_lo:.4g}"),
                                             key=f"hs_lo_{col}", format="%.4g")
                with hsc2:
                    hs_hi = st.number_input("Hill S max", 1e-9, 1e12,
                                             float(f"{def_s_hi:.4g}"),
                                             key=f"hs_hi_{col}", format="%.4g")
                per_channel_bounds[col]["hill_s"] = (hs_lo, hs_hi)

                st.caption(f"✅ Bounds set for: {', '.join(per_channel_bounds[col].keys())}")
    else:
        st.info("Select media channels in Section A first.")

    if per_channel_bounds:
        rows = [
            {"Channel": col,
             "Type": "Competitor" if col in comp_media else "Own Media",
             "Parameter": param,
             "Min": f"{v[0]:.4g}",
             "Max": f"{v[1]:.4g}" if v[1] is not None else "∞"}
            for col, bdict in per_channel_bounds.items()
            for param, v in bdict.items()
        ]
        if rows:
            with st.expander("📋 All per-variable bounds summary", expanded=False):
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.divider()

    # ── E. Train / Test Split ─────────────────────────────────────────
    st.markdown("### E · Train / Test Split")
    train_ratio = st.slider("Training proportion", 0.50, 0.95, 0.80, 0.05,
                             format="%.0f%%", key="train_ratio")
    n_total = len(df); n_train = int(n_total * train_ratio); n_test = n_total - n_train
    c1, c2, c3 = st.columns(3)
    c1.metric("Total", n_total); c2.metric("Train", n_train); c3.metric("Test", n_test)

    st.divider()

    # ── F. Additional Options ─────────────────────────────────────────
    st.markdown("### F · Additional Options")
    c1, c2 = st.columns(2)
    with c1: use_organic = st.checkbox("Organic drift (μ) in intercept state", key="use_organic")
    with c2: use_price   = st.checkbox("Include price effects", value=bool(price_vars), key="use_price")
    intercept_effector_options = list(media) + list(non_media)
    intercept_effectors = safe_multiselect(
        "Intercept effectors (media or non-media/organic variables boosting baseline)",
        options=intercept_effector_options, default=list(media), key="intercept_eff")
    if any(c in non_media for c in intercept_effectors):
        info(
            "Non-media effectors boost the intercept using their <b>raw value</b> each "
            "period (no Hill/saturation curve is applied — consistent with how "
            "non-media variables are treated everywhere else in the model)."
        )

    if st.button("💾 Save Configuration", type="primary", use_container_width=True):
        if not media:
            st.error("Select at least one media channel.")
        else:
            n_bounds_set = sum(len(v) for v in per_channel_bounds.values())
            prophet_in_model = [c for c in non_media if c.startswith("prophet_")]
            st.session_state.config = {
                "target": target,
                "media": media,
                "non_media": non_media,
                "price": price_vars if use_price else [],
                "comp_media": comp_media,
                "comp_nonmedia": comp_nonmedia,
                "dummy_cols": [],
                "intercept_effectors": intercept_effectors,
                "cross_media_map": cross_map,
                "use_organic": use_organic,
                "use_price": use_price,
                "train_ratio": train_ratio,
                "n_train": n_train,
                "n_test": n_test,
                "positive_beta_cols": positive_beta_cols,
                "per_channel_bounds": per_channel_bounds,
                "initial_media_betas":         {c: 0.0     for c in media},
                "initial_comp_betas":          {c: -0.0001 for c in comp_media},
                "initial_own_nonmedia_betas":  {c: 0.0     for c in non_media},
                "initial_comp_nonmedia_betas": {c: -0.01   for c in comp_nonmedia},
                "initial_price_beta":          {c: -0.1    for c in price_vars},
            }
            st.success(
                f"✅ Saved — {len(media)} own-media · {len(comp_media)} competitor · "
                f"{len(non_media)} non-media "
                f"({len(prophet_in_model)} prophet col{'s' if len(prophet_in_model)!=1 else ''}) · "
                f"train/test: **{n_train}/{n_test}** · "
                f"positive-beta: **{len(positive_beta_cols)}** · "
                f"per-variable bound params: **{n_bounds_set}**"
            )
            if prophet_in_model:
                prophet_info(
                    f"📌 Prophet controls included in model: "
                    f"<b>{', '.join(prophet_in_model)}</b>"
                )