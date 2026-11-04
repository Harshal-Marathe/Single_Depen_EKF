"""
Tab 4 — Model Configuration: variable roles, positive-beta constraints,
cross-media learning, adstock type, transformation type, lag count,
per-variable hyperparameter bounds, train/test split, and saving config.
"""

import numpy as np
import pandas as pd
import streamlit as st

from modules.ui_helpers import (
    section, info, positive_info, per_channel_info, prophet_info,
    weibull_placeholder, need_data, safe_multiselect,
)


def render_tab4():
    section("04", "Model Configuration")
    if st.session_state.df is None: need_data()

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

    # ── A2. Spend Mapping (for ROI) ───────────────────────────────────
    st.markdown("### A2 · Spend Mapping (for ROI)")
    info(
        "For every media / competitor-media variable, tell us whether the modelling "
        "column itself already <b>is</b> Spend, or whether it's an "
        "<b>Impressions / GRP / NGRP</b> metric — in which case you pick the raw "
        "Spend column that goes with it. Nothing here is auto-detected; you choose "
        "explicitly for every variable. This mapping drives the <b>Sum of Spends</b> "
        "and <b>ROI</b> columns shown later in the Contribution and ROI tables."
    )

    spend_map_vars = list(media) + list(comp_media)
    media_spend_map: dict = {}
    spend_col_candidates = [c for c in num_cols if c != target]

    with st.expander("📊 Configure spend per media channel", expanded=True):
        if spend_map_vars:
            for i, var in enumerate(spend_map_vars):
                is_comp_var = var in comp_media
                st.markdown(f"**{var}**{'  *(competitor)*' if is_comp_var else ''}")
                q1, q2 = st.columns([1, 1.2])
                with q1:
                    choice = st.radio(
                        f"Is '{var}' itself a spend column?",
                        ["Yes — it is spend", "No — it is impressions / GRP"],
                        key=f"spend_isspend_{var}",
                    )
                is_spend = choice.startswith("Yes")
                with q2:
                    if is_spend:
                        st.markdown(
                            f'<div style="background:#052e1a;border-left:4px solid #22c55e;'
                            f'padding:8px 12px;border-radius:4px;margin-top:28px;">'
                            f'Spend = raw values of <b>{var}</b></div>',
                            unsafe_allow_html=True,
                        )
                        spend_col = var
                    else:
                        others = [c for c in spend_col_candidates if c != var]
                        spend_col = st.selectbox(
                            f"Select the Spend column for '{var}'",
                            options=others,
                            index=None,
                            placeholder="Select spend column…",
                            key=f"spend_col_{var}",
                        )
                media_spend_map[var] = {"is_spend": is_spend, "spend_col": spend_col}
                if i < len(spend_map_vars) - 1:
                    st.divider()
        else:
            st.info("Select media or competitor-media channels in Section A first.")

    st.divider()

    # ── B. Beta Sign Constraints ─────────────────────────────────────
    st.markdown("### B · Beta Sign Constraints")

    all_own   = list(media) + list(non_media)
    all_comp  = list(comp_media) + list(comp_nonmedia)
    all_price = list(price_vars)
    all_sign_cols = all_own + all_comp + all_price

    bcol1, bcol2 = st.columns(2)

    with bcol1:
        positive_info(
            "📈 <b>Positive Beta Enforcement</b> — Variables selected here must "
            "contribute <b>positively</b> to the KPI. The optimizer enforces "
            "non-negative <code>delta</code> bounds and floors filtered betas at zero."
        )
        if all_own:
            positive_beta_cols = safe_multiselect(
                "📈 Variables that must have POSITIVE betas",
                options=all_own, default=list(media), key="positive_beta_cols")
        else:
            positive_beta_cols = []
            st.info("Select media or non-media channels in Section A first.")

    with bcol2:
        st.markdown(
            '<div style="background:#1e293b;border-left:4px solid #ef4444;'
            'padding:8px 12px;border-radius:4px;margin-bottom:8px;">'
            '📉 <b>Negative Beta Enforcement</b> — Variables selected here must '
            'contribute <b>negatively</b> to the KPI. The optimizer enforces '
            'non-positive <code>delta</code> bounds and caps filtered betas at zero.'
            '</div>', unsafe_allow_html=True)
        neg_candidates = all_comp + all_price + all_own
        # default: competitor media + price are naturally negative
        neg_defaults = list(comp_media) + list(comp_nonmedia) + list(price_vars)
        neg_defaults = [c for c in neg_defaults if c in neg_candidates]
        if neg_candidates:
            negative_beta_cols = safe_multiselect(
                "📉 Variables that must have NEGATIVE betas",
                options=neg_candidates, default=neg_defaults, key="negative_beta_cols")
            # Remove overlap — positive takes precedence
            negative_beta_cols = [c for c in negative_beta_cols
                                   if c not in positive_beta_cols]
        else:
            negative_beta_cols = []
            st.info("Select channels in Section A first.")

    if positive_beta_cols or negative_beta_cols:
        st.caption(
            f"🔒 Positive: **{', '.join(positive_beta_cols) or 'none'}**  |  "
            f"📉 Negative: **{', '.join(negative_beta_cols) or 'none'}**"
        )

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

    # ── D. Adstock & Transformation ───────────────────────────────────
    st.markdown("### D · Adstock Function & Media Transformation")

    info(
        "Choose <b>Adstock</b> (how past spend carries over) and <b>Transformation</b> "
        "(how raw spend maps to marketing effectiveness). These two choices define the "
        "state equation used for all media beta coefficients."
    )

    dcol1, dcol2 = st.columns(2)

    with dcol1:
        st.markdown("#### Adstock (Carry-over)")
        adstock_choice = st.radio(
            "Adstock type",
            ["Instant (Nerlove-Arrow / geometric)", "Delayed (Weibull distributed lag)"],
            horizontal=False,
            key="adstock_type_radio",
            help=(
                "**Instant**: β_t = λ·β_{t-1} + δ·f(x_t). "
                "Fast geometric decay — parameter λ ∈ (0,1).\n\n"
                "**Delayed (Weibull)**: β_t = Σ w_l·x_{t-l} + δ·f(x_t). "
                "Weighted lag distribution — parameters shape k, scale λ, number of lags."
            ),
        )
        use_weibull = adstock_choice.startswith("Delayed")

        if use_weibull:
            n_lags = st.number_input(
                "Number of lags to consider (L)",
                min_value=1, max_value=52, value=8, step=1,
                key="adstock_n_lags",
                help="Weibull weights are computed for lag = 0, 1, …, L-1 and normalised to sum to 1.",
            )
            st.caption(
                f"📐 Weibull PDF: w_lag = (k/λ) · ((lag+1)/λ)^(k−1), "
                f"normalised over {n_lags} lags. Parameters **shape k** and **scale λ** "
                "are fitted per channel (set bounds below)."
            )
        else:
            n_lags = 8  # default, unused for instant

    with dcol2:
        st.markdown("#### Transformation (Response Curve)")
        transform_choice = st.radio(
            "Transformation type",
            ["Hill (S-curve saturation)", "Power (diminishing returns)"],
            horizontal=False,
            key="transform_type_radio",
            help=(
                "**Hill**: f(x) = x^n / (x^n + S^n). "
                "S-shaped saturation curve — parameters n ∈ [1,15], S > 0.\n\n"
                "**Power**: f(x) = x^n. "
                "Pure diminishing returns — parameter n ∈ (0,1]."
            ),
        )
        use_hill = transform_choice.startswith("Hill")

    # Summary box showing the active state equation
    adstock_label = "Delayed (Weibull)" if use_weibull else "Instant (λ)"
    transform_label = "Hill(x; n, S)" if use_hill else "x^n"
    transform_type_str = "hill" if use_hill else "power"
    adstock_type_str   = "weibull" if use_weibull else "instant"

    if use_weibull and use_hill:
        eq_text = (
            "β_{i,t} = **Σ w_l · x_{i,t-l}** + δ_i · **Hill(x_{i,t})** "
            "+ Σ_j δ_{ij} · Hill(x_{j,t})"
        )
        params_text = "Parameters: shape k, scale λ (per lag group), n, S per channel"
    elif use_weibull and not use_hill:
        eq_text = (
            "β_{i,t} = **Σ w_l · x_{i,t-l}** + δ_i · **x_{i,t}^n** "
            "+ Σ_j δ_{ij} · x_{j,t}^n"
        )
        params_text = "Parameters: shape k, scale λ, n per channel  (n ∈ (0,1))"
    elif not use_weibull and use_hill:
        eq_text = (
            "β_{i,t} = **λ_i · β_{i,t-1}** + δ_i · **Hill(x_{i,t})** "
            "+ Σ_j δ_{ij} · Hill(x_{j,t})"
        )
        params_text = "Parameters: λ (decay), n, S per channel"
    else:
        eq_text = (
            "β_{i,t} = **λ_i · β_{i,t-1}** + δ_i · **x_{i,t}^n** "
            "+ Σ_j δ_{ij} · x_{j,t}^n"
        )
        params_text = "Parameters: λ (decay), n per channel  (n ∈ (0,1))"

    st.info(f"**Active state equation:** {eq_text}\n\n*{params_text}*")

    st.markdown(
        "**Intercept state (all modes):** "
        "I_t = G₀ · I_{t-1} + Σ_k γ_k · media_k,t^{n_k}"
    )

    st.divider()

    # ── D2. Per-Variable Hyperparameter Bounds ───────────────────────
    st.markdown("### D2 · Per-Variable Hyperparameter Bounds")
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

                # Beta persistence
                st.markdown("**Beta Persistence (Ls)**")
                lsc1, lsc2 = st.columns(2)
                with lsc1:
                    ls_lo = st.number_input("Ls min", 0.0, 1.0,
                                             0.8 if is_comp else 0.5, 0.01, key=f"ls_lo_{col}")
                with lsc2:
                    ls_hi = st.number_input("Ls max", 0.0, 1.0, 1.0, 0.01, key=f"ls_hi_{col}")
                per_channel_bounds[col]["ls"] = (ls_lo, ls_hi)
                st.divider()

                # Transformation parameters
                if use_hill:
                    st.markdown("**Hill Slope (n)** — controls steepness of S-curve")
                    hnc1, hnc2 = st.columns(2)
                    with hnc1:
                        hn_lo = st.number_input("n min", 0.01, 15.0, 1.0, 0.1, key=f"hn_lo_{col}")
                    with hnc2:
                        hn_hi = st.number_input("n max", 0.01, 15.0, 15.0, 0.5, key=f"hn_hi_{col}")
                    per_channel_bounds[col]["hill_n"] = (hn_lo, hn_hi)
                    st.divider()

                    st.markdown(f"**Hill Half-Saturation (S)** — data range: 0 – {col_max:,.0f}")
                    hsc1, hsc2 = st.columns(2)
                    with hsc1:
                        hs_lo = st.number_input("S min", 1e-9, 1e12,
                                                 float(f"{def_s_lo:.4g}"),
                                                 key=f"hs_lo_{col}", format="%.4g")
                    with hsc2:
                        hs_hi = st.number_input("S max", 1e-9, 1e12,
                                                 float(f"{def_s_hi:.4g}"),
                                                 key=f"hs_hi_{col}", format="%.4g")
                    per_channel_bounds[col]["hill_s"] = (hs_lo, hs_hi)
                else:
                    st.markdown("**Power Exponent (n)** — n ∈ (0,1] for diminishing returns")
                    pnc1, pnc2 = st.columns(2)
                    with pnc1:
                        pn_lo = st.number_input("n min", 0.001, 1.0, 0.01, 0.01,
                                                 key=f"pn_lo_{col}")
                    with pnc2:
                        pn_hi = st.number_input("n max", 0.001, 1.0, 1.0, 0.01,
                                                 key=f"pn_hi_{col}")
                    per_channel_bounds[col]["transform_n"] = (pn_lo, pn_hi)
                st.divider()

                # Adstock parameters
                if use_weibull:
                    st.markdown("**Weibull Adstock — Shape (k) and Scale (λ)**")
                    st.caption(
                        "Shape k controls the weight distribution across lags "
                        "(k < 1: front-loaded, k = 1: exponential, k > 1: bell-shaped). "
                        "Scale λ controls how quickly weights decay."
                    )
                    wa1, wa2, wa3, wa4 = st.columns(4)
                    with wa1: wa_klo = st.number_input("k min", 0.01, 10.0, 0.1, 0.1,
                                                        key=f"wa_klo_{col}")
                    with wa2: wa_khi = st.number_input("k max", 0.01, 10.0, 5.0, 0.1,
                                                        key=f"wa_khi_{col}")
                    with wa3: wl_lo  = st.number_input("λ min", 0.01, 10.0, 0.1, 0.1,
                                                        key=f"wl_lo_{col}")
                    with wa4: wl_hi  = st.number_input("λ max", 0.01, 10.0, 5.0, 0.1,
                                                        key=f"wl_hi_{col}")
                    per_channel_bounds[col]["adstock_shape"] = (wa_klo, wa_khi)
                    per_channel_bounds[col]["adstock_scale"] = (wl_lo,  wl_hi)
                else:
                    st.markdown("**Instant Adstock Decay (λ)** — λ ∈ [0,1)")
                    na1, na2 = st.columns(2)
                    with na1: na_lo = st.number_input("λ min", 0.0, 0.999, 0.0,  0.01,
                                                       key=f"na_lo_{col}")
                    with na2: na_hi = st.number_input("λ max", 0.0, 0.999, 0.99, 0.01,
                                                       key=f"na_hi_{col}")
                    per_channel_bounds[col]["adstock_lambda"] = (na_lo, na_hi)

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
    # Any numeric column in the dataset (except the target KPI) can boost the
    # intercept — not just the variables picked as media / non-media above.
    intercept_effector_options = [c for c in num_cols if c != target]
    intercept_effectors = safe_multiselect(
        "Intercept effectors (any variable from the data that boosts baseline)",
        options=intercept_effector_options, default=list(media), key="intercept_eff")
    if intercept_effectors:
        info(
            "Every intercept effector — whether it is a media channel, a non-media "
            "variable, or any other column from your data — gets the <b>same "
            "Hill/Power transformation</b> when boosting the intercept, each with its "
            "own fitted exponent (<code>n_intercept</code>), half-saturation/scale "
            "(<code>S_intercept</code>) and boost coefficient (<code>gamma</code>)."
        )

    if st.button("💾 Save Configuration", type="primary", use_container_width=True):
        missing_spend = [
            v for v, d in media_spend_map.items()
            if not d["is_spend"] and not d["spend_col"]
        ]
        if not media:
            st.error("Select at least one media channel.")
        elif missing_spend:
            st.error(
                "⚠️ Pick a Spend column for every impression/GRP variable in "
                f"**A2 · Spend Mapping**: `{'`, `'.join(missing_spend)}`"
            )
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
                "adstock_type": adstock_type_str,
                "transform_type": transform_type_str,
                "adstock_n_lags": int(n_lags),
                "use_organic": use_organic,
                "use_price": use_price,
                "train_ratio": train_ratio,
                "n_train": n_train,
                "n_test": n_test,
                "positive_beta_cols": positive_beta_cols,
                "negative_beta_cols": negative_beta_cols,
                "per_channel_bounds": per_channel_bounds,
                "media_spend_map": media_spend_map,
                "initial_media_betas":         {c: 0.0     for c in media},
                "initial_comp_betas":          {c: -0.0001 for c in comp_media},
                "initial_own_nonmedia_betas":  {c: 0.0     for c in non_media},
                "initial_comp_nonmedia_betas": {c: -0.01   for c in comp_nonmedia},
                "initial_price_beta":          {c: -0.1    for c in price_vars},
            }
            combo_label = f"{'Weibull' if use_weibull else 'Instant'} × {'Hill' if use_hill else 'Power'}"
            st.success(
                f"✅ Saved — {len(media)} own-media · {len(comp_media)} competitor · "
                f"{len(non_media)} non-media "
                f"({len(prophet_in_model)} prophet col{'s' if len(prophet_in_model)!=1 else ''}) · "
                f"adstock×transform: **{combo_label}** · "
                f"{'lags: ' + str(n_lags) + ' · ' if use_weibull else ''}"
                f"train/test: **{n_train}/{n_test}** · "
                f"positive-beta: **{len(positive_beta_cols)}** · "
                f"negative-beta: **{len(negative_beta_cols)}** · "
                f"per-variable bound params: **{n_bounds_set}**"
            )
            if prophet_in_model:
                prophet_info(
                    f"📌 Prophet controls included in model: "
                    f"<b>{', '.join(prophet_in_model)}</b>"
                )
