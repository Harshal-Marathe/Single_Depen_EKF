"""
Nevergrad multi-objective optimizer support.

Loss = -w_loglik * loglik
       + w_mape * max(0, MAPE - mape_target)^2
       + w_r2   * max(0, r2_target - R2)^2
       + w_pos  * positivity_violations
"""

import numpy as np
import streamlit as st

from modules.params import unpack_theta
from modules.kalman import run_kalman_filter


def _composite_loss(theta, df_train, g, ng_weights):
    w_loglik = ng_weights.get("w_loglik", 1.0); w_mape = ng_weights.get("w_mape", 10.0)
    w_r2     = ng_weights.get("w_r2",     5.0); w_pos  = ng_weights.get("w_pos",  100.0)
    mape_tgt = ng_weights.get("mape_target", 0.10); r2_tgt = ng_weights.get("r2_target", 0.80)
    try:
        p = unpack_theta(theta, g)
        yhat, resid, _, _, _, _, _, _, loglik = run_kalman_filter(df_train, p, g)
        tv   = df_train[g["TARGET_COL"]].values
        mape = float(np.mean(np.abs(resid / (tv + 1e-12))))
        r2   = 1.0 - np.sum(resid**2) / (np.sum((tv - tv.mean())**2) + 1e-12)
        loss = -w_loglik * loglik
        loss += w_mape * max(0, mape - mape_tgt) ** 2
        loss += w_r2  * max(0, r2_tgt - r2) ** 2
        for i, col in enumerate(g["MEDIA_COLS"]):
            if col in g.get("POSITIVE_BETA_COLS", []):
                loss += w_pos * max(0, -p["delta"][i]) ** 2
        for j in range(g["N_COMP"]):
            loss += w_pos * max(0, p["delta_comp"][j]) ** 2
        return float(loss)
    except Exception:
        return 1e12


def run_nevergrad_optimizer(df_train, g, theta0, bounds, ng_cfg):
    import nevergrad as ng
    strategy_name = ng_cfg.get("strategy", "NgIohh"); budget = ng_cfg.get("budget", 500)
    num_workers = ng_cfg.get("num_workers", 1); ng_weights = ng_cfg.get("ng_weights", {})
    lows  = np.array([b[0] if b[0] is not None else -1e6 for b in bounds])
    highs = np.array([b[1] if b[1] is not None else  1e6 for b in bounds])
    param = ng.p.Array(init=theta0).set_bounds(lows, highs)
    optimizer_cls = getattr(ng.optimizers, strategy_name, None) or ng.optimizers.NgIohh
    optimizer = optimizer_cls(parametrization=param, budget=budget, num_workers=num_workers)
    best_loss = np.inf; best_theta = theta0.copy()
    progress = st.progress(0, text=f"Nevergrad [{strategy_name}] — 0/{budget} evals")
    for i, cand in enumerate(optimizer.ask() for _ in range(budget)):
        loss = _composite_loss(cand.value, df_train, g, ng_weights)
        optimizer.tell(cand, loss)
        if loss < best_loss: best_loss = loss; best_theta = cand.value.copy()
        if i % max(1, budget // 50) == 0:
            progress.progress(int(i/budget*100),
                              text=f"Nevergrad [{strategy_name}] — {i}/{budget} | best: {best_loss:.4f}")
    progress.progress(100, text=f"✅ Nevergrad done — best loss: {best_loss:.4f}")
    return best_theta, best_loss
