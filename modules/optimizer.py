"""
Nevergrad optimizer support (same optimizer as the 2-dependent app).

Loss = -loglik   (identical objective to the L-BFGS-B / SLSQP path)

The search runs in the per-parameter-normalized space from
modules/bounds.py::build_normalized_problem, candidates are evaluated
`num_workers` at a time on a thread pool, and the default strategy is NGOpt.
(This replaces the previous weighted multi-objective loss — loglik + MAPE /
R² target penalties + positivity penalties. Sign constraints are enforced by
the parameter bounds and the filter's beta floors, so the penalty terms are
no longer needed to keep betas on the right side of zero.)
"""

import numpy as np
import streamlit as st
from concurrent.futures import ThreadPoolExecutor

from modules.params import unpack_theta
from modules.bounds import build_normalized_problem
from modules.kalman import run_kalman_filter


def _composite_loss(theta, df_train, g):
    try:
        p = unpack_theta(theta, g)
        _, _, _, _, _, _, _, _, loglik = run_kalman_filter(df_train, p, g)
        return -loglik
    except Exception:
        return 1e12


def _ask_eval_tell_loop(optimizer, budget, num_workers, progress_label, loss_fn):
    """
    Shared ask/evaluate/tell driver for both the single-equation and joint
    Nevergrad optimizers.

    Previously this always asked for and evaluated ONE candidate at a time,
    regardless of the `num_workers` setting in the UI — so raising "workers"
    had zero effect on wall-clock time. This now genuinely batches
    `num_workers` candidates per round and evaluates them concurrently via a
    thread pool (nevergrad's ask-many/tell-many pattern is explicitly
    designed for this). With num_workers=1 (the default) this is exactly
    the old serial behaviour — nothing changes unless you raise it.

    Note: the Kalman filter's per-timestep loop is Python-level, not a
    single vectorized numpy call, so GIL contention limits how much thread
    parallelism can help — the numpy matrix multiplies inside each step do
    release the GIL, so there's a real, if likely partial (not N×), speedup
    from raising num_workers. It is, at minimum, no longer a no-op.
    """
    best_loss = np.inf; best_theta = None
    progress = st.progress(0, text=f"{progress_label} — 0/{budget} evals")
    executor = ThreadPoolExecutor(max_workers=num_workers) if num_workers > 1 else None
    evaluated = 0
    report_every = max(1, budget // 50)
    try:
        while evaluated < budget:
            batch_n = min(num_workers, budget - evaluated)
            cands = [optimizer.ask() for _ in range(batch_n)]
            if executor is not None:
                losses = list(executor.map(lambda c: loss_fn(c.value), cands))
            else:
                losses = [loss_fn(c.value) for c in cands]
            for cand, loss in zip(cands, losses):
                optimizer.tell(cand, loss)
                if loss < best_loss:
                    best_loss = loss; best_theta = cand.value.copy()
            evaluated += batch_n
            if (evaluated // report_every) != ((evaluated - batch_n) // report_every):
                progress.progress(min(100, int(evaluated / budget * 100)),
                                  text=f"{progress_label} — {evaluated}/{budget} | best: {best_loss:.4f}")
    finally:
        if executor is not None:
            executor.shutdown(wait=False)
    progress.progress(100, text=f"✅ {progress_label} done — best loss: {best_loss:.4f}")
    return best_theta, best_loss


def _ng_bounds_arrays(norm_bounds, sentinel=100.0):
    """
    ng.p.Array.set_bounds() needs finite numeric arrays, but
    build_normalized_problem() legitimately leaves a side as None when the
    ORIGINAL parameter bound was None on that side (see its docstring —
    that's a real "no constraint here" that we don't want to silently
    reintroduce, so we don't turn it into a tight box). Since every
    dimension is already normalized to a comparable, roughly-O(1) scale by
    that point (unlike the raw ±1e6-in-real-units sentinel this replaces),
    a single generous, uniform normalized-space sentinel works fine here —
    it's proportionate for every dimension instead of swamping the ones
    that started out small in real units.
    """
    lows  = np.array([b[0] if b[0] is not None else -sentinel for b in norm_bounds])
    highs = np.array([b[1] if b[1] is not None else  sentinel for b in norm_bounds])
    return lows, highs


def run_nevergrad_optimizer(df_train, g, theta0, bounds, ng_cfg):
    import nevergrad as ng
    strategy_name = ng_cfg.get("strategy", "NGOpt"); budget = ng_cfg.get("budget", 500)
    num_workers = max(1, int(ng_cfg.get("num_workers", 1)))

    # Search in per-parameter-normalized space instead of raw theta (see
    # modules/bounds.py::build_normalized_problem and the note in the
    # 2-dependent app's optimizer): without it every unbounded dimension
    # (gamma, sigma_y, several deltas) gets an arbitrary huge box that lets
    # those dimensions dominate a population/mutation-based search and drown
    # out small-range ones like Hill's n (1-15).
    theta0_norm, norm_bounds, unscale = build_normalized_problem(theta0, bounds)
    lows, highs = _ng_bounds_arrays(norm_bounds)
    param = ng.p.Array(init=theta0_norm).set_bounds(lows, highs)
    optimizer_cls = getattr(ng.optimizers, strategy_name, None) or ng.optimizers.NGOpt
    optimizer = optimizer_cls(parametrization=param, budget=budget, num_workers=num_workers)

    loss_fn = lambda theta_norm: _composite_loss(unscale(theta_norm), df_train, g)
    best_theta_norm, best_loss = _ask_eval_tell_loop(
        optimizer, budget, num_workers, f"Nevergrad [{strategy_name}]", loss_fn)
    best_theta = unscale(best_theta_norm) if best_theta_norm is not None else theta0.copy()
    return best_theta, best_loss
