"""
Bootstrap prediction intervals for mortality forecasting models.
"""

from __future__ import annotations

import numpy as np


def residual_bootstrap(
    kt: np.ndarray,
    kt_central: np.ndarray,
    residuals: np.ndarray,
    n_bootstrap: int,
    horizon: int,
    arima_order: tuple[int, int, int] = (0, 1, 0),
    ci: float = 0.95,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Residual bootstrap for the Lee-Carter κ_t forecast.

    Resamples the standardised residuals from the κ_t ARIMA fit and
    generates n_bootstrap trajectory replicates of the forecast path.
    Returns (lower, upper) percentile bands.

    Parameters
    ----------
    kt : np.ndarray
        Fitted κ_t series (calibration period).
    kt_central : np.ndarray
        Central forecast (length = horizon).
    residuals : np.ndarray
        ARIMA innovation residuals of the kt fit.
    n_bootstrap : int
        Number of bootstrap samples.
    horizon : int
        Forecast horizon.
    arima_order : tuple
        ARIMA order (p, d, q) — only (0,1,0) is currently used.
    ci : float
        Confidence level.
    seed : int or None
        Random seed.

    Returns
    -------
    lower, upper : np.ndarray, np.ndarray
        Lower and upper percentile kt forecast paths.
    """
    from statsmodels.tsa.arima.model import ARIMA

    rng = np.random.default_rng(seed)
    resid_1d = residuals.ravel()
    resid_1d = resid_1d[np.isfinite(resid_1d)]
    if len(resid_1d) == 0:
        raise ValueError("No finite residuals available for bootstrap")

    # Fit ARIMA to calibration kt to get innovation std
    model = ARIMA(kt, order=arima_order)
    result = model.fit()
    arima_resids = np.asarray(result.resid)
    arima_resids = arima_resids[np.isfinite(arima_resids)]
    if len(arima_resids) == 0:
        arima_resids = np.array([0.0])

    kt_boot = np.empty((n_bootstrap, horizon))
    for b in range(n_bootstrap):
        # Resample ARIMA residuals
        innovations = rng.choice(arima_resids, size=horizon, replace=True)
        # Random walk with drift: kt[t] = kt[t-1] + drift + innovation
        drift = float(np.diff(kt).mean()) if len(kt) > 1 else 0.0
        path = np.empty(horizon)
        prev = float(kt[-1])
        for h in range(horizon):
            prev = prev + drift + innovations[h]
            path[h] = prev
        kt_boot[b] = path

    alpha = (1.0 - ci) / 2.0
    lower = np.quantile(kt_boot, alpha, axis=0)
    upper = np.quantile(kt_boot, 1.0 - alpha, axis=0)
    return lower, upper
