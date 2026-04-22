"""
Lee-Carter mortality model (Lee & Carter 1992).

Model:  log(m_{x,t}) = α_x + β_x κ_t + ε_{x,t}

Fitting convention (matching R's demography::lca and StMoMo::lc):
  - α_x = row mean of log(m_{x,t})
  - β_x, κ_t from rank-1 SVD of the centred log-rate matrix
  - β_x normalised so sum(β_x) = 1; κ_t rescaled accordingly
  - κ_t centred so mean(κ_t) = 0 (optional, controlled by centre_kappa)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .rates import MortalityRates
from .bootstrap import residual_bootstrap


class LeeCarterForecast:
    """Forecast object returned by LeeCarter.forecast()."""

    def __init__(
        self,
        ages: np.ndarray,
        years_calib: np.ndarray,
        years_forecast: np.ndarray,
        ax: np.ndarray,
        bx: np.ndarray,
        kt_calib: np.ndarray,
        kt_central: np.ndarray,
        kt_lower: np.ndarray | None,
        kt_upper: np.ndarray | None,
        ci: float,
        n_bootstrap: int,
    ) -> None:
        self.ages = ages
        self.years_calib = years_calib
        self.years_forecast = years_forecast
        self.ax = ax
        self.bx = bx
        self.kt_calib = kt_calib
        self.kt_central = kt_central
        self.kt_lower = kt_lower
        self.kt_upper = kt_upper
        self.ci = ci
        self.n_bootstrap = n_bootstrap

    def log_mx(self, year: int) -> np.ndarray:
        """Predicted log(m_x) for a given forecast year."""
        if year in self.years_calib:
            t_idx = list(self.years_calib).index(year)
            kt = self.kt_calib[t_idx]
        else:
            t_idx = list(self.years_forecast).index(year)
            kt = self.kt_central[t_idx]
        return self.ax + self.bx * kt

    def mx(self, year: int) -> np.ndarray:
        return np.exp(self.log_mx(year))

    def qx(self, year: int) -> np.ndarray:
        """Approximate q_x from m_x via UDD: q_x = m_x / (1 + 0.5 m_x)."""
        mx_val = self.mx(year)
        return mx_val / (1.0 + 0.5 * mx_val)

    def to_dataframe(self) -> pd.DataFrame:
        """Return a long-format DataFrame (year, age, log_mx_central, log_mx_lower, log_mx_upper)."""
        rows = []
        all_years = np.concatenate([self.years_calib, self.years_forecast])
        for yr in all_years:
            log_mu = self.log_mx(yr)
            for i, age in enumerate(self.ages):
                row = {"year": yr, "age": age, "log_mx_central": log_mu[i]}
                rows.append(row)
        return pd.DataFrame(rows)


class LeeCarter:
    """
    Lee-Carter mortality model.

    Usage
    -----
    ::

        lc = LeeCarter().fit(rates)   # rates: MortalityRates
        forecast = lc.forecast(horizon=30, n_bootstrap=500, ci=0.95)
    """

    def __init__(self, centre_kappa: bool = True) -> None:
        self.centre_kappa = centre_kappa
        self._fitted = False

    def fit(self, rates: MortalityRates) -> LeeCarter:
        """
        Fit the Lee-Carter model by SVD.

        Parameters
        ----------
        rates : MortalityRates
            Log mortality rates surface (ages × years).
        """
        log_mx = rates.log_mx  # shape (n_ages, n_years)
        ages = rates.ages
        years = rates.years

        # Handle NaN (open age groups, censored data) by mean-imputation per age
        col_valid = np.all(np.isfinite(log_mx), axis=0)
        if not np.all(col_valid):
            log_mx = log_mx[:, col_valid]
            years = years[col_valid]

        n_ages, n_years = log_mx.shape

        # α_x = row means
        ax = log_mx.mean(axis=1)

        # Centre the matrix
        centred = log_mx - ax[:, np.newaxis]

        # Rank-1 SVD
        U, s, Vt = np.linalg.svd(centred, full_matrices=False)
        bx_raw = U[:, 0]
        kt_raw = s[0] * Vt[0, :]

        # Normalise: sum(bx) = 1
        bx_sum = bx_raw.sum()
        if bx_sum == 0.0:
            raise ValueError("SVD produced zero bx — check input data")
        bx = bx_raw / bx_sum
        kt = kt_raw * bx_sum

        if self.centre_kappa:
            kt_mean = kt.mean()
            ax = ax + bx * kt_mean
            kt = kt - kt_mean

        self.ax = ax
        self.bx = bx
        self.kt = kt
        self.ages = ages
        self.years = years
        self._residuals = centred - np.outer(bx_raw, kt_raw / bx_sum * bx_sum)
        self._log_mx_fit = log_mx
        self._fitted = True
        return self

    def fitted_log_mx(self) -> np.ndarray:
        """Fitted log(m_{x,t}) = α_x + β_x κ_t."""
        if not self._fitted:
            raise RuntimeError("Model not fitted — call .fit() first")
        return self.ax[:, np.newaxis] + np.outer(self.bx, self.kt)

    def forecast(
        self,
        horizon: int = 30,
        arima_order: tuple[int, int, int] = (0, 1, 0),
        n_bootstrap: int = 0,
        ci: float = 0.95,
        seed: int | None = None,
    ) -> LeeCarterForecast:
        """
        Forecast kt forward using ARIMA and return a LeeCarterForecast.

        Parameters
        ----------
        horizon : int
            Number of years ahead.
        arima_order : (p, d, q)
            ARIMA order for kt. Default (0,1,0) = random walk with drift.
        n_bootstrap : int
            Number of bootstrap samples for prediction intervals. 0 = no PI.
        ci : float
            Confidence level for prediction intervals (e.g. 0.95).
        seed : int or None
            Random seed for bootstrap reproducibility.
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted — call .fit() first")

        from statsmodels.tsa.arima.model import ARIMA

        model = ARIMA(self.kt, order=arima_order)
        result = model.fit()

        forecast_obj = result.forecast(steps=horizon)
        kt_central = np.asarray(forecast_obj)

        last_year = int(self.years[-1])
        years_forecast = np.arange(last_year + 1, last_year + horizon + 1)

        kt_lower: np.ndarray | None = None
        kt_upper: np.ndarray | None = None

        if n_bootstrap > 0:
            alpha = 1.0 - ci
            kt_lower, kt_upper = residual_bootstrap(
                kt=self.kt,
                kt_central=kt_central,
                residuals=self._residuals,
                n_bootstrap=n_bootstrap,
                horizon=horizon,
                arima_order=arima_order,
                ci=ci,
                seed=seed,
            )

        return LeeCarterForecast(
            ages=self.ages,
            years_calib=self.years,
            years_forecast=years_forecast,
            ax=self.ax,
            bx=self.bx,
            kt_calib=self.kt,
            kt_central=kt_central,
            kt_lower=kt_lower,
            kt_upper=kt_upper,
            ci=ci,
            n_bootstrap=n_bootstrap,
        )

    def __repr__(self) -> str:
        if not self._fitted:
            return "LeeCarter(not fitted)"
        return (
            f"LeeCarter(ages={self.ages[0]}–{self.ages[-1]}, "
            f"years={self.years[0]}–{self.years[-1]})"
        )
