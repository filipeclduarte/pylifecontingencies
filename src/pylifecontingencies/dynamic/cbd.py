"""
Cairns-Blake-Dowd (CBD) M5 mortality model.

Model:  logit(q_{x,t}) = κ^1_t + (x - x̄) κ^2_t

Fitting: per-year OLS of logit(q_{x,t}) on [1, (x - x̄)].
Forecast: bivariate random walk with drift on (κ^1_t, κ^2_t).

Reference: Cairns, Blake, Dowd (2006) "A Two-Factor Model for Stochastic
Mortality with Parameter Uncertainty", JRI 73(4):687-718.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .rates import MortalityRates


class CBDForecast:
    """Forecast object returned by CBD.forecast()."""

    def __init__(
        self,
        ages: np.ndarray,
        x_bar: float,
        years_calib: np.ndarray,
        years_forecast: np.ndarray,
        k1_calib: np.ndarray,
        k2_calib: np.ndarray,
        k1_central: np.ndarray,
        k2_central: np.ndarray,
        k1_lower: np.ndarray | None,
        k2_lower: np.ndarray | None,
        k1_upper: np.ndarray | None,
        k2_upper: np.ndarray | None,
        ci: float,
    ) -> None:
        self.ages = ages
        self.x_bar = x_bar
        self.years_calib = years_calib
        self.years_forecast = years_forecast
        self.k1_calib = k1_calib
        self.k2_calib = k2_calib
        self.k1_central = k1_central
        self.k2_central = k2_central
        self.k1_lower = k1_lower
        self.k2_lower = k2_lower
        self.k1_upper = k1_upper
        self.k2_upper = k2_upper
        self.ci = ci

    def _get_kappa(self, year: int) -> tuple[float, float]:
        if year in self.years_calib:
            idx = list(self.years_calib).index(year)
            return float(self.k1_calib[idx]), float(self.k2_calib[idx])
        idx = list(self.years_forecast).index(year)
        return float(self.k1_central[idx]), float(self.k2_central[idx])

    def logit_qx(self, year: int) -> np.ndarray:
        k1, k2 = self._get_kappa(year)
        return k1 + (self.ages - self.x_bar) * k2

    def qx(self, year: int) -> np.ndarray:
        """Predicted q_x for a given year."""
        lgt = self.logit_qx(year)
        return 1.0 / (1.0 + np.exp(-lgt))

    def mx(self, year: int) -> np.ndarray:
        """Convert q_x to m_x via UDD approximation."""
        qx = self.qx(year)
        return qx / (1.0 - 0.5 * qx)


class CBD:
    """
    Cairns-Blake-Dowd (CBD) M5 two-factor mortality model.

    Usage
    -----
    ::

        cbd = CBD().fit(rates)
        forecast = cbd.forecast(horizon=30, ci=0.95)
    """

    def __init__(self) -> None:
        self._fitted = False

    def fit(self, rates: MortalityRates) -> CBD:
        """
        Fit the CBD M5 model.

        Parameters
        ----------
        rates : MortalityRates
            Log mortality rate surface. The model internally works with
            q_x values derived from m_x via UDD.
        """
        ages = rates.ages
        years = rates.years
        mx_mat = rates.mx  # shape (n_ages, n_years)

        # Convert mx → qx via UDD
        qx_mat = mx_mat / (1.0 + 0.5 * mx_mat)
        # Clip for numerical stability (logit requires strict (0,1))
        qx_mat = np.clip(qx_mat, 1e-10, 1.0 - 1e-10)

        logit_qx = np.log(qx_mat / (1.0 - qx_mat))

        x_bar = float(ages.mean())
        n_years = len(years)

        k1 = np.empty(n_years)
        k2 = np.empty(n_years)

        age_centred = ages - x_bar
        X = np.column_stack([np.ones(len(ages)), age_centred])
        XtX_inv = np.linalg.inv(X.T @ X)

        for t_idx in range(n_years):
            y = logit_qx[:, t_idx]
            valid = np.isfinite(y)
            if valid.sum() < 2:
                k1[t_idx] = np.nan
                k2[t_idx] = np.nan
                continue
            beta = XtX_inv @ (X[valid].T @ y[valid])
            k1[t_idx] = beta[0]
            k2[t_idx] = beta[1]

        self.ages = ages
        self.x_bar = x_bar
        self.years = years
        self.k1 = k1
        self.k2 = k2
        self._fitted = True
        return self

    def forecast(
        self,
        horizon: int = 30,
        n_bootstrap: int = 0,
        ci: float = 0.95,
        seed: int | None = None,
    ) -> CBDForecast:
        """
        Forecast κ^1_t and κ^2_t as a bivariate random walk with drift.

        Drift and covariance are estimated from the historical differences
        Δκ_t = κ_t - κ_{t-1}.
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted — call .fit() first")

        valid = np.isfinite(self.k1) & np.isfinite(self.k2)
        k1_v = self.k1[valid]
        k2_v = self.k2[valid]

        # Estimate drift and innovation covariance from first differences
        dk1 = np.diff(k1_v)
        dk2 = np.diff(k2_v)
        drift1 = dk1.mean()
        drift2 = dk2.mean()
        resid = np.column_stack([dk1 - drift1, dk2 - drift2])
        Sigma = (resid.T @ resid) / len(resid)

        last_year = int(self.years[-1])
        years_forecast = np.arange(last_year + 1, last_year + horizon + 1)

        k1_central = np.empty(horizon)
        k2_central = np.empty(horizon)
        k1_v_last = k1_v[-1]
        k2_v_last = k2_v[-1]

        for h in range(horizon):
            k1_central[h] = k1_v_last + (h + 1) * drift1
            k2_central[h] = k2_v_last + (h + 1) * drift2

        k1_lower = k1_upper = k2_lower = k2_upper = None

        if n_bootstrap > 0:
            rng = np.random.default_rng(seed)
            k1_boot = np.empty((n_bootstrap, horizon))
            k2_boot = np.empty((n_bootstrap, horizon))

            chol = np.linalg.cholesky(Sigma + 1e-12 * np.eye(2))
            for b in range(n_bootstrap):
                k1_path = k1_v_last
                k2_path = k2_v_last
                for h in range(horizon):
                    z = chol @ rng.standard_normal(2)
                    k1_path = k1_path + drift1 + z[0]
                    k2_path = k2_path + drift2 + z[1]
                    k1_boot[b, h] = k1_path
                    k2_boot[b, h] = k2_path

            alpha = (1.0 - ci) / 2.0
            k1_lower = np.quantile(k1_boot, alpha, axis=0)
            k1_upper = np.quantile(k1_boot, 1.0 - alpha, axis=0)
            k2_lower = np.quantile(k2_boot, alpha, axis=0)
            k2_upper = np.quantile(k2_boot, 1.0 - alpha, axis=0)

        return CBDForecast(
            ages=self.ages,
            x_bar=self.x_bar,
            years_calib=self.years,
            years_forecast=years_forecast,
            k1_calib=self.k1,
            k2_calib=self.k2,
            k1_central=k1_central,
            k2_central=k2_central,
            k1_lower=k1_lower,
            k2_lower=k2_lower,
            k1_upper=k1_upper,
            k2_upper=k2_upper,
            ci=ci,
        )

    def __repr__(self) -> str:
        if not self._fitted:
            return "CBD(not fitted)"
        return (
            f"CBD(ages={self.ages[0]}–{self.ages[-1]}, "
            f"years={self.years[0]}–{self.years[-1]}, x_bar={self.x_bar:.1f})"
        )
