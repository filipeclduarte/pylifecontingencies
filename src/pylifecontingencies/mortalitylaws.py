"""
Parametric mortality laws and fitting helpers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .lifetable import LifeTable


def _clip_qx(qx: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(qx, dtype=float), 1e-12, 1.0 - 1e-12)


@dataclass
class MortalityLawFit:
    """Fitted mortality-law result."""

    law_name: str
    param_names: tuple[str, ...]
    params: np.ndarray
    ages: np.ndarray
    observed_qx: np.ndarray
    fitted_qx: np.ndarray
    loglik: float

    def __post_init__(self) -> None:
        self.params = np.asarray(self.params, dtype=float)
        self.ages = np.asarray(self.ages, dtype=int)
        self.observed_qx = np.asarray(self.observed_qx, dtype=float)
        self.fitted_qx = np.asarray(self.fitted_qx, dtype=float)

    @property
    def n_params(self) -> int:
        return len(self.params)

    @property
    def residuals(self) -> np.ndarray:
        return self.observed_qx - self.fitted_qx

    @property
    def sse(self) -> float:
        return float(np.sum(self.residuals ** 2))

    @property
    def rmse(self) -> float:
        return float(np.sqrt(np.mean(self.residuals ** 2)))

    @property
    def mae(self) -> float:
        return float(np.mean(np.abs(self.residuals)))

    @property
    def aic(self) -> float:
        return float(2.0 * self.n_params - 2.0 * self.loglik)

    @property
    def bic(self) -> float:
        return float(np.log(len(self.ages)) * self.n_params - 2.0 * self.loglik)

    def as_dict(self) -> dict[str, float]:
        out = {name: float(value) for name, value in zip(self.param_names, self.params)}
        out.update(
            loglik=self.loglik,
            aic=self.aic,
            bic=self.bic,
            rmse=self.rmse,
            mae=self.mae,
            sse=self.sse,
        )
        return out

    @property
    def params_dict(self) -> dict[str, float]:
        return {name: float(value) for name, value in zip(self.param_names, self.params)}

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "age": self.ages,
                "qx_observed": self.observed_qx,
                "qx_fitted": self.fitted_qx,
                "residual": self.residuals,
            }
        )

    def __repr__(self) -> str:
        return (
            f"MortalityLawFit(law={self.law_name!r}, rmse={self.rmse:.6g}, "
            f"aic={self.aic:.3f}, n={len(self.ages)})"
        )


class GompertzMakeham:
    """
    Gompertz-Makeham law with force of mortality mu_x = A + B * c^x.
    """

    param_names = ("A", "B", "c")

    def __init__(self, A: float | None = None, B: float | None = None, c: float | None = None) -> None:
        self.params = None if None in (A, B, c) else np.array([A, B, c], dtype=float)

    def mu_x(self, ages: np.ndarray, params: np.ndarray | None = None) -> np.ndarray:
        p = self.params if params is None else np.asarray(params, dtype=float)
        A, B, c = p
        ages = np.asarray(ages, dtype=float)
        return A + B * np.power(c, ages)

    def qx(self, ages: np.ndarray, params: np.ndarray | None = None) -> np.ndarray:
        p = self.params if params is None else np.asarray(params, dtype=float)
        A, B, c = p
        ages = np.asarray(ages, dtype=float)
        integral = A + B * np.power(c, ages) * (c - 1.0) / np.log(c)
        return 1.0 - np.exp(-integral)

    def _initial_params(self, ages: np.ndarray, qx: np.ndarray) -> np.ndarray:
        qx = _clip_qx(qx)
        mu = -np.log1p(-qx)
        A0 = max(float(np.quantile(mu, 0.05)) * 0.5, 1e-7)
        gm = np.clip(mu - A0, 1e-10, None)
        slope, intercept = np.polyfit(ages, np.log(gm), 1)
        c0 = float(np.clip(np.exp(slope), 1.01, 1.30))
        phi = (c0 - 1.0) / np.log(c0)
        B0 = max(float(np.exp(intercept) / phi), 1e-10)
        return np.array([A0, B0, c0], dtype=float)

    def fit(self, lt: LifeTable, ages: np.ndarray | None = None) -> MortalityLawFit:
        ages_arr, qx_obs, exposure, deaths = _life_table_fit_inputs(lt, ages)
        x0 = self._initial_params(ages_arr, qx_obs)
        bounds = [(1e-12, 1.0), (1e-12, 10.0), (1.0001, 2.0)]

        def objective(params: np.ndarray) -> float:
            q_hat = _clip_qx(self.qx(ages_arr, params))
            ll = np.sum(deaths * np.log(q_hat) + (exposure - deaths) * np.log1p(-q_hat))
            return float(-ll)

        res = minimize(objective, x0=x0, method="L-BFGS-B", bounds=bounds)
        if not res.success:
            raise RuntimeError(f"GompertzMakeham fit failed: {res.message}")

        self.params = np.asarray(res.x, dtype=float)
        fitted_qx = _clip_qx(self.qx(ages_arr))
        loglik = -float(res.fun)
        return MortalityLawFit(
            law_name="GompertzMakeham",
            param_names=self.param_names,
            params=self.params.copy(),
            ages=ages_arr,
            observed_qx=qx_obs,
            fitted_qx=fitted_qx,
            loglik=loglik,
        )

    def __repr__(self) -> str:
        if self.params is None:
            return "GompertzMakeham(unfitted)"
        A, B, c = self.params
        return f"GompertzMakeham(A={A:.6g}, B={B:.6g}, c={c:.6g})"


class HeligmanPollard:
    """
    Heligman-Pollard 8-parameter law.

    Uses the standard odds formulation:
    q_x / (1 - q_x) =
        A^(x + B)^C + D exp(-E (log x - log F)^2) + G H^x
    """

    param_names = ("A", "B", "C", "D", "E", "F", "G", "H")

    def __init__(
        self,
        A: float | None = None,
        B: float | None = None,
        C: float | None = None,
        D: float | None = None,
        E: float | None = None,
        F: float | None = None,
        G: float | None = None,
        H: float | None = None,
    ) -> None:
        self.params = None if None in (A, B, C, D, E, F, G, H) else np.array(
            [A, B, C, D, E, F, G, H], dtype=float
        )

    def qx(self, ages: np.ndarray, params: np.ndarray | None = None) -> np.ndarray:
        p = self.params if params is None else np.asarray(params, dtype=float)
        A, B, C, D, E, F, G, H = p
        ages = np.asarray(ages, dtype=float)
        ages_safe = np.maximum(ages, 1e-6)
        infant = np.power(A, np.power(ages_safe + B, C))
        accident = D * np.exp(-E * np.square(np.log(ages_safe) - np.log(F)))
        senescent = G * np.power(H, ages_safe)
        odds = infant + accident + senescent
        return odds / (1.0 + odds)

    def _initial_params(self, ages: np.ndarray, qx: np.ndarray) -> np.ndarray:
        qx = _clip_qx(qx)
        old_idx = max(int(len(ages) * 0.75), 1)
        ages_old = ages[old_idx:]
        qx_old = qx[old_idx:]
        slope, intercept = np.polyfit(ages_old, np.log(qx_old), 1)
        H0 = float(np.clip(np.exp(slope), 1.01, 1.20))
        G0 = max(float(np.exp(intercept)), 1e-8)
        peak_age = float(ages[np.argmax(qx)])
        return np.array(
            [0.15, 1.0, 0.12, 0.001, 10.0, max(peak_age, 18.0), G0, H0],
            dtype=float,
        )

    def fit(self, lt: LifeTable, ages: np.ndarray | None = None) -> MortalityLawFit:
        ages_arr, qx_obs, exposure, deaths = _life_table_fit_inputs(lt, ages)
        x0 = self._initial_params(ages_arr, qx_obs)
        bounds = [
            (1e-6, 0.999),
            (1e-4, 50.0),
            (1e-4, 5.0),
            (1e-8, 1.0),
            (1e-4, 100.0),
            (1e-3, 120.0),
            (1e-8, 1.0),
            (1.0001, 2.0),
        ]

        def objective(params: np.ndarray) -> float:
            q_hat = _clip_qx(self.qx(ages_arr, params))
            ll = np.sum(deaths * np.log(q_hat) + (exposure - deaths) * np.log1p(-q_hat))
            logit_hat = np.log(q_hat / (1.0 - q_hat))
            logit_obs = np.log(qx_obs / (1.0 - qx_obs))
            penalty = 1e4 * np.mean((logit_hat - logit_obs) ** 2)
            return float(-ll + penalty)

        res = minimize(objective, x0=x0, method="L-BFGS-B", bounds=bounds)
        if not res.success:
            raise RuntimeError(f"HeligmanPollard fit failed: {res.message}")

        self.params = np.asarray(res.x, dtype=float)
        fitted_qx = _clip_qx(self.qx(ages_arr))
        q_hat = _clip_qx(fitted_qx)
        loglik = float(np.sum(deaths * np.log(q_hat) + (exposure - deaths) * np.log1p(-q_hat)))
        return MortalityLawFit(
            law_name="HeligmanPollard",
            param_names=self.param_names,
            params=self.params.copy(),
            ages=ages_arr,
            observed_qx=qx_obs,
            fitted_qx=fitted_qx,
            loglik=loglik,
        )

    def __repr__(self) -> str:
        if self.params is None:
            return "HeligmanPollard(unfitted)"
        params = ", ".join(
            f"{name}={value:.4g}" for name, value in zip(self.param_names, self.params)
        )
        return f"HeligmanPollard({params})"


def _life_table_fit_inputs(
    lt: LifeTable,
    ages: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ages_arr = np.asarray(lt.ages[:-1] if ages is None else ages, dtype=int)
    if ages_arr.ndim != 1 or len(ages_arr) == 0:
        raise ValueError("ages must be a non-empty 1-D array")
    if np.any(ages_arr < lt.x_min) or np.any(ages_arr >= lt.omega):
        raise ValueError(f"ages must lie within [{lt.x_min}, {lt.omega - 1}]")

    exposure = np.asarray(lt.lx(ages_arr), dtype=float)
    deaths = np.asarray(lt.dx(ages_arr), dtype=float)
    mask = exposure > 0
    ages_arr = ages_arr[mask]
    exposure = exposure[mask]
    deaths = deaths[mask]
    qx_obs = _clip_qx(deaths / exposure)
    return ages_arr, qx_obs, exposure, deaths


def fit_mortality_law(
    lt: LifeTable,
    law: str | GompertzMakeham | HeligmanPollard,
    ages: np.ndarray | None = None,
) -> MortalityLawFit:
    """
    Fit a mortality law to life-table q_x values.

    Examples
    --------
    >>> lt = load_table("soa_ilt")
    >>> fit = fit_mortality_law(lt, "gompertz_makeham", ages=np.arange(40, 90))
    >>> fit.params_dict["c"], fit.rmse
    """
    if isinstance(law, str):
        key = law.lower().replace("-", "").replace("_", "")
        if key in {"gompertzmakeham", "gompertz", "makeham"}:
            law = GompertzMakeham()
        elif key in {"heligmanpollard", "heligman"}:
            law = HeligmanPollard()
        else:
            raise ValueError(f"Unknown mortality law: {law!r}")

    return law.fit(lt, ages=ages)
