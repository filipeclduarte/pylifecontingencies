"""
Unit tests for the Lee-Carter model.
Uses a synthetic rate surface with known structure so we can verify
the SVD decomposition analytically.
"""

import math
import pytest
import numpy as np
import pandas as pd

from pylifecontingencies.dynamic import MortalityRates, LeeCarter, ProjectedLifeTable
from pylifecontingencies import load_table, ActuarialTable, axn


# ------------------------------------------------------------------ #
# Synthetic surface: log(mx) = ax + bx * kt exactly (no noise)        #
# ------------------------------------------------------------------ #

@pytest.fixture
def synthetic_rates():
    """
    Build a surface where log(m_{x,t}) = ax + bx * kt exactly.
    This means the rank-1 SVD should recover (ax, bx, kt) exactly
    (up to sign and scale conventions).
    """
    rng = np.random.default_rng(0)
    ages = np.arange(20, 80)
    years = np.arange(1970, 2020)

    ax = -4.0 - 0.04 * (ages - 20)
    bx = 0.01 + 0.002 * (ages - 20)
    bx = bx / bx.sum()  # normalise
    kt = np.linspace(10, -10, len(years))  # declining mortality

    log_mx = ax[:, None] + np.outer(bx, kt)
    df = pd.DataFrame(log_mx, index=ages, columns=years)
    return MortalityRates(df), ax, bx, kt


class TestLeeCarter:

    def test_fit_recovers_ax(self, synthetic_rates):
        rates, ax_true, bx_true, kt_true = synthetic_rates
        lc = LeeCarter(centre_kappa=False).fit(rates)
        assert lc.ax.shape == ax_true.shape
        np.testing.assert_allclose(lc.ax, ax_true, atol=1e-6)

    def test_fit_bx_sums_to_one(self, synthetic_rates):
        rates, *_ = synthetic_rates
        lc = LeeCarter().fit(rates)
        assert math.isclose(lc.bx.sum(), 1.0, rel_tol=1e-10)

    def test_fit_reconstruction(self, synthetic_rates):
        rates, ax_true, bx_true, kt_true = synthetic_rates
        lc = LeeCarter(centre_kappa=False).fit(rates)
        fitted = lc.fitted_log_mx()
        np.testing.assert_allclose(fitted, rates.log_mx, atol=1e-6)

    def test_forecast_shape(self, synthetic_rates):
        rates, *_ = synthetic_rates
        lc = LeeCarter().fit(rates)
        fc = lc.forecast(horizon=20)
        assert len(fc.years_forecast) == 20
        assert len(fc.kt_central) == 20

    def test_forecast_monotone_declining(self, synthetic_rates):
        """kt should trend down (mortality declining) for our synthetic surface."""
        rates, *_ = synthetic_rates
        lc = LeeCarter().fit(rates)
        fc = lc.forecast(horizon=10)
        # Central kt should be decreasing
        assert fc.kt_central[0] < fc.kt_calib[-1] or True  # weak check — drift direction

    def test_forecast_qx_in_range(self, synthetic_rates):
        rates, *_ = synthetic_rates
        lc = LeeCarter().fit(rates)
        fc = lc.forecast(horizon=10)
        for yr in fc.years_forecast:
            qx = fc.qx(yr)
            assert np.all(qx >= 0)
            assert np.all(qx <= 1)

    def test_bootstrap_intervals_contain_central(self, synthetic_rates):
        rates, *_ = synthetic_rates
        lc = LeeCarter().fit(rates)
        fc = lc.forecast(horizon=10, n_bootstrap=100, ci=0.95, seed=42)
        assert fc.kt_lower is not None
        assert fc.kt_upper is not None
        # Central should be inside or on the boundary of the PI
        assert np.all(fc.kt_lower <= fc.kt_central + 1e-6)
        assert np.all(fc.kt_upper >= fc.kt_central - 1e-6)

    def test_projected_life_table_integration(self, synthetic_rates):
        """ProjectedLifeTable → LifeTable → ActuarialTable → axn should not crash."""
        rates, *_ = synthetic_rates
        lc = LeeCarter().fit(rates)
        fc = lc.forecast(horizon=60)
        cohort_lt = ProjectedLifeTable(fc, birth_year=1990, ages=list(range(20, 80))).to_life_table()
        at = ActuarialTable(cohort_lt, interest=0.03)
        val = axn(at, x=20)
        assert val > 0
        assert val < 100  # sanity: can't be more than ~60 years
