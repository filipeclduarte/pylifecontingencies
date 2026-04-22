"""
Unit tests for the CBD M5 model.
Uses a synthetic surface where logit(q_{x,t}) = k1_t + (x - x_bar) k2_t exactly.
"""

import math
import pytest
import numpy as np
import pandas as pd

from pylifecontingencies.dynamic import MortalityRates, CBD, ProjectedLifeTable
from pylifecontingencies import load_table, ActuarialTable, axn


@pytest.fixture
def synthetic_cbd_rates():
    """
    Synthetic log(m_x) surface derived from a known CBD logit(q_x) surface.
    k1 and k2 follow linear trends.
    """
    ages = np.arange(40, 80)
    years = np.arange(1980, 2010)
    x_bar = float(ages.mean())

    k1 = np.linspace(-8, -9, len(years))       # declining mortality
    k2 = np.linspace(0.08, 0.09, len(years))   # age slope

    qx_mat = np.empty((len(ages), len(years)))
    for t_idx, (k1t, k2t) in enumerate(zip(k1, k2)):
        logit_qx = k1t + (ages - x_bar) * k2t
        qx_mat[:, t_idx] = 1.0 / (1.0 + np.exp(-logit_qx))

    mx_mat = qx_mat / (1.0 - 0.5 * qx_mat)
    log_mx = np.log(mx_mat)
    df = pd.DataFrame(log_mx, index=ages, columns=years)
    return MortalityRates(df), k1, k2, x_bar


class TestCBD:

    def test_fit_recovers_kappa(self, synthetic_cbd_rates):
        rates, k1_true, k2_true, x_bar_true = synthetic_cbd_rates
        cbd = CBD().fit(rates)
        np.testing.assert_allclose(cbd.k1, k1_true, atol=1e-6)
        np.testing.assert_allclose(cbd.k2, k2_true, atol=1e-6)
        assert math.isclose(cbd.x_bar, x_bar_true, rel_tol=1e-12)

    def test_forecast_shape(self, synthetic_cbd_rates):
        rates, *_ = synthetic_cbd_rates
        cbd = CBD().fit(rates)
        fc = cbd.forecast(horizon=20)
        assert len(fc.k1_central) == 20
        assert len(fc.k2_central) == 20
        assert len(fc.years_forecast) == 20

    def test_forecast_qx_in_range(self, synthetic_cbd_rates):
        rates, *_ = synthetic_cbd_rates
        cbd = CBD().fit(rates)
        fc = cbd.forecast(horizon=10)
        for yr in fc.years_forecast:
            qx = fc.qx(yr)
            assert np.all(qx >= 0)
            assert np.all(qx <= 1)

    def test_bootstrap_intervals(self, synthetic_cbd_rates):
        rates, *_ = synthetic_cbd_rates
        cbd = CBD().fit(rates)
        fc = cbd.forecast(horizon=10, n_bootstrap=100, ci=0.90, seed=7)
        assert fc.k1_lower is not None
        assert fc.k1_upper is not None
        assert np.all(fc.k1_lower <= fc.k1_central + 1e-6)
        assert np.all(fc.k1_upper >= fc.k1_central - 1e-6)

    def test_projected_table_integration(self, synthetic_cbd_rates):
        rates, *_ = synthetic_cbd_rates
        cbd = CBD().fit(rates)
        fc = cbd.forecast(horizon=50)
        lt = ProjectedLifeTable(fc, period_year=2020, ages=list(range(40, 80))).to_life_table()
        at = ActuarialTable(lt, interest=0.03)
        val = axn(at, x=40)
        assert val > 0
