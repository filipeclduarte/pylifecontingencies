"""
Tests for the consolidated ProjectedLifeTable.

Covers:
1. Raw DataFrame constructors (from_mx, from_qx, from_log_mx)
2. Prediction intervals (lower / upper)
3. Multi-scenario stochastic analysis
4. Extrapolation strategies (clamp, constant_slope, none)
5. Integration with DynamicActuarialTable
6. Backward compatibility with forecast objects
"""

import math

import numpy as np
import pandas as pd
import pytest

from pylifecontingencies import LifeTable, ActuarialTable
from pylifecontingencies.dynamic import (
    ProjectedLifeTable,
    DynamicActuarialTable,
    StochasticResult,
)


# ------------------------------------------------------------------ #
# Fixtures                                                             #
# ------------------------------------------------------------------ #

@pytest.fixture
def flat_mx_df():
    """Constant mx = 0.01 for ages 0-99, years 2000-2060."""
    ages = np.arange(0, 100)
    years = np.arange(2000, 2061)
    return pd.DataFrame(
        np.full((len(ages), len(years)), 0.01), index=ages, columns=years
    )


@pytest.fixture
def gompertz_mx_df():
    """Gompertz mx with time improvement, ages 20-99, years 2000-2060."""
    ages = np.arange(20, 100)
    years = np.arange(2000, 2061)
    mx = np.empty((len(ages), len(years)))
    for i, age in enumerate(ages):
        for j, year in enumerate(years):
            improvement = -0.01 * max(0, year - 2020)
            mx[i, j] = 0.0003 * np.exp(0.07 * age + improvement)
    return pd.DataFrame(np.clip(mx, 1e-8, 10.0), index=ages, columns=years)


@pytest.fixture
def flat_qx_df():
    """Constant qx = 0.01 for ages 0-99, years 2000-2060."""
    ages = np.arange(0, 100)
    years = np.arange(2000, 2061)
    return pd.DataFrame(
        np.full((len(ages), len(years)), 0.01), index=ages, columns=years
    )


@pytest.fixture
def flat_log_mx_df():
    """Constant log(mx) = log(0.01) for ages 0-99, years 2000-2060."""
    ages = np.arange(0, 100)
    years = np.arange(2000, 2061)
    return pd.DataFrame(
        np.full((len(ages), len(years)), np.log(0.01)), index=ages, columns=years
    )


# ------------------------------------------------------------------ #
# from_mx / from_qx / from_log_mx basic tests                         #
# ------------------------------------------------------------------ #

class TestFromDataFrame:
    """Test raw DataFrame constructors."""

    def test_from_mx_period(self, flat_mx_df):
        plt = ProjectedLifeTable.from_mx(flat_mx_df, period_year=2030)
        lt = plt.to_life_table()
        assert isinstance(lt, LifeTable)
        assert lt.x_min == 0

    def test_from_mx_cohort(self, flat_mx_df):
        plt = ProjectedLifeTable.from_mx(flat_mx_df, birth_year=2000)
        lt = plt.to_life_table()
        assert isinstance(lt, LifeTable)

    def test_from_mx_qx_values(self, flat_mx_df):
        plt = ProjectedLifeTable.from_mx(flat_mx_df, period_year=2030)
        lt = plt.to_life_table()
        expected_qx = 0.01 / (1.0 + 0.5 * 0.01)
        assert math.isclose(lt.qx(40), expected_qx, rel_tol=1e-10)

    def test_from_qx(self, flat_qx_df):
        plt = ProjectedLifeTable.from_qx(flat_qx_df, period_year=2030)
        lt = plt.to_life_table()
        assert math.isclose(lt.qx(40), 0.01, rel_tol=1e-10)

    def test_from_log_mx(self, flat_log_mx_df):
        plt = ProjectedLifeTable.from_log_mx(flat_log_mx_df, period_year=2030)
        lt = plt.to_life_table()
        expected_qx = 0.01 / (1.0 + 0.5 * 0.01)
        assert math.isclose(lt.qx(40), expected_qx, rel_tol=1e-10)

    def test_subset_ages(self, flat_mx_df):
        plt = ProjectedLifeTable.from_mx(
            flat_mx_df, period_year=2030, ages=list(range(20, 80))
        )
        lt = plt.to_life_table()
        assert lt.x_min == 20

    def test_full_pipeline(self, gompertz_mx_df):
        plt = ProjectedLifeTable.from_mx(
            gompertz_mx_df, birth_year=1985, ages=list(range(20, 90))
        )
        lt = plt.to_life_table()
        at = ActuarialTable(lt, interest=0.03)
        result = at.axn(x=40)
        assert 0 < result < 50


# ------------------------------------------------------------------ #
# Prediction interval tests                                            #
# ------------------------------------------------------------------ #

class TestPredictionIntervals:
    """Test lower/upper prediction interval support."""

    def test_pi_creates_three_tables(self, flat_mx_df):
        # Lower: half the mortality, Upper: double
        lower = flat_mx_df * 0.5
        upper = flat_mx_df * 2.0

        plt = ProjectedLifeTable.from_mx(
            flat_mx_df, lower=lower, upper=upper, period_year=2030
        )
        assert plt.has_prediction_interval
        assert plt.n_scenarios == 3
        assert plt.lower is not None
        assert plt.upper is not None

    def test_pi_central_accessible(self, flat_mx_df):
        lower = flat_mx_df * 0.5
        upper = flat_mx_df * 2.0

        plt = ProjectedLifeTable.from_mx(
            flat_mx_df, lower=lower, upper=upper, period_year=2030
        )
        lt = plt.lifetable  # central
        expected_qx = 0.01 / (1.0 + 0.5 * 0.01)
        assert math.isclose(lt.qx(40), expected_qx, rel_tol=1e-10)

    def test_pi_ordering(self, flat_mx_df):
        """Lower mortality → higher survival → lower qx at each age."""
        lower = flat_mx_df * 0.5
        upper = flat_mx_df * 2.0

        plt = ProjectedLifeTable.from_mx(
            flat_mx_df, lower=lower, upper=upper, period_year=2030
        )
        # Lower mx should give lower qx
        qx_lo = plt.lower.qx(40)
        qx_central = plt.lifetable.qx(40)
        qx_hi = plt.upper.qx(40)
        assert qx_lo < qx_central < qx_hi

    def test_pi_to_life_table_returns_central(self, flat_mx_df):
        lower = flat_mx_df * 0.5
        upper = flat_mx_df * 2.0

        plt = ProjectedLifeTable.from_mx(
            flat_mx_df, lower=lower, upper=upper, period_year=2030
        )
        lt = plt.to_life_table()
        assert lt is plt.lifetable

    def test_pi_with_dynamic_actuarial(self, flat_mx_df):
        """PI tables should work with DynamicActuarialTable → StochasticResult."""
        lower = flat_mx_df * 0.5
        upper = flat_mx_df * 2.0

        plt = ProjectedLifeTable.from_mx(
            flat_mx_df, lower=lower, upper=upper,
            period_year=2030, ages=list(range(0, 90))
        )
        dat = DynamicActuarialTable(plt, interest=0.06)
        result = dat.axn(x=40)
        assert isinstance(result, StochasticResult)
        assert result.n == 3  # lower, central, upper

    def test_must_provide_both_or_neither(self, flat_mx_df):
        with pytest.raises(ValueError, match="both lower and upper"):
            ProjectedLifeTable.from_mx(
                flat_mx_df, lower=flat_mx_df * 0.5, period_year=2030
            )

    def test_pi_with_qx(self, flat_qx_df):
        """PI should also work with from_qx."""
        lower = flat_qx_df * 0.5
        upper = flat_qx_df * 2.0

        plt = ProjectedLifeTable.from_qx(
            flat_qx_df, lower=lower, upper=upper, period_year=2030
        )
        assert plt.has_prediction_interval
        assert plt.lower.qx(40) < plt.lifetable.qx(40) < plt.upper.qx(40)


# ------------------------------------------------------------------ #
# Multi-scenario (stochastic) tests                                    #
# ------------------------------------------------------------------ #

class TestScenarios:
    """Test from_scenarios and from_scenarios_array."""

    def test_from_scenarios_basic(self, flat_mx_df):
        # 5 scenarios with slightly different mortality
        scenarios = [flat_mx_df * (1 + 0.1 * i) for i in range(5)]
        plt = ProjectedLifeTable.from_scenarios(
            scenarios, period_year=2030
        )
        assert plt.is_stochastic
        assert plt.n_scenarios == 5
        assert not plt.has_prediction_interval

    def test_from_scenarios_dynamic_actuarial(self, flat_mx_df):
        scenarios = [flat_mx_df * (1 + 0.1 * i) for i in range(10)]
        plt = ProjectedLifeTable.from_scenarios(
            scenarios, period_year=2030, ages=list(range(0, 90))
        )
        dat = DynamicActuarialTable(plt, interest=0.06)
        result = dat.axn(x=40)
        assert isinstance(result, StochasticResult)
        assert result.n == 10

    def test_from_scenarios_array(self, flat_mx_df):
        ages = np.arange(0, 100)
        years = np.arange(2000, 2061)
        arr = np.stack([flat_mx_df.values * (1 + 0.1 * i) for i in range(5)])

        plt = ProjectedLifeTable.from_scenarios_array(
            arr, ages=ages, years=years, period_year=2030
        )
        assert plt.is_stochastic
        assert plt.n_scenarios == 5

    def test_scenarios_lifetable_raises(self, flat_mx_df):
        """Accessing .lifetable on a stochastic table should raise."""
        scenarios = [flat_mx_df * (1 + 0.1 * i) for i in range(3)]
        plt = ProjectedLifeTable.from_scenarios(scenarios, period_year=2030)
        with pytest.raises(ValueError, match="multiple scenarios"):
            plt.lifetable


# ------------------------------------------------------------------ #
# Extrapolation tests                                                  #
# ------------------------------------------------------------------ #

class TestExtrapolation:

    def test_clamp_default(self, flat_mx_df):
        plt = ProjectedLifeTable.from_mx(flat_mx_df, birth_year=1920)
        lt = plt.to_life_table()
        assert isinstance(lt, LifeTable)

    def test_none_raises(self, flat_mx_df):
        with pytest.raises(ValueError, match="outside"):
            ProjectedLifeTable.from_mx(
                flat_mx_df, birth_year=1920, extrapolation="none"
            )

    def test_constant_slope(self, gompertz_mx_df):
        plt = ProjectedLifeTable.from_mx(
            gompertz_mx_df, birth_year=1950,
            ages=list(range(20, 90)), extrapolation="constant_slope",
        )
        lt = plt.to_life_table()
        assert isinstance(lt, LifeTable)

    def test_in_range_modes_agree(self, flat_mx_df):
        """When no extrapolation needed, modes give same result."""
        lt_clamp = ProjectedLifeTable.from_mx(
            flat_mx_df, birth_year=2000, ages=list(range(0, 60)),
            extrapolation="clamp",
        ).to_life_table()
        lt_slope = ProjectedLifeTable.from_mx(
            flat_mx_df, birth_year=2000, ages=list(range(0, 60)),
            extrapolation="constant_slope",
        ).to_life_table()
        for age in range(0, 59):
            assert math.isclose(lt_clamp.qx(age), lt_slope.qx(age), rel_tol=1e-12)


# ------------------------------------------------------------------ #
# Backward compatibility                                               #
# ------------------------------------------------------------------ #

class TestBackwardCompat:

    def test_forecast_object(self):
        """Old-style forecast objects still work."""

        class MockForecast:
            ages = np.arange(20, 80)
            years_calib = np.arange(2000, 2020)
            years_forecast = np.arange(2020, 2050)

            def qx(self, year):
                return np.full(len(self.ages), 0.01)

        fc = MockForecast()
        plt = ProjectedLifeTable(fc, period_year=2030)
        lt = plt.to_life_table()
        assert isinstance(lt, LifeTable)
        assert math.isclose(lt.qx(40), 0.01, rel_tol=1e-10)

    def test_dynamic_lifetable_with_dat(self):
        """DynamicLifeTable still works with DynamicActuarialTable."""
        from pylifecontingencies.dynamic import DynamicLifeTable

        ages = np.arange(0, 100)
        years = np.arange(2000, 2061)
        df = pd.DataFrame(
            np.full((len(ages), len(years)), 0.01), index=ages, columns=years
        )
        dlt = DynamicLifeTable.from_forecast_mx(df, period_year=2030)
        dat = DynamicActuarialTable(dlt, interest=0.06)
        result = dat.axn(x=40)
        assert isinstance(result, float)
        assert result > 0


# ------------------------------------------------------------------ #
# Constructor validation                                               #
# ------------------------------------------------------------------ #

class TestValidation:

    def test_must_provide_one_year(self, flat_mx_df):
        with pytest.raises(ValueError, match="exactly one"):
            ProjectedLifeTable.from_mx(flat_mx_df)

    def test_cannot_provide_both(self, flat_mx_df):
        with pytest.raises(ValueError, match="exactly one"):
            ProjectedLifeTable.from_mx(
                flat_mx_df, birth_year=1985, period_year=2030
            )

    def test_repr_single(self, flat_mx_df):
        plt = ProjectedLifeTable.from_mx(flat_mx_df, period_year=2030)
        assert "single-path" in repr(plt)

    def test_repr_pi(self, flat_mx_df):
        plt = ProjectedLifeTable.from_mx(
            flat_mx_df, lower=flat_mx_df * 0.5, upper=flat_mx_df * 2.0,
            period_year=2030,
        )
        assert "PI" in repr(plt)

    def test_repr_stochastic(self, flat_mx_df):
        scenarios = [flat_mx_df for _ in range(3)]
        plt = ProjectedLifeTable.from_scenarios(scenarios, period_year=2030)
        assert "stochastic" in repr(plt)
