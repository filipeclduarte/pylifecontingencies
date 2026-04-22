"""
Tests for DynamicLifeTable and DynamicActuarialTable.

Covers:
- Single-path from mx / qx / log_mx
- Cohort vs period extraction
- Stochastic from list of DataFrames and from 3-D array
- DynamicActuarialTable returns float (single) / StochasticResult (stochastic)
- Numerical consistency: results must equal static ActuarialTable for identical qx
- StochasticResult statistics
- Reserve recursion (scalar and stochastic)
- Edge cases: clamping warning, terminal age enforcement
"""

import math
import warnings

import numpy as np
import pandas as pd
import pytest

from pylifecontingencies import (
    LifeTable,
    ActuarialTable,
    load_table,
    axn,
    Axn,
    Exn,
)
from pylifecontingencies.dynamic import (
    DynamicLifeTable,
    DynamicActuarialTable,
    StochasticResult,
)


# ------------------------------------------------------------------ #
# Fixtures                                                             #
# ------------------------------------------------------------------ #

AGES = list(range(0, 100))
YEARS = list(range(2020, 2071))


@pytest.fixture
def flat_mx_df():
    """
    Constant mx surface: m_x = 0.02 for all ages and years.
    qx = 0.02 / 1.01 ≈ 0.019802 (UDD).
    """
    mx = 0.02
    return pd.DataFrame(
        np.full((len(AGES), len(YEARS)), mx),
        index=AGES,
        columns=YEARS,
    )


@pytest.fixture
def soa_ilt_as_mx_df():
    """
    Replicate the SOA ILT as a constant-year mx surface so we can compare
    DynamicActuarialTable against the static ActuarialTable exactly.
    """
    lt = load_table("soa_ilt")
    df_lt = lt.to_dataframe()
    ages = df_lt["age"].values.astype(int)
    # Use UDD: mx = qx / (1 - 0.5 * qx)
    mx_vals = df_lt["qx"].values / (1.0 - 0.5 * df_lt["qx"].values)

    # Build a surface where every year column has the same mx
    mx_matrix = np.outer(mx_vals, np.ones(len(YEARS)))
    return pd.DataFrame(mx_matrix, index=ages, columns=YEARS)


@pytest.fixture
def scenario_list(flat_mx_df):
    """10 identical scenarios — stochastic result should equal single-path."""
    return [flat_mx_df.copy() for _ in range(10)]


# ------------------------------------------------------------------ #
# DynamicLifeTable construction                                        #
# ------------------------------------------------------------------ #

class TestDynamicLifeTableConstruction:

    def test_from_forecast_mx_cohort(self, flat_mx_df):
        dlt = DynamicLifeTable.from_forecast_mx(flat_mx_df, birth_year=1985)
        assert not dlt.is_stochastic
        assert dlt.n_scenarios == 1
        lt = dlt.lifetable
        assert lt.x_min == 0
        # qx ≈ 0.02 / 1.01 for all ages
        for x in [20, 40, 60]:
            assert math.isclose(lt.qx(x), 0.02 / 1.01, rel_tol=1e-6)

    def test_from_forecast_mx_period(self, flat_mx_df):
        dlt = DynamicLifeTable.from_forecast_mx(flat_mx_df, period_year=2040)
        assert not dlt.is_stochastic
        lt = dlt.lifetable
        for x in [20, 50, 80]:
            assert math.isclose(lt.qx(x), 0.02 / 1.01, rel_tol=1e-6)

    def test_from_forecast_qx(self):
        qx_val = 0.01
        df_qx = pd.DataFrame(
            np.full((len(AGES), len(YEARS)), qx_val),
            index=AGES, columns=YEARS,
        )
        dlt = DynamicLifeTable.from_forecast_qx(df_qx, period_year=2030)
        lt = dlt.lifetable
        for x in [10, 40, 70]:
            assert math.isclose(lt.qx(x), qx_val, rel_tol=1e-6)

    def test_from_forecast_log_mx(self, flat_mx_df):
        log_mx_df = np.log(flat_mx_df)
        dlt = DynamicLifeTable.from_forecast_log_mx(log_mx_df, period_year=2035)
        lt = dlt.lifetable
        expected_qx = 0.02 / 1.01
        for x in [10, 50, 90]:
            assert math.isclose(lt.qx(x), expected_qx, rel_tol=1e-6)

    def test_invalid_input_type(self, flat_mx_df):
        with pytest.raises(ValueError, match="input_type"):
            DynamicLifeTable.from_scenarios([flat_mx_df], period_year=2030, input_type="bad")

    def test_mutual_exclusion_birth_period(self, flat_mx_df):
        with pytest.raises(ValueError):
            DynamicLifeTable.from_forecast_mx(flat_mx_df)  # neither
        with pytest.raises(ValueError):
            DynamicLifeTable.from_forecast_mx(flat_mx_df, birth_year=1985, period_year=2040)

    def test_clamping_warning_issued(self, flat_mx_df):
        # birth_year=1990, ages start at 0 → year 1990 < 2020, should warn
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            DynamicLifeTable.from_forecast_mx(flat_mx_df, birth_year=1990)
        assert any("Clamping" in str(warning.message) for warning in w)

    def test_from_scenarios_stochastic(self, scenario_list):
        dlt = DynamicLifeTable.from_scenarios(scenario_list, period_year=2030)
        assert dlt.is_stochastic
        assert dlt.n_scenarios == 10

    def test_from_scenarios_array(self):
        arr = np.full((5, len(AGES), len(YEARS)), 0.02)
        dlt = DynamicLifeTable.from_scenarios_array(
            arr, ages=AGES, years=YEARS, period_year=2030
        )
        assert dlt.is_stochastic
        assert dlt.n_scenarios == 5

    def test_from_scenarios_array_shape_error(self):
        arr = np.ones((5, 80))  # 2-D, wrong
        with pytest.raises(ValueError, match="3-D"):
            DynamicLifeTable.from_scenarios_array(arr, ages=AGES[:80], years=YEARS, period_year=2030)

    def test_lifetable_raises_on_stochastic(self, scenario_list):
        dlt = DynamicLifeTable.from_scenarios(scenario_list, period_year=2030)
        with pytest.raises(ValueError, match="multiple scenarios"):
            _ = dlt.lifetable


# ------------------------------------------------------------------ #
# DynamicActuarialTable — single-path numerical consistency            #
# ------------------------------------------------------------------ #

class TestDynamicActuarialTableSinglePath:

    def test_axn_matches_static(self, soa_ilt_as_mx_df):
        """Single-path DynamicActuarialTable must exactly match static ActuarialTable."""
        # Build dynamic
        dlt = DynamicLifeTable.from_forecast_mx(soa_ilt_as_mx_df, period_year=2030)
        dat = DynamicActuarialTable(dlt, interest=0.06)

        # Build static from the same qx vector
        lt_dyn = dlt.lifetable
        at_static = ActuarialTable(lt_dyn, interest=0.06)

        for x in [20, 40, 60]:
            py_val = dat.axn(x=x)
            static_val = axn(at_static, x=x)
            assert isinstance(py_val, float), "single-path should return float"
            assert math.isclose(py_val, static_val, rel_tol=1e-12), (
                f"axn(x={x}): dynamic={py_val}, static={static_val}"
            )

    def test_Axn_matches_static(self, soa_ilt_as_mx_df):
        dlt = DynamicLifeTable.from_forecast_mx(soa_ilt_as_mx_df, period_year=2030)
        dat = DynamicActuarialTable(dlt, interest=0.06)
        lt_dyn = dlt.lifetable
        at_static = ActuarialTable(lt_dyn, interest=0.06)

        for x, n in [(30, 20), (45, 10), (60, 5)]:
            assert math.isclose(dat.Axn(x=x, n=n), Axn(at_static, x=x, n=n), rel_tol=1e-12)

    def test_Exn_matches_static(self, soa_ilt_as_mx_df):
        dlt = DynamicLifeTable.from_forecast_mx(soa_ilt_as_mx_df, period_year=2030)
        dat = DynamicActuarialTable(dlt, interest=0.06)
        lt_dyn = dlt.lifetable
        at_static = ActuarialTable(lt_dyn, interest=0.06)

        for x, n in [(30, 20), (50, 15)]:
            assert math.isclose(dat.Exn(x=x, n=n), Exn(at_static, x=x, n=n), rel_tol=1e-12)

    def test_insurance_annuity_relationship(self, flat_mx_df):
        dlt = DynamicLifeTable.from_forecast_mx(flat_mx_df, period_year=2030)
        dat = DynamicActuarialTable(dlt, interest=0.05)
        i = dat.i
        d = i / (1 + i)
        for x in [30, 50, 70]:
            A = dat.Axn(x=x)
            a = dat.axn(x=x)
            assert math.isclose(A, 1.0 - d * a, rel_tol=1e-10), f"A=1-d*a failed at x={x}"

    def test_endowment_decomposition(self, flat_mx_df):
        dlt = DynamicLifeTable.from_forecast_mx(flat_mx_df, period_year=2030)
        dat = DynamicActuarialTable(dlt, interest=0.05)
        for x, n in [(30, 20), (50, 10)]:
            assert math.isclose(
                dat.AExn(x=x, n=n),
                dat.Axn(x=x, n=n) + dat.Exn(x=x, n=n),
                rel_tol=1e-12,
            )

    def test_net_premium_returns_float(self, flat_mx_df):
        dlt = DynamicLifeTable.from_forecast_mx(flat_mx_df, period_year=2030)
        dat = DynamicActuarialTable(dlt, interest=0.05)
        P = dat.net_premium(x=40, n=20)
        assert isinstance(P, float)
        assert P > 0

    def test_reserve_recursion_length(self, flat_mx_df):
        dlt = DynamicLifeTable.from_forecast_mx(flat_mx_df, period_year=2030)
        dat = DynamicActuarialTable(dlt, interest=0.05)
        reserves = dat.reserve_recursion(x=40, n=20)
        assert len(reserves) == 21
        assert math.isclose(reserves[0], 0.0, abs_tol=1e-10)


# ------------------------------------------------------------------ #
# DynamicActuarialTable — stochastic                                   #
# ------------------------------------------------------------------ #

class TestDynamicActuarialTableStochastic:

    def test_axn_returns_stochastic_result(self, scenario_list):
        dlt = DynamicLifeTable.from_scenarios(scenario_list, period_year=2030)
        dat = DynamicActuarialTable(dlt, interest=0.05)
        result = dat.axn(x=40)
        assert isinstance(result, StochasticResult)
        assert result.n == 10

    def test_identical_scenarios_zero_std(self, scenario_list):
        """10 identical scenarios → std must be 0."""
        dlt = DynamicLifeTable.from_scenarios(scenario_list, period_year=2030)
        dat = DynamicActuarialTable(dlt, interest=0.05)
        result = dat.axn(x=40)
        assert math.isclose(result.std, 0.0, abs_tol=1e-10)

    def test_identical_scenarios_mean_equals_single(self, flat_mx_df, scenario_list):
        """Mean across identical scenarios == single-path result."""
        dlt_single = DynamicLifeTable.from_forecast_mx(flat_mx_df, period_year=2030)
        dat_single = DynamicActuarialTable(dlt_single, interest=0.05)

        dlt_stoch = DynamicLifeTable.from_scenarios(scenario_list, period_year=2030)
        dat_stoch = DynamicActuarialTable(dlt_stoch, interest=0.05)

        for x in [30, 50, 70]:
            assert math.isclose(
                dat_stoch.axn(x=x).mean,
                dat_single.axn(x=x),
                rel_tol=1e-10,
            )

    def test_varying_scenarios_nonzero_std(self):
        """Scenarios with different mortality → std > 0."""
        rng = np.random.default_rng(42)
        scenarios = [
            pd.DataFrame(
                np.full((len(AGES), len(YEARS)), mx),
                index=AGES, columns=YEARS,
            )
            for mx in rng.uniform(0.01, 0.05, size=50)
        ]
        dlt = DynamicLifeTable.from_scenarios(scenarios, period_year=2030)
        dat = DynamicActuarialTable(dlt, interest=0.05)
        result = dat.axn(x=40)
        assert result.std > 0.0
        assert result.n == 50

    def test_stochastic_ci_contains_mean(self):
        rng = np.random.default_rng(7)
        scenarios = [
            pd.DataFrame(
                np.full((len(AGES), len(YEARS)), mx),
                index=AGES, columns=YEARS,
            )
            for mx in rng.uniform(0.005, 0.05, size=200)
        ]
        dlt = DynamicLifeTable.from_scenarios(scenarios, period_year=2030)
        dat = DynamicActuarialTable(dlt, interest=0.04)
        result = dat.axn(x=40)
        lo, hi = result.ci(0.95)
        assert lo <= result.mean <= hi

    def test_reserve_recursion_stochastic(self, scenario_list):
        dlt = DynamicLifeTable.from_scenarios(scenario_list, period_year=2030)
        dat = DynamicActuarialTable(dlt, interest=0.05)
        reserves = dat.reserve_recursion(x=40, n=10)
        assert len(reserves) == 11
        assert isinstance(reserves[5], StochasticResult)
        # All identical scenarios → std = 0
        for r in reserves:
            assert math.isclose(r.std, 0.0, abs_tol=1e-10)

    def test_net_premium_stochastic(self):
        rng = np.random.default_rng(99)
        scenarios = [
            pd.DataFrame(
                np.full((len(AGES), len(YEARS)), mx),
                index=AGES, columns=YEARS,
            )
            for mx in rng.uniform(0.01, 0.04, size=30)
        ]
        dlt = DynamicLifeTable.from_scenarios(scenarios, period_year=2030)
        dat = DynamicActuarialTable(dlt, interest=0.04)
        result = dat.net_premium(x=40, n=20, benefit="endowment")
        assert isinstance(result, StochasticResult)
        assert result.mean > 0


# ------------------------------------------------------------------ #
# StochasticResult                                                     #
# ------------------------------------------------------------------ #

class TestStochasticResult:

    def test_basic_stats(self):
        samples = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        r = StochasticResult(samples)
        assert math.isclose(r.mean, 3.0)
        assert math.isclose(r.median, 3.0)
        assert math.isclose(r.min, 1.0)
        assert math.isclose(r.max, 5.0)
        assert r.n == 5

    def test_quantile(self):
        samples = np.arange(1, 101, dtype=float)
        r = StochasticResult(samples)
        assert math.isclose(r.quantile(0.5), 50.5, rel_tol=1e-3)
        assert math.isclose(r.quantile(0.0), 1.0)
        assert math.isclose(r.quantile(1.0), 100.0)

    def test_ci_symmetric(self):
        samples = np.arange(1, 101, dtype=float)
        r = StochasticResult(samples)
        lo, hi = r.ci(0.90)
        assert lo < hi
        assert lo <= r.mean <= hi

    def test_float_coercion(self):
        r = StochasticResult(np.array([2.5, 3.5]))
        assert math.isclose(float(r), 3.0)

    def test_summary_keys(self):
        r = StochasticResult(np.arange(10, dtype=float))
        s = r.summary()
        for key in ("mean", "std", "median", "p05", "p25", "p75", "p95", "min", "max", "n"):
            assert key in s

    def test_non_1d_raises(self):
        with pytest.raises(ValueError):
            StochasticResult(np.ones((3, 4)))
