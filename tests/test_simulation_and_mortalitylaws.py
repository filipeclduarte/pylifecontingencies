"""
Tests for Monte Carlo PV simulation and parametric mortality-law fitting.
"""

import math

import numpy as np
import pytest

from pylifecontingencies import (
    ActuarialTable,
    AExn,
    Axn,
    DAxn,
    Exn,
    GompertzMakeham,
    HeligmanPollard,
    IAxn,
    available_mortality_laws,
    axn,
    fit_mortality_law,
    simulate_pv,
)


class TestSimulatePV:

    # ------------------------------------------------------------------ #
    # Closed-form parity for all 7 benefit types                         #
    # ------------------------------------------------------------------ #

    def test_term_insurance_mean_matches_closed_form(self, soa_ilt):
        at = ActuarialTable(soa_ilt, interest=0.03)
        result = simulate_pv(
            at, x=40, n=20, benefit="term", n_sim=20_000, random_state=0
        )
        assert len(result.samples) == 20_000
        assert result.std > 0.0
        assert math.isclose(result.mean, Axn(at, x=40, n=20), abs_tol=0.01)

    def test_whole_life_insurance_mean_matches_closed_form(self, soa_ilt):
        at = ActuarialTable(soa_ilt, interest=0.03)
        result = simulate_pv(
            at, x=40, n=None, benefit="insurance", n_sim=20_000, random_state=1
        )
        assert math.isclose(result.mean, Axn(at, x=40), abs_tol=0.01)

    def test_whole_life_insurance_alias(self, soa_ilt):
        at = ActuarialTable(soa_ilt, interest=0.03)
        r1 = simulate_pv(at, x=40, n=None, benefit="whole_life", n_sim=5_000, random_state=0)
        r2 = simulate_pv(at, x=40, n=None, benefit="whole", n_sim=5_000, random_state=0)
        # Same seed → identical samples, just checking both aliases work
        assert math.isclose(r1.mean, r2.mean, abs_tol=1e-12)

    def test_annuity_mean_matches_closed_form(self, soa_ilt):
        at = ActuarialTable(soa_ilt, interest=0.03)
        result = at.simulate_pv(
            x=40, n=20, benefit="annuity", n_sim=20_000, random_state=0
        )
        assert len(result.samples) == 20_000
        assert math.isclose(result.mean, axn(at, x=40, n=20), abs_tol=0.15)

    def test_pure_endowment_mean_matches_closed_form(self, soa_ilt):
        at = ActuarialTable(soa_ilt, interest=0.03)
        result = simulate_pv(
            at, x=40, n=20, benefit="pure_endowment", n_sim=30_000, random_state=2
        )
        assert math.isclose(result.mean, Exn(at, x=40, n=20), abs_tol=0.01)

    def test_endowment_mean_matches_closed_form(self, soa_ilt):
        at = ActuarialTable(soa_ilt, interest=0.03)
        result = simulate_pv(
            at, x=40, n=20, benefit="endowment", n_sim=30_000, random_state=3
        )
        assert math.isclose(result.mean, AExn(at, x=40, n=20), abs_tol=0.01)

    def test_increasing_insurance_mean_matches_closed_form(self, soa_ilt):
        at = ActuarialTable(soa_ilt, interest=0.03)
        result = simulate_pv(
            at, x=40, n=20, benefit="increasing", n_sim=30_000, random_state=4
        )
        assert math.isclose(result.mean, IAxn(at, x=40, n=20), abs_tol=0.05)

    def test_decreasing_insurance_mean_matches_closed_form(self, soa_ilt):
        at = ActuarialTable(soa_ilt, interest=0.03)
        result = simulate_pv(
            at, x=40, n=20, benefit="decreasing", n_sim=30_000, random_state=5
        )
        assert math.isclose(result.mean, DAxn(at, x=40, n=20), abs_tol=0.05)

    # ------------------------------------------------------------------ #
    # Edge cases / input validation                                       #
    # ------------------------------------------------------------------ #

    def test_endowment_requires_finite_term(self, soa_ilt):
        at = ActuarialTable(soa_ilt, interest=0.03)
        with pytest.raises(ValueError):
            simulate_pv(at, x=40, benefit="endowment", n_sim=100, random_state=0)

    def test_pure_endowment_requires_finite_term(self, soa_ilt):
        at = ActuarialTable(soa_ilt, interest=0.03)
        with pytest.raises(ValueError):
            simulate_pv(at, x=40, benefit="pure_endowment", n_sim=100, random_state=0)

    def test_decreasing_requires_finite_term(self, soa_ilt):
        at = ActuarialTable(soa_ilt, interest=0.03)
        with pytest.raises(ValueError):
            simulate_pv(at, x=40, benefit="decreasing", n_sim=100, random_state=0)

    def test_invalid_n_sim_raises(self, soa_ilt):
        at = ActuarialTable(soa_ilt, interest=0.03)
        with pytest.raises(ValueError, match="n_sim"):
            simulate_pv(at, x=40, n=20, benefit="term", n_sim=0)

    def test_negative_n_raises(self, soa_ilt):
        at = ActuarialTable(soa_ilt, interest=0.03)
        with pytest.raises(ValueError, match="n must be positive"):
            simulate_pv(at, x=40, n=-5, benefit="term", n_sim=100)

    def test_age_out_of_range_raises(self, soa_ilt):
        at = ActuarialTable(soa_ilt, interest=0.03)
        with pytest.raises(ValueError):
            simulate_pv(at, x=999, n=20, benefit="term", n_sim=100)

    def test_unknown_benefit_raises(self, soa_ilt):
        at = ActuarialTable(soa_ilt, interest=0.03)
        with pytest.raises(ValueError, match="Unknown benefit"):
            simulate_pv(at, x=40, n=20, benefit="foobar", n_sim=100)

    def test_benefit_aliases_are_accepted(self, soa_ilt):
        at = ActuarialTable(soa_ilt, interest=0.03)
        r1 = simulate_pv(at, x=40, n=20, benefit="term_insurance", n_sim=1000, random_state=0)
        r2 = simulate_pv(at, x=40, n=None, benefit="whole_life", n_sim=1000, random_state=0)
        assert len(r1.samples) == 1000
        assert len(r2.samples) == 1000

    # ------------------------------------------------------------------ #
    # Reproducibility                                                     #
    # ------------------------------------------------------------------ #

    def test_reproducibility_with_int_seed(self, soa_ilt):
        at = ActuarialTable(soa_ilt, interest=0.03)
        r1 = simulate_pv(at, x=40, n=20, benefit="term", n_sim=500, random_state=42)
        r2 = simulate_pv(at, x=40, n=20, benefit="term", n_sim=500, random_state=42)
        np.testing.assert_array_equal(r1.samples, r2.samples)

    def test_generator_random_state_accepted(self, soa_ilt):
        at = ActuarialTable(soa_ilt, interest=0.03)
        rng = np.random.default_rng(99)
        result = simulate_pv(at, x=40, n=20, benefit="term", n_sim=500, random_state=rng)
        assert len(result.samples) == 500

    # ------------------------------------------------------------------ #
    # StochasticResult API                                                #
    # ------------------------------------------------------------------ #

    def test_stochastic_result_summary_keys(self, soa_ilt):
        at = ActuarialTable(soa_ilt, interest=0.03)
        result = simulate_pv(at, x=40, n=20, benefit="term", n_sim=1000, random_state=0)
        s = result.summary()
        expected = {"mean", "std", "median", "p05", "p25", "p75", "p95", "min", "max", "n"}
        assert expected == set(s.keys())

    def test_stochastic_result_ci_contains_mean(self, soa_ilt):
        at = ActuarialTable(soa_ilt, interest=0.03)
        result = simulate_pv(at, x=40, n=20, benefit="term", n_sim=5000, random_state=0)
        lo, hi = result.ci(0.95)
        assert lo < result.mean < hi

    def test_stochastic_result_quantiles_monotone(self, soa_ilt):
        at = ActuarialTable(soa_ilt, interest=0.03)
        result = simulate_pv(at, x=40, n=20, benefit="term", n_sim=5000, random_state=0)
        assert result.quantile(0.05) <= result.quantile(0.50) <= result.quantile(0.95)

    def test_stochastic_result_float_coercion(self, soa_ilt):
        at = ActuarialTable(soa_ilt, interest=0.03)
        result = simulate_pv(at, x=40, n=20, benefit="term", n_sim=1000, random_state=0)
        assert math.isclose(float(result), result.mean)

    def test_stochastic_result_label_contains_params(self, soa_ilt):
        at = ActuarialTable(soa_ilt, interest=0.03)
        result = simulate_pv(at, x=40, n=20, benefit="term", n_sim=500, random_state=0)
        assert "x=40" in result.label
        assert "n=20" in result.label
        assert "term" in result.label

    def test_stochastic_result_n_property(self, soa_ilt):
        at = ActuarialTable(soa_ilt, interest=0.03)
        result = simulate_pv(at, x=40, n=20, benefit="term", n_sim=777, random_state=0)
        assert result.n == 777

    def test_method_on_actuarialtable_matches_function(self, soa_ilt):
        at = ActuarialTable(soa_ilt, interest=0.03)
        r_func = simulate_pv(at, x=40, n=20, benefit="term", n_sim=500, random_state=7)
        r_meth = at.simulate_pv(x=40, n=20, benefit="term", n_sim=500, random_state=7)
        np.testing.assert_array_equal(r_func.samples, r_meth.samples)


    # ------------------------------------------------------------------ #
    # k-thly (fractional) annuity payments                               #
    # ------------------------------------------------------------------ #

    def test_annuity_monthly_k12_mean_matches_closed_form(self, soa_ilt):
        at = ActuarialTable(soa_ilt, interest=0.03)
        result = simulate_pv(
            at, x=40, n=20, benefit="annuity", k=12, n_sim=100_000, random_state=0
        )
        expected = axn(at, x=40, n=20, k=12)
        assert math.isclose(result.mean, expected, abs_tol=0.02)

    def test_annuity_whole_life_k12_matches_closed_form(self, soa_ilt):
        at = ActuarialTable(soa_ilt, interest=0.03)
        result = simulate_pv(
            at, x=40, n=None, benefit="annuity", k=12, n_sim=100_000, random_state=1
        )
        expected = axn(at, x=40, n=None, k=12)
        assert math.isclose(result.mean, expected, abs_tol=0.05)

    def test_k_greater_than_1_non_annuity_raises(self, soa_ilt):
        at = ActuarialTable(soa_ilt, interest=0.03)
        with pytest.raises(ValueError, match="k > 1"):
            simulate_pv(at, x=40, n=20, benefit="term", k=12, n_sim=100)

    def test_k1_behavior_unchanged_same_seed(self, soa_ilt):
        """k=1 must give bit-for-bit identical output to the no-k call."""
        at = ActuarialTable(soa_ilt, interest=0.03)
        r_default = simulate_pv(at, x=40, n=20, benefit="annuity", n_sim=500, random_state=7)
        r_k1 = simulate_pv(at, x=40, n=20, benefit="annuity", k=1, n_sim=500, random_state=7)
        np.testing.assert_array_equal(r_default.samples, r_k1.samples)

    def test_invalid_k_raises(self, soa_ilt):
        at = ActuarialTable(soa_ilt, interest=0.03)
        with pytest.raises(ValueError, match="k"):
            simulate_pv(at, x=40, n=20, benefit="annuity", k=0, n_sim=100)

    # ------------------------------------------------------------------ #
    # Deferred annuity (m > 0)                                           #
    # ------------------------------------------------------------------ #

    def test_deferred_annuity_term_matches_closed_form(self, soa_ilt):
        """m=10 deferred 10-year annuity: _10E_x * ä_{x+10:10|}."""
        at = ActuarialTable(soa_ilt, interest=0.03)
        from pylifecontingencies import Exn
        expected = Exn(at, x=40, n=10) * axn(at, x=50, n=10)
        result = simulate_pv(
            at, x=40, n=10, benefit="annuity", m=10, n_sim=100_000, random_state=2
        )
        assert math.isclose(result.mean, expected, abs_tol=0.05)

    def test_deferred_whole_life_annuity_matches_closed_form(self, soa_ilt):
        """m=10 deferred whole-life annuity: _10E_x * ä_{x+10}."""
        at = ActuarialTable(soa_ilt, interest=0.03)
        from pylifecontingencies import Exn
        expected = Exn(at, x=40, n=10) * axn(at, x=50, n=None)
        result = simulate_pv(
            at, x=40, n=None, benefit="annuity", m=10, n_sim=100_000, random_state=3
        )
        assert math.isclose(result.mean, expected, abs_tol=0.05)

    def test_deferred_monthly_annuity_matches_closed_form(self, soa_ilt):
        """m=5 deferred k=12 10-year annuity: _5E_x * ä^(12)_{x+5:10|}."""
        at = ActuarialTable(soa_ilt, interest=0.03)
        from pylifecontingencies import Exn
        expected = Exn(at, x=40, n=5) * axn(at, x=45, n=10, k=12)
        result = simulate_pv(
            at, x=40, n=10, benefit="annuity", k=12, m=5, n_sim=100_000, random_state=4
        )
        assert math.isclose(result.mean, expected, abs_tol=0.05)

    def test_deferred_term_insurance_matches_closed_form(self, soa_ilt):
        """m-year deferred term insurance: _mE_x * A^1_{x+m:n|}."""
        at = ActuarialTable(soa_ilt, interest=0.03)
        from pylifecontingencies import Axn, Exn
        expected = Exn(at, x=40, n=10) * Axn(at, x=50, n=10)
        result = simulate_pv(
            at, x=40, n=10, benefit="term", m=10, n_sim=100_000, random_state=5
        )
        assert math.isclose(result.mean, expected, abs_tol=0.01)

    def test_impossible_deferral_gives_zero_pv(self, soa_ilt):
        """When m >= omega - x, no scenario survives deferral → all PVs = 0."""
        at = ActuarialTable(soa_ilt, interest=0.03)
        # x=95, omega=100 → max curtate lifetime K ≤ 4; m=5 is unreachable
        result = simulate_pv(
            at, x=95, n=None, benefit="annuity", m=5, n_sim=10_000, random_state=6
        )
        assert result.mean == 0.0

    def test_invalid_m_raises(self, soa_ilt):
        at = ActuarialTable(soa_ilt, interest=0.03)
        with pytest.raises(ValueError, match="m"):
            simulate_pv(at, x=40, n=20, benefit="annuity", m=-1, n_sim=100)

    # ------------------------------------------------------------------ #
    # StochasticResult.to_dataframe                                       #
    # ------------------------------------------------------------------ #

    def test_to_dataframe_columns_and_length(self, soa_ilt):
        at = ActuarialTable(soa_ilt, interest=0.03)
        result = simulate_pv(at, x=40, n=20, benefit="term", n_sim=500, random_state=0)
        df = result.to_dataframe()
        import pandas as pd
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["pv"]
        assert len(df) == 500

    def test_to_dataframe_values_match_samples(self, soa_ilt):
        at = ActuarialTable(soa_ilt, interest=0.03)
        result = simulate_pv(at, x=40, n=20, benefit="term", n_sim=200, random_state=0)
        df = result.to_dataframe()
        np.testing.assert_array_equal(df["pv"].to_numpy(), result.samples)

    def test_to_dataframe_supports_pandas_operations(self, soa_ilt):
        at = ActuarialTable(soa_ilt, interest=0.03)
        result = simulate_pv(at, x=40, n=20, benefit="annuity", n_sim=1000, random_state=0)
        df = result.to_dataframe()
        assert abs(df["pv"].mean() - result.mean) < 1e-10
        assert df["pv"].min() >= 0.0


class TestMortalityLaws:

    def test_available_mortality_laws(self):
        laws = available_mortality_laws()
        assert "GompertzMakeham" in laws
        assert "HeligmanPollard" in laws

    def test_gompertz_makeham_fit_on_soa_table(self, soa_ilt):
        fit = fit_mortality_law(
            soa_ilt,
            GompertzMakeham(),
            ages=np.arange(40, 90),
        )
        assert fit.law_name == "GompertzMakeham"
        assert fit.rmse < 0.002
        assert np.all(np.isfinite(fit.params))
        assert fit.aic < fit.bic

    def test_heligman_pollard_fit_on_soa_table(self, soa_ilt):
        fit = fit_mortality_law(
            soa_ilt,
            HeligmanPollard(),
            ages=np.arange(1, 90),
        )
        assert fit.law_name == "HeligmanPollard"
        assert fit.rmse < 0.01
        assert np.all(np.isfinite(fit.params))
        assert fit.fitted_qx.shape == fit.observed_qx.shape

    def test_string_dispatch_for_fit_mortality_law(self, soa_ilt):
        fit = fit_mortality_law(soa_ilt, "gompertz_makeham", ages=np.arange(40, 90))
        assert fit.law_name == "GompertzMakeham"

    def test_fit_helpers_expose_params_and_dataframe(self, soa_ilt):
        fit = fit_mortality_law(soa_ilt, GompertzMakeham(), ages=np.arange(40, 90))
        df = fit.to_dataframe()
        assert {"age", "qx_observed", "qx_fitted", "residual"}.issubset(df.columns)
        assert set(fit.param_names) == set(fit.params_dict)

    def test_rejects_non_native_law_name(self, soa_ilt):
        with pytest.raises(ValueError, match="Unknown mortality law"):
            fit_mortality_law(soa_ilt, "siler", ages=np.arange(40, 90))
