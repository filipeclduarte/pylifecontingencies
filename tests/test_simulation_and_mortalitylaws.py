"""
Tests for Monte Carlo PV simulation and parametric mortality-law fitting.
"""

import math

import numpy as np
import pytest

from pylifecontingencies import (
    ActuarialTable,
    Axn,
    GompertzMakeham,
    HeligmanPollard,
    available_mortality_laws,
    axn,
    fit_mortality_law,
    simulate_pv,
)


class TestSimulatePV:

    def test_term_insurance_mean_matches_closed_form(self, soa_ilt):
        at = ActuarialTable(soa_ilt, interest=0.03)
        result = simulate_pv(
            at, x=40, n=20, benefit="term", n_sim=20_000, random_state=0
        )

        assert len(result.samples) == 20_000
        assert result.std > 0.0
        assert math.isclose(result.mean, Axn(at, x=40, n=20), abs_tol=0.01)

    def test_annuity_mean_matches_closed_form(self, soa_ilt):
        at = ActuarialTable(soa_ilt, interest=0.03)
        result = at.simulate_pv(
            x=40, n=20, benefit="annuity", n_sim=20_000, random_state=0
        )

        assert len(result.samples) == 20_000
        assert math.isclose(result.mean, axn(at, x=40, n=20), abs_tol=0.15)

    def test_endowment_requires_finite_term(self, soa_ilt):
        at = ActuarialTable(soa_ilt, interest=0.03)
        with pytest.raises(ValueError):
            simulate_pv(at, x=40, benefit="endowment", n_sim=100, random_state=0)

    def test_benefit_aliases_are_accepted(self, soa_ilt):
        at = ActuarialTable(soa_ilt, interest=0.03)
        r1 = simulate_pv(at, x=40, n=20, benefit="term_insurance", n_sim=1000, random_state=0)
        r2 = simulate_pv(at, x=40, n=None, benefit="whole_life", n_sim=1000, random_state=0)

        assert len(r1.samples) == 1000
        assert len(r2.samples) == 1000


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

        assert set(["age", "qx_observed", "qx_fitted", "residual"]).issubset(df.columns)
        assert set(fit.param_names) == set(fit.params_dict)

    def test_rejects_non_native_law_name(self, soa_ilt):
        with pytest.raises(ValueError, match="Unknown mortality law"):
            fit_mortality_law(soa_ilt, "siler", ages=np.arange(40, 90))
