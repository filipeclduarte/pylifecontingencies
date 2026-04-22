"""
Unit tests for InterestRate and financial conversion functions.
All closed-form: no life table dependency.
"""

import math
import pytest
import numpy as np

from pylifecontingencies import (
    InterestRate,
    convertible2Effective,
    effective2Convertible,
    discount2Interest,
    interest2Discount,
    intensity2Interest,
    interest2Intensity,
    nominal2Real,
    real2Nominal,
)


class TestInterestRate:

    def test_basic_construction(self):
        ir = InterestRate(i=0.05)
        assert ir.i == 0.05

    def test_invalid_rate(self):
        with pytest.raises(ValueError):
            InterestRate(i=-1.0)
        with pytest.raises(ValueError):
            InterestRate(i=-1.5)

    def test_v(self):
        ir = InterestRate(i=0.05)
        assert math.isclose(ir.v, 1 / 1.05, rel_tol=1e-12)

    def test_discount_rate(self):
        ir = InterestRate(i=0.05)
        d = ir.d
        assert math.isclose(d, 0.05 / 1.05, rel_tol=1e-12)
        # d = 1 - v
        assert math.isclose(d, 1.0 - ir.v, rel_tol=1e-12)

    def test_delta(self):
        ir = InterestRate(i=0.05)
        assert math.isclose(ir.delta, math.log(1.05), rel_tol=1e-12)

    def test_from_delta_roundtrip(self):
        ir = InterestRate.from_delta(0.04879016)
        assert math.isclose(ir.i, 0.05, rel_tol=1e-6)

    def test_from_discount_roundtrip(self):
        d = 0.05 / 1.05
        ir = InterestRate.from_discount(d)
        assert math.isclose(ir.i, 0.05, rel_tol=1e-12)

    def test_from_discount_invalid(self):
        with pytest.raises(ValueError):
            InterestRate.from_discount(0.0)
        with pytest.raises(ValueError):
            InterestRate.from_discount(1.0)

    def test_from_nominal_interest(self):
        # i^(12) = 12 * ((1+i)^(1/12) - 1)
        ir = InterestRate(i=0.05)
        i_12 = ir.i_m(12)
        ir_back = InterestRate.from_nominal_interest(i_12, 12)
        assert math.isclose(ir_back.i, 0.05, rel_tol=1e-12)

    def test_i_m_annual(self):
        ir = InterestRate(i=0.06)
        assert math.isclose(ir.i_m(1), 0.06, rel_tol=1e-12)

    def test_d_m_annual(self):
        ir = InterestRate(i=0.06)
        # d^(1) = d
        assert math.isclose(ir.d_m(1), ir.d, rel_tol=1e-12)

    def test_accumulate_discount_inverse(self):
        ir = InterestRate(i=0.06)
        t = 10.5
        assert math.isclose(ir.accumulate(t) * ir.discount_factor(t), 1.0, rel_tol=1e-12)

    def test_alpha_beta_udd(self):
        ir = InterestRate(i=0.06)
        # For k=1: alpha = id/(i*d) = 1 but i^(1)=i, d^(1)=d
        assert math.isclose(ir.alpha(1), 1.0, rel_tol=1e-12)
        assert math.isclose(ir.beta(1), 0.0, abs_tol=1e-12)

    def test_real_rate_fisher(self):
        ir = InterestRate(i=0.05)
        inflation = 0.02
        r = ir.real_rate(inflation)
        # Fisher: (1+r)(1+inf) = (1+i)
        assert math.isclose((1.0 + r) * (1.0 + inflation), 1.05, rel_tol=1e-12)


class TestConversionFunctions:

    def test_discount_interest_roundtrip(self):
        i = 0.06
        d = interest2Discount(i)
        assert math.isclose(discount2Interest(d), i, rel_tol=1e-12)

    def test_intensity_interest_roundtrip(self):
        i = 0.05
        delta = interest2Intensity(i)
        assert math.isclose(intensity2Interest(delta), i, rel_tol=1e-12)

    def test_nominal_effective_roundtrip(self):
        i = 0.06
        m = 4
        i_m = effective2Convertible(i, m)
        assert math.isclose(convertible2Effective(i_m, m), i, rel_tol=1e-12)

    def test_nominal_real_roundtrip(self):
        i_real = 0.03
        inflation = 0.02
        i_nom = real2Nominal(i_real, inflation)
        assert math.isclose(nominal2Real(i_nom, inflation), i_real, rel_tol=1e-12)
