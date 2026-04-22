"""
Unit tests for LifeTable and ActuarialTable, plus core actuarial values.
Uses the de Moivre table for closed-form verification.
"""

import math
import pytest
import numpy as np

from pylifecontingencies import (
    LifeTable,
    ActuarialTable,
    InterestRate,
    load_table,
    axn,
    Axn,
    Exn,
    AExn,
    IAxn,
    DAxn,
    pxt,
    qxt,
    exn,
)


class TestLifeTable:

    def test_from_qx_basic(self, de_moivre_100):
        lt = de_moivre_100
        assert lt.x_min == 0
        assert lt.omega == 100
        # l_0 = 100, l_50 = 50
        assert math.isclose(lt.lx(0), 100.0)
        assert math.isclose(lt.lx(50), 50.0)
        assert math.isclose(lt.lx(100), 0.0)

    def test_from_qx_monotone_check(self):
        # non-monotone lx should raise
        with pytest.raises(ValueError):
            LifeTable(np.array([100.0, 110.0, 50.0, 0.0]))

    def test_from_qx_terminal_zero(self):
        with pytest.raises(ValueError):
            LifeTable(np.array([100.0, 80.0, 50.0, 10.0]))  # last != 0

    def test_from_qx_adds_terminal(self):
        qx = np.array([0.1, 0.2, 0.3])
        # No terminal 1.0 supplied — from_qx appends 1.0, giving ages 0..4, omega=4
        lt = LifeTable.from_qx(qx)
        assert lt.omega == 4

    def test_px_qx_sum_to_one(self, de_moivre_100):
        for x in [0, 20, 50, 80, 99]:
            assert math.isclose(de_moivre_100.px(x) + de_moivre_100.qx(x), 1.0)

    def test_npx_de_moivre(self, de_moivre_100):
        # De Moivre: _n p_x = (omega - x - n) / (omega - x)
        lt = de_moivre_100
        omega = 100
        x, n = 40, 20
        expected = (omega - x - n) / (omega - x)
        assert math.isclose(lt.npx(x, n), expected, rel_tol=1e-12)

    def test_get_omega(self, de_moivre_100):
        assert de_moivre_100.getOmega() == 100

    def test_to_dataframe(self, soa_ilt):
        df = soa_ilt.to_dataframe()
        assert "age" in df.columns
        assert "lx" in df.columns
        assert "qx" in df.columns
        assert (df["qx"] >= 0).all()
        assert (df["qx"] <= 1).all()

    def test_load_soa_ilt(self, soa_ilt):
        assert soa_ilt.x_min == 0
        assert soa_ilt.omega == 100


class TestActuarialTable:

    def test_commutation_Dx(self, de_moivre_100):
        at = ActuarialTable(de_moivre_100, interest=0.06)
        v = 1 / 1.06
        # D_0 = v^0 * l_0 = 100
        assert math.isclose(at.Dx(0), 100.0 * v ** 0)
        # D_50 = v^50 * l_50 = v^50 * 50
        assert math.isclose(at.Dx(50), (v ** 50) * 50.0, rel_tol=1e-10)

    def test_Nx_ge_Dx(self, de_moivre_100):
        at = ActuarialTable(de_moivre_100, interest=0.06)
        # N_x = sum D_{x}, D_{x+1}, ... >= D_x
        for x in [0, 20, 50]:
            assert at.Nx(x) >= at.Dx(x)

    def test_Mx_ge_zero(self, de_moivre_100):
        at = ActuarialTable(de_moivre_100, interest=0.06)
        for x in [0, 20, 50]:
            assert at.Mx(x) >= 0.0


class TestActuarialFunctions:

    def test_Exn_de_moivre(self, de_moivre_100):
        """_n E_x = v^n * _n p_x = D_{x+n}/D_x."""
        at = ActuarialTable(de_moivre_100, interest=0.06)
        x, n = 30, 20
        v = 1 / 1.06
        p = de_moivre_100.npx(x, n)
        expected = v ** n * p
        assert math.isclose(Exn(at, x=x, n=n), expected, rel_tol=1e-10)

    def test_Exn_beyond_omega_is_zero(self, de_moivre_100):
        at = ActuarialTable(de_moivre_100, interest=0.06)
        assert Exn(at, x=90, n=20) == 0.0

    def test_insurance_annuity_relationship(self, at_soa_6pct):
        """A_x = 1 - d * ä_x."""
        at = at_soa_6pct
        i = at.interest.i
        d = i / (1 + i)
        for x in [30, 45, 60]:
            A = Axn(at, x=x)
            a = axn(at, x=x)
            assert math.isclose(A, 1.0 - d * a, rel_tol=1e-10), f"failed at x={x}"

    def test_endowment_decomposition(self, at_soa_6pct):
        """AExn(x,n) = Axn(x,n) + Exn(x,n)."""
        at = at_soa_6pct
        for x, n in [(30, 20), (45, 10), (60, 5)]:
            assert math.isclose(
                AExn(at, x=x, n=n),
                Axn(at, x=x, n=n) + Exn(at, x=x, n=n),
                rel_tol=1e-12,
            )

    def test_axn_whole_vs_term_limit(self, at_soa_6pct):
        """axn(x, n=omega-x) should equal axn(x) whole-life."""
        at = at_soa_6pct
        x = 40
        n = at.omega - x
        a_term = axn(at, x=x, n=n)
        a_whole = axn(at, x=x)
        assert math.isclose(a_term, a_whole, rel_tol=1e-10)

    def test_DA_IA_identity(self, at_soa_6pct):
        """(DA)^1 + (IA)^1 = (n+1) A^1_{x:n|}."""
        at = at_soa_6pct
        x, n = 40, 20
        da = DAxn(at, x=x, n=n)
        ia = IAxn(at, x=x, n=n)
        a = Axn(at, x=x, n=n)
        assert math.isclose(da + ia, (n + 1) * a, rel_tol=1e-10)

    def test_exn_de_moivre(self, de_moivre_100):
        """De Moivre curtate expectation: e_x = (omega - x - 1) / 2."""
        at = ActuarialTable(de_moivre_100, interest=0.06)
        x = 30
        omega = 100
        expected = (omega - x - 1) / 2
        assert math.isclose(exn(at, x=x), expected, rel_tol=1e-10)

    def test_demographic_pxt_qxt(self, soa_ilt):
        for x in [20, 40, 60]:
            p = pxt(soa_ilt, x=x, t=10)
            q = qxt(soa_ilt, x=x, t=10)
            assert math.isclose(p + q, 1.0, rel_tol=1e-12)
            assert 0.0 <= p <= 1.0
