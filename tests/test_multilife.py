"""
Unit tests for multi-life actuarial functions.

Tests cover:
1. De Moivre analytic results (closed-form joint-life values)
2. Inclusion-exclusion identities: ä_{x̄ȳ} = äₓ + ä_y − ä_{xy}
3. Boundary cases: n=1, same life/table, symmetry
4. SOA ILT-based numerical checks
"""

import math

import numpy as np
import pytest

from pylifecontingencies import (
    LifeTable,
    ActuarialTable,
    load_table,
    axn,
    Axn,
    Exn,
    exn,
)
from pylifecontingencies.multilife import (
    pxyt,
    qxyt,
    exyt,
    axyn,
    Axyn as Axyn_ml,
    Exyn,
    AExyn,
)


# ------------------------------------------------------------------ #
# Fixtures                                                             #
# ------------------------------------------------------------------ #

@pytest.fixture
def de_moivre_100():
    """De Moivre table: l_x = 100 - x, omega = 100."""
    lx = np.array([float(100 - x) for x in range(101)])
    return LifeTable(lx, x_min=0, name="deMoivre100")


@pytest.fixture
def soa_ilt():
    return load_table("soa_ilt")


@pytest.fixture
def at_soa_6(soa_ilt):
    return ActuarialTable(soa_ilt, interest=0.06)


@pytest.fixture
def at_soa_3(soa_ilt):
    return ActuarialTable(soa_ilt, interest=0.03)


@pytest.fixture
def at_dm(de_moivre_100):
    return ActuarialTable(de_moivre_100, interest=0.06)


# ------------------------------------------------------------------ #
# pxyt tests                                                           #
# ------------------------------------------------------------------ #

class TestPxyt:
    """Tests for joint/last-survivor survival probabilities."""

    def test_joint_basic(self, de_moivre_100):
        """ₜp_{xy} = ₜpₓ · ₜp_y for De Moivre."""
        lt = de_moivre_100
        x, y, t = 40, 50, 10
        px = lt.npx(x, t)  # (100-50)/(100-40) = 50/60
        py = lt.npx(y, t)  # (100-60)/(100-50) = 40/50
        expected = px * py
        assert math.isclose(pxyt(lt, lt, x, y, t, "joint"), expected, rel_tol=1e-12)

    def test_last_basic(self, de_moivre_100):
        """ₜp_{x̄ȳ} = ₜpₓ + ₜp_y − ₜpₓ·ₜp_y."""
        lt = de_moivre_100
        x, y, t = 40, 50, 10
        px = lt.npx(x, t)
        py = lt.npx(y, t)
        expected = px + py - px * py
        assert math.isclose(pxyt(lt, lt, x, y, t, "last"), expected, rel_tol=1e-12)

    def test_qxyt_complement(self, de_moivre_100):
        """qxyt = 1 - pxyt."""
        lt = de_moivre_100
        for status in ("joint", "last"):
            p = pxyt(lt, lt, 40, 50, 10, status)
            q = qxyt(lt, lt, 40, 50, 10, status)
            assert math.isclose(p + q, 1.0, abs_tol=1e-15)

    def test_joint_t0(self, de_moivre_100):
        """ₜ=0: both alive with certainty → p = 1."""
        assert pxyt(de_moivre_100, de_moivre_100, 40, 50, 0, "joint") == 1.0

    def test_last_t0(self, de_moivre_100):
        """ₜ=0: at least one alive with certainty → p = 1."""
        assert pxyt(de_moivre_100, de_moivre_100, 40, 50, 0, "last") == 1.0

    def test_joint_geq_last(self, soa_ilt):
        """Joint survival ≤ each single survival ≤ last-survivor survival."""
        lt = soa_ilt
        x, y, t = 40, 50, 10
        p_joint = pxyt(lt, lt, x, y, t, "joint")
        p_last = pxyt(lt, lt, x, y, t, "last")
        px = lt.npx(x, t)
        py = lt.npx(y, t)
        assert p_joint <= min(px, py) + 1e-15
        assert p_last >= max(px, py) - 1e-15

    def test_same_age_symmetry(self, soa_ilt):
        """pxyt(lt, lt, x, x, t) = pxyt(lt, lt, x, x, t) — symmetric."""
        lt = soa_ilt
        p1 = pxyt(lt, lt, 40, 50, 10, "joint")
        p2 = pxyt(lt, lt, 50, 40, 10, "joint")
        assert math.isclose(p1, p2, rel_tol=1e-12)

    def test_invalid_status(self, de_moivre_100):
        with pytest.raises(ValueError, match="status"):
            pxyt(de_moivre_100, de_moivre_100, 40, 50, 10, "invalid")

    def test_accepts_actuarialtable(self, at_soa_6):
        """pxyt should accept ActuarialTable objects transparently."""
        p = pxyt(at_soa_6, at_soa_6, 40, 50, 10, "joint")
        assert 0 < p < 1


# ------------------------------------------------------------------ #
# exyt tests                                                           #
# ------------------------------------------------------------------ #

class TestExyt:
    """Tests for joint/last-survivor life expectation."""

    def test_de_moivre_joint(self, de_moivre_100):
        """De Moivre joint-life: e_{xy} = Σ ₜp_{xy}."""
        lt = de_moivre_100
        x, y = 40, 50
        # Manual calculation
        max_t = min(100 - x, 100 - y)  # min(60, 50) = 50
        manual = sum(lt.npx(x, t) * lt.npx(y, t) for t in range(1, max_t + 1))
        assert math.isclose(exyt(lt, lt, x, y, "joint"), manual, rel_tol=1e-10)

    def test_last_inclusion_exclusion(self, soa_ilt):
        """e_{x̄ȳ} = eₓ + e_y − e_{xy}."""
        lt = soa_ilt
        x, y = 40, 50
        e_last = exyt(lt, lt, x, y, "last")
        ex = exn(lt, x)
        ey = exn(lt, y)
        exy = exyt(lt, lt, x, y, "joint")
        assert math.isclose(e_last, ex + ey - exy, rel_tol=1e-10)

    def test_joint_leq_single(self, soa_ilt):
        """e_{xy} ≤ min(eₓ, e_y)."""
        lt = soa_ilt
        x, y = 40, 50
        exy = exyt(lt, lt, x, y, "joint")
        ex = exn(lt, x)
        ey = exn(lt, y)
        assert exy <= min(ex, ey) + 1e-10

    def test_last_geq_single(self, soa_ilt):
        """e_{x̄ȳ} ≥ max(eₓ, e_y)."""
        lt = soa_ilt
        x, y = 40, 50
        e_last = exyt(lt, lt, x, y, "last")
        ex = exn(lt, x)
        ey = exn(lt, y)
        assert e_last >= max(ex, ey) - 1e-10


# ------------------------------------------------------------------ #
# Exyn tests                                                           #
# ------------------------------------------------------------------ #

class TestExyn:
    """Tests for joint/last-survivor pure endowment."""

    def test_joint_basic(self, at_dm):
        """ₙE_{xy} = vⁿ · ₙp_{xy}."""
        at = at_dm
        x, y, n = 40, 50, 10
        v = at.v
        p = pxyt(at, at, x, y, n, "joint")
        assert math.isclose(Exyn(at, at, x, y, n, "joint"), v**n * p, rel_tol=1e-12)

    def test_last_basic(self, at_dm):
        """ₙE_{x̄ȳ} = vⁿ · ₙp_{x̄ȳ}."""
        at = at_dm
        x, y, n = 40, 50, 10
        v = at.v
        p = pxyt(at, at, x, y, n, "last")
        assert math.isclose(Exyn(at, at, x, y, n, "last"), v**n * p, rel_tol=1e-12)

    def test_joint_leq_single(self, at_soa_6):
        """ₙE_{xy} ≤ min(ₙEₓ, ₙE_y)."""
        at = at_soa_6
        x, y, n = 40, 50, 10
        exy = Exyn(at, at, x, y, n, "joint")
        ex = Exn(at, x, n)
        ey = Exn(at, y, n)
        assert exy <= min(ex, ey) + 1e-12


# ------------------------------------------------------------------ #
# axyn tests                                                           #
# ------------------------------------------------------------------ #

class TestAxyn:
    """Tests for joint/last-survivor annuities."""

    def test_joint_whole_life(self, at_soa_6):
        """Joint whole-life annuity: ä_{xy} via direct summation."""
        at = at_soa_6
        x, y = 40, 50
        lt = at.lifetable
        v = at.v
        max_t = min(lt.omega - x, lt.omega - y)

        manual = sum(
            v**t * (lt.npx(x, t) if t > 0 else 1.0) * (lt.npx(y, t) if t > 0 else 1.0)
            for t in range(max_t)
        )
        result = axyn(at, at, x, y, status="joint")
        assert math.isclose(result, manual, rel_tol=1e-10)

    def test_joint_term(self, at_soa_6):
        """Joint 20-year annuity-due."""
        at = at_soa_6
        x, y, n = 40, 50, 20
        lt = at.lifetable
        v = at.v

        manual = sum(
            v**t * (lt.npx(x, t) if t > 0 else 1.0) * (lt.npx(y, t) if t > 0 else 1.0)
            for t in range(n)
        )
        result = axyn(at, at, x, y, n=20, status="joint")
        assert math.isclose(result, manual, rel_tol=1e-10)

    def test_last_inclusion_exclusion(self, at_soa_6):
        """ä_{x̄ȳ} = äₓ + ä_y − ä_{xy} for whole-life."""
        at = at_soa_6
        x, y = 40, 50

        ax_val = axn(at, x)
        ay_val = axn(at, y)
        axy_val = axyn(at, at, x, y, status="joint")
        a_last = axyn(at, at, x, y, status="last")

        assert math.isclose(a_last, ax_val + ay_val - axy_val, rel_tol=1e-10)

    def test_last_term_inclusion_exclusion(self, at_soa_6):
        """ä_{x̄ȳ:n|} = äₓ_{:n|} + ä_y_{:n|} − ä_{xy:n|} for term."""
        at = at_soa_6
        x, y, n = 40, 50, 20

        ax_val = axn(at, x, n=n)
        ay_val = axn(at, y, n=n)
        axy_val = axyn(at, at, x, y, n=n, status="joint")
        a_last = axyn(at, at, x, y, n=n, status="last")

        assert math.isclose(a_last, ax_val + ay_val - axy_val, rel_tol=1e-10)

    def test_joint_leq_single(self, at_soa_6):
        """ä_{xy} ≤ min(äₓ, ä_y)."""
        at = at_soa_6
        x, y = 40, 50
        axy = axyn(at, at, x, y, status="joint")
        ax_val = axn(at, x)
        ay_val = axn(at, y)
        assert axy <= min(ax_val, ay_val) + 1e-10

    def test_last_geq_single(self, at_soa_6):
        """ä_{x̄ȳ} ≥ max(äₓ, ä_y)."""
        at = at_soa_6
        x, y = 40, 50
        a_last = axyn(at, at, x, y, status="last")
        ax_val = axn(at, x)
        ay_val = axn(at, y)
        assert a_last >= max(ax_val, ay_val) - 1e-10

    def test_n1_equals_1(self, at_soa_6):
        """ä_{xy:1|} = 1 (first payment is certain, annuity-due)."""
        at = at_soa_6
        assert math.isclose(axyn(at, at, 40, 50, n=1, status="joint"), 1.0, rel_tol=1e-12)

    def test_same_ages_same_table(self, at_soa_6):
        """Same ages + same table: ä_{xx} symmetric."""
        at = at_soa_6
        a1 = axyn(at, at, 40, 50, status="joint")
        a2 = axyn(at, at, 50, 40, status="joint")
        assert math.isclose(a1, a2, rel_tol=1e-12)

    def test_kthly_joint(self, at_soa_6):
        """k-thly annuity via UDD: ä^(2)_{xy} should be close to but not equal ä_{xy}."""
        at = at_soa_6
        x, y = 40, 50
        a1 = axyn(at, at, x, y, status="joint", k=1)
        a2 = axyn(at, at, x, y, status="joint", k=2)
        # Semi-annual payments should give a value close to annual
        assert abs(a2 - a1) < 1.0
        assert a2 != a1  # but not identical


# ------------------------------------------------------------------ #
# Axyn tests                                                           #
# ------------------------------------------------------------------ #

class TestAxynInsurance:
    """Tests for joint/last-survivor life insurance."""

    def test_joint_whole_life(self, at_soa_6):
        """Joint whole-life insurance: direct summation."""
        at = at_soa_6
        x, y = 40, 50
        lt = at.lifetable
        v = at.v
        max_t = min(lt.omega - x, lt.omega - y)

        manual = 0.0
        for t in range(max_t):
            tpxy = lt.npx(x, t) * lt.npx(y, t) if t > 0 else 1.0
            px_t = lt.npx(x + t, 1) if x + t < lt.omega else 0.0
            py_t = lt.npx(y + t, 1) if y + t < lt.omega else 0.0
            q_joint = 1.0 - px_t * py_t
            manual += v**(t+1) * tpxy * q_joint

        result = Axyn_ml(at, at, x, y, status="joint")
        assert math.isclose(result, manual, rel_tol=1e-10)

    def test_last_inclusion_exclusion(self, at_soa_6):
        """A_{x̄ȳ} = Aₓ + A_y − A_{xy}."""
        at = at_soa_6
        x, y = 40, 50

        Ax_val = Axn(at, x)
        Ay_val = Axn(at, y)
        Axy_val = Axyn_ml(at, at, x, y, status="joint")
        A_last = Axyn_ml(at, at, x, y, status="last")

        assert math.isclose(A_last, Ax_val + Ay_val - Axy_val, rel_tol=1e-10)

    def test_last_term_inclusion_exclusion(self, at_soa_6):
        """A_{x̄ȳ:n|} = Aₓ_{:n|} + A_y_{:n|} − A_{xy:n|}."""
        at = at_soa_6
        x, y, n = 40, 50, 20

        Ax_val = Axn(at, x, n=n)
        Ay_val = Axn(at, y, n=n)
        Axy_val = Axyn_ml(at, at, x, y, n=n, status="joint")
        A_last = Axyn_ml(at, at, x, y, n=n, status="last")

        assert math.isclose(A_last, Ax_val + Ay_val - Axy_val, rel_tol=1e-10)

    def test_insurance_annuity_identity(self, at_soa_6):
        """A_{xy} + d · ä_{xy} = 1 (whole-life identity for joint-life)."""
        at = at_soa_6
        x, y = 40, 50
        d = at.interest.d
        Axy = Axyn_ml(at, at, x, y, status="joint")
        axy = axyn(at, at, x, y, status="joint")
        # A_{xy} = 1 - d · ä_{xy}
        assert math.isclose(Axy + d * axy, 1.0, abs_tol=1e-8)

    def test_between_0_and_1(self, at_soa_6):
        """Insurance value should be between 0 and 1."""
        at = at_soa_6
        for status in ("joint", "last"):
            A = Axyn_ml(at, at, 40, 50, status=status)
            assert 0 < A < 1

    def test_kthly_joint(self, at_soa_6):
        """k-thly insurance via UDD."""
        at = at_soa_6
        A1 = Axyn_ml(at, at, 40, 50, status="joint", k=1)
        A2 = Axyn_ml(at, at, 40, 50, status="joint", k=2)
        # They should be close
        assert abs(A2 - A1) < 0.05
        assert A2 != A1


# ------------------------------------------------------------------ #
# AExyn tests                                                         #
# ------------------------------------------------------------------ #

class TestAExyn:
    """Tests for joint/last-survivor endowment insurance."""

    def test_endowment_decomposition(self, at_soa_6):
        """A_{xy:n|} = A¹_{xy:n|} + ₙE_{xy}."""
        at = at_soa_6
        x, y, n = 40, 50, 20
        for status in ("joint", "last"):
            endow = AExyn(at, at, x, y, n, status=status)
            term = Axyn_ml(at, at, x, y, n=n, status=status)
            pure = Exyn(at, at, x, y, n, status=status)
            assert math.isclose(endow, term + pure, rel_tol=1e-12)

    def test_between_0_and_1(self, at_soa_6):
        """Endowment value should be between 0 and 1."""
        at = at_soa_6
        for status in ("joint", "last"):
            AE = AExyn(at, at, 40, 50, 20, status=status)
            assert 0 < AE < 1


# ------------------------------------------------------------------ #
# Cross-validation: different interest rates                          #
# ------------------------------------------------------------------ #

class TestCrossValidation:
    """Additional cross-checks across interest rates."""

    @pytest.mark.parametrize("i", [0.01, 0.03, 0.06, 0.10])
    def test_insurance_annuity_identity_parametric(self, soa_ilt, i):
        """A_{xy} + d·ä_{xy} = 1 for various interest rates."""
        at = ActuarialTable(soa_ilt, interest=i)
        x, y = 30, 40
        d = at.interest.d
        Axy = Axyn_ml(at, at, x, y, status="joint")
        axy = axyn(at, at, x, y, status="joint")
        assert math.isclose(Axy + d * axy, 1.0, abs_tol=1e-8)

    @pytest.mark.parametrize("i", [0.01, 0.03, 0.06])
    def test_last_survivor_identity_parametric(self, soa_ilt, i):
        """Inclusion-exclusion for various interest rates."""
        at = ActuarialTable(soa_ilt, interest=i)
        x, y = 30, 40

        # Annuity
        a_last = axyn(at, at, x, y, status="last")
        ax_val = axn(at, x)
        ay_val = axn(at, y)
        axy = axyn(at, at, x, y, status="joint")
        assert math.isclose(a_last, ax_val + ay_val - axy, rel_tol=1e-10)

        # Insurance
        A_last = Axyn_ml(at, at, x, y, status="last")
        Ax_val = Axn(at, x)
        Ay_val = Axn(at, y)
        Axy = Axyn_ml(at, at, x, y, status="joint")
        assert math.isclose(A_last, Ax_val + Ay_val - Axy, rel_tol=1e-10)
