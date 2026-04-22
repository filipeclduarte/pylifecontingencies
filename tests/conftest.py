"""
Pytest fixtures shared across the test suite.
"""

import numpy as np
import pytest

from pylifecontingencies import LifeTable, ActuarialTable, InterestRate, load_table


# ------------------------------------------------------------------ #
# Simple analytic life tables for fast deterministic testing          #
# ------------------------------------------------------------------ #

@pytest.fixture
def de_moivre_100():
    """De Moivre table with omega=100: l_x = 100 - x, q_x = 1/(100-x)."""
    lx = np.array([float(100 - x) for x in range(101)])
    return LifeTable(lx, x_min=0, name="deMoivre100")


@pytest.fixture
def soa_ilt():
    """Bundled SOA Illustrative Life Table."""
    return load_table("soa_ilt")


@pytest.fixture
def at_soa_6pct(soa_ilt):
    """ActuarialTable with SOA ILT at 6% interest."""
    return ActuarialTable(soa_ilt, interest=0.06)


@pytest.fixture
def at_soa_3pct(soa_ilt):
    """ActuarialTable with SOA ILT at 3% interest."""
    return ActuarialTable(soa_ilt, interest=0.03)


# ------------------------------------------------------------------ #
# rpy2 / R fixtures (skipped automatically if rpy2 is not installed) #
# ------------------------------------------------------------------ #

def pytest_configure(config):
    config.addinivalue_line(
        "markers", "rpy2: marks tests that require rpy2 and R lifecontingencies"
    )


@pytest.fixture(scope="session")
def r_lifecontingencies():
    """
    Load R lifecontingencies package via rpy2.

    Tests decorated with this fixture are automatically skipped if
    rpy2 or R is not available.
    """
    pytest.importorskip("rpy2", reason="rpy2 not installed — skipping R parity tests")
    try:
        from rpy2.robjects.packages import importr
        from rpy2 import robjects as ro

        lc = importr("lifecontingencies")
        return lc, ro
    except Exception as exc:
        pytest.skip(f"R lifecontingencies not available: {exc}")


@pytest.fixture(scope="session")
def r_soa_ilt(r_lifecontingencies):
    """SOA ILT as an R actuarialtable object at i=0.06."""
    lc, ro = r_lifecontingencies
    ro.r('data("soa08")')
    # Use soa08 which is the standard test table in lifecontingencies
    at = ro.r('soa08Act <- new("actuarialtable", x=soa08@x, lx=soa08@lx, interest=0.06)')
    return at
