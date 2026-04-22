"""
rpy2-backed parity tests against R lifecontingencies.

These tests are skipped automatically when rpy2 or R is not installed.
To run:  pytest tests/test_actuarial_vs_r.py -v -s

The grid covers:
  - SOA ILT (Python) vs soa08 (R)  — same underlying table
  - ages:   [20, 40, 60]
  - terms:  [1, 10, 20, whole-life]
  - i:      [0.01, 0.03, 0.06]
"""

import math
import pytest
import numpy as np

pytestmark = pytest.mark.rpy2

AGES = [20, 40, 60]
TERMS = [1, 10, 20, None]  # None = whole-life
INTEREST_RATES = [0.01, 0.03, 0.06]
ATOL = 1e-8  # numerical tolerance


@pytest.fixture(scope="module")
def r_env(r_lifecontingencies):
    """Set up the R environment with soa08 table."""
    lc, ro = r_lifecontingencies
    ro.r('library(lifecontingencies)')
    ro.r('data("soa08")')
    ro.r('soa08Act <- new("actuarialtable", x=soa08@x, lx=soa08@lx, interest=0.06)')
    return lc, ro


def r_at(ro, i: float):
    """Create R actuarialtable at interest i."""
    ro.r(f'at_test <- new("actuarialtable", x=soa08@x, lx=soa08@lx, interest={i})')
    return ro.r("at_test")


def py_at(soa_ilt, i: float):
    from pylifecontingencies import ActuarialTable
    return ActuarialTable(soa_ilt, interest=i)


# ------------------------------------------------------------------ #
# axn                                                                  #
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("i", INTEREST_RATES)
@pytest.mark.parametrize("x", AGES)
@pytest.mark.parametrize("n", TERMS)
def test_axn_parity(r_env, soa_ilt, x, n, i):
    lc, ro = r_env
    r_at_obj = r_at(ro, i)
    at = py_at(soa_ilt, i)

    if n is None:
        r_val = float(ro.r(f'axn(at_test, x={x})')[0])
        py_val = at.axn(x=x)
    else:
        if x + n > 99:
            pytest.skip("term exceeds table")
        r_val = float(ro.r(f'axn(at_test, x={x}, n={n})')[0])
        py_val = at.axn(x=x, n=n)

    assert math.isclose(py_val, r_val, abs_tol=ATOL), (
        f"axn(x={x}, n={n}, i={i}): Python={py_val:.10f}, R={r_val:.10f}"
    )


# ------------------------------------------------------------------ #
# Axn                                                                  #
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("i", INTEREST_RATES)
@pytest.mark.parametrize("x", AGES)
@pytest.mark.parametrize("n", TERMS)
def test_Axn_parity(r_env, soa_ilt, x, n, i):
    lc, ro = r_env
    r_at_obj = r_at(ro, i)
    at = py_at(soa_ilt, i)

    if n is None:
        r_val = float(ro.r(f'Axn(at_test, x={x})')[0])
        py_val = at.Axn(x=x)
    else:
        if x + n > 99:
            pytest.skip("term exceeds table")
        r_val = float(ro.r(f'Axn(at_test, x={x}, n={n})')[0])
        py_val = at.Axn(x=x, n=n)

    assert math.isclose(py_val, r_val, abs_tol=ATOL), (
        f"Axn(x={x}, n={n}, i={i}): Python={py_val:.10f}, R={r_val:.10f}"
    )


# ------------------------------------------------------------------ #
# Exn                                                                  #
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("i", INTEREST_RATES)
@pytest.mark.parametrize("x", AGES)
@pytest.mark.parametrize("n", [1, 10, 20])
def test_Exn_parity(r_env, soa_ilt, x, n, i):
    lc, ro = r_env
    r_at_obj = r_at(ro, i)
    at = py_at(soa_ilt, i)

    if x + n > 99:
        pytest.skip("term exceeds table")

    r_val = float(ro.r(f'Exn(at_test, x={x}, n={n})')[0])
    py_val = at.Exn(x=x, n=n)

    assert math.isclose(py_val, r_val, abs_tol=ATOL), (
        f"Exn(x={x}, n={n}, i={i}): Python={py_val:.10f}, R={r_val:.10f}"
    )


# ------------------------------------------------------------------ #
# exn (life expectation)                                               #
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("x", AGES)
def test_exn_parity(r_env, soa_ilt, x):
    lc, ro = r_env
    ro.r('at_test <- new("actuarialtable", x=soa08@x, lx=soa08@lx, interest=0.06)')
    at = py_at(soa_ilt, 0.06)

    r_val = float(ro.r(f'exn(at_test, x={x})')[0])
    from pylifecontingencies import exn
    py_val = exn(at, x=x)

    assert math.isclose(py_val, r_val, abs_tol=ATOL), (
        f"exn(x={x}): Python={py_val:.10f}, R={r_val:.10f}"
    )
