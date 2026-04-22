"""
rpy2-backed parity tests for multi-life functions against R lifecontingencies.

These tests are skipped automatically when rpy2 or R is not installed.
To run:  pytest tests/test_multilife_vs_r.py -v -s

The grid covers:
  - SOA 08 table (same for both lives)
  - ages:   (x, y) pairs from [30, 40, 50, 60]
  - terms:  [5, 10, 20, whole-life]
  - status: ["joint", "last"]
  - i:      from actuarialtable (0.06)
"""

import math
import pytest

pytestmark = pytest.mark.rpy2

AGES_XY = [(30, 40), (40, 50), (50, 60), (40, 40)]
TERMS = [5, 10, 20, None]  # None = whole-life
STATUSES = ["joint", "last"]
ATOL = 1e-6  # tolerance (R uses summation-based approach too)


@pytest.fixture(scope="module")
def r_env(r_lifecontingencies):
    """Set up the R environment with soa08 table."""
    lc, ro = r_lifecontingencies
    ro.r('library(lifecontingencies)')
    ro.r('data("soa08")')
    ro.r('soa08Act <- new("actuarialtable", x=soa08@x, lx=soa08@lx, interest=0.06)')
    return lc, ro


@pytest.fixture(scope="module")
def py_at(soa_ilt):
    from pylifecontingencies import ActuarialTable
    return ActuarialTable(soa_ilt, interest=0.06)


# ------------------------------------------------------------------ #
# pxyt parity                                                          #
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("status", STATUSES)
@pytest.mark.parametrize("xy", AGES_XY)
@pytest.mark.parametrize("t", [5, 10, 20])
def test_pxyt_parity(r_env, soa_ilt, xy, t, status):
    lc, ro = r_env
    x, y = xy

    if x + t > 99 or y + t > 99:
        pytest.skip("term exceeds table")

    r_val = float(ro.r(
        f'pxyt(soa08Act, soa08Act, x={x}, y={y}, t={t}, status="{status}")'
    )[0])

    from pylifecontingencies.multilife import pxyt
    py_val = pxyt(soa_ilt, soa_ilt, x, y, t, status)

    assert math.isclose(py_val, r_val, abs_tol=ATOL), (
        f"pxyt(x={x}, y={y}, t={t}, status={status}): Python={py_val:.10f}, R={r_val:.10f}"
    )


# ------------------------------------------------------------------ #
# axyn parity                                                          #
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("status", STATUSES)
@pytest.mark.parametrize("xy", AGES_XY)
@pytest.mark.parametrize("n", TERMS)
def test_axyn_parity(r_env, soa_ilt, xy, n, status):
    lc, ro = r_env
    x, y = xy
    from pylifecontingencies import ActuarialTable
    at = ActuarialTable(soa_ilt, interest=0.06)

    if n is not None and (x + n > 99 or y + n > 99):
        pytest.skip("term exceeds table")

    if n is None:
        r_cmd = f'axyn(soa08Act, soa08Act, x={x}, y={y}, status="{status}")'
    else:
        r_cmd = f'axyn(soa08Act, soa08Act, x={x}, y={y}, n={n}, status="{status}")'

    r_val = float(ro.r(r_cmd)[0])

    from pylifecontingencies.multilife import axyn
    py_val = axyn(at, at, x, y, n=n, status=status)

    assert math.isclose(py_val, r_val, abs_tol=ATOL), (
        f"axyn(x={x}, y={y}, n={n}, status={status}): Python={py_val:.10f}, R={r_val:.10f}"
    )


# ------------------------------------------------------------------ #
# Axyn parity                                                          #
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("status", STATUSES)
@pytest.mark.parametrize("xy", AGES_XY)
@pytest.mark.parametrize("n", TERMS)
def test_Axyn_parity(r_env, soa_ilt, xy, n, status):
    lc, ro = r_env
    x, y = xy
    from pylifecontingencies import ActuarialTable
    at = ActuarialTable(soa_ilt, interest=0.06)

    if n is not None and (x + n > 99 or y + n > 99):
        pytest.skip("term exceeds table")

    if n is None:
        r_cmd = f'Axyn(soa08Act, soa08Act, x={x}, y={y}, status="{status}")'
    else:
        r_cmd = f'Axyn(soa08Act, soa08Act, x={x}, y={y}, n={n}, status="{status}")'

    r_val = float(ro.r(r_cmd)[0])

    from pylifecontingencies.multilife import Axyn
    py_val = Axyn(at, at, x, y, n=n, status=status)

    assert math.isclose(py_val, r_val, abs_tol=ATOL), (
        f"Axyn(x={x}, y={y}, n={n}, status={status}): Python={py_val:.10f}, R={r_val:.10f}"
    )
