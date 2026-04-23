"""
Tests for the BR-EMS bundled life tables.

Validation strategy:
  1. load_table(name) succeeds and returns a LifeTable.
  2. The table is loadable via list_tables().
  3. qx(0) matches the printed value from the official SUSEP spreadsheet.
  4. Curtate life expectancy e_0 (rounded to 1 decimal) matches the printed
     ex column in the spreadsheet (available for 2015 and 2021 tables only).
  5. Monotone age range: ages start at 0 and are consecutive.
  6. qx values are in (0, 1] and the terminal age has qx = 1.0.
"""

from __future__ import annotations

import pytest

from pylifecontingencies import load_table, list_tables, exn, ex_complete


# ---------------------------------------------------------------------------
# Expected values from the official SUSEP spreadsheet (row 7, i.e. age 0).
# ---------------------------------------------------------------------------

# (table_name, expected_qx_age0, expected_ex_age0_or_None, use_complete_ex)
# 2021 tables: spreadsheet prints curtate e_x  → use exn()
# 2015 tables: spreadsheet prints complete ê_x → use ex_complete()
# 2010 tables: no ex column                    → skip
_TABLE_SPECS: list[tuple[str, float, float | None, bool]] = [
    ("br_emssb_2021_m", 0.000352673,  81.1, False),
    ("br_emssb_2021_f", 0.0002926024, 86.5, False),
    ("br_emsmt_2021_m", 0.0003707975, 78.3, False),
    ("br_emsmt_2021_f", 0.0003545253, 83.2, False),
    ("br_emssb_2015_m", 0.0003372,    82.4, True),
    ("br_emssb_2015_f", 0.0003438,    87.8, True),
    ("br_emsmt_2015_m", 0.0003911,    79.9, True),
    ("br_emsmt_2015_f", 0.0004151,    84.7, True),
    ("br_emssb_2010_m", 0.002,        None, False),
    ("br_emssb_2010_f", 0.00038,      None, False),
    ("br_emsmt_2010_m", 0.00274,      None, False),
    ("br_emsmt_2010_f", 0.00128,      None, False),
]

_ALL_NAMES = [spec[0] for spec in _TABLE_SPECS]


@pytest.mark.parametrize("name", _ALL_NAMES)
def test_load_succeeds(name):
    lt = load_table(name)
    assert lt is not None


@pytest.mark.parametrize("name", _ALL_NAMES)
def test_in_list_tables(name):
    assert name in list_tables()


@pytest.mark.parametrize("name,expected_qx0,_ex,_c", _TABLE_SPECS)
def test_qx_age0(name, expected_qx0, _ex, _c):
    lt = load_table(name)
    assert lt.nqx(0, 1) == pytest.approx(expected_qx0, rel=1e-5)


@pytest.mark.parametrize("name,_qx,expected_ex,complete", _TABLE_SPECS)
def test_ex_age0(name, _qx, expected_ex, complete):
    if expected_ex is None:
        pytest.skip("no printed ex for 2010 tables")
    lt = load_table(name)
    fn = ex_complete if complete else exn
    computed = round(fn(lt, 0), 1)
    assert computed == expected_ex, (
        f"{name}: computed e_0={computed}, spreadsheet e_0={expected_ex}"
    )


@pytest.mark.parametrize("name", _ALL_NAMES)
def test_ages_start_at_zero_and_consecutive(name):
    lt = load_table(name)
    assert lt.x_min == 0
    ages = lt.ages
    assert int(ages[0]) == 0
    for i in range(1, len(ages)):
        assert int(ages[i]) == int(ages[i - 1]) + 1


@pytest.mark.parametrize("name", _ALL_NAMES)
def test_qx_in_range(name):
    lt = load_table(name)
    for x in lt.ages[:-1]:  # exclude terminal age
        q = lt.nqx(int(x), 1)
        assert 0 < q <= 1, f"{name}: qx({x}) = {q} out of range"


@pytest.mark.parametrize("name", _ALL_NAMES)
def test_terminal_qx_is_one(name):
    lt = load_table(name)
    # The second-to-last age should have qx = 1.0 (sets lx at omega to 0)
    omega_minus_1 = lt.omega - 1
    assert lt.nqx(omega_minus_1, 1) == pytest.approx(1.0, abs=1e-9), (
        f"{name}: qx at omega-1={omega_minus_1} is not 1.0"
    )
