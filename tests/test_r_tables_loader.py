"""
Tests for bundled parquet tables converted from the R lifecontingencies package.
"""

from __future__ import annotations

import pytest

from pylifecontingencies import LifeTable, list_columns, list_tables, load_table


_MULTI_COLUMN_CASES = [
    ("demoUsa", "USSS2007M", "lx"),
    ("demoUk", "AM92", "lx"),
    ("demoFrance", "TH00_02", "lx"),
    ("demoIta", "SIM92", "lx"),
    ("demoGermany", "qxMale", "qx"),
    ("demoJapan", "JP8587M", "qx"),
    ("demoChina", "CL1", "qx"),
    ("demoCanada", "up94M", "qx"),
]


@pytest.mark.parametrize("name,_column,_kind", _MULTI_COLUMN_CASES)
def test_multi_column_tables_are_listed(name, _column, _kind):
    assert name in list_tables()


@pytest.mark.parametrize("name,column,_kind", _MULTI_COLUMN_CASES)
def test_list_columns_contains_expected_column(name, column, _kind):
    cols = list_columns(name)
    assert column in cols
    assert len(cols) >= 2


@pytest.mark.parametrize("name,column,kind", _MULTI_COLUMN_CASES)
def test_load_table_with_column_returns_lifetable(name, column, kind):
    lt = load_table(name, column=column)

    assert isinstance(lt, LifeTable)
    assert lt.name == f"{name}/{column}"
    assert lt.x_min >= 0

    first_qx = lt.nqx(lt.x_min, 1)
    assert 0.0 <= first_qx <= 1.0

    if kind == "lx":
        assert lt.lx(lt.x_min) > 1.0


@pytest.mark.parametrize("name,_column,_kind", _MULTI_COLUMN_CASES)
def test_load_table_requires_column_for_multi_column_parquet(name, _column, _kind):
    with pytest.raises(ValueError, match="multiple data columns"):
        load_table(name)


@pytest.mark.parametrize("name,_column,_kind", _MULTI_COLUMN_CASES)
def test_load_table_rejects_unknown_column(name, _column, _kind):
    with pytest.raises(ValueError, match="not found"):
        load_table(name, column="__does_not_exist__")


def test_single_column_parquet_does_not_require_column():
    lt = load_table("soa08")
    assert isinstance(lt, LifeTable)
    assert lt.name == "soa08/lx"


def test_list_columns_for_single_column_parquet():
    assert list_columns("soa08") == ["lx"]
