"""Smoke tests for all bundled life tables in tabuas_csv/."""

import pytest
from pylifecontingencies.data import list_tables, load_table


# Only test CSV-based tables (root + tabuas_csv); skip parquet to avoid
# requiring R-generated files that may not be present in CI.
_CSV_TABLES = [
    name for name in list_tables()
    if not (
        name.startswith("demo")
        or name in {"soa08", "AM92Lt", "AF92Lt"}
    )
]


@pytest.mark.parametrize("name", _CSV_TABLES)
def test_load_table_basic(name):
    t = load_table(name)
    assert t.omega > t.x_min, f"{name}: omega={t.omega} not > x_min={t.x_min}"
    qx = t.nqx(t.x_min, 1)
    assert 0.0 <= qx <= 1.0, f"{name}: qx={qx} out of [0, 1]"


def test_list_tables_includes_tabuas():
    names = list_tables()
    assert "soa_ilt" in names
    assert "at_2000_female" in names
    assert "ibge_2020_homens" in names
    assert "br_emssb_2021_m" in names
    assert len(names) >= 100
