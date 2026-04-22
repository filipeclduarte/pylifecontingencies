"""
SOA XTbML table loader via the optional `pymort` dependency.

Install: pip install "pylifecontingencies[soa]"
"""

from __future__ import annotations


def load_soa_xtbml(table_id: int | str) -> "LifeTable":
    """
    Load an SOA table by ID using pymort.

    Parameters
    ----------
    table_id : int or str
        SOA mortality table ID (e.g. 5 for the 1958 CSO, 1 for the 1941 CSO).
        See https://mort.soa.org for the full catalogue.

    Returns
    -------
    LifeTable
    """
    try:
        import pymort
    except ImportError:
        raise ImportError(
            "pymort is required for SOA XTbML table loading. "
            "Install it with: pip install 'pylifecontingencies[soa]'"
        ) from None

    from ..lifetable import LifeTable
    import numpy as np

    xml_table = pymort.MortXML(table_id)
    # pymort returns Tables list; take the first select-or-ultimate table
    tbl = xml_table.Tables[0]
    df = tbl.Values
    # Column names vary by table; normalise to age/qx
    df = df.reset_index()
    if "Age" in df.columns:
        df = df.rename(columns={"Age": "age"})
    # Select ultimate q column (name varies); fall back to first numeric column
    q_col = next(
        (c for c in df.columns if c.lower() in ("qx", "ultimate", "q")), None
    )
    if q_col is None:
        numeric_cols = df.select_dtypes("number").columns.tolist()
        age_col = [c for c in numeric_cols if "age" in c.lower()]
        q_col = next(c for c in numeric_cols if c not in age_col)

    return LifeTable.from_dataframe(df, age_col="age", qx_col=q_col, name=str(table_id))
