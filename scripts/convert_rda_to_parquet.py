#!/usr/bin/env python3
"""
Convert R lifecontingencies life tables (.rda) to parquet files for bundling.

Requires:
  - R installed (https://www.r-project.org)
  - lifecontingencies CRAN package: Rscript -e "install.packages('lifecontingencies')"
  - rpy2:    pip install rpy2
  - pyarrow: pip install pyarrow

Usage:
  python scripts/convert_rda_to_parquet.py
  python scripts/convert_rda_to_parquet.py --tables soa08 AM92Lt AF92Lt
  python scripts/convert_rda_to_parquet.py --out-dir /custom/output/path

Table types in lifecontingencies
---------------------------------
S4 lifetable  (soa08, AM92Lt, AF92Lt, soaLt)
  Slots: @x (ages), @lx (survivors)
  Saved as: age + lx columns

data.frame with lx columns  (demoUsa, demoFrance, demoIta, demoUk)
  First column = age; remaining columns = lx values (~100 000 radix)
  Saved as-is; load_table(..., column=<col>) selects a sub-table.

data.frame with qx columns  (demoGermany, demoJapan, demoChina, demoCanada)
  First column = age; remaining columns = annual mortality rates
  Saved as-is; column values < 1 → treated as qx by load_table.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_TABLES = [
    "soa08",
    "AM92Lt",
    "AF92Lt",
    "demoUsa",
    "demoUk",   # note: lowercase 'k' — that's the R object name
    "demoIta",
    "demoFrance",
    "demoGermany",
    "demoJapan",
    "demoChina",
    "demoCanada",
]

DEFAULT_OUT_DIR = (
    Path(__file__).parent.parent / "src" / "pylifecontingencies" / "data"
)

# Canonical age-column names used by the various data.frames in lifecontingencies
_AGE_COL_CANDIDATES = {"age", "Age", "x", "X"}


def _load_r_object(ro, r_name: str):
    """Load an R dataset by name and return the rpy2 object."""
    try:
        ro.r(f'data("{r_name}")')
    except Exception:
        ro.r(f'data("{r_name}", package="lifecontingencies")')
    return ro.r(r_name)


def _convert_s4_lifetable(ro, r_name: str, out_dir: Path) -> None:
    """Handle S4 lifetable objects (soa08, AM92Lt, AF92Lt …)."""
    import pandas as pd

    _load_r_object(ro, r_name)
    x  = [int(v)   for v in ro.r(f"{r_name}@x")]
    lx = [float(v) for v in ro.r(f"{r_name}@lx")]
    df = pd.DataFrame({"age": x, "lx": lx})
    _write(df, r_name, out_dir)


def _convert_dataframe(ro, r_name: str, out_dir: Path) -> None:
    """
    Handle data.frame objects (demoUsa, demoUk, demoIta, demoFrance,
    demoGermany, demoJapan, demoChina, demoCanada).

    Extracts columns manually to avoid rpy2 version-dependent converter APIs.
    Normalises the age column to 'age' and keeps all data columns.
    """
    import pandas as pd

    _load_r_object(ro, r_name)

    r_df = ro.r(r_name)
    col_names = list(r_df.names)
    data = {col: list(r_df.rx2(col)) for col in col_names}
    df = pd.DataFrame(data)

    # Normalise age column name to 'age'
    age_col = next(
        (c for c in df.columns if c in _AGE_COL_CANDIDATES),
        df.columns[0],
    )
    if age_col != "age":
        df = df.rename(columns={age_col: "age"})

    df["age"] = df["age"].astype(int)
    df = df.sort_values("age").reset_index(drop=True)

    _write(df, r_name, out_dir)
    data_cols = [c for c in df.columns if c != "age"]
    print(f"    columns: {data_cols}")


def _write(df, name: str, out_dir: Path) -> None:
    import pandas as pd
    out_path = out_dir / f"{name}.parquet"
    df.to_parquet(out_path, index=False)
    print(f"  Written: {out_path} ({len(df)} rows)")


def convert_table(ro, r_name: str, out_dir: Path) -> None:
    """Dispatch to the right converter based on the R object's class."""
    _load_r_object(ro, r_name)
    r_class = str(ro.r(f'class({r_name})[1]')[0])

    if r_class == "lifetable":
        _convert_s4_lifetable(ro, r_name, out_dir)
    elif r_class == "data.frame":
        _convert_dataframe(ro, r_name, out_dir)
    else:
        raise ValueError(f"Unsupported R class '{r_class}' for table '{r_name}'")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tables",
        nargs="+",
        default=DEFAULT_TABLES,
        help="List of R table names to convert",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory for parquet files",
    )
    args = parser.parse_args()

    try:
        import rpy2.robjects as ro
        from rpy2.robjects.packages import importr
        from rpy2.robjects import pandas2ri  # noqa: F401 — trigger import check
    except ImportError:
        print("ERROR: rpy2 is required. Install with: pip install rpy2", file=sys.stderr)
        sys.exit(1)

    try:
        importr("lifecontingencies")
    except Exception as e:
        print(f"ERROR: Could not load R lifecontingencies: {e}", file=sys.stderr)
        print(
            "Install it in R with: install.packages('lifecontingencies')",
            file=sys.stderr,
        )
        sys.exit(1)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {args.out_dir}")
    print(f"Converting {len(args.tables)} tables...\n")

    failed = []
    for name in args.tables:
        try:
            convert_table(ro, name, args.out_dir)
        except Exception as e:
            print(f"  FAILED {name}: {e}")
            failed.append(name)

    print()
    if failed:
        print(f"Failed tables: {failed}", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"Done. {len(args.tables)} tables converted.")


if __name__ == "__main__":
    main()
