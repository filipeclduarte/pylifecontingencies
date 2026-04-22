#!/usr/bin/env python3
"""
Convert R lifecontingencies life tables (.rda) to parquet files for bundling.

Requires:
  - R installed (https://www.r-project.org)
  - lifecontingencies CRAN package: Rscript -e "install.packages('lifecontingencies')"
  - rpy2: pip install rpy2

Usage:
  python scripts/convert_rda_to_parquet.py
  python scripts/convert_rda_to_parquet.py --tables soa08 AM92Lt AF92Lt
  python scripts/convert_rda_to_parquet.py --out-dir /custom/output/path
"""

import argparse
import sys
from pathlib import Path

DEFAULT_TABLES = [
    "soa08",
    "AM92Lt",
    "AF92Lt",
    "demoUsa",
    "demoUK",
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


def convert_lifetable(ro, lc, r_name: str, out_dir: Path) -> None:
    """Convert a single R lifetable/actuarialtable object to parquet."""
    import pandas as pd

    try:
        ro.r(f'data("{r_name}")')
    except Exception:
        ro.r(f'data("{r_name}", package="lifecontingencies")')

    obj = ro.r(r_name)

    # Extract x (ages) and lx from the S4 object
    x = list(ro.r(f'{r_name}@x'))
    lx = list(ro.r(f'{r_name}@lx'))

    df = pd.DataFrame({"age": [int(v) for v in x], "lx": [float(v) for v in lx]})

    out_path = out_dir / f"{r_name}.parquet"
    df.to_parquet(out_path, index=False)
    print(f"  Written: {out_path} ({len(df)} rows)")


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
    except ImportError:
        print("ERROR: rpy2 is required. Install with: pip install rpy2", file=sys.stderr)
        sys.exit(1)

    try:
        lc = importr("lifecontingencies")
    except Exception as e:
        print(f"ERROR: Could not load R lifecontingencies: {e}", file=sys.stderr)
        print(
            "Install it in R with: install.packages('lifecontingencies')",
            file=sys.stderr,
        )
        sys.exit(1)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {args.out_dir}")
    print(f"Converting {len(args.tables)} tables...")

    failed = []
    for name in args.tables:
        try:
            convert_lifetable(ro, lc, name, args.out_dir)
        except Exception as e:
            print(f"  FAILED {name}: {e}")
            failed.append(name)

    if failed:
        print(f"\nFailed tables: {failed}", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"\nDone. {len(args.tables)} tables converted.")


if __name__ == "__main__":
    main()
