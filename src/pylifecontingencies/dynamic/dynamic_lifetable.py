"""
DynamicLifeTable: build a LifeTable (or a collection of them for stochastic
analysis) from an external mortality-rate forecast supplied as a DataFrame.

This is the entry point for users who bring their own model predictions —
from a neural net, gradient-boosted model, external R/Julia model, or any
other source — and want to plug them into the pylifecontingencies actuarial
machinery.

Supported inputs
----------------
- ``pd.DataFrame`` (ages × years) of central death rates m_x or mortality
  rates q_x — for a **single forecast path**.
- A list of such DataFrames, or a 3-D numpy array (n_samples × n_ages ×
  n_years) — for **stochastic / scenario analysis**.

Extraction modes
----------------
- **Cohort** (``birth_year`` supplied): for age x use forecast year
  ``birth_year + x``. Ages where the target year falls outside the
  forecast range are clamped to the nearest available year.
- **Period** (``period_year`` supplied): all ages use the same calendar
  year column.
"""

from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
import pandas as pd

from ..lifetable import LifeTable
from ..fractional import FractionalAge


# ------------------------------------------------------------------ #
# Private helpers                                                      #
# ------------------------------------------------------------------ #

def _normalise_df(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure integer index (ages) and integer columns (years)."""
    df = df.copy()
    df.index = np.array(df.index, dtype=int)
    df.columns = np.array(df.columns, dtype=int)
    return df


def _extract_qx_vector(
    df: pd.DataFrame,
    *,
    birth_year: int | None,
    period_year: int | None,
    input_type: str,
    fractional: FractionalAge,
) -> tuple[np.ndarray, int]:
    """
    Extract a 1-D qx vector from a (ages × years) forecast DataFrame.

    Returns
    -------
    qx : np.ndarray
    x_min : int
    """
    if (birth_year is None) == (period_year is None):
        raise ValueError("Provide exactly one of birth_year or period_year")

    df = _normalise_df(df)
    ages = np.array(df.index, dtype=int)
    year_min = int(df.columns.min())
    year_max = int(df.columns.max())

    # Warn if cohort diagonal requires clamping
    if birth_year is not None:
        target_years = birth_year + ages
        n_clamped = int(np.sum((target_years < year_min) | (target_years > year_max)))
        if n_clamped > 0:
            warnings.warn(
                f"{n_clamped} of {len(ages)} ages fall outside the forecast range "
                f"[{year_min}, {year_max}] for birth_year={birth_year}. "
                "Clamping to the nearest available year.",
                stacklevel=4,
            )

    raw_values = np.empty(len(ages))
    for i, age in enumerate(ages):
        if period_year is not None:
            year = int(np.clip(period_year, year_min, year_max))
        else:
            year = int(np.clip(birth_year + age, year_min, year_max))
        raw_values[i] = float(df.loc[age, year])

    # Convert to qx
    if input_type == "qx":
        qx = np.clip(raw_values, 0.0, 1.0)
    elif input_type == "mx":
        qx = raw_values / (1.0 + 0.5 * raw_values)
        qx = np.clip(qx, 0.0, 1.0)
    elif input_type == "log_mx":
        mx = np.exp(raw_values)
        qx = mx / (1.0 + 0.5 * mx)
        qx = np.clip(qx, 0.0, 1.0)
    else:
        raise ValueError(f"input_type must be 'qx', 'mx', or 'log_mx', got {input_type!r}")

    return qx, int(ages[0])


def _build_lifetable(
    df: pd.DataFrame,
    *,
    birth_year: int | None,
    period_year: int | None,
    input_type: str,
    fractional: FractionalAge,
    name: str = "",
) -> LifeTable:
    qx, x_min = _extract_qx_vector(
        df,
        birth_year=birth_year,
        period_year=period_year,
        input_type=input_type,
        fractional=fractional,
    )
    return LifeTable.from_qx(qx, x_min=x_min, name=name, fractional=fractional)


# ------------------------------------------------------------------ #
# Public class                                                         #
# ------------------------------------------------------------------ #

class DynamicLifeTable:
    """
    A life table (or collection of life tables) built from an external
    mortality-rate forecast.

    Parameters
    ----------
    tables : list[LifeTable]
        One LifeTable per scenario. Single-path tables have ``len(tables) == 1``.
    is_stochastic : bool
        True when multiple scenarios are stored.

    See Also
    --------
    DynamicActuarialTable : wraps this with an interest rate and computes EPVs.
    """

    def __init__(self, tables: list[LifeTable], is_stochastic: bool) -> None:
        if not tables:
            raise ValueError("tables must be a non-empty list")
        self._tables = tables
        self.is_stochastic = is_stochastic

    # ------------------------------------------------------------------ #
    # Constructors                                                         #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_forecast_mx(
        cls,
        df_mx: pd.DataFrame,
        *,
        birth_year: int | None = None,
        period_year: int | None = None,
        fractional: FractionalAge = FractionalAge.UDD,
        name: str = "",
    ) -> DynamicLifeTable:
        """
        Build a single-path DynamicLifeTable from central death rates m_x.

        Parameters
        ----------
        df_mx : pd.DataFrame
            Ages as index (int), calendar years as columns (int),
            values are central death rates m_{x,t}.
        birth_year : int or None
            Extract cohort diagonal (age x → year birth_year + x).
        period_year : int or None
            Extract a period column (all ages from the same year).
        fractional : FractionalAge
            Fractional-age assumption for the resulting LifeTable.
        name : str
            Optional label for the LifeTable.

        Examples
        --------
        ::

            dlt = DynamicLifeTable.from_forecast_mx(df_mx, birth_year=1985)
            dat = DynamicActuarialTable(dlt, i=0.03)
            dat.axn(x=40)
        """
        lt = _build_lifetable(
            df_mx,
            birth_year=birth_year,
            period_year=period_year,
            input_type="mx",
            fractional=fractional,
            name=name or _auto_name(birth_year, period_year),
        )
        return cls([lt], is_stochastic=False)

    @classmethod
    def from_forecast_qx(
        cls,
        df_qx: pd.DataFrame,
        *,
        birth_year: int | None = None,
        period_year: int | None = None,
        fractional: FractionalAge = FractionalAge.UDD,
        name: str = "",
    ) -> DynamicLifeTable:
        """
        Build a single-path DynamicLifeTable from mortality rates q_x.

        Parameters
        ----------
        df_qx : pd.DataFrame
            Ages as index (int), calendar years as columns (int),
            values are annual mortality rates q_{x,t}.
        """
        lt = _build_lifetable(
            df_qx,
            birth_year=birth_year,
            period_year=period_year,
            input_type="qx",
            fractional=fractional,
            name=name or _auto_name(birth_year, period_year),
        )
        return cls([lt], is_stochastic=False)

    @classmethod
    def from_forecast_log_mx(
        cls,
        df_log_mx: pd.DataFrame,
        *,
        birth_year: int | None = None,
        period_year: int | None = None,
        fractional: FractionalAge = FractionalAge.UDD,
        name: str = "",
    ) -> DynamicLifeTable:
        """
        Build a single-path DynamicLifeTable from log central death rates log(m_x).

        This is the natural output of Lee-Carter and similar log-linear models.
        """
        lt = _build_lifetable(
            df_log_mx,
            birth_year=birth_year,
            period_year=period_year,
            input_type="log_mx",
            fractional=fractional,
            name=name or _auto_name(birth_year, period_year),
        )
        return cls([lt], is_stochastic=False)

    @classmethod
    def from_scenarios(
        cls,
        scenarios: Sequence[pd.DataFrame],
        *,
        birth_year: int | None = None,
        period_year: int | None = None,
        input_type: str = "mx",
        fractional: FractionalAge = FractionalAge.UDD,
    ) -> DynamicLifeTable:
        """
        Build a stochastic DynamicLifeTable from multiple forecast paths.

        Parameters
        ----------
        scenarios : sequence of pd.DataFrame
            Each DataFrame is a (ages × years) forecast surface. All must
            share the same index (ages) and columns (years).
        input_type : {'mx', 'qx', 'log_mx'}
            What the DataFrame values represent.

        Examples
        --------
        ::

            # 500 bootstrap paths
            dlt = DynamicLifeTable.from_scenarios(list_of_dfs, birth_year=1985)
            dat = DynamicActuarialTable(dlt, i=0.03)
            result = dat.axn(x=40)   # StochasticResult
            print(result.mean, result.std, result.ci(0.95))
        """
        tables = [
            _build_lifetable(
                df,
                birth_year=birth_year,
                period_year=period_year,
                input_type=input_type,
                fractional=fractional,
                name=f"scenario_{i}",
            )
            for i, df in enumerate(scenarios)
        ]
        return cls(tables, is_stochastic=True)

    @classmethod
    def from_scenarios_array(
        cls,
        arr: np.ndarray,
        *,
        ages: np.ndarray | list[int],
        years: np.ndarray | list[int],
        birth_year: int | None = None,
        period_year: int | None = None,
        input_type: str = "mx",
        fractional: FractionalAge = FractionalAge.UDD,
    ) -> DynamicLifeTable:
        """
        Build a stochastic DynamicLifeTable from a 3-D numpy array.

        Parameters
        ----------
        arr : np.ndarray, shape (n_samples, n_ages, n_years)
            Mortality-rate cube (central rates, raw rates, or log rates).
        ages : array-like of int
            Age labels for axis 1.
        years : array-like of int
            Year labels for axis 2.
        input_type : {'mx', 'qx', 'log_mx'}

        Examples
        --------
        ::

            # arr shape: (500, 80, 50)  — 500 scenarios, ages 20-99, years 2025-2074
            dlt = DynamicLifeTable.from_scenarios_array(
                arr, ages=range(20, 100), years=range(2025, 2075), birth_year=1985
            )
        """
        arr = np.asarray(arr, dtype=float)
        if arr.ndim != 3:
            raise ValueError(f"arr must be 3-D (n_samples, n_ages, n_years), got shape {arr.shape}")

        ages_arr = np.asarray(ages, dtype=int)
        years_arr = np.asarray(years, dtype=int)

        scenarios = [
            pd.DataFrame(arr[i], index=ages_arr, columns=years_arr)
            for i in range(arr.shape[0])
        ]
        return cls.from_scenarios(
            scenarios,
            birth_year=birth_year,
            period_year=period_year,
            input_type=input_type,
            fractional=fractional,
        )

    # ------------------------------------------------------------------ #
    # Properties                                                           #
    # ------------------------------------------------------------------ #

    @property
    def tables(self) -> list[LifeTable]:
        return self._tables

    @property
    def n_scenarios(self) -> int:
        return len(self._tables)

    @property
    def lifetable(self) -> LifeTable:
        """The single LifeTable — raises if stochastic."""
        if self.is_stochastic:
            raise ValueError(
                "This DynamicLifeTable has multiple scenarios. "
                "Access individual tables via .tables[i] or use DynamicActuarialTable."
            )
        return self._tables[0]

    def __repr__(self) -> str:
        lt = self._tables[0]
        tag = f"stochastic, n={self.n_scenarios}" if self.is_stochastic else "single-path"
        return (
            f"DynamicLifeTable({tag}, ages={lt.x_min}–{lt.omega - 1}, "
            f"fractional={lt.fractional.value})"
        )


def _auto_name(birth_year: int | None, period_year: int | None) -> str:
    if birth_year is not None:
        return f"cohort_{birth_year}"
    if period_year is not None:
        return f"period_{period_year}"
    return ""
