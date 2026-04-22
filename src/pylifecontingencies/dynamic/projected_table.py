"""
ProjectedLifeTable: the single, model-agnostic entry point for building
life tables from any mortality-rate forecast.

Users can bring forecasts from **any** source — Lee-Carter, CBD, neural nets,
gradient boosting, external R/Julia models, or even hand-crafted assumptions —
and immediately plug them into the pylifecontingencies actuarial machinery.

Supported inputs
----------------
- **Central forecast** as a DataFrame (ages × years) of m_x, q_x, or log(m_x).
- **Central + prediction intervals**: pass ``lower`` and ``upper`` DataFrames
  alongside the central forecast.
- **Multiple scenarios**: pass a list of DataFrames for full stochastic analysis.
- **Model forecast objects**: LeeCarterForecast, CBDForecast, or any object
  with a ``.qx(year)`` method (backward compatibility).

Extraction modes
----------------
- **Cohort** (``birth_year``): for age x, use forecast year ``birth_year + x``.
- **Period** (``period_year``): all ages use the same calendar year.

Extrapolation
-------------
When cohort ages map to years outside the forecast range:
- ``"clamp"`` (default): use the nearest available year.
- ``"constant_slope"``: extrapolate log(m_x) with the slope from the
  last two available years.
- ``"none"``: raise an error.

Examples
--------
::

    # Central forecast only
    plt = ProjectedLifeTable.from_mx(df_mx, birth_year=1985)
    lt = plt.to_life_table()
    at = ActuarialTable(lt, i=0.03)

    # With prediction intervals
    plt = ProjectedLifeTable.from_mx(
        df_mx_central, lower=df_mx_lower, upper=df_mx_upper,
        birth_year=1985,
    )
    lt_lo, lt_central, lt_hi = plt.lower, plt.lifetable, plt.upper

    # Multiple scenarios (stochastic)
    plt = ProjectedLifeTable.from_scenarios([df1, df2, ...], birth_year=1985)
    dat = DynamicActuarialTable(plt, i=0.03)
    result = dat.axn(x=40)   # → StochasticResult

    # From a fitted model (backward compat)
    forecast = LeeCarter().fit(rates).forecast(horizon=50)
    plt = ProjectedLifeTable(forecast, birth_year=1985)
"""

from __future__ import annotations

import warnings
from typing import Literal, Sequence

import numpy as np
import pandas as pd

from ..lifetable import LifeTable
from ..fractional import FractionalAge

# Supported extrapolation modes
ExtrapolationMode = Literal["clamp", "constant_slope", "none"]


class ProjectedLifeTable:
    """
    Model-agnostic life table built from any mortality-rate forecast.

    Stores one or more ``LifeTable`` objects internally:
    - **Single-path**: one table (central forecast).
    - **With prediction intervals**: three tables (lower, central, upper).
    - **Stochastic**: N tables (one per scenario).

    Parameters
    ----------
    forecast : object with ``.qx(year)``, ``.ages``, ``.years_calib``, ``.years_forecast``
        A model forecast object (LeeCarterForecast, CBDForecast, or compatible).
        For raw DataFrame input, use the ``from_mx`` / ``from_qx`` class methods instead.
    birth_year : int or None
        Birth year for cohort extraction.
    period_year : int or None
        Calendar year for period extraction.
    ages : list[int] or None
        Ages to include. Defaults to all available ages.
    fractional : FractionalAge
        Fractional-age assumption for the resulting LifeTable(s).
    extrapolation : str
        ``"clamp"``, ``"constant_slope"``, or ``"none"``.
    """

    def __init__(
        self,
        forecast=None,
        *,
        birth_year: int | None = None,
        period_year: int | None = None,
        ages: list[int] | None = None,
        fractional: FractionalAge = FractionalAge.UDD,
        extrapolation: ExtrapolationMode = "clamp",
        # Internal — used by class methods:
        _tables: list[LifeTable] | None = None,
        _is_stochastic: bool = False,
        _has_pi: bool = False,
    ) -> None:
        if _tables is not None:
            # Internal construction from class methods
            self._tables = _tables
            self._is_stochastic = _is_stochastic
            self._has_pi = _has_pi
            self.birth_year = birth_year
            self.period_year = period_year
            self.fractional = fractional
            self.extrapolation = extrapolation
            return

        # Public construction from a forecast object (backward compat)
        if forecast is None:
            raise TypeError(
                "Provide a forecast object, or use a class method "
                "(from_mx, from_qx, from_scenarios, etc.)"
            )
        if (birth_year is None) == (period_year is None):
            raise ValueError("Provide exactly one of birth_year or period_year")

        self.birth_year = birth_year
        self.period_year = period_year
        self.fractional = fractional
        self.extrapolation = extrapolation
        self._has_pi = False
        self._is_stochastic = False

        # Wrap the forecast object in a _ForecastAdapter
        adapter = _ForecastAdapter(forecast)
        f_ages = np.asarray(ages, dtype=int) if ages is not None else forecast.ages
        lt = self._build_lifetable(adapter, f_ages)
        self._tables = [lt]

    # ================================================================== #
    # Class methods: raw DataFrame input                                   #
    # ================================================================== #

    @classmethod
    def from_mx(
        cls,
        df_mx: pd.DataFrame,
        *,
        lower: pd.DataFrame | None = None,
        upper: pd.DataFrame | None = None,
        birth_year: int | None = None,
        period_year: int | None = None,
        ages: list[int] | None = None,
        fractional: FractionalAge = FractionalAge.UDD,
        extrapolation: ExtrapolationMode = "clamp",
    ) -> ProjectedLifeTable:
        """
        Build from a DataFrame of central death rates m_{x,t}.

        Parameters
        ----------
        df_mx : pd.DataFrame
            Ages as index (int), calendar years as columns (int),
            values are central death rates m_{x,t}.
        lower : pd.DataFrame or None
            Lower prediction interval (same shape as df_mx).
        upper : pd.DataFrame or None
            Upper prediction interval (same shape as df_mx).
        birth_year, period_year, ages, fractional, extrapolation :
            See class docstring.

        Examples
        --------
        ::

            # Central only
            plt = ProjectedLifeTable.from_mx(df_mx, birth_year=1985)

            # With 95% prediction intervals
            plt = ProjectedLifeTable.from_mx(
                df_central, lower=df_lo, upper=df_hi, birth_year=1985
            )
        """
        return cls._from_df(
            df_mx, input_type="mx", lower=lower, upper=upper,
            birth_year=birth_year, period_year=period_year,
            ages=ages, fractional=fractional, extrapolation=extrapolation,
        )

    @classmethod
    def from_qx(
        cls,
        df_qx: pd.DataFrame,
        *,
        lower: pd.DataFrame | None = None,
        upper: pd.DataFrame | None = None,
        birth_year: int | None = None,
        period_year: int | None = None,
        ages: list[int] | None = None,
        fractional: FractionalAge = FractionalAge.UDD,
        extrapolation: ExtrapolationMode = "clamp",
    ) -> ProjectedLifeTable:
        """
        Build from a DataFrame of mortality rates q_{x,t}.

        Parameters
        ----------
        df_qx : pd.DataFrame
            Ages as index (int), calendar years as columns (int),
            values are annual mortality rates q_{x,t}.
        lower, upper : pd.DataFrame or None
            Prediction interval bounds.
        """
        return cls._from_df(
            df_qx, input_type="qx", lower=lower, upper=upper,
            birth_year=birth_year, period_year=period_year,
            ages=ages, fractional=fractional, extrapolation=extrapolation,
        )

    @classmethod
    def from_log_mx(
        cls,
        df_log_mx: pd.DataFrame,
        *,
        lower: pd.DataFrame | None = None,
        upper: pd.DataFrame | None = None,
        birth_year: int | None = None,
        period_year: int | None = None,
        ages: list[int] | None = None,
        fractional: FractionalAge = FractionalAge.UDD,
        extrapolation: ExtrapolationMode = "clamp",
    ) -> ProjectedLifeTable:
        """
        Build from a DataFrame of log(m_{x,t}).

        The natural output of Lee-Carter and similar log-linear models.
        """
        return cls._from_df(
            df_log_mx, input_type="log_mx", lower=lower, upper=upper,
            birth_year=birth_year, period_year=period_year,
            ages=ages, fractional=fractional, extrapolation=extrapolation,
        )

    @classmethod
    def from_scenarios(
        cls,
        scenarios: Sequence[pd.DataFrame],
        *,
        birth_year: int | None = None,
        period_year: int | None = None,
        input_type: str = "mx",
        ages: list[int] | None = None,
        fractional: FractionalAge = FractionalAge.UDD,
        extrapolation: ExtrapolationMode = "clamp",
    ) -> ProjectedLifeTable:
        """
        Build from multiple forecast paths (stochastic analysis).

        Parameters
        ----------
        scenarios : sequence of pd.DataFrame
            Each DataFrame is an (ages × years) forecast surface.
        input_type : {'mx', 'qx', 'log_mx'}
            What the DataFrame values represent.

        Examples
        --------
        ::

            # 500 bootstrap paths
            plt = ProjectedLifeTable.from_scenarios(
                list_of_dfs, birth_year=1985, input_type="mx"
            )
            dat = DynamicActuarialTable(plt, i=0.03)
            result = dat.axn(x=40)   # StochasticResult
        """
        if (birth_year is None) == (period_year is None):
            raise ValueError("Provide exactly one of birth_year or period_year")
        if not scenarios:
            raise ValueError("scenarios must be a non-empty sequence")

        tables = []
        for i, df in enumerate(scenarios):
            adapter = _DataFrameAdapter(df, input_type=input_type)
            f_ages = (
                np.asarray(ages, dtype=int) if ages is not None
                else np.array(df.index, dtype=int)
            )
            instance = cls.__new__(cls)
            instance.birth_year = birth_year
            instance.period_year = period_year
            instance.fractional = fractional
            instance.extrapolation = extrapolation
            lt = instance._build_lifetable(adapter, f_ages, name=f"scenario_{i}")
            tables.append(lt)

        return cls(
            birth_year=birth_year, period_year=period_year,
            fractional=fractional, extrapolation=extrapolation,
            _tables=tables, _is_stochastic=True, _has_pi=False,
        )

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
        extrapolation: ExtrapolationMode = "clamp",
    ) -> ProjectedLifeTable:
        """
        Build from a 3-D numpy array (n_samples × n_ages × n_years).

        Parameters
        ----------
        arr : np.ndarray, shape (n_samples, n_ages, n_years)
        ages : array-like of int
        years : array-like of int
        """
        arr = np.asarray(arr, dtype=float)
        if arr.ndim != 3:
            raise ValueError(
                f"arr must be 3-D (n_samples, n_ages, n_years), got shape {arr.shape}"
            )
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
            ages=list(ages_arr),
            fractional=fractional,
            extrapolation=extrapolation,
        )

    # ================================================================== #
    # Internal: build from DataFrames with optional PI                     #
    # ================================================================== #

    @classmethod
    def _from_df(
        cls,
        df: pd.DataFrame,
        *,
        input_type: str,
        lower: pd.DataFrame | None,
        upper: pd.DataFrame | None,
        birth_year: int | None,
        period_year: int | None,
        ages: list[int] | None,
        fractional: FractionalAge,
        extrapolation: ExtrapolationMode,
    ) -> ProjectedLifeTable:
        """Shared implementation for from_mx / from_qx / from_log_mx."""
        if (birth_year is None) == (period_year is None):
            raise ValueError("Provide exactly one of birth_year or period_year")

        has_pi = (lower is not None) and (upper is not None)
        if (lower is None) != (upper is None):
            raise ValueError("Provide both lower and upper, or neither")

        dfs = [df] if not has_pi else [lower, df, upper]
        labels = ["central"] if not has_pi else ["lower", "central", "upper"]

        tables = []
        for sub_df, label in zip(dfs, labels):
            adapter = _DataFrameAdapter(sub_df, input_type=input_type)
            f_ages = (
                np.asarray(ages, dtype=int) if ages is not None
                else np.array(sub_df.index, dtype=int)
            )
            instance = cls.__new__(cls)
            instance.birth_year = birth_year
            instance.period_year = period_year
            instance.fractional = fractional
            instance.extrapolation = extrapolation
            lt = instance._build_lifetable(adapter, f_ages, name=label)
            tables.append(lt)

        return cls(
            birth_year=birth_year, period_year=period_year,
            fractional=fractional, extrapolation=extrapolation,
            _tables=tables, _is_stochastic=has_pi, _has_pi=has_pi,
        )

    # ================================================================== #
    # Properties                                                           #
    # ================================================================== #

    @property
    def tables(self) -> list[LifeTable]:
        """All underlying LifeTable objects."""
        return self._tables

    @property
    def n_scenarios(self) -> int:
        """Number of scenarios / tables stored."""
        return len(self._tables)

    @property
    def is_stochastic(self) -> bool:
        """True when multiple tables are stored (PI or scenarios)."""
        return self._is_stochastic

    @property
    def has_prediction_interval(self) -> bool:
        """True when lower/central/upper prediction interval tables are stored."""
        return self._has_pi

    @property
    def lifetable(self) -> LifeTable:
        """
        The central (or only) LifeTable.

        For single-path: the sole table.
        For PI: the central table (index 1 of [lower, central, upper]).
        For scenarios: raises ValueError (use ``.tables`` instead).
        """
        if self._has_pi:
            return self._tables[1]
        if self._is_stochastic and not self._has_pi:
            raise ValueError(
                "This ProjectedLifeTable has multiple scenarios. "
                "Use .tables[i] or pass to DynamicActuarialTable."
            )
        return self._tables[0]

    @property
    def lower(self) -> LifeTable | None:
        """Lower prediction interval LifeTable, or None."""
        return self._tables[0] if self._has_pi else None

    @property
    def upper(self) -> LifeTable | None:
        """Upper prediction interval LifeTable, or None."""
        return self._tables[2] if self._has_pi else None

    # ================================================================== #
    # Conversion                                                           #
    # ================================================================== #

    def to_life_table(self, name: str = "") -> LifeTable:
        """
        Return the central LifeTable (backward compatibility).

        Identical to ``.lifetable`` for single-path and PI tables.
        """
        lt = self.lifetable
        if name:
            lt.name = name
        return lt

    # ================================================================== #
    # Core: build a LifeTable from a forecast adapter                      #
    # ================================================================== #

    def _build_lifetable(
        self,
        adapter,
        ages: np.ndarray,
        name: str = "",
    ) -> LifeTable:
        """Build a single LifeTable from a forecast-like adapter."""
        all_years = np.concatenate([adapter.years_calib, adapter.years_forecast])
        year_min = int(all_years.min())
        year_max = int(all_years.max())

        qx_values = []
        for age in ages:
            if self.birth_year is not None:
                target_year = self.birth_year + age
            else:
                target_year = self.period_year

            if target_year < year_min or target_year > year_max:
                qx_val = self._extrapolate_qx(
                    adapter, ages, age, target_year, year_min, year_max
                )
            else:
                qx_val = _lookup_qx(adapter, age, target_year)

            qx_values.append(np.clip(qx_val, 0.0, 1.0))

        qx_arr = np.asarray(qx_values)
        if qx_arr[-1] < 1.0:
            qx_arr[-1] = 1.0

        x_min = int(ages[0])
        label = name or (
            f"cohort_{self.birth_year}" if self.birth_year else f"period_{self.period_year}"
        )
        return LifeTable.from_qx(
            qx_arr, x_min=x_min, name=label, fractional=self.fractional
        )

    def _extrapolate_qx(
        self, adapter, ages, age, target_year, year_min, year_max
    ) -> float:
        """Handle out-of-range years according to self.extrapolation."""
        if self.extrapolation == "none":
            raise ValueError(
                f"Age {age} maps to year {target_year} which is outside the "
                f"forecast range [{year_min}, {year_max}]. "
                "Set extrapolation='clamp' or 'constant_slope'."
            )

        if self.extrapolation == "clamp":
            clamped = int(np.clip(target_year, year_min, year_max))
            return _lookup_qx(adapter, age, clamped)

        if self.extrapolation == "constant_slope":
            return _extrapolate_constant_slope(
                adapter, age, target_year, year_min, year_max
            )

        raise ValueError(
            f"extrapolation must be 'clamp', 'constant_slope', or 'none', "
            f"got {self.extrapolation!r}"
        )

    def __repr__(self) -> str:
        lt = self._tables[0]
        mode = f"birth={self.birth_year}" if self.birth_year else f"period={self.period_year}"
        if self._has_pi:
            tag = "PI: lower/central/upper"
        elif self._is_stochastic:
            tag = f"stochastic, n={self.n_scenarios}"
        else:
            tag = "single-path"
        return f"ProjectedLifeTable({mode}, {tag}, ages={lt.x_min}–{lt.omega - 1})"


# ================================================================== #
# Adapters: make DataFrames and forecast objects look the same          #
# ================================================================== #

class _DataFrameAdapter:
    """Wraps a raw (ages × years) DataFrame to quack like a forecast object."""

    def __init__(self, df: pd.DataFrame, input_type: str = "mx") -> None:
        self._df = df.copy()
        self._df.index = np.array(df.index, dtype=int)
        self._df.columns = np.array(df.columns, dtype=int)
        self._input_type = input_type

        self.ages = np.array(self._df.index, dtype=int)
        self.years_calib = np.array(self._df.columns, dtype=int)
        self.years_forecast = np.array([], dtype=int)

    def qx(self, year: int) -> np.ndarray:
        col = self._df[year].values.astype(float)
        if self._input_type == "qx":
            return np.clip(col, 0.0, 1.0)
        elif self._input_type == "mx":
            qx = col / (1.0 + 0.5 * col)
            return np.clip(qx, 0.0, 1.0)
        elif self._input_type == "log_mx":
            mx = np.exp(col)
            qx = mx / (1.0 + 0.5 * mx)
            return np.clip(qx, 0.0, 1.0)
        raise ValueError(f"Unknown input_type: {self._input_type!r}")

    def mx(self, year: int) -> np.ndarray:
        col = self._df[year].values.astype(float)
        if self._input_type == "mx":
            return col
        elif self._input_type == "log_mx":
            return np.exp(col)
        elif self._input_type == "qx":
            qx = np.clip(col, 0.0, 1.0 - 1e-15)
            return qx / (1.0 - 0.5 * qx)
        raise ValueError(f"Unknown input_type: {self._input_type!r}")


class _ForecastAdapter:
    """
    Thin wrapper around a model forecast object (LeeCarterForecast, CBDForecast).
    Passes through attributes directly.
    """

    def __init__(self, forecast) -> None:
        self._fc = forecast

    @property
    def ages(self) -> np.ndarray:
        return self._fc.ages

    @property
    def years_calib(self) -> np.ndarray:
        return self._fc.years_calib

    @property
    def years_forecast(self) -> np.ndarray:
        return self._fc.years_forecast

    def qx(self, year: int) -> np.ndarray:
        return self._fc.qx(year)

    def mx(self, year: int) -> np.ndarray:
        if hasattr(self._fc, "mx"):
            return self._fc.mx(year)
        qx = self.qx(year)
        qx = np.clip(qx, 0.0, 1.0 - 1e-15)
        return qx / (1.0 - 0.5 * qx)


# ================================================================== #
# Free helpers                                                         #
# ================================================================== #

def _lookup_qx(adapter, age: int, year: int) -> float:
    """Look up q_x for a given age in a given year from an adapter."""
    qx_year = adapter.qx(year)
    age_idx = int(np.searchsorted(adapter.ages, age))
    if age_idx >= len(qx_year):
        return 1.0
    return float(qx_year[age_idx])


def _extrapolate_constant_slope(
    adapter, age: int, target_year: int, year_min: int, year_max: int
) -> float:
    """Extrapolate using constant log-mortality slope."""
    if target_year > year_max:
        y1 = max(year_min, year_max - 1)
        y2 = year_max
    else:
        y1 = year_min
        y2 = min(year_max, year_min + 1)

    qx_y1 = _lookup_qx(adapter, age, y1)
    qx_y2 = _lookup_qx(adapter, age, y2)

    mx_y1 = qx_y1 / (1.0 - 0.5 * qx_y1) if qx_y1 < 1.0 else 100.0
    mx_y2 = qx_y2 / (1.0 - 0.5 * qx_y2) if qx_y2 < 1.0 else 100.0

    log_mx_y1 = np.log(max(mx_y1, 1e-15))
    log_mx_y2 = np.log(max(mx_y2, 1e-15))

    slope = (log_mx_y2 - log_mx_y1) / (y2 - y1) if y2 != y1 else 0.0

    edge_log = log_mx_y2 if target_year > year_max else log_mx_y1
    edge_year = year_max if target_year > year_max else year_min
    dt = target_year - edge_year
    log_mx_ext = edge_log + slope * dt
    mx_ext = np.exp(log_mx_ext)

    qx_ext = mx_ext / (1.0 + 0.5 * mx_ext)
    return float(np.clip(qx_ext, 0.0, 1.0))
