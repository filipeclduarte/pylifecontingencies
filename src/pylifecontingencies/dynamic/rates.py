"""
MortalityRates: a thin wrapper around a 2-D (age × year) DataFrame
of log central death rates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path


class MortalityRates:
    """
    Mortality rate surface: log(m_{x,t}) indexed by (age, year).

    The underlying data is a DataFrame with ages as the index and
    calendar years as columns.

    Parameters
    ----------
    data : pd.DataFrame
        log(m_{x,t}) with integer ages as index and integer years as columns.
    label : str
        Optional description (e.g. "USA Female 1950-2020").
    """

    def __init__(self, data: pd.DataFrame, label: str = "") -> None:
        if not isinstance(data.index, pd.Index) or not isinstance(data.columns, pd.Index):
            raise TypeError("data must be a DataFrame with age index and year columns")
        self.data = data.copy().astype(float)
        self.label = label

    # ------------------------------------------------------------------ #
    # Constructors                                                         #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        label: str = "",
    ) -> MortalityRates:
        """
        Construct from a DataFrame already shaped (ages × years) of log(mx).

        Index should be integer ages; columns should be integer years.
        """
        return cls(df, label=label)

    @classmethod
    def from_mx_dataframe(
        cls,
        df: pd.DataFrame,
        label: str = "",
    ) -> MortalityRates:
        """Construct from a (ages × years) DataFrame of raw m_x values (not log)."""
        log_df = np.log(df.replace(0, np.nan))
        return cls(log_df, label=label)

    @classmethod
    def from_qx_dataframe(
        cls,
        df: pd.DataFrame,
        label: str = "",
    ) -> MortalityRates:
        """Construct from a (ages × years) DataFrame of q_x values."""
        mx_df = df / (1.0 - 0.5 * df)
        return cls.from_mx_dataframe(mx_df, label=label)

    @classmethod
    def from_hmd_file(
        cls,
        path: str | Path,
        *,
        sex: str = "Female",
        label: str = "",
    ) -> MortalityRates:
        """
        Load from an HMD-format Mx (central death rates) flat text file.

        HMD files have columns: Year, Age, Female, Male, Total.
        Download from https://www.mortality.org (requires free account).
        """
        path = Path(path)
        df_raw = pd.read_csv(path, sep=r"\s+", skiprows=2, na_values=".")
        df_raw.columns = df_raw.columns.str.strip()
        df_raw["Age"] = (
            df_raw["Age"].astype(str).str.replace("+", "", regex=False).astype(int)
        )
        df_raw[sex] = pd.to_numeric(df_raw[sex], errors="coerce")
        pivot = df_raw.pivot(index="Age", columns="Year", values=sex).astype(float)
        log_mx = np.log(pivot.replace(0, np.nan))
        return cls(log_mx, label=label or f"HMD {sex}")

    # ------------------------------------------------------------------ #
    # Properties                                                           #
    # ------------------------------------------------------------------ #

    @property
    def ages(self) -> np.ndarray:
        return np.array(self.data.index, dtype=int)

    @property
    def years(self) -> np.ndarray:
        return np.array(self.data.columns, dtype=int)

    @property
    def log_mx(self) -> np.ndarray:
        """log(mx) matrix, shape (n_ages, n_years)."""
        return self.data.values

    @property
    def mx(self) -> np.ndarray:
        """Central death rates m_x, shape (n_ages, n_years)."""
        return np.exp(self.data.values)

    def subset(
        self,
        ages: list[int] | None = None,
        years: list[int] | None = None,
    ) -> MortalityRates:
        """Return a sub-surface restricted to the given ages and/or years."""
        df = self.data
        if ages is not None:
            df = df.loc[ages]
        if years is not None:
            df = df[years]
        return MortalityRates(df, label=self.label)

    # ------------------------------------------------------------------ #
    # Serialisation                                                        #
    # ------------------------------------------------------------------ #

    def to_dataframe(self) -> pd.DataFrame:
        """Return the underlying DataFrame (ages × years)."""
        return self.data.copy()

    def __repr__(self) -> str:
        return (
            f"MortalityRates(ages={self.ages[0]}–{self.ages[-1]}, "
            f"years={self.years[0]}–{self.years[-1]}, label={self.label!r})"
        )
