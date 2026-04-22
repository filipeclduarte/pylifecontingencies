from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

from .fractional import FractionalAge


class LifeTable:
    """
    Single-decrement life table.

    Internally stores lx as a float64 array of length (omega - x_min + 1),
    so lx[0] = l_{x_min} and lx[-1] = l_{omega} = 0.

    Attributes
    ----------
    x_min : int
        Lowest age in the table.
    omega : int
        Limiting age: l_{omega} = 0.
    name : str
        Optional label.
    fractional : FractionalAge
        Assumption used for fractional ages (default: UDD).
    """

    def __init__(
        self,
        lx: np.ndarray,
        x_min: int = 0,
        name: str = "",
        fractional: FractionalAge = FractionalAge.UDD,
    ) -> None:
        lx_arr = np.asarray(lx, dtype=float)
        if lx_arr.ndim != 1 or len(lx_arr) < 2:
            raise ValueError("lx must be a 1-D array with at least two elements")
        if np.any(lx_arr < 0):
            raise ValueError("lx values must be non-negative")
        if lx_arr[-1] != 0.0:
            raise ValueError("lx[-1] (l_omega) must be 0")
        if np.any(np.diff(lx_arr) > 0):
            raise ValueError("lx must be non-increasing")

        self._lx = lx_arr.copy()
        self._x_min = int(x_min)
        self.name = name
        self.fractional = fractional

    # ------------------------------------------------------------------ #
    # Alternative constructors                                             #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_qx(
        cls,
        qx: np.ndarray,
        x_min: int = 0,
        radix: float = 100_000.0,
        name: str = "",
        fractional: FractionalAge = FractionalAge.UDD,
    ) -> LifeTable:
        """Construct from a vector of annual mortality rates q_x."""
        qx_arr = np.asarray(qx, dtype=float)
        if np.any(qx_arr < 0) or np.any(qx_arr > 1):
            raise ValueError("qx values must be in [0, 1]")
        if qx_arr[-1] != 1.0:
            # force terminal decrement
            qx_arr = np.append(qx_arr, 1.0)

        lx = np.empty(len(qx_arr) + 1)
        lx[0] = radix
        for k, q in enumerate(qx_arr):
            lx[k + 1] = lx[k] * (1.0 - q)
        lx[-1] = 0.0
        return cls(lx, x_min=x_min, name=name, fractional=fractional)

    @classmethod
    def from_mx(
        cls,
        mx: np.ndarray,
        x_min: int = 0,
        radix: float = 100_000.0,
        name: str = "",
        fractional: FractionalAge = FractionalAge.UDD,
    ) -> LifeTable:
        """Construct from central death rates m_x using the UDD approximation qx = mx/(1+0.5*mx)."""
        mx_arr = np.asarray(mx, dtype=float)
        qx = mx_arr / (1.0 + 0.5 * mx_arr)
        return cls.from_qx(qx, x_min=x_min, radix=radix, name=name, fractional=fractional)

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        age_col: str = "age",
        lx_col: Optional[str] = None,
        qx_col: Optional[str] = None,
        radix: float = 100_000.0,
        name: str = "",
        fractional: FractionalAge = FractionalAge.UDD,
    ) -> LifeTable:
        """Construct from a DataFrame with an age column and either lx or qx."""
        df = df.sort_values(age_col).reset_index(drop=True)
        x_min = int(df[age_col].iloc[0])
        if lx_col is not None:
            lx = np.append(df[lx_col].values.astype(float), 0.0)
            return cls(lx, x_min=x_min, name=name, fractional=fractional)
        if qx_col is not None:
            return cls.from_qx(
                df[qx_col].values, x_min=x_min, radix=radix, name=name, fractional=fractional
            )
        raise ValueError("Must supply either lx_col or qx_col")

    # ------------------------------------------------------------------ #
    # Basic properties                                                     #
    # ------------------------------------------------------------------ #

    @property
    def x_min(self) -> int:
        return self._x_min

    @property
    def omega(self) -> int:
        """Limiting age (l_omega = 0)."""
        return self._x_min + len(self._lx) - 1

    @property
    def ages(self) -> np.ndarray:
        """Integer ages from x_min to omega (inclusive)."""
        return np.arange(self._x_min, self.omega + 1)

    # ------------------------------------------------------------------ #
    # lx access                                                            #
    # ------------------------------------------------------------------ #

    def lx(self, x: int | np.ndarray) -> float | np.ndarray:
        """Survivors at exact integer age x."""
        x = np.asarray(x, dtype=int)
        scalar = x.ndim == 0
        x = np.atleast_1d(x)
        idx = x - self._x_min
        if np.any(idx < 0) or np.any(idx >= len(self._lx)):
            raise ValueError(f"Age out of range [{self._x_min}, {self.omega}]")
        result = self._lx[idx]
        return float(result[0]) if scalar else result

    # ------------------------------------------------------------------ #
    # Integer-year probabilities                                           #
    # ------------------------------------------------------------------ #

    def dx(self, x: int | np.ndarray) -> float | np.ndarray:
        """Deaths between age x and x+1: d_x = l_x - l_{x+1}."""
        return self.lx(x) - self.lx(np.asarray(x) + 1)

    def px(self, x: int | np.ndarray) -> float | np.ndarray:
        """One-year survival probability p_x = l_{x+1}/l_x."""
        return self.lx(np.asarray(x) + 1) / self.lx(x)

    def qx(self, x: int | np.ndarray) -> float | np.ndarray:
        """One-year death probability q_x = 1 - p_x."""
        return 1.0 - self.px(x)

    # ------------------------------------------------------------------ #
    # Multi-year probabilities (fractional n supported)                   #
    # ------------------------------------------------------------------ #

    def npx(self, x: int, n: float) -> float:
        """
        n-year survival probability _n p_x.

        For integer n uses exact lx values.
        For fractional n = floor + s applies the fractional-age assumption
        to the last year.
        """
        n_int = int(np.floor(n))
        s = n - n_int
        if s == 0.0:
            return float(self.lx(x + n_int) / self.lx(x))
        # _n p_x = _{n_int} p_x * _s p_{x+n_int}
        p_int = float(self.lx(x + n_int) / self.lx(x))
        px_frac = float(self.px(x + n_int))
        return p_int * self.fractional.npx(px_frac, s)

    def nqx(self, x: int, n: float) -> float:
        """n-year death probability _n q_x = 1 - _n p_x."""
        return 1.0 - self.npx(x, n)

    # ------------------------------------------------------------------ #
    # Central death rate                                                   #
    # ------------------------------------------------------------------ #

    def mx(self, x: int | np.ndarray) -> float | np.ndarray:
        """Central death rate m_x = d_x / L_x  (assuming UDD: L_x ≈ l_x - 0.5*d_x)."""
        dx = self.dx(x)
        lx = self.lx(x)
        Lx = lx - 0.5 * dx
        return dx / Lx

    # ------------------------------------------------------------------ #
    # Serialisation                                                        #
    # ------------------------------------------------------------------ #

    def to_dataframe(self) -> pd.DataFrame:
        """Return a DataFrame with columns age, lx, dx, qx, px."""
        ages_int = np.arange(self._x_min, self.omega)
        lx = self._lx[:-1]
        dx = np.diff(self._lx) * -1  # positive deaths
        qx = dx / lx
        return pd.DataFrame(
            {"age": ages_int, "lx": lx, "dx": dx, "qx": qx, "px": 1.0 - qx}
        )

    def getOmega(self) -> int:
        """Limiting age — matches R's getOmega()."""
        return self.omega

    def __repr__(self) -> str:
        return (
            f"LifeTable(name={self.name!r}, x_min={self._x_min}, "
            f"omega={self.omega}, radix={self._lx[0]:.0f})"
        )
