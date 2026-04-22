from __future__ import annotations

import functools

import numpy as np

from .interest import InterestRate
from .lifetable import LifeTable


class ActuarialTable:
    """
    Life table combined with an interest rate.

    Provides commutation columns (Dx, Nx, Cx, Mx, Rx) and delegates to
    the module-level actuarial functions (axn, Axn, …) via convenience methods.

    Parameters
    ----------
    lifetable : LifeTable
    interest : float or InterestRate
        Annual effective interest rate (float) or InterestRate instance.
    name : str
        Optional label.
    """

    def __init__(
        self,
        lifetable: LifeTable,
        interest: float | InterestRate,
        name: str = "",
    ) -> None:
        self.lifetable = lifetable
        if isinstance(interest, (int, float)):
            interest = InterestRate(i=float(interest))
        self.interest = interest
        self.name = name or lifetable.name

    # ------------------------------------------------------------------ #
    # Convenience aliases matching R constructor keyword argument          #
    # ------------------------------------------------------------------ #

    @property
    def i(self) -> float:
        return self.interest.i

    @property
    def v(self) -> float:
        return self.interest.v

    @property
    def x_min(self) -> int:
        return self.lifetable.x_min

    @property
    def omega(self) -> int:
        return self.lifetable.omega

    # ------------------------------------------------------------------ #
    # Commutation columns (vectorised, cached)                            #
    # ------------------------------------------------------------------ #

    @functools.cached_property
    def _comm(self) -> dict[str, np.ndarray]:
        """Compute all commutation columns at once and cache them."""
        lt = self.lifetable
        v = self.interest.v

        ages = np.arange(lt.x_min, lt.omega + 1)
        lx_full = np.array([lt.lx(x) for x in ages], dtype=float)
        dx_full = -np.diff(lx_full)  # length = omega - x_min (no terminal age)

        # Dx = v^x * lx  (length: omega - x_min + 1, includes omega)
        Dx = v ** ages * lx_full

        # Nx = cumulative sum of Dx from right, length = omega - x_min + 1
        # N_x = sum_{t=x}^{omega} D_t  (includes D_omega = 0)
        Nx = np.cumsum(Dx[::-1])[::-1]

        # Cx = v^{x+1} * dx  (length: omega - x_min)
        Cx = v ** (ages[:-1] + 1) * dx_full

        # Mx = cumulative sum of Cx from right
        # M_x = sum_{t=x}^{omega-1} C_t
        Mx = np.concatenate([np.cumsum(Cx[::-1])[::-1], [0.0]])

        # Rx = cumulative sum of Mx from right  (same length as Mx)
        # R_x = sum_{t=x}^{omega-1} M_t  (R_omega = 0)
        Rx = np.concatenate([np.cumsum(Mx[:-1][::-1])[::-1], [0.0]])

        return {
            "ages": ages,
            "Dx": Dx,
            "Nx": Nx,
            "Cx": Cx,
            "Mx": Mx,
            "Rx": Rx,
        }

    def _idx(self, x: int) -> int:
        """Translate age to array index."""
        idx = int(x) - self.x_min
        if idx < 0 or idx >= len(self._comm["Dx"]):
            raise ValueError(
                f"Age {x} out of table range [{self.x_min}, {self.omega}]"
            )
        return idx

    def Dx(self, x: int) -> float:
        return float(self._comm["Dx"][self._idx(x)])

    def Nx(self, x: int) -> float:
        return float(self._comm["Nx"][self._idx(x)])

    def Cx(self, x: int) -> float:
        idx = self._idx(x)
        Cx_arr = self._comm["Cx"]
        if idx >= len(Cx_arr):
            return 0.0
        return float(Cx_arr[idx])

    def Mx(self, x: int) -> float:
        return float(self._comm["Mx"][self._idx(x)])

    def Rx(self, x: int) -> float:
        return float(self._comm["Rx"][self._idx(x)])

    # ------------------------------------------------------------------ #
    # Convenience method wrappers (import deferred to avoid circular)     #
    # ------------------------------------------------------------------ #

    def axn(self, x: int, n: float | None = None, k: int = 1) -> float:
        from .actuarial import axn
        return axn(self, x=x, n=n, k=k)

    def Axn(self, x: int, n: float | None = None, k: int = 1) -> float:
        from .actuarial import Axn
        return Axn(self, x=x, n=n, k=k)

    def Exn(self, x: int, n: int) -> float:
        from .actuarial import Exn
        return Exn(self, x=x, n=n)

    def AExn(self, x: int, n: int, k: int = 1) -> float:
        from .actuarial import AExn
        return AExn(self, x=x, n=n, k=k)

    def IAxn(self, x: int, n: float | None = None, k: int = 1) -> float:
        from .actuarial import IAxn
        return IAxn(self, x=x, n=n, k=k)

    def DAxn(self, x: int, n: int, k: int = 1) -> float:
        from .actuarial import DAxn
        return DAxn(self, x=x, n=n, k=k)

    def exn(self, x: int, n: float | None = None) -> float:
        from .actuarial import exn
        return exn(self, x=x, n=n)

    def __repr__(self) -> str:
        return (
            f"ActuarialTable(name={self.name!r}, "
            f"x_min={self.x_min}, omega={self.omega}, i={self.i:.4g})"
        )
