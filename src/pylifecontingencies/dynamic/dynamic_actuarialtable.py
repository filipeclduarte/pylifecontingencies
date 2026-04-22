"""
DynamicActuarialTable: actuarial present values on top of a
ProjectedLifeTable or DynamicLifeTable.

- **Single-path** → all methods return ``float``, behaving identically
  to ``ActuarialTable``.
- **Stochastic / PI** → all methods return ``StochasticResult``, giving
  mean, std, percentiles, and the full sample vector across scenarios.

The class is a drop-in replacement for ``ActuarialTable`` in the
single-path case — same method signatures, same return types.
"""

from __future__ import annotations

from typing import Union

import numpy as np

from ..actuarialtable import ActuarialTable
from ..interest import InterestRate
from .dynamic_lifetable import DynamicLifeTable
from .projected_table import ProjectedLifeTable
from .stochastic import StochasticResult

# Type alias for return values
_Scalar = float
_Result = Union[float, StochasticResult]
_ListResult = Union[list[float], list[StochasticResult]]


class DynamicActuarialTable:
    """
    Actuarial table built on top of a ProjectedLifeTable or DynamicLifeTable.

    Wraps one or more ActuarialTable instances (one per scenario) and
    exposes the same EPV API. Returns ``float`` for single-path tables
    and ``StochasticResult`` for stochastic/PI tables.

    Parameters
    ----------
    source : ProjectedLifeTable, DynamicLifeTable, or LifeTable list
    interest : float or InterestRate
        Annual effective interest rate.

    Examples
    --------
    From a ProjectedLifeTable::

        plt = ProjectedLifeTable.from_mx(df, birth_year=1985)
        dat = DynamicActuarialTable(plt, i=0.03)
        dat.axn(x=40)   # → float

    With prediction intervals::

        plt = ProjectedLifeTable.from_mx(
            df, lower=df_lo, upper=df_hi, birth_year=1985
        )
        dat = DynamicActuarialTable(plt, i=0.03)
        r = dat.axn(x=40)   # → StochasticResult (3 values: lo, central, hi)

    Stochastic (N scenarios)::

        plt = ProjectedLifeTable.from_scenarios(paths, birth_year=1985)
        dat = DynamicActuarialTable(plt, i=0.03)
        r = dat.axn(x=40)
        r.mean, r.std, r.ci(0.95)  # → StochasticResult
    """

    def __init__(
        self,
        source: ProjectedLifeTable | DynamicLifeTable,
        interest: float | InterestRate,
        name: str = "",
    ) -> None:
        # Accept both ProjectedLifeTable and DynamicLifeTable
        if isinstance(source, ProjectedLifeTable):
            self._source = source
            tables = source.tables
            self._is_stochastic = source.is_stochastic
        elif isinstance(source, DynamicLifeTable):
            self._source = source
            tables = source.tables
            self._is_stochastic = source.is_stochastic
        else:
            raise TypeError(
                f"Expected ProjectedLifeTable or DynamicLifeTable, "
                f"got {type(source).__name__}"
            )

        # Keep backward compat attribute
        self.dynamic_lifetable = source

        if isinstance(interest, (int, float)):
            interest = InterestRate(i=float(interest))
        self.interest = interest
        self.name = name

        # Pre-build one ActuarialTable per scenario
        self._ats: list[ActuarialTable] = [
            ActuarialTable(lt, interest) for lt in tables
        ]

    # ------------------------------------------------------------------ #
    # Internal dispatch                                                    #
    # ------------------------------------------------------------------ #

    def _map(self, fn, *args, label: str = "", **kwargs) -> _Result:
        """
        Apply fn(ActuarialTable, *args, **kwargs) across all scenarios.
        Returns float for single-path, StochasticResult for stochastic.
        """
        results = [fn(at, *args, **kwargs) for at in self._ats]
        if not self._is_stochastic:
            return results[0]
        return StochasticResult(np.array(results, dtype=float), label=label)

    # ------------------------------------------------------------------ #
    # Convenience properties                                               #
    # ------------------------------------------------------------------ #

    @property
    def i(self) -> float:
        return self.interest.i

    @property
    def is_stochastic(self) -> bool:
        return self._is_stochastic

    @property
    def n_scenarios(self) -> int:
        return len(self._ats)

    # ------------------------------------------------------------------ #
    # Pure endowment                                                       #
    # ------------------------------------------------------------------ #

    def Exn(self, x: int, n: int) -> _Result:
        """n-year pure endowment _n E_x."""
        from ..actuarial import Exn as _Exn
        return self._map(_Exn, x=x, n=n, label=f"Exn(x={x}, n={n})")

    # ------------------------------------------------------------------ #
    # Life annuities                                                       #
    # ------------------------------------------------------------------ #

    def axn(self, x: int, n: float | None = None, k: int = 1) -> _Result:
        """
        Actuarial present value of a life annuity-due ä_{x:n|}.

        Parameters
        ----------
        x : int   Attained age.
        n : float or None   Term (None = whole-life).
        k : int   Payment frequency per year.
        """
        from ..actuarial import axn as _axn
        label = f"axn(x={x}" + (f", n={n}" if n is not None else "") + f", k={k})"
        return self._map(_axn, x=x, n=n, k=k, label=label)

    # ------------------------------------------------------------------ #
    # Life insurances                                                      #
    # ------------------------------------------------------------------ #

    def Axn(self, x: int, n: float | None = None, k: int = 1) -> _Result:
        """Term (or whole-life) insurance A^1_{x:n|}."""
        from ..actuarial import Axn as _Axn
        label = f"Axn(x={x}" + (f", n={n}" if n is not None else "") + f", k={k})"
        return self._map(_Axn, x=x, n=n, k=k, label=label)

    def AExn(self, x: int, n: int, k: int = 1) -> _Result:
        """Endowment insurance A_{x:n|}."""
        from ..actuarial import AExn as _AExn
        return self._map(_AExn, x=x, n=n, k=k, label=f"AExn(x={x}, n={n}, k={k})")

    def IAxn(self, x: int, n: float | None = None, k: int = 1) -> _Result:
        """Increasing insurance (IA)_{x:n|}."""
        from ..actuarial import IAxn as _IAxn
        label = f"IAxn(x={x}" + (f", n={n}" if n is not None else "") + ")"
        return self._map(_IAxn, x=x, n=n, k=k, label=label)

    def DAxn(self, x: int, n: int, k: int = 1) -> _Result:
        """Decreasing term insurance (DA)^1_{x:n|}."""
        from ..actuarial import DAxn as _DAxn
        return self._map(_DAxn, x=x, n=n, k=k, label=f"DAxn(x={x}, n={n})")

    # ------------------------------------------------------------------ #
    # Life expectation                                                     #
    # ------------------------------------------------------------------ #

    def exn(self, x: int, n: float | None = None) -> _Result:
        """Curtate future lifetime expectation e_x (or temporary e_{x:n|})."""
        from ..actuarial import exn as _exn
        label = f"exn(x={x}" + (f", n={n}" if n is not None else "") + ")"
        return self._map(_exn, x=x, n=n, label=label)

    # ------------------------------------------------------------------ #
    # Premiums                                                             #
    # ------------------------------------------------------------------ #

    def net_premium(
        self,
        x: int,
        n: int | None = None,
        k: int = 1,
        benefit: str = "term",
    ) -> _Result:
        """Net (benefit) premium via the equivalence principle."""
        from ..premiums import net_premium as _np
        return self._map(
            lambda at: _np(at, x=x, n=n, k=k, benefit=benefit),
            label=f"net_premium(x={x}, n={n}, benefit={benefit!r})",
        )

    def gross_premium(
        self,
        x: int,
        n: int | None = None,
        k: int = 1,
        benefit: str = "term",
        expense_ratio: float = 0.0,
        per_policy: float = 0.0,
    ) -> _Result:
        """Gross premium including expense loading."""
        from ..premiums import gross_premium as _gp
        return self._map(
            lambda at: _gp(
                at, x=x, n=n, k=k, benefit=benefit,
                expense_ratio=expense_ratio, per_policy=per_policy,
            ),
            label=f"gross_premium(x={x}, n={n})",
        )

    # ------------------------------------------------------------------ #
    # Reserves                                                             #
    # ------------------------------------------------------------------ #

    def prospective_reserve(
        self,
        x: int,
        n: int,
        t: int,
        k: int = 1,
        benefit: str = "term",
        premium: float | None = None,
    ) -> _Result:
        """Prospective net reserve _t V at duration t."""
        from ..reserves import prospective_reserve as _pr
        return self._map(
            lambda at: _pr(at, x=x, n=n, t=t, k=k, benefit=benefit, premium=premium),
            label=f"reserve(x={x}, n={n}, t={t})",
        )

    def reserve_recursion(
        self,
        x: int,
        n: int,
        k: int = 1,
        benefit: str = "term",
    ) -> list[_Result]:
        """
        Full reserve vector [_0 V, _1 V, ..., _n V].

        Returns a list of floats (single-path) or a list of StochasticResult
        (stochastic), one element per duration t = 0, ..., n.
        """
        from ..reserves import reserve_recursion as _rr

        if not self._is_stochastic:
            return _rr(self._ats[0], x=x, n=n, k=k, benefit=benefit)

        # For each t collect values across scenarios
        per_scenario = [_rr(at, x=x, n=n, k=k, benefit=benefit) for at in self._ats]
        # Transpose: list[scenario][t] -> list[t][scenario]
        return [
            StochasticResult(
                np.array([per_scenario[s][t] for s in range(len(self._ats))]),
                label=f"reserve(t={t})",
            )
            for t in range(n + 1)
        ]

    # ------------------------------------------------------------------ #
    # Commutation columns (single-path only)                              #
    # ------------------------------------------------------------------ #

    def Dx(self, x: int) -> float:
        return self._ats[0].Dx(x)

    def Nx(self, x: int) -> float:
        return self._ats[0].Nx(x)

    def Mx(self, x: int) -> float:
        return self._ats[0].Mx(x)

    # ------------------------------------------------------------------ #
    # Repr                                                                 #
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        lt = self._ats[0].lifetable
        if self._is_stochastic:
            tag = f"stochastic, n={self.n_scenarios}"
        else:
            tag = "single-path"
        return (
            f"DynamicActuarialTable({tag}, "
            f"ages={lt.x_min}–{lt.omega - 1}, i={self.i:.4g})"
        )
