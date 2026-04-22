"""
Demographic functions operating on a LifeTable.

All functions match the R lifecontingencies interface:
  pxt(lt, x, t)  ->  _t p_x
  qxt(lt, x, t)  ->  _t q_x
  etc.
"""

from __future__ import annotations

import numpy as np

from .lifetable import LifeTable


def pxt(lt: LifeTable, x: int, t: float = 1) -> float:
    """t-year survival probability _t p_x."""
    return lt.npx(x, t)


def qxt(lt: LifeTable, x: int, t: float = 1) -> float:
    """t-year death probability _t q_x = 1 - _t p_x."""
    return lt.nqx(x, t)


def dxt(lt: LifeTable, x: int, t: float = 1) -> float:
    """Expected deaths between age x and x+t (per-survivor basis × lx)."""
    return float(lt.lx(x) * lt.nqx(x, t))


def mxt(lt: LifeTable, x: int) -> float:
    """Central death rate m_x at integer age x."""
    return float(lt.mx(x))


def Lxt(lt: LifeTable, x: int) -> float:
    """
    Person-years lived between ages x and x+1.

    Under UDD: L_x = l_x - 0.5 d_x = 0.5 (l_x + l_{x+1}).
    """
    lx = float(lt.lx(x))
    lx1 = float(lt.lx(x + 1))
    return 0.5 * (lx + lx1)


def Tx(lt: LifeTable, x: int) -> float:
    """
    Total person-years lived from age x onwards:  T_x = sum_{t=x}^{omega-1} L_t.
    """
    total = 0.0
    for age in range(x, lt.omega):
        total += Lxt(lt, age)
    return total


def exn(
    lt,
    x: int,
    n: float | None = None,
) -> float:
    """
    Curtate future lifetime expectation e_x (or curtate temporary e_{x:n|}).

    Accepts either a LifeTable or an ActuarialTable.

    e_x = sum_{k=1}^{omega-x} _k p_x = N_{x+1} / D_x  (when ActuarialTable)
    e_{x:n|} = sum_{k=1}^{n} _k p_x

    Parameters
    ----------
    lt : LifeTable or ActuarialTable
    x : int
        Attained age.
    n : float or None
        Term. If None, whole-life expectation is returned.
    """
    # Unpack ActuarialTable to its underlying LifeTable
    from .actuarialtable import ActuarialTable
    if isinstance(lt, ActuarialTable):
        lt = lt.lifetable

    # e_x = sum_{k=1}^{omega-x} _k p_x = (l_{x+1} + ... + l_omega) / l_x
    # e_{x:n|} = sum_{k=1}^{n} _k p_x
    x_idx = x - lt.x_min
    lx_val = lt._lx[x_idx]
    if lx_val == 0.0:
        return 0.0

    if n is None:
        return float(lt._lx[x_idx + 1:].sum() / lx_val)

    max_k = int(n)
    end_idx = min(x_idx + max_k + 1, len(lt._lx))
    return float(lt._lx[x_idx + 1: end_idx].sum() / lx_val)


def ex_complete(lt: LifeTable, x: int) -> float:
    """
    Complete future lifetime expectation ê_x.

    Under UDD:  ê_x = e_x + 0.5
    """
    return exn(lt, x) + 0.5


def mx2qx(mx_val: float) -> float:
    """Convert central death rate m_x to q_x under UDD: q_x = m_x / (1 + 0.5 m_x)."""
    return mx_val / (1.0 + 0.5 * mx_val)


def qx2mx(qx_val: float) -> float:
    """Convert q_x to central death rate m_x under UDD: m_x = q_x / (1 - 0.5 q_x)."""
    return qx_val / (1.0 - 0.5 * qx_val)


def getOmega(lt: LifeTable) -> int:
    """Return the limiting age omega of a LifeTable."""
    return lt.omega


def probs2lifetable(
    qx: np.ndarray,
    x_min: int = 0,
    radix: float = 100_000.0,
    name: str = "",
) -> LifeTable:
    """Convert a vector of qx values to a LifeTable (convenience wrapper)."""
    from .lifetable import LifeTable
    return LifeTable.from_qx(qx, x_min=x_min, radix=radix, name=name)
