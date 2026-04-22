"""
Single-life actuarial present values.

All functions accept an ActuarialTable as first argument and return floats,
matching the R lifecontingencies interface exactly.

Formulas use commutation columns from ActuarialTable for integer k=1 cases.
For k>1 (m-thly payments) the UDD approximation is applied.
"""

from __future__ import annotations

import math

import numpy as np

from .actuarialtable import ActuarialTable


# ------------------------------------------------------------------ #
# Pure endowment                                                       #
# ------------------------------------------------------------------ #

def Exn(at: ActuarialTable, x: int, n: int) -> float:
    """
    n-year pure endowment  _n E_x = v^n * _n p_x = D_{x+n} / D_x.

    Parameters
    ----------
    at : ActuarialTable
    x : int
        Attained age.
    n : int
        Term in years.
    """
    if x + n > at.omega:
        return 0.0
    return at.Dx(x + n) / at.Dx(x)


# ------------------------------------------------------------------ #
# Life annuities                                                       #
# ------------------------------------------------------------------ #

def axn(
    at: ActuarialTable,
    x: int,
    n: float | None = None,
    k: int = 1,
) -> float:
    """
    Actuarial present value of a life annuity-due.

    Parameters
    ----------
    at : ActuarialTable
    x : int
        Attained age.
    n : float or None
        Term in years. ``None`` (default) = whole-life.
    k : int
        Payment frequency per year (k=1 annual-due, k=2 semi-annual, …).
        For k>1 the UDD approximation is used.
    """
    whole_life = (n is None) or math.isinf(float(n))
    n_int = at.omega - x if whole_life else int(n)

    if n_int <= 0:
        return 0.0

    # Annual annuity-due (exact via commutation columns)
    if k == 1:
        Nx_val = at.Nx(x)
        Nx_n = at.Nx(x + n_int) if (x + n_int) <= at.omega else 0.0
        Dx_val = at.Dx(x)
        if whole_life:
            return Nx_val / Dx_val
        return (Nx_val - Nx_n) / Dx_val

    # k-thly via UDD: ä^(k) = α(k) ä - β(k)(1 - _n E_x)
    ann_annual = axn(at, x, n=None if whole_life else n_int, k=1)
    alpha = at.interest.alpha(k)
    beta = at.interest.beta(k)
    nEx = 0.0 if whole_life else Exn(at, x, n_int)
    return alpha * ann_annual - beta * (1.0 - nEx)


# ------------------------------------------------------------------ #
# Life insurances                                                      #
# ------------------------------------------------------------------ #

def Axn(
    at: ActuarialTable,
    x: int,
    n: float | None = None,
    k: int = 1,
) -> float:
    """
    Actuarial present value of a term (or whole-life) insurance.

    Pays 1 at end of year of death if death occurs within n years.
    If n is None or inf, a whole-life insurance is returned.

    Parameters
    ----------
    k : int
        For k>1: A^(k) = (i / i^(k)) * A  under UDD.
    """
    whole_life = (n is None) or math.isinf(float(n))
    n_int = at.omega - x if whole_life else int(n)

    if n_int <= 0:
        return 0.0

    Mx_val = at.Mx(x)
    Mx_n = at.Mx(x + n_int) if (x + n_int) <= at.omega else 0.0
    Dx_val = at.Dx(x)

    if k == 1:
        return (Mx_val - Mx_n) / Dx_val

    # UDD k-thly: A^(k) = (i / i^(k)) * A^(1)
    A_annual = (Mx_val - Mx_n) / Dx_val
    i = at.interest.i
    i_k = at.interest.i_m(k)
    return (i / i_k) * A_annual


def AExn(
    at: ActuarialTable,
    x: int,
    n: int,
    k: int = 1,
) -> float:
    """
    Endowment insurance  A_{x:n|} = A^1_{x:n|} + _n E_x.

    Pays 1 at end of year of death (if within n years) or at age x+n.
    """
    return Axn(at, x, n=n, k=k) + Exn(at, x, n)


# ------------------------------------------------------------------ #
# Increasing/decreasing insurances                                     #
# ------------------------------------------------------------------ #

def IAxn(
    at: ActuarialTable,
    x: int,
    n: float | None = None,
    k: int = 1,
) -> float:
    """
    Increasing insurance (IA)_{x:n|}.

    Pays (k+1) if death occurs in year k+1  (k = 0, 1, …, n-1).
    For whole-life (n=None): (IA)_x = R_x / D_x.
    For term:  (IA)^1_{x:n|} = (R_x - R_{x+n} - n M_{x+n}) / D_x.

    k parameter: k>1 uses UDD scaling as for Axn.
    """
    whole_life = (n is None) or math.isinf(float(n))
    n_int = at.omega - x if whole_life else int(n)

    if n_int <= 0:
        return 0.0

    Rx_val = at.Rx(x)
    Dx_val = at.Dx(x)

    if whole_life:
        result = Rx_val / Dx_val
    else:
        Rx_n = at.Rx(x + n_int) if (x + n_int) <= at.omega else 0.0
        Mx_n = at.Mx(x + n_int) if (x + n_int) <= at.omega else 0.0
        result = (Rx_val - Rx_n - n_int * Mx_n) / Dx_val

    if k == 1:
        return result
    i = at.interest.i
    i_k = at.interest.i_m(k)
    return (i / i_k) * result


def DAxn(
    at: ActuarialTable,
    x: int,
    n: int,
    k: int = 1,
) -> float:
    """
    Decreasing term insurance (DA)^1_{x:n|}.

    Pays (n-k) if death occurs in year k+1 (k = 0, …, n-1).
    Identity: (DA)^1 + (IA)^1 = (n+1) A^1_{x:n|}.
    """
    if n <= 0:
        return 0.0
    A_term = Axn(at, x, n=n, k=1)
    IA_term = IAxn(at, x, n=n, k=1)
    result = (n + 1) * A_term - IA_term

    if k == 1:
        return result
    i = at.interest.i
    i_k = at.interest.i_m(k)
    return (i / i_k) * result


# ------------------------------------------------------------------ #
# Life expectation                                                     #
# ------------------------------------------------------------------ #

def exn(
    at: ActuarialTable,
    x: int,
    n: float | None = None,
) -> float:
    """
    Curtate future lifetime expectation e_x (or temporary e_{x:n|}).

    e_x = (l_{x+1} + l_{x+2} + ... + l_omega) / l_x  — no discounting.
    e_{x:n|} = sum_{k=1}^{n} _k p_x.

    Note: N_{x+1}/D_x gives the interest-discounted annuity-immediate, NOT e_x.
    """
    from .demographic import exn as _exn_demo
    return _exn_demo(at, x=x, n=n)
