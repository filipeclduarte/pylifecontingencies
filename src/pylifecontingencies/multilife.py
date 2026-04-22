"""
Multi-life actuarial functions (joint-life and last-survivor).

All functions assume independent future lifetimes and accept two separate
life table objects, enabling different mortality for each life (e.g., male
and female).

Demographic functions operate on LifeTable objects:
  pxyt, qxyt, exyt

Actuarial present value functions operate on ActuarialTable objects:
  axyn, Axyn, Exyn, AExyn

The ``status`` parameter is always a string:
  - ``"joint"`` — joint-life (first-to-die / both alive)
  - ``"last"``  — last-survivor (at least one alive)

Matches the R lifecontingencies interface:
  R:      axyn(tablex, tabley, x, y, n, status="joint")
  Python: axyn(atx, aty, x, y, n, status="joint")

References
----------
Bowers, N.L. et al. "Actuarial Mathematics" (2nd ed.), Chapters 9–10.
"""

from __future__ import annotations

import math
from typing import Literal

from .actuarialtable import ActuarialTable
from .lifetable import LifeTable

# Valid status values
Status = Literal["joint", "last"]

_VALID_STATUSES = {"joint", "last"}


def _validate_status(status: str) -> str:
    """Normalise and validate the status parameter."""
    s = status.lower().strip()
    if s not in _VALID_STATUSES:
        raise ValueError(
            f"status must be 'joint' or 'last', got {status!r}"
        )
    return s


# ------------------------------------------------------------------ #
# Helper: extract LifeTable from either LifeTable or ActuarialTable  #
# ------------------------------------------------------------------ #

def _as_lifetable(obj) -> LifeTable:
    """Return a LifeTable from either a LifeTable or an ActuarialTable."""
    if isinstance(obj, ActuarialTable):
        return obj.lifetable
    if isinstance(obj, LifeTable):
        return obj
    raise TypeError(
        f"Expected LifeTable or ActuarialTable, got {type(obj).__name__}"
    )


# ------------------------------------------------------------------ #
# Demographic functions                                               #
# ------------------------------------------------------------------ #

def pxyt(
    ltx: LifeTable,
    lty: LifeTable,
    x: int,
    y: int,
    t: int,
    status: str = "joint",
) -> float:
    """
    t-year survival probability for a two-life status.

    Parameters
    ----------
    ltx : LifeTable or ActuarialTable
        Life table for life (x).
    lty : LifeTable or ActuarialTable
        Life table for life (y).
    x : int
        Age of life (x).
    y : int
        Age of life (y).
    t : int
        Term in years.
    status : str
        ``"joint"`` — both alive after t years.
        ``"last"``  — at least one alive after t years.

    Returns
    -------
    float
        Probability.

    Notes
    -----
    Joint:  ₜp_{xy} = ₜpₓ · ₜp_y   (independence)
    Last:   ₜp_{x̄ȳ} = ₜpₓ + ₜp_y − ₜpₓ · ₜp_y
    """
    status = _validate_status(status)
    ltx = _as_lifetable(ltx)
    lty = _as_lifetable(lty)

    px = ltx.npx(x, t)
    py = lty.npx(y, t)

    if status == "joint":
        return px * py
    else:  # last
        return px + py - px * py


def qxyt(
    ltx: LifeTable,
    lty: LifeTable,
    x: int,
    y: int,
    t: int,
    status: str = "joint",
) -> float:
    """
    t-year death probability for a two-life status.

    Parameters
    ----------
    ltx, lty, x, y, t, status : see pxyt

    Returns
    -------
    float
        Probability that the status has failed by time t.

    Notes
    -----
    Joint: ₜq_{xy} = 1 − ₜp_{xy}  (at least one has died)
    Last:  ₜq_{x̄ȳ} = 1 − ₜp_{x̄ȳ}  (both have died)
    """
    return 1.0 - pxyt(ltx, lty, x, y, t, status)


def exyt(
    ltx: LifeTable,
    lty: LifeTable,
    x: int,
    y: int,
    status: str = "joint",
) -> float:
    """
    Curtate joint-life expectation e_{xy} or last-survivor e_{x̄ȳ}.

    Parameters
    ----------
    ltx, lty : LifeTable or ActuarialTable
    x, y : int
        Ages of the two lives.
    status : str
        ``"joint"`` or ``"last"``.

    Returns
    -------
    float
        Expected curtate future lifetime of the status.

    Notes
    -----
    e_{xy} = Σ_{t=1}^{ω−1} ₜp_{xy}
    e_{x̄ȳ} = eₓ + e_y − e_{xy}   (inclusion-exclusion)
    """
    status = _validate_status(status)
    ltx_lt = _as_lifetable(ltx)
    lty_lt = _as_lifetable(lty)

    # Maximum possible term
    max_t = min(ltx_lt.omega - x, lty_lt.omega - y)

    if status == "joint":
        total = 0.0
        for t in range(1, max_t + 1):
            p = ltx_lt.npx(x, t) * lty_lt.npx(y, t)
            total += p
        return total
    else:  # last
        # e_{x̄ȳ} = eₓ + e_y − e_{xy}
        from .demographic import exn
        ex = exn(ltx_lt, x)
        ey = exn(lty_lt, y)
        exy = exyt(ltx, lty, x, y, status="joint")
        return ex + ey - exy


# ------------------------------------------------------------------ #
# Actuarial present value functions                                   #
# ------------------------------------------------------------------ #

def Exyn(
    atx: ActuarialTable,
    aty: ActuarialTable,
    x: int,
    y: int,
    n: int,
    status: str = "joint",
) -> float:
    """
    n-year pure endowment for a two-life status.

    Parameters
    ----------
    atx : ActuarialTable
        Actuarial table for life (x).
    aty : ActuarialTable
        Actuarial table for life (y).
    x, y : int
        Ages.
    n : int
        Term in years.
    status : str
        ``"joint"`` or ``"last"``.

    Returns
    -------
    float
        vⁿ · ₙp_{xy} (joint) or vⁿ · ₙp_{x̄ȳ} (last).

    Notes
    -----
    Both actuarial tables must share the same interest rate.
    The discount factor v is taken from ``atx``.
    """
    _validate_status(status)
    v = atx.v
    p = pxyt(atx, aty, x, y, n, status)
    return v ** n * p


def axyn(
    atx: ActuarialTable,
    aty: ActuarialTable,
    x: int,
    y: int,
    n: float | None = None,
    k: int = 1,
    status: str = "joint",
) -> float:
    """
    Actuarial present value of a life annuity-due on two lives.

    Parameters
    ----------
    atx, aty : ActuarialTable
        Actuarial tables for each life.
    x, y : int
        Ages of the two lives.
    n : float or None
        Term in years. ``None`` = whole-life.
    k : int
        Payment frequency per year (k=1 annual, k=2 semi-annual, …).
        For k>1 the UDD approximation is used.
    status : str
        ``"joint"`` — payments while both alive.
        ``"last"``  — payments while at least one alive.

    Returns
    -------
    float

    Notes
    -----
    Joint:  ä_{xy:n|} = Σ_{t=0}^{n−1} vᵗ · ₜp_{xy}
    Last:   ä_{x̄ȳ:n|} = äₓ_{:n|} + ä_y_{:n|} − ä_{xy:n|}
                          (inclusion-exclusion)

    For k-thly payments (k>1):
      ä^(k)_{xy:n|} = α(k)·ä_{xy:n|} − β(k)·(1 − ₙE_{xy})
    """
    status = _validate_status(status)
    whole_life = (n is None) or math.isinf(float(n))

    if status == "last":
        # Inclusion-exclusion:  ä_{x̄ȳ} = äₓ + ä_y − ä_{xy}
        from .actuarial import axn as _axn
        ax_val = _axn(atx, x, n=n, k=k)
        ay_val = _axn(aty, y, n=n, k=k)
        axy_val = axyn(atx, aty, x, y, n=n, k=k, status="joint")
        return ax_val + ay_val - axy_val

    # --- Joint-life status: direct summation ---
    v = atx.v
    ltx = atx.lifetable
    lty = aty.lifetable

    max_t = min(ltx.omega - x, lty.omega - y)
    n_int = max_t if whole_life else min(int(n), max_t)

    if n_int <= 0:
        return 0.0

    # Annual annuity-due: ä = Σ_{t=0}^{n-1} v^t · ₜp_{xy}
    if k == 1:
        total = 0.0
        for t in range(n_int):
            px = ltx.npx(x, t) if t > 0 else 1.0
            py = lty.npx(y, t) if t > 0 else 1.0
            total += v ** t * px * py
        return total

    # k-thly via UDD: ä^(k) = α(k)·ä − β(k)·(1 − ₙE_{xy})
    ann_annual = axyn(atx, aty, x, y, n=None if whole_life else n_int, k=1, status="joint")
    alpha = atx.interest.alpha(k)
    beta = atx.interest.beta(k)
    nExy = 0.0 if whole_life else Exyn(atx, aty, x, y, n_int, status="joint")
    return alpha * ann_annual - beta * (1.0 - nExy)


def Axyn(
    atx: ActuarialTable,
    aty: ActuarialTable,
    x: int,
    y: int,
    n: float | None = None,
    k: int = 1,
    status: str = "joint",
) -> float:
    """
    Actuarial present value of a life insurance on two lives.

    Pays 1 at end of year of first status change (first death for joint,
    last death for last-survivor).

    Parameters
    ----------
    atx, aty : ActuarialTable
    x, y : int
    n : float or None
        Term. ``None`` = whole-life.
    k : int
        For k>1: A^(k) = (i/i^(k)) · A (UDD approximation).
    status : str
        ``"joint"`` — pays at first death.
        ``"last"``  — pays at last death.

    Returns
    -------
    float

    Notes
    -----
    Joint: A_{xy:n|} = Σ_{t=0}^{n−1} v^{t+1} · ₜp_{xy} · q_{x+t,y+t|joint}
      where q_{x+t,y+t|joint} = 1 − p_{x+t} · p_{y+t}
      (probability that at least one dies in year [t, t+1])

    Last:  A_{x̄ȳ} = Aₓ + A_y − A_{xy}   (inclusion-exclusion)
    """
    status = _validate_status(status)
    whole_life = (n is None) or math.isinf(float(n))

    if status == "last":
        # Inclusion-exclusion
        from .actuarial import Axn as _Axn
        Ax_val = _Axn(atx, x, n=n, k=k)
        Ay_val = _Axn(aty, y, n=n, k=k)
        Axy_val = Axyn(atx, aty, x, y, n=n, k=k, status="joint")
        return Ax_val + Ay_val - Axy_val

    # --- Joint-life: direct summation ---
    v = atx.v
    ltx = atx.lifetable
    lty = aty.lifetable

    max_t = min(ltx.omega - x, lty.omega - y)
    n_int = max_t if whole_life else min(int(n), max_t)

    if n_int <= 0:
        return 0.0

    # A = Σ_{t=0}^{n-1} v^{t+1} · ₜp_{xy} · (1 − p_{x+t}·p_{y+t})
    if k == 1:
        total = 0.0
        for t in range(n_int):
            # ₜp_{xy}
            tpxy = (ltx.npx(x, t) * lty.npx(y, t)) if t > 0 else 1.0

            # q_{x+t,y+t | joint} = prob at least one dies in [t, t+1]
            # given both alive at t
            age_x = x + t
            age_y = y + t
            # Guard against ages at or beyond omega
            px_t = ltx.npx(age_x, 1) if age_x < ltx.omega else 0.0
            py_t = lty.npx(age_y, 1) if age_y < lty.omega else 0.0
            q_joint = 1.0 - px_t * py_t

            total += v ** (t + 1) * tpxy * q_joint
        return total

    # k-thly via UDD: A^(k) = (i / i^(k)) · A
    A_annual = Axyn(atx, aty, x, y, n=None if whole_life else n_int, k=1, status="joint")
    i = atx.interest.i
    i_k = atx.interest.i_m(k)
    return (i / i_k) * A_annual


def AExyn(
    atx: ActuarialTable,
    aty: ActuarialTable,
    x: int,
    y: int,
    n: int,
    k: int = 1,
    status: str = "joint",
) -> float:
    """
    Endowment insurance on two lives.

    A_{xy:n|} = A¹_{xy:n|} + ₙE_{xy}

    Pays 1 at end of year of status change if within n years,
    or at time n if status is still active.
    """
    return Axyn(atx, aty, x, y, n=n, k=k, status=status) + Exyn(atx, aty, x, y, n, status=status)
