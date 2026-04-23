"""
Monte Carlo present value simulation for static actuarial tables.
"""

from __future__ import annotations

import numpy as np

from .actuarialtable import ActuarialTable
from .dynamic.stochastic import StochasticResult

_BENEFIT_ALIASES = {
    "term": "term",
    "term_insurance": "term",
    "whole": "insurance",
    "whole_life": "insurance",
    "whole_life_insurance": "insurance",
    "insurance": "insurance",
    "annuity": "annuity",
    "life_annuity": "annuity",
    "pure_endowment": "pure_endowment",
    "endowment": "endowment",
    "endowment_insurance": "endowment",
    "increasing": "increasing",
    "decreasing": "decreasing",
}


def _normalize_benefit(benefit: str) -> str:
    key = benefit.strip().lower()
    if key not in _BENEFIT_ALIASES:
        choices = ", ".join(sorted(_BENEFIT_ALIASES))
        raise ValueError(f"Unknown benefit type: {benefit!r}. Expected one of: {choices}")
    return _BENEFIT_ALIASES[key]


def _death_year_pmf(at: ActuarialTable, x: int) -> np.ndarray:
    """Return P[K_x = k] for k = 0, ..., omega-x-1."""
    ages = np.arange(x, at.omega)
    lx_x = at.lifetable.lx(x)
    dx = at.lifetable.dx(ages)
    return np.asarray(dx, dtype=float) / float(lx_x)


def _pv_from_death_years(
    at: ActuarialTable,
    death_years: np.ndarray,
    benefit: str,
    n: int | None,
) -> np.ndarray:
    """Map curtate future lifetimes to present values."""
    v = at.interest.v
    K = np.asarray(death_years, dtype=int)
    whole_life = n is None
    term = None if whole_life else int(n)

    if benefit == "annuity":
        payments = K + 1 if whole_life else np.minimum(K + 1, term)
        if np.isclose(1.0 - v, 0.0):
            return payments.astype(float)
        return (1.0 - v ** payments) / (1.0 - v)

    death_in_term = np.ones_like(K, dtype=bool) if whole_life else (K < term)

    if benefit in {"term", "whole", "insurance"}:
        return np.where(death_in_term, v ** (K + 1), 0.0)
    if benefit == "pure_endowment":
        if whole_life:
            raise ValueError("pure_endowment requires a finite term n")
        return np.where(K >= term, v ** term, 0.0)
    if benefit == "endowment":
        if whole_life:
            raise ValueError("endowment requires a finite term n")
        return np.where(K < term, v ** (K + 1), v ** term)
    if benefit == "increasing":
        return np.where(death_in_term, (K + 1) * v ** (K + 1), 0.0)
    if benefit == "decreasing":
        if whole_life:
            raise ValueError("decreasing insurance requires a finite term n")
        return np.where(K < term, (term - K) * v ** (K + 1), 0.0)

    raise ValueError(f"Unknown benefit type: {benefit!r}")


def simulate_pv(
    at: ActuarialTable,
    x: int,
    n: int | None = None,
    benefit: str = "term",
    n_sim: int = 10_000,
    random_state: int | np.random.Generator | None = None,
) -> StochasticResult:
    """
    Simulate a present-value distribution via Monte Carlo.

    Parameters
    ----------
    at : ActuarialTable
        Static actuarial table.
    x : int
        Attained age at issue.
    n : int or None
        Term in years. ``None`` gives a whole-life contract when supported.
    benefit : str
        One of ``"term"``, ``"whole"``, ``"insurance"``, ``"annuity"``,
        ``"pure_endowment"``, ``"endowment"``, ``"increasing"``,
        or ``"decreasing"``. Common aliases such as ``"whole_life"``
        and ``"term_insurance"`` are also accepted.
    n_sim : int
        Number of Monte Carlo draws.
    random_state : int, Generator, or None
        Seed or NumPy generator for reproducibility.

    Examples
    --------
    >>> lt = load_table("soa_ilt")
    >>> at = ActuarialTable(lt, 0.03)
    >>> r = simulate_pv(at, x=40, n=20, benefit="term", n_sim=5000, random_state=0)
    >>> r.mean, r.std
    """
    if n_sim <= 0:
        raise ValueError("n_sim must be positive")
    if x < at.x_min or x >= at.omega:
        raise ValueError(f"Age {x} out of table range [{at.x_min}, {at.omega - 1}]")
    if n is not None and n <= 0:
        raise ValueError("n must be positive when supplied")

    benefit = _normalize_benefit(benefit)
    if benefit == "insurance" and n is None:
        n = None

    pmf = _death_year_pmf(at, x)
    cdf = np.cumsum(pmf)

    if isinstance(random_state, np.random.Generator):
        rng = random_state
    else:
        rng = np.random.default_rng(random_state)

    u = rng.random(n_sim)
    death_years = np.searchsorted(cdf, u, side="right")
    death_years = np.minimum(death_years, len(pmf) - 1)
    samples = _pv_from_death_years(at, death_years, benefit=benefit, n=n)

    label = f"simulate_pv(x={x}, n={n}, benefit={benefit}, n_sim={n_sim})"
    return StochasticResult(np.asarray(samples, dtype=float), label=label)
