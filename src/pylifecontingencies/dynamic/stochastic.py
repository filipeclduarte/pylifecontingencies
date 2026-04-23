"""
StochasticResult: thin wrapper around a 1-D sample array returned by
DynamicActuarialTable when the underlying DynamicLifeTable is stochastic.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class StochasticResult:
    """
    Distribution of a scalar actuarial quantity across N scenarios.

    Returned by DynamicActuarialTable methods when the table was built
    from multiple forecast paths.

    Attributes
    ----------
    samples : np.ndarray
        1-D array of length N (one value per scenario).
    label : str
        Optional label (e.g. ``"axn(x=40, n=25)"``).
    """

    samples: np.ndarray
    label: str = ""

    def __post_init__(self) -> None:
        self.samples = np.asarray(self.samples, dtype=float)
        if self.samples.ndim != 1:
            raise ValueError("samples must be a 1-D array")

    # ------------------------------------------------------------------ #
    # Summary statistics                                                   #
    # ------------------------------------------------------------------ #

    @property
    def n(self) -> int:
        """Number of scenarios."""
        return len(self.samples)

    @property
    def mean(self) -> float:
        return float(np.mean(self.samples))

    @property
    def std(self) -> float:
        return float(np.std(self.samples, ddof=1) if self.n > 1 else 0.0)

    @property
    def median(self) -> float:
        return float(np.median(self.samples))

    @property
    def min(self) -> float:
        return float(np.min(self.samples))

    @property
    def max(self) -> float:
        return float(np.max(self.samples))

    def quantile(self, q: float) -> float:
        """Return the q-th quantile (0 ≤ q ≤ 1)."""
        return float(np.quantile(self.samples, q))

    def ci(self, level: float = 0.95) -> tuple[float, float]:
        """Symmetric (1-level)/2 credible interval: (lower, upper)."""
        alpha = (1.0 - level) / 2.0
        return self.quantile(alpha), self.quantile(1.0 - alpha)

    def summary(self) -> dict[str, float]:
        """Return a dict with key percentiles — useful for quick inspection."""
        return {
            "mean":   self.mean,
            "std":    self.std,
            "median": self.median,
            "p05":    self.quantile(0.05),
            "p25":    self.quantile(0.25),
            "p75":    self.quantile(0.75),
            "p95":    self.quantile(0.95),
            "min":    self.min,
            "max":    self.max,
            "n":      float(self.n),
        }

    def to_dataframe(self) -> "pd.DataFrame":
        """Return samples as a one-column ``pandas.DataFrame`` with column ``'pv'``."""
        import pandas as pd
        return pd.DataFrame({"pv": self.samples})

    # ------------------------------------------------------------------ #
    # Operator interop                                                     #
    # ------------------------------------------------------------------ #

    def __float__(self) -> float:
        """Coerce to float via the mean — allows use in scalar contexts."""
        return self.mean

    def __repr__(self) -> str:
        tag = f" [{self.label}]" if self.label else ""
        lo, hi = self.ci(0.95)
        return (
            f"StochasticResult{tag}("
            f"mean={self.mean:.4f}, std={self.std:.4f}, "
            f"95% CI=[{lo:.4f}, {hi:.4f}], n={self.n})"
        )
