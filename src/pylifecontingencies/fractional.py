from __future__ import annotations

from enum import Enum

import numpy as np


class FractionalAge(str, Enum):
    """Fractional-age assumption for interpolating survival probabilities."""

    UDD = "udd"           # Uniform Distribution of Deaths
    CONST_FORCE = "cf"    # Constant Force of Mortality
    BALDUCCI = "balducci" # Hyperbolic / Balducci

    # ------------------------------------------------------------------
    # Core: fractional survival probability given integer-year _1p_x
    # ------------------------------------------------------------------

    def npx(self, px_int: float, s: float) -> float:
        """
        Return _s p_x given _1 p_x = px_int and 0 <= s <= 1.

        Parameters
        ----------
        px_int : float
            One-year survival probability _1 p_x (in [0, 1]).
        s : float
            Fractional duration within the year (in [0, 1]).
        """
        s = float(s)
        px = float(px_int)
        if s == 0.0:
            return 1.0
        if s == 1.0:
            return px
        if self is FractionalAge.UDD:
            return 1.0 - s * (1.0 - px)
        if self is FractionalAge.CONST_FORCE:
            return px ** s
        if self is FractionalAge.BALDUCCI:
            # _s p_x = px / (1 - (1-s)*(1-px))
            qx = 1.0 - px
            return px / (1.0 - (1.0 - s) * qx)
        raise NotImplementedError(self)

    def nqx(self, px_int: float, s: float) -> float:
        return 1.0 - self.npx(px_int, s)

    # ------------------------------------------------------------------
    # Deferred fractional probability  _s|t q_x  for integer t
    # ------------------------------------------------------------------

    def mu(self, px_int: float, s: float) -> float:
        """
        Force of mortality μ_{x+s} given integer-year _1 p_x.

        Parameters
        ----------
        px_int : float
            One-year survival probability.
        s : float
            Fractional age within the year.
        """
        px, s = float(px_int), float(s)
        if self is FractionalAge.UDD:
            qx = 1.0 - px
            return qx / (1.0 - s * qx)
        if self is FractionalAge.CONST_FORCE:
            return -np.log(px)
        if self is FractionalAge.BALDUCCI:
            qx = 1.0 - px
            return qx / (px + (1.0 - s) * qx)
        raise NotImplementedError(self)
