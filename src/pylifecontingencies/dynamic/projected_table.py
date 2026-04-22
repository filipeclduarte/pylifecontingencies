"""
ProjectedLifeTable: extract a cohort (or period) life table from a
mortality-model forecast and return a plain LifeTable for actuarial
present-value calculations.
"""

from __future__ import annotations

import numpy as np

from ..lifetable import LifeTable
from ..fractional import FractionalAge


class ProjectedLifeTable:
    """
    Extract a life table from a mortality forecast surface.

    Supports two extraction modes:

    - **Cohort** (``birth_year`` supplied): for a cohort born in ``birth_year``,
      the q_x at age x comes from the forecast for calendar year
      ``birth_year + x``.
    - **Period** (``period_year`` supplied): all q_x values come from the
      same calendar year.

    Parameters
    ----------
    forecast : LeeCarterForecast or CBDForecast
        A fitted-and-forecasted model object with a ``qx(year)`` method.
    birth_year : int or None
        Birth year for cohort extraction. Mutually exclusive with period_year.
    period_year : int or None
        Calendar year for period extraction. Mutually exclusive with birth_year.
    ages : list[int] or None
        Ages to include. Defaults to all ages available in the forecast.
    fractional : FractionalAge
        Fractional-age assumption for the resulting LifeTable.
    """

    def __init__(
        self,
        forecast,
        *,
        birth_year: int | None = None,
        period_year: int | None = None,
        ages: list[int] | None = None,
        fractional: FractionalAge = FractionalAge.UDD,
    ) -> None:
        if (birth_year is None) == (period_year is None):
            raise ValueError("Provide exactly one of birth_year or period_year")
        self.forecast = forecast
        self.birth_year = birth_year
        self.period_year = period_year
        self._ages = np.asarray(ages, dtype=int) if ages is not None else forecast.ages
        self.fractional = fractional

    def to_life_table(self, name: str = "") -> LifeTable:
        """
        Build and return a LifeTable from the projection.

        For ages where the forecast year falls outside the forecasted range,
        the nearest available forecast is used.
        """
        qx_values = []
        all_years = np.concatenate(
            [self.forecast.years_calib, self.forecast.years_forecast]
        )
        year_min = int(all_years.min())
        year_max = int(all_years.max())

        for age in self._ages:
            if self.birth_year is not None:
                target_year = self.birth_year + age
            else:
                target_year = self.period_year

            # Clamp to available years
            target_year = int(np.clip(target_year, year_min, year_max))
            qx_year = self.forecast.qx(target_year)
            # qx_year is an array indexed by the forecast ages
            age_idx = int(np.searchsorted(self.forecast.ages, age))
            if age_idx >= len(qx_year):
                qx_val = 1.0
            else:
                qx_val = float(qx_year[age_idx])
            qx_values.append(np.clip(qx_val, 0.0, 1.0))

        qx_arr = np.asarray(qx_values)
        # Ensure the table is terminal
        if qx_arr[-1] < 1.0:
            qx_arr[-1] = 1.0

        x_min = int(self._ages[0])
        label = name or (
            f"cohort_{self.birth_year}" if self.birth_year else f"period_{self.period_year}"
        )
        return LifeTable.from_qx(qx_arr, x_min=x_min, name=label, fractional=self.fractional)

    def __repr__(self) -> str:
        mode = f"birth={self.birth_year}" if self.birth_year else f"period={self.period_year}"
        return f"ProjectedLifeTable({mode}, ages={self._ages[0]}–{self._ages[-1]})"
