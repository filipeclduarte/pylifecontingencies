"""
Human Mortality Database (HMD) flat-file loader.

HMD files are available at https://www.mortality.org after registering
for a free account. Download the country-specific "Mx_1x1.txt" files
(central death rates, 1-year age and period groups).
"""

from __future__ import annotations

from pathlib import Path

from ..dynamic.rates import MortalityRates


def load_hmd(
    path: str | Path,
    sex: str = "Female",
    age_min: int | None = None,
    age_max: int | None = None,
    year_min: int | None = None,
    year_max: int | None = None,
) -> MortalityRates:
    """
    Load an HMD Mx flat text file into a MortalityRates surface.

    Parameters
    ----------
    path : str or Path
        Path to the HMD ``Mx_1x1.txt`` file.
    sex : {'Female', 'Male', 'Total'}
        Which sex column to load.
    age_min, age_max : int or None
        Optional age range filter.
    year_min, year_max : int or None
        Optional year range filter.

    Returns
    -------
    MortalityRates
        Log mortality rate surface.

    Examples
    --------
    ::

        from pylifecontingencies.io import load_hmd
        rates = load_hmd("USA_Mx_1x1.txt", sex="Female", age_min=0, age_max=99)
    """
    mr = MortalityRates.from_hmd_file(path, sex=sex)
    ages = mr.ages
    years = mr.years

    if age_min is not None or age_max is not None:
        lo = age_min if age_min is not None else int(ages[0])
        hi = age_max if age_max is not None else int(ages[-1])
        ages_sel = [a for a in ages if lo <= a <= hi]
        mr = mr.subset(ages=ages_sel)

    if year_min is not None or year_max is not None:
        lo = year_min if year_min is not None else int(years[0])
        hi = year_max if year_max is not None else int(years[-1])
        years_sel = [y for y in years if lo <= y <= hi]
        mr = mr.subset(years=years_sel)

    return mr
