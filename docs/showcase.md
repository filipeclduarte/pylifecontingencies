# Overview: pylifecontingencies

`pylifecontingencies` is a native Python implementation of the functionalities found in the R `lifecontingencies` package. The project focuses on providing tools for actuarial calculations and mortality forecasting using the NumPy and Pandas ecosystem, completely independent of the R runtime.

---

## Key Features

*   **R-Compatible API**: Uses the same function nomenclature (`axn`, `Axn`, `Exn`, `pxt`, etc.) to facilitate transitioning existing workflows.
*   **Environment Independence**: Pure Python implementation (NumPy + Pandas). R is only used in the testing suite for result validation.
*   **Mortality Forecasting**: Includes modules for Lee-Carter and CBD (M5) models, enabling the creation of projected tables and cohort analysis.
*   **Built-in Data**: Contains standard SOA tables and Brazilian tables **BR-EMS (2010, 2015, and 2021)**.
*   **Stochastic Simulation**: Monte Carlo simulation of Actuarial Present Values (APV) with support for k-thly payment frequency, deferral period, and pandas export via `StochasticResult.to_dataframe()`.

---

## Usage Examples

### Basic Actuarial Calculations
Calculating annuities and insurances using static tables:

```python
from pylifecontingencies import load_table, ActuarialTable, axn, Axn

# Loading and configuring the table
lt = load_table("soa_ilt")
at = ActuarialTable(lt, interest=0.06)

# ä_40 (whole life annuity-due)
vpa_annuity = axn(at, x=40) 

# A^1_{40:20|} (20-year term insurance)
vpa_insurance = Axn(at, x=40, n=20)
```

### Stochastic PV Simulation — Monthly Annuity and Deferral

```python
from pylifecontingencies import load_table, ActuarialTable, simulate_pv

lt = load_table("soa_ilt")
at = ActuarialTable(lt, interest=0.03)

# Monthly pension starting in 10 years (k=12 payments/year, m=10 deferral)
r = simulate_pv(at, x=40, n=20, benefit="annuity", k=12, m=10, n_sim=100_000, random_state=42)
print(r.mean, r.std)
lo, hi = r.ci(0.95)

# Pandas export for custom analysis or visualisation
df = r.to_dataframe()
df["pv"].describe()
df["pv"].hist(bins=30)
```

### Dynamic Forecasting with Lee-Carter
Example of model fitting and cohort table generation:

```python
from pylifecontingencies.dynamic import LeeCarter, ProjectedLifeTable

# Model fitting on historical log mortality rates matrix
lc = LeeCarter().fit(log_mx_matrix)

# Projection and conversion to a cohort life table
forecast = lc.forecast(horizon=50)
cohort_lt = ProjectedLifeTable(forecast, birth_year=1990).to_life_table()
```

---

## Technical Comparison

| Feature | R (`lifecontingencies`) | Python (`pylifecontingencies`) |
| :--- | :--- | :--- |
| Whole Life Annuity | `axn(at, x=40)` | `axn(at, x=40)` |
| Term Insurance | `Axn(at, x=40, n=20)` | `Axn(at, x=40, n=20)` |
| Dynamic Modeling | Requires extra packages | Native in `dynamic` |
| Database | `.rda` files | CSV and Parquet (optimized) |

---

## Installation and License

To install via pip:
```bash
pip install pylifecontingencies
```

The project is licensed under **GPL-2.0**, maintaining compatibility with the original R package license. Additional documentation on mathematical reserves, premiums, and mortality law fitting is available in the repository.
