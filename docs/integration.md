# Integration Guide: pylifecontingencies

This document is aimed at **Developers, Data Engineers, and Software Architects** who need to integrate the actuarial logic of `pylifecontingencies` into production systems, APIs, or data pipelines.

Built entirely on **NumPy and Pandas**, the package is easily vectorized and ideal for large-scale processing or as a calculation engine in modern backends.

---

## 1. API Integration (e.g., FastAPI / Flask)

The most common use case in modern applications is exposing actuarial functions as a Pricing API.

**FastAPI Example:**

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pylifecontingencies import load_table, ActuarialTable, axn, Axn

app = FastAPI(title="Pricing API - Actuarial")

# Load the table into memory at startup (caching)
# In production, this could be a local file or S3
lt_soa = load_table("soa_ilt")

class PricingRequest(BaseModel):
    age: int
    term: int
    interest_rate: float
    benefit_amount: float

@app.post("/calculate-term-insurance")
def calculate_insurance(req: PricingRequest):
    if req.age < 0 or req.age > 100:
        raise HTTPException(status_code=400, detail="Invalid age.")
        
    # Instantiate the table with the requested interest rate
    at = ActuarialTable(lt_soa, interest=req.interest_rate)
    
    # Calculate the Actuarial Present Value factor
    apv_factor = Axn(at, x=req.age, n=req.term)
    
    return {
        "apv_factor": apv_factor,
        "premium": apv_factor * req.benefit_amount
    }
```

**Advantages:** 
* No R dependency, meaning minimal cold start times.
* Easily containerized (Docker) and scaled on Kubernetes or AWS Lambda.

---

## 2. Data Pipeline Integration (e.g., Pandas / PySpark)

For batch processes (e.g., recalculating reserves for an entire portfolio overnight), the package can be applied efficiently to DataFrames.

**Batch processing example with Pandas:**

```python
import pandas as pd
from pylifecontingencies import load_table, ActuarialTable, Axn

# 1. Load portfolio data (database, parquet, etc.)
portfolio = pd.DataFrame({
    'policy_id': [1, 2, 3],
    'age': [35, 45, 55],
    'remaining_term': [10, 15, 5],
    'benefit_amount': [100000, 200000, 50000]
})

# 2. Configure the actuarial engine
lt = load_table("soa_ilt")
at = ActuarialTable(lt, interest=0.03)

# 3. Function to apply to the portfolio
def calculate_apv(row):
    factor = Axn(at, x=row['age'], n=row['remaining_term'])
    return factor * row['benefit_amount']

# 4. Apply the calculation (vectorization or apply)
portfolio['mathematical_reserve'] = portfolio.apply(calculate_apv, axis=1)

print(portfolio)
```

> **Performance Tip:** For massive portfolios (1M+ rows), it is recommended to group (`groupby`) by `(age, remaining_term)`, calculate the factor once for each combination, and perform a `merge/join` back into the original table.

---

## 3. Architecture and Table Caching

When integrating `pylifecontingencies`, managing table loading is crucial for performance.

* **Static Tables:** Use the `load_table()` method at worker startup (startup event in FastAPI, or start of an Airflow script). Tables are already optimized in `.parquet` or `.csv` formats.
* **Dynamic Tables (Forecasting):** If using the `dynamic` module (e.g., Lee-Carter), the `.fit()` step consumes CPU. It is recommended to run `.fit()` in an asynchronous process (e.g., Celery) or an Airflow DAG, save the resulting table (rates matrix), and consume it read-only in the pricing API.

```python
# Offline Script (Airflow / Cron)
lc = LeeCarter().fit(historical_data)
forecast = lc.forecast(horizon=30)
# Save the result to parquet or database
forecast.to_dataframe().to_parquet("forecast_2024.parquet")
```

---

## Summary of System Requirements

* **Language:** Python 3.10+
* **Core Dependencies:** `numpy`, `pandas`, `scipy`
* **Environment:** Works on Linux, Windows, macOS, and serverless environments (AWS Lambda, Cloud Functions) due to the absence of complex C/C++ dependencies or R instances.
