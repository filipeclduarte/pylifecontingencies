# Guia de Integração: pylifecontingencies

Este documento é voltado para **Desenvolvedores, Engenheiros de Dados e Arquitetos de Software** que precisam integrar a lógica atuarial do `pylifecontingencies` em sistemas produtivos, APIs ou pipelines de dados.

Por ser construído inteiramente sobre **NumPy e Pandas**, o pacote é facilmente vetorizável e ideal para processamento em larga escala ou como motor de cálculo em backends modernos.

---

## 1. Integração com APIs (Ex: FastAPI / Flask)

O uso mais comum em aplicações modernas é expor as funções atuariais como um serviço de precificação (Pricing API).

**Exemplo com FastAPI:**

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pylifecontingencies import load_table, ActuarialTable, axn, Axn

app = FastAPI(title="Pricing API - Atuária")

# Carrega a tábua em memória na inicialização (cache)
# Em produção, pode ser um arquivo local ou S3
lt_br = load_table("br_emssb_2021_m")

class PricingRequest(BaseModel):
    age: int
    term: int
    interest_rate: float
    benefit_amount: float

@app.post("/calculate-term-insurance")
def calculate_insurance(req: PricingRequest):
    if req.age < 0 or req.age > 100:
        raise HTTPException(status_code=400, detail="Idade inválida.")
        
    # Instancia a tabela com a taxa de juros da requisição
    at = ActuarialTable(lt_br, interest=req.interest_rate)
    
    # Calcula o Fator de Valor Presente Atuarial
    apv_factor = Axn(at, x=req.age, n=req.term)
    
    return {
        "apv_factor": apv_factor,
        "premium": apv_factor * req.benefit_amount
    }
```

**Vantagens:** 
* Sem dependência do R, o tempo de inicialização (cold start) é mínimo.
* Pode ser facilmente containerizado (Docker) e escalado no Kubernetes ou AWS Lambda.

---

## 2. Integração em Pipelines de Dados (Ex: Pandas / PySpark)

Para processos batch (ex: recálculo de reservas de uma carteira inteira à noite), o pacote pode ser aplicado de forma eficiente em DataFrames.

**Exemplo de processamento em lote com Pandas:**

```python
import pandas as pd
from pylifecontingencies import load_table, ActuarialTable, Axn

# 1. Carrega dados da carteira (banco de dados, parquet, etc.)
carteira = pd.DataFrame({
    'apolice_id': [1, 2, 3],
    'idade': [35, 45, 55],
    'prazo_restante': [10, 15, 5],
    'capital_segurado': [100000, 200000, 50000]
})

# 2. Configura o motor atuarial
lt = load_table("br_emssb_2021_m")
at = ActuarialTable(lt, interest=0.03)

# 3. Função para aplicar na carteira
def calcular_vpa(row):
    fator = Axn(at, x=row['idade'], n=row['prazo_restante'])
    return fator * row['capital_segurado']

# 4. Aplica o cálculo (vetorização ou apply)
carteira['reserva_matematica'] = carteira.apply(calcular_vpa, axis=1)

print(carteira)
```

> **Dica de Performance:** Para carteiras massivas (+1 Milhão de linhas), recomenda-se agrupar (groupby) por `(idade, prazo_restante)`, calcular o fator uma única vez para cada combinação e fazer um `merge/join` de volta na tabela original.

---

## 3. Arquitetura e Cache de Tábuas

Ao integrar o `pylifecontingencies`, gerenciar o carregamento das tábuas é crucial para a performance.

* **Tábuas Estáticas:** Utilize o método `load_table()` na inicialização do seu worker (startup event no FastAPI, ou início do script Airflow). As tábuas já vêm otimizadas em formato `.parquet` ou `.csv`.
* **Tábuas Dinâmicas (Projeções):** Se utilizar o módulo `dynamic` (ex: Lee-Carter), a etapa de `.fit()` consome CPU. Recomenda-se rodar o `.fit()` em um processo assíncrono (ex: Celery) ou DAG do Airflow, salvar a tábua resultante (matriz de taxas) e consumi-la apenas para leitura na API de precificação.

```python
# Script Offline (Airflow / Cron)
lc = LeeCarter().fit(dados_historicos)
forecast = lc.forecast(horizon=30)
# Salva o resultado em parquet ou banco de dados
forecast.to_dataframe().to_parquet("forecast_2024.parquet")
```

---

## Resumo dos Requisitos de Sistema

* **Linguagem:** Python 3.10+
* **Dependências Base:** `numpy`, `pandas`, `scipy`
* **Ambiente:** Funciona em Linux, Windows, macOS, e ambientes serverless (AWS Lambda, Cloud Functions) devido à ausência de dependências C/C++ complexas ou instâncias do R.
