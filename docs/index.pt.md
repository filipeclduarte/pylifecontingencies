# pylifecontingencies

Um port nativo em Python do pacote R [`lifecontingencies`](https://github.com/spedygiorgio/lifecontingencies), estendido com **tábuas de vida dinâmicas e projeção de taxas de mortalidade** (Lee-Carter, CBD M5).

Não requer o ambiente R. Funciona puramente com NumPy + Pandas; o `rpy2` é usado apenas na suíte de testes de validação.

---

## O que está incluído

- Valores presentes atuariais para uma vida: anuidades, seguros, dotações, benefícios crescentes/decrescentes, prêmios e reservas
- Funções demográficas e de mortalidade: `pxt`, `qxt`, `mxt`, `Lxt`, `Tx`, `exn`
- Projeção de mortalidade dinâmica: Lee-Carter, CBD M5, tábuas de vida projetadas, suporte a cenários estocásticos
- Simulação de Monte Carlo para VPA via `StochasticResult` — suporta frequência de pagamentos fracionária (`k`), período de diferimento (`m`) e exportação para pandas
- Graduação paramétrica de mortalidade com `GompertzMakeham` e `HeligmanPollard`
- Dados embutidos: SOA ILT, tábuas **BR-EMS** e tábuas R convertidas para parquet (ex: `soa08`, `AM92Lt`, `demoUsa`)

## Instalação

Pacote principal:

```bash
pip install pylifecontingencies
```

Com suporte a XTbML da SOA:

```bash
pip install "pylifecontingencies[soa]"
```

---

## Exemplos Rápidos

### Valores atuariais estáticos

```python
from pylifecontingencies import load_table, ActuarialTable, axn, Axn, AExn

lt = load_table("soa_ilt")
at = ActuarialTable(lt, interest=0.06)

axn(at, x=40)          # anuidade vitalícia antecipada
Axn(at, x=40, n=20)    # seguro temporário de 20 anos
AExn(at, x=40, n=20)   # dotação mista de 20 anos
```

### Tábuas R Embutidas

```python
from pylifecontingencies import load_table, list_columns

list_columns("demoUsa")
lt_usa = load_table("demoUsa", column="USSS2007M")
lt_soa = load_table("soa08")
```

### Tábuas BR-EMS

```python
from pylifecontingencies import load_table, ActuarialTable, axn, qxt

# BR-EMS Sobrevivência 2021, masculino
lt_br = load_table("br_emssb_2021_m")
at_br = ActuarialTable(lt_br, interest=0.03)

qxt(lt_br, x=40, t=10)   # Probabilidade de morte em 10 anos aos 40
axn(at_br, x=65)         # anuidade vitalícia antecipada aos 65
```

### Simulação Estocástica

```python
from pylifecontingencies import simulate_pv

# Seguro temporário anual
r = simulate_pv(at, x=40, n=20, benefit="term", n_sim=50_000, random_state=42)
r.mean, r.std, r.quantile(0.95)

# Anuidade mensal (k=12 pagamentos por ano)
r_mensal = simulate_pv(at, x=40, n=20, benefit="annuity", k=12, n_sim=50_000)

# Anuidade vitalícia com diferimento de 10 anos (previdência)
r_diferido = simulate_pv(at, x=40, benefit="annuity", m=10, n_sim=50_000)

# Exportar amostras para pandas
df = r.to_dataframe()   # DataFrame com coluna "pv"
```

### Projeções Dinâmicas (Lee-Carter)

```python
from pylifecontingencies.dynamic import MortalityRates, LeeCarter, ProjectedLifeTable

# Constrói uma superfície de taxas a partir de um DataFrame
rates = MortalityRates.from_dataframe(df_log_mx)

# Ajusta o modelo Lee-Carter
lc = LeeCarter().fit(rates)

# Previsão para 50 anos com intervalo de predição via bootstrap (95%)
forecast = lc.forecast(horizon=50, n_bootstrap=500, ci=0.95)

# Constrói uma tábua de coorte para alguém nascido em 1985
cohort_lt = ProjectedLifeTable(forecast, birth_year=1985).to_life_table()
at_cohort = ActuarialTable(cohort_lt, interest=0.03)
axn(at_cohort, x=40)   # anuidade real de coorte aos 40
```

---

## Comparação com R lifecontingencies

| R | Python |
|---|--------|
| `axn(at, x=40, n=20)` | `axn(at, x=40, n=20)` |
| `Axn(at, x=40, n=20)` | `Axn(at, x=40, n=20)` |
| `Exn(at, x=40, n=20)` | `Exn(at, x=40, n=20)` |
| `exn(lt, x=40)` | `exn(lt, x=40)` |
| `pxt(lt, x=40, t=5)` | `pxt(lt, x=40, t=5)` |
| `new("lifetable", x=..., lx=..., name=...)` | `LifeTable(lx, x_min=0, name=...)` |
| `new("actuarialtable", ..., interest=0.06)` | `ActuarialTable(lt, interest=0.06)` |

---

## Licença

GPL-2.0 — compatível com o pacote `lifecontingencies` original em R.
