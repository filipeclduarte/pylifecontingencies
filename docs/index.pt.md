# pylifecontingencies

Um port nativo em Python do pacote R [`lifecontingencies`](https://github.com/spedygiorgio/lifecontingencies), estendido com **tábuas de vida dinâmicas e projeção de taxas de mortalidade** (Lee-Carter, CBD M5).

Não requer o ambiente R. Funciona puramente com NumPy + Pandas; o `rpy2` é usado apenas na suíte de testes de validação.

---

## O que está incluído

- Valores presentes atuariais para uma vida: anuidades, seguros, dotações, benefícios crescentes/decrescentes, prêmios e reservas
- Funções demográficas e de mortalidade: `pxt`, `qxt`, `mxt`, `Lxt`, `Tx`, `exn`
- Projeção de mortalidade dinâmica: Lee-Carter, CBD M5, tábuas de vida projetadas, suporte a cenários estocásticos
- Simulação de Monte Carlo para VPA via `StochasticResult` — pagamentos fracionários (`k`), diferimento (`m`), exportação pandas, visualização (`hist`, `plot`) e métricas de risco (`var`, `tvar`)
- Graduação paramétrica de mortalidade com `GompertzMakeham` e `HeligmanPollard`
- Dados embutidos: mais de 100 tábuas de mortalidade via `load_table()` / `list_tables()` — SOA ILT, série **BR-EMS** (2010–2021), AT, UP, RP, GAM, CSO, IBGE e várias tábuas históricas; além de parquets R (`soa08`, `AM92Lt`, `demoUsa`, etc.)

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

# Visualização
r.hist()           # histograma com média e IC 95 %
r.plot()           # FDA empírica

# Métricas de risco (Solvência II)
r.var(0.995)       # Value at Risk a 99,5 %
r.tvar(0.995)      # Tail VaR (Expected Shortfall) a 99,5 %

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

## Tábuas embutidas

Mais de 100 tábuas de mortalidade são distribuídas com o pacote e descobertas automaticamente em tempo de execução:

```python
from pylifecontingencies import list_tables, load_table

list_tables()                        # lista todos os nomes disponíveis
lt = load_table("at_2000_female")
lt = load_table("ibge_2020_homens")
lt = load_table("up_94_male")
```

Todas as tábuas CSV têm colunas `age` e `qx`; o `x_min` é inferido automaticamente da primeira linha.

#### SOA

| Nome | Descrição |
|------|-----------|
| `soa_ilt` | SOA Illustrative Life Table (Bowers et al., idades 0–99) |

#### AT — Annuity Tables

| Nome |
|------|
| `at_49_female` |
| `at_49_male` |
| `at_50` |
| `at_55` |
| `at_71` |
| `at_83_female_basic` |
| `at_83_female_iam` |
| `at_83_male_basic` |
| `at_83_male_iam` |
| `at_2000_female` |
| `at_2000_male` |
| `at2000_suavizada_10_fem` |
| `at2000_suavizada_10_mas` |

#### UP / RP — Tábuas de renda de grupo

| Nome |
|------|
| `up_84_f` |
| `up_84_m` |
| `up84_mas_fem` |
| `up_94_female` |
| `up_94_male` |
| `rp_2000_female` |
| `rp_2000male` |
| `rp_2000_disabled_female` |
| `rp_2000_disabled_male` |

#### GAM — Group Annuity Mortality

| Nome |
|------|
| `gam_71_female` |
| `gam_71_male` |
| `gam83_basica_female` |
| `gam83_basica_masc` |
| `gam_83_female_suav_10` |
| `gam_83_masc_suav_10` |
| `gam_94_female` |
| `gam_94male` |

#### CSO — Commissioners Standard Ordinary

| Nome |
|------|
| `cso_41` |
| `cso_58` |
| `cso58_female` |
| `cso58_male` |
| `cso58_fem_age_last` |
| `cso58_fem_age_nearest` |
| `cso58_mas_age_last` |
| `cso58_mas_age_nearest` |
| `cso80` |
| `csg_60` |

#### GKM / GKF / GR — Tábuas alemãs

| Nome |
|------|
| `gkm_70` |
| `gkm_80` |
| `gkm_95` |
| `gkf_95` |
| `gr_95_male` |
| `gr_95female` |

#### BR-EMS — Tábuas de Experiência Brasileira

| Nome |
|------|
| `br_emssb_2010_m` |
| `br_emssb_2010_f` |
| `br_emssb_2015_m` |
| `br_emssb_2015_f` |
| `br_emssb_2021_m` |
| `br_emssb_2021_f` |
| `br_emsmt_2010_m` |
| `br_emsmt_2010_f` |
| `br_emsmt_2015_m` |
| `br_emsmt_2015_f` |
| `br_emsmt_2021_m` |
| `br_emsmt_2021_f` |

#### IBGE — Tábuas populacionais brasileiras

| Nome |
|------|
| `ibge_2006_ambos_os_sexos` |
| `ibge_2007_ambos_os_sexos` |
| `ibge_2008_ambos_os_sexos` |
| `ibge_2009_ambos_os_sexos` |
| `ibge_2020_homens` |
| `ibge_2020_mulheres` |

#### Tábuas brasileiras específicas por setor

| Nome |
|------|
| `iba_ferroviarios` |
| `iba_ferroviarios_v2` |
| `iapb_57_forte` |
| `iapb_57_fraca` |
| `iapc` |
| `light_forte` |
| `light_media` |
| `prudential_50` |
| `prudential_ferr_aposent` |
| `rgps_99_02_m_m` |
| `experiencia_cap` |
| `alvaro_vindas` |
| `grupal_americana` |
| `grupal_americana_v2` |
| `grupo_americana` |

#### RRB — Railroad Retirement Board

| Nome |
|------|
| `rrb_44` |
| `rrb_1944_mod_fem` |
| `rrb_1944_mod_masc` |

#### SGB — Tábuas suíças

| Nome |
|------|
| `sgb_51` |
| `sgb_71` |
| `sgb_75` |

#### Tábuas históricas e internacionais

| Nome |
|------|
| `allg_72` |
| `american_experience` |
| `bentzien` |
| `eb7_75` |
| `hunter_s` |
| `hunter_semitropical` |
| `muller` |
| `muller_v2` |
| `rentiers_francais` |
| `tasa_1927` |
| `tasa_1927_v2` |
| `ustp_61` |
| `winklevoss` |
| `wyatt_1985` |
| `x_17` |
| `zimmermann` |
| `zimmermann_empr_escrit` |
| `zimmermann_ferr_alemaes` |

Tábuas parquet R (multi-coluna) exigem seleção de coluna:

```python
from pylifecontingencies import load_table, list_columns

list_columns("demoUsa")
lt_usa = load_table("demoUsa", column="USSS2007M")
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
