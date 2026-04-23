# TODO — Roadmap do pylifecontingencies

## Alta Prioridade (Próxima Sprint)

### 1. ~~DynamicLifeTable — aceitar projeções externas diretamente~~ ✅ FEITO

Implementado em `src/pylifecontingencies/dynamic/` com 31 testes unitários (`tests/test_dynamic_lifetable.py`).

**Arquivos criados:**

- `dynamic_lifetable.py` — Classe `DynamicLifeTable` com construtores:
  `from_forecast_mx`, `from_forecast_qx`, `from_forecast_log_mx`,
  `from_scenarios`, `from_scenarios_array`
- `dynamic_actuarialtable.py` — `DynamicActuarialTable` (encapsula DynamicLifeTable + taxa de juros);
  cenário único retorna float, estocástico retorna `StochasticResult`
- `stochastic.py` — Contêiner `StochasticResult` com `mean`, `std`, `quantile(q)`, `ci(level)`
- `dynamic/__init__.py` — re-exporta as novas classes

---

### 2. ~~Melhorias em ProjectedLifeTable~~ ✅ FEITO

Consolidou `ProjectedLifeTable` e `DynamicLifeTable` em uma única classe agnóstica de modelo (`projected_table.py`). 29 testes (`tests/test_projected_table.py`). Retrocompatível com objetos de previsão de modelo e `DynamicLifeTable`.

**Métodos da classe** (aceitam DataFrames brutos de qualquer modelo):

- `ProjectedLifeTable.from_mx(df, lower=..., upper=..., birth_year=...)`
- `ProjectedLifeTable.from_qx(df, lower=..., upper=..., birth_year=...)`
- `ProjectedLifeTable.from_log_mx(df, lower=..., upper=..., birth_year=...)`
- `ProjectedLifeTable.from_scenarios(list_of_dfs, birth_year=...)`
- `ProjectedLifeTable.from_scenarios_array(arr, ages=..., years=..., birth_year=...)`

**Intervalos de predição**: passe DataFrames `lower` e `upper` para qualquer construtor. Produz propriedades `.lower`, `.lifetable` (central), `.upper`.

**Estratégias de extrapolação**: `"clamp"` (padrão), `"constant_slope"`, `"none"`.

---

### 3. ~~Adicionar tábuas embutidas~~ ✅ FEITO

**BR-EMS (SUSEP)** — 80 testes em `tests/test_br_ems.py`

12 tábuas via `scripts/convert_br_ems.py` → CSV em `src/pylifecontingencies/data/`:

- `br_emssb_2021_m/f`, `br_emsmt_2021_m/f` — BR-EMS 2021 (idades 0–117/116)
- `br_emssb_2015_m/f`, `br_emsmt_2015_m/f` — BR-EMS 2015 (idades 0–118)
- `br_emssb_2010_m/f`, `br_emsmt_2010_m/f` — BR-EMS 2010 (idades 0–116/113)

**Tábuas do R lifecontingencies** — via `scripts/convert_rda_to_parquet.py` → parquet:

- `soa08` — 2001 CSO (Tábua de vida S4, idades 0–140)
- `AM92Lt`, `AF92Lt` — Reino Unido AM92/AF92 (Tábua de vida S4)
- Diversas outras demográficas (`demoUsa`, etc).

---

## Prioridade Média (v2)

### 4. ~~Funções atuariais para múltiplas vidas~~ ✅ FEITO

Implementado em `src/pylifecontingencies/multilife.py` com testes de paridade R.

### 5. Tábuas de Múltiplo Decremento (MDT)

- Classe `MultiDecrementTable`
- `qxt_prime`, `qxt_fromQxprime` — decrementos independentes/dependentes
- `Axn_mdt` — VPA do benefício no decremento j

### 6. ~~Simulação Estocástica de VPA~~ ✅ FEITO

- `simulate_pv(at, x, n, benefit, n_sim)` implementado
- Retorna `StochasticResult` com `mean`, `std`, quantis e as `samples` completas

### 7. ~~Ajuste de Leis de Mortalidade~~ ✅ FEITO

- Implementados `GompertzMakeham` e `HeligmanPollard`

---

## Prioridade Menor (v2+)

### 8. Modelos de Projeção Adicionais (Família StMoMo)

- `RenshawHaberman` — LC com efeito de coorte
- `APC` — Age-Period-Cohort
- `CBD_M6`, `CBD_M7`, `CBD_M8` — variantes estendidas do CBD
- `Plat` — Modelo de quatro fatores de Plat (2009)

### 9. Fórmula de Woolhouse para anuidades k-anuais

Substituir a aproximação UDD pela fórmula de três termos de Woolhouse para maior precisão em anuidades mensais/contínuas.

### 10. Funções Contínuas

- `abar_x`, `Abar_x` — equivalentes contínuos ($\delta$ como força de juros)

### 11. ~~CI/CD e Publicação~~ ✅ FEITO
