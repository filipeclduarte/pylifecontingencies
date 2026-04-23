# Visão Geral: pylifecontingencies

O `pylifecontingencies` é uma implementação nativa em Python das funcionalidades do pacote R `lifecontingencies`. O foco do projeto é fornecer ferramentas para cálculos atuariais e projeção de mortalidade utilizando o ecossistema NumPy e Pandas, sem dependência do runtime de R.

---

## Características Principais

*   **API Compatível com R**: Utiliza a mesma nomenclatura de funções (`axn`, `Axn`, `Exn`, `pxt`, etc.) para facilitar a transição de fluxos de trabalho existentes.
*   **Independência de Ambiente**: Implementação puramente em Python (NumPy + Pandas). O R é utilizado apenas na suíte de testes para validação de resultados.
*   **Projeção de Mortalidade**: Inclui módulos para modelos de Lee-Carter e CBD (M5), permitindo a criação de tábuas projetadas e análises de coorte.
*   **Dados Integrados**: Contém tábuas padrão da SOA e tábuas brasileiras **BR-EMS (2010, 2015 e 2021)**.
*   **Simulação Estocástica**: Simulação de Monte Carlo do VPA — pagamentos fracionários (`k`), diferimento (`m`), visualização (`hist`, `plot`), métricas de risco (`var`, `tvar`) e exportação para pandas via `StochasticResult`.

---

## Exemplos de Uso

### Cálculos Atuariais Básicos
Cálculo de anuidades e seguros utilizando tábuas estáticas:

```python
from pylifecontingencies import load_table, ActuarialTable, axn, Axn

# Carregamento e configuração da tábua
lt = load_table("soa_ilt")
at = ActuarialTable(lt, interest=0.06)

# ä_40 (anuidade vitalícia antecipada)
vpa_anuidade = axn(at, x=40) 

# A^1_{40:20|} (seguro temporário de 20 anos)
vpa_seguro = Axn(at, x=40, n=20)
```

### Simulação Estocástica — Anuidade Mensal, Métricas de Risco e Visualização

```python
from pylifecontingencies import load_table, ActuarialTable, simulate_pv

lt = load_table("soa_ilt")
at = ActuarialTable(lt, interest=0.03)

# Aposentadoria mensal com início em 10 anos (k=12, diferimento m=10)
r = simulate_pv(at, x=40, n=20, benefit="annuity", k=12, m=10, n_sim=100_000, random_state=42)
print(r.mean, r.std)
lo, hi = r.ci(0.95)

# Visualização
r.hist()      # histograma com média e linhas de IC 95 %
r.plot()      # FDA empírica

# Métricas de risco para Solvência II
r.var(0.995)   # Value at Risk a 99,5 %
r.tvar(0.995)  # Tail VaR (Expected Shortfall) a 99,5 %

# Exportar para pandas
df = r.to_dataframe()
df["pv"].describe()
```

### Utilização de Tábuas Brasileiras (BR-EMS)
Acesso direto às tábuas nacionais para cálculos de probabilidade e VPA:

```python
from pylifecontingencies import load_table, qxt

# BR-EMS Sobrevivência 2021 Masculina
lt_br = load_table("br_emssb_2021_m")

# Probabilidade de morte (_10 q_40)
prob = qxt(lt_br, x=40, t=10)
```

### Projeção com Lee-Carter
Exemplo de ajuste de modelo e geração de tábua de coorte:

```python
from pylifecontingencies.dynamic import LeeCarter, ProjectedLifeTable

# Ajuste do modelo em matriz de taxas históricas
lc = LeeCarter().fit(matriz_log_mx)

# Projeção e conversão para tábua de vida de coorte
forecast = lc.forecast(horizon=50)
cohort_lt = ProjectedLifeTable(forecast, birth_year=1990).to_life_table()
```

---

## Comparativo Técnico

| Funcionalidade | R (`lifecontingencies`) | Python (`pylifecontingencies`) |
| :--- | :--- | :--- |
| Anuidade Vitalícia | `axn(at, x=40)` | `axn(at, x=40)` |
| Seguro Temporário | `Axn(at, x=40, n=20)` | `Axn(at, x=40, n=20)` |
| Modelagem Dinâmica | Requer pacotes adicionais | Nativa em `dynamic` |
| Base de Dados | Arquivos `.rda` | CSV e Parquet (otimizado) |

---

## Instalação e Licença

Para instalar via pip:
```bash
pip install pylifecontingencies
```

O projeto está licenciado sob **GPL-2.0**, mantendo a compatibilidade com a licença do pacote original em R. Documentação adicional sobre reservas matemáticas, prêmios e ajustes de leis de mortalidade está disponível no repositório.
