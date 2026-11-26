# MeteoPel — Previsão Meteorológica Paralela para Pelotas, RS

Este repositório contém **duas frentes de execução claramente separadas**, mantendo total retrocompatibilidade com a atividade original da disciplina de Processamento Paralelo e Distribuído (IPPD — UFPel):

| Frente | Objetivo | Como executar | Relatório correspondente |
| :--- | :--- | :--- | :--- |
| **Execução 1 — Atividade Base (Entrega Obrigatória)** | Transcrição fiel da inferência original de OpenMP (`minha_previsao.c`) para CUDA sobre `dados_pelotas.csv` (floresta sintética `seed=42`, kernel direto 1 thread/amostra). | Notebook [`inferencia_cuda.ipynb`](inferencia_cuda.ipynb) no Google Colab (GPU) ou `./meteo_omp` vs. `./meteo_cuda --kernel global`. | [`relatorios/relatorio_entrega_cuda.pdf`](relatorios/relatorio_entrega_cuda.pdf) \| [`.md`](relatorios/relatorio_entrega_cuda.md) |
| **Execução 2 — Contribuições Extras (Fork / Bônus AE2)** | Arquitetura nativa completa (`meteo`, `meteo_omp`, `meteo_mpi`, `meteo_cuda` com `Shared Memory + SoA + Pinned`), paralelização de faixas/construção, treinamento real de Pelotas (Estação `A887` UFPel/INMET ou Open-Meteo Archive) e suite de benchmark. | `Makefile` + executáveis nativos (`--kernel shared`, `--model models/floresta_pelotas.rf`, `--forecast`, `mpirun`) + `benchmark_all.py`. | [`relatorios/relatorio_contribuicoes_fork.pdf`](relatorios/relatorio_contribuicoes_fork.pdf) \| [`.md`](relatorios/relatorio_contribuicoes_fork.md) |

---

## Estrutura do Repositório

```text
.
├── inferencia_cuda.ipynb                      # [Execução 1] Notebook autocontido da entrega CUDA (já validado em Tesla T4)
├── relatorios/
│   ├── relatorio_entrega_cuda.pdf             # [Execução 1] Relatório obrigatório em PDF
│   ├── relatorio_entrega_cuda.md              # [Execução 1] Fonte Markdown do relatório obrigatório
│   ├── relatorio_contribuicoes_fork.pdf       # [Execução 2] Relatório das contribuições do fork em PDF
│   └── relatorio_contribuicoes_fork.md        # [Execução 2] Fonte Markdown do relatório das contribuições
├── saidas/
│   ├── dados_pelotas.csv                      # Dados reais de Pelotas (357h) da execução validada no Colab
│   ├── resultados_inferencia.csv              # Tabela de desempenho OpenMP vs. CUDA (Tesla T4)
│   ├── desempenho_cuda_vs_openmp.png          # Gráficos de tempo e speedup (Tesla T4)
│   ├── resultados_benchmark_local.csv         # Tabela Sequencial vs. OpenMP 12T (validação local)
│   └── desempenho_cpu_local.png               # Gráfico Sequencial vs. OpenMP 12T
├── include/
│   └── forest_common.h                        # [Execução 2] ABI única (Node=20B, RandomForest=30.400B), I/O, PRNG stateless e CLI
├── src/
│   ├── meteo.c                                # [Execução 2] Versão sequencial de referência (./meteo)
│   ├── meteo_mpi.c                            # [Execução 2] Versão distribuída em MPI + OpenMP híbrido (./meteo_mpi)
│   └── meteo_cuda.cu                          # [Execução 2] CUDA C++ nativo: --kernel global e --kernel shared (SoA + Pinned)
├── minha_previsao.c                           # OpenMP (retrocompatível + faixas com reduction e construção paralela)
├── get_meteo_data.py                          # Coleta dados reais de Pelotas via API Open-Meteo -> dados_pelotas.csv
├── train_forest_pelotas.py                    # [Execução 2] Treina Random Forest real (Estação A887 UFPel/INMET ou Open-Meteo)
├── models/
│   └── floresta_pelotas.rf                    # [Execução 2] Floresta real treinada em 26.300h de Pelotas (30.400 bytes)
├── benchmark_all.py                           # [Execução 2] Validação numérica cruzada + benchmark automatizado
├── Makefile                                   # [Execução 2] Build unificado (all, seq, omp, mpi, cuda, data, train, test, benchmark)
└── requirements.txt                           # Dependências Python
```

---

## Execução 1 — Atividade Base (Transcrição OpenMP → CUDA)

Esta execução reproduz exatamente o problema original (`get_meteo_data.py` + `minha_previsao.c` com floresta sintética de 100 árvores × 15 nós) e mede a transcrição direta para CUDA (1 thread por amostra, `256` threads/bloco, memória global).

### Opção A — Via Google Colab (Recomendada para reproduzir o relatório)
1. Abra [`inferencia_cuda.ipynb`](inferencia_cuda.ipynb) no Google Colab;
2. Selecione **Ambiente de execução → Alterar tipo de ambiente de execução → GPU** (ex.: Tesla T4);
3. Clique em **Ambiente de execução → Executar tudo**.

O notebook baixa `dados_pelotas.csv` via API Open-Meteo, compila o `minha_previsao.c` original com `gcc -O3 -fopenmp`, compila o kernel CUDA via CuPy/NVRTC, valida `erro_max = 0` em múltiplos tamanhos de lote, exibe a previsão meteorológica em OpenMP e CUDA e gera `resultados_inferencia.csv` e `desempenho_cuda_vs_openmp.png`.

### Opção B — Via Terminal (Binários Nativos no Modo Padrão)
```bash
pip install -r requirements.txt
python3 get_meteo_data.py
make omp cuda
./meteo_omp dados_pelotas.csv --legacy-rand
./meteo_cuda dados_pelotas.csv --kernel global --legacy-rand
```

> **Retrocompatibilidade garantida:** o comando original `gcc -O3 -fopenmp -o minha_previsao minha_previsao.c -lm` continua compilando diretamente da raiz do repositório.

### Resultados Validados da Execução 1 (Google Colab — Tesla T4 vs. Xeon 2T)
Dados completos em [`saidas/resultados_inferencia.csv`](saidas/resultados_inferencia.csv) e discutidos em [`relatorios/relatorio_entrega_cuda.md`](relatorios/relatorio_entrega_cuda.md) (`erro_max = 0` em todos os lotes):

| Amostras (`N`) | CPU 1T (ms) | OpenMP 2T (ms) | Kernel CUDA (ms) | GPU Total (ms) | Speedup Kernel | Speedup Total |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **357** | 0,862 | 0,596 | 0,260 | 0,316 | 2,29× | 1,89× |
| **10.000** | 21,872 | 13,599 | 0,265 | 0,374 | 51,30× | 36,36× |
| **100.000** | 214,011 | 138,442 | 0,764 | 1,396 | 181,18× | 99,20× |
| **1.000.000** | 2170,566 | 1841,795 | 6,488 | 12,012 | **283,86×** | **153,33×** |

---

## Execução 2 — Contribuições Extras do Fork (HPC Completo + Modelo Real)

Esta execução ativa todas as extensões desenvolvidas para o *fork*, detalhadas em [`relatorios/relatorio_contribuicoes_fork.md`](relatorios/relatorio_contribuicoes_fork.md):

1. **Compilar todos os motores disponíveis no ambiente (`meteo`, `meteo_omp`, `meteo_mpi`, `meteo_cuda`):**
   ```bash
   pip install -r requirements.txt
   make all
   make data
   ```
   *(Em Windows com MinGW-GCC 32-bit: `gcc -std=c11 -O3 -o meteo.exe src/meteo.c -lm` e `gcc -std=c11 -O3 -fopenmp -static -o meteo_omp.exe minha_previsao.c -lm`.)*

2. **Treinar (ou atualizar) o modelo real de Pelotas (`models/floresta_pelotas.rf` — 30.400 bytes):**
   ```bash
   # Automático via Open-Meteo Historical Archive (3 anos = ~26.300 horas reais de Pelotas):
   make train

   # Ou usando CSVs/ZIPs oficiais da Estação Automática A887 (Capão do Leão / Pelotas — Convênio UFPel/Embrapa/INMET):
   # Download oficial em: https://portal.inmet.gov.br/dadoshistoricos
   python3 train_forest_pelotas.py --inmet caminho/para/dados_inmet_A887/
   ```

3. **Executar cada motor paralelo com o modelo real e previsão das próximas 24h (`--forecast`):**
   ```bash
   # Sequencial (linha de base)
   ./meteo dados_pelotas.csv --model models/floresta_pelotas.rf --forecast

   # OpenMP (faixas com reduction + construção paralela + inferência paralela)
   ./meteo_omp dados_pelotas.csv --model models/floresta_pelotas.rf --forecast

   # MPI distribuído (MPI_Bcast + MPI_Scatterv + MPI_Gatherv)
   mpirun -np 2 ./meteo_mpi dados_pelotas.csv --model models/floresta_pelotas.rf --forecast

   # CUDA otimizado (__shared__ 30,4 KB + layout SoA coalescido + Pinned Memory)
   ./meteo_cuda dados_pelotas.csv --model models/floresta_pelotas.rf --kernel shared --forecast
   ```

4. **Rodar validação numérica cruzada e benchmark completo:**
   ```bash
   make test       # verifica erro_max < 1e-5 de todos os motores contra ./meteo
   make benchmark  # executa benchmark completo (N=357 até 1.000.000) e gera gráficos
   ```

---

## Requisitos do Sistema

- **Python 3.10+** com `pip` (`pip install -r requirements.txt`)
- **Compilador C11 com OpenMP** (`gcc` ou `clang`) — obrigatório para `./meteo` e `./meteo_omp`
- **OpenMPI / MPICH** (`mpicc`, `mpirun`) — opcional para `./meteo_mpi`
- **NVIDIA CUDA Toolkit** (`nvcc`) — opcional para `./meteo_cuda` nativo (ou use o Google Colab via `inferencia_cuda.ipynb`)

## Classificação de Risco de Chuva

| Precipitação (mm/h) | Classificação |
| :--- | :--- |
| `< 1.0` | Tempo Seco / Estável |
| `1.0 - 5.0` | Chuva Leve / Garoa |
| `5.0 - 15.0` | Chuva Moderada |
| `15.0 - 30.0` | ALERTA: Chuva Forte |
| `≥ 30.0` | PERIGO: Tempestade Severa |
