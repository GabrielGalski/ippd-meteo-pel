CC        ?= gcc
MPICC     ?= mpicc
NVCC      ?= nvcc
PYTHON    ?= python3
CUDA_ARCH ?= -arch=native

CFLAGS    := -std=c11 -O3 -Wall -Wextra
OMPFLAGS  := -fopenmp
LDFLAGS   := -lm
NVCCFLAGS := -O3 $(CUDA_ARCH)

ifeq ($(OS),Windows_NT)
OMPFLAGS  += -static
endif

HAS_MPICC := $(shell command -v $(MPICC) 2>/dev/null)
HAS_NVCC  := $(shell command -v $(NVCC) 2>/dev/null)

.PHONY: all seq omp mpi cuda data train test benchmark clean help

all: seq omp
ifneq ($(HAS_MPICC),)
	@$(MAKE) mpi
else
	@echo "[Aviso] $(MPICC) nao encontrado; alvo 'mpi' ignorado."
endif
ifneq ($(HAS_NVCC),)
	@$(MAKE) cuda
else
	@echo "[Aviso] $(NVCC) nao encontrado; alvo 'cuda' ignorado."
endif

seq: meteo
omp: meteo_omp
mpi: meteo_mpi
cuda: meteo_cuda

meteo: src/meteo.c include/forest_common.h
	$(CC) $(CFLAGS) -o $@ src/meteo.c $(LDFLAGS)

meteo_omp: minha_previsao.c include/forest_common.h
	$(CC) $(CFLAGS) $(OMPFLAGS) -o $@ minha_previsao.c $(LDFLAGS)

meteo_mpi: src/meteo_mpi.c include/forest_common.h
	$(MPICC) $(CFLAGS) $(OMPFLAGS) -o $@ src/meteo_mpi.c $(LDFLAGS)

meteo_cuda: src/meteo_cuda.cu include/forest_common.h
	$(NVCC) $(NVCCFLAGS) -o $@ src/meteo_cuda.cu

data:
	$(PYTHON) get_meteo_data.py

train:
	$(PYTHON) train_forest_pelotas.py

test: all data
	$(PYTHON) benchmark_all.py --validate-only

benchmark: all data
	$(PYTHON) benchmark_all.py

clean:
	rm -f meteo meteo_omp meteo_mpi meteo_cuda minha_previsao *.tmp.bin

help:
	@echo "Alvos disponiveis:"
	@echo "  make all        - Compila meteo, meteo_omp e (se disponiveis) meteo_mpi e meteo_cuda"
	@echo "  make seq        - Compila versao sequencial (./meteo)"
	@echo "  make omp        - Compila versao OpenMP (./meteo_omp)"
	@echo "  make mpi        - Compila versao distribuida MPI (./meteo_mpi)"
	@echo "  make cuda       - Compila versao nativa CUDA (./meteo_cuda)"
	@echo "  make data       - Baixa dados reais de Pelotas via API Open-Meteo"
	@echo "  make train      - Treina floresta real e gera models/floresta_pelotas.rf"
	@echo "  make test       - Valida equivalencia numerica entre todos os binarios"
	@echo "  make benchmark  - Executa validacao + benchmark completo e gera graficos"
	@echo "  make clean      - Remove executaveis e temporarios"
