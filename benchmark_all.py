"""
Suite automatizada de validação numérica cruzada e benchmark de escalabilidade
para os executáveis nativos do projeto MeteoPel:
  - ./meteo      (Sequencial CPU 1 thread)
  - ./meteo_omp  (OpenMP CPU multithread)
  - ./meteo_mpi  (MPI distribuído)
  - ./meteo_cuda (CUDA GPU: --kernel global e --kernel shared)
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RE_INF = re.compile(r"Tempo de inferencia:\s*([0-9.]+)\s*segundos")
RE_CUDA_K = re.compile(r"Tempo de inferencia \(kernel CUDA\):\s*([0-9.]+)\s*segundos")
RE_CUDA_T = re.compile(r"Tempo de inferencia \(total H2D\+kernel\+D2H\):\s*([0-9.]+)\s*segundos")
RE_MPI_C = re.compile(r"Tempo de inferencia \(computacao local max\):\s*([0-9.]+)\s*segundos")
RE_MPI_T = re.compile(r"Tempo de inferencia \(total com Bcast\+Scatterv\+Gatherv\):\s*([0-9.]+)\s*segundos")


def _exe(nome: str) -> Path | None:
    for cand in (Path(f"./{nome}"), Path(f"./{nome}.exe")):
        if cand.exists():
            return cand.resolve()
    return None


def _mpirun_cmd(mpirun: str, mpi_procs: int, mpi_exe: Path, args: list[str]) -> list[str]:
    ver = subprocess.run([mpirun, "--version"], capture_output=True, text=True)
    extra = ["--oversubscribe"] if "open mpi" in (ver.stdout + ver.stderr).lower() else []
    if os.geteuid() == 0 if hasattr(os, "geteuid") else False:
        if extra:
            extra.append("--allow-run-as-root")
    return [mpirun, *extra, "-np", str(mpi_procs), str(mpi_exe), *args]


def _rodar(cmd: list[str], env: dict[str, str] | None = None) -> str:
    res = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True)
    return res.stdout


def _ler_bin(path: Path) -> np.ndarray:
    return np.fromfile(path, dtype=np.float32)


def _extrair(padrao: re.Pattern[str], texto: str) -> float:
    m = padrao.search(texto)
    if not m:
        raise ValueError(f"Tempo não encontrado na saída para padrão {padrao.pattern}:\n{texto}")
    return float(m.group(1)) * 1000.0  # ms


def validar_todos(
    csv_path: Path,
    model_path: Path | None,
    omp_threads: int,
    mpi_procs: int,
) -> None:
    seq = _exe("meteo")
    if not seq:
        raise FileNotFoundError("Binário './meteo' não encontrado. Execute 'make all' primeiro.")

    extra_model = ["--model", str(model_path)] if model_path and model_path.exists() else []
    with tempfile.TemporaryDirectory(prefix="meteopel_val_") as tmp:
        tmp_dir = Path(tmp)
        ref_bin = tmp_dir / "ref.bin"
        _rodar([str(seq), str(csv_path), *extra_model, "--dump", str(ref_bin)])
        ref = _ler_bin(ref_bin)
        print(f"[OK] Referência sequencial ({len(ref)} amostras) gerada.")

        alvos: list[tuple[str, list[str], dict[str, str] | None]] = []
        omp = _exe("meteo_omp")
        if omp:
            env = {**os.environ, "OMP_NUM_THREADS": str(omp_threads)}
            alvos.append(("OpenMP", [str(omp), str(csv_path), *extra_model], env))

        mpi = _exe("meteo_mpi")
        mpirun = shutil.which("mpirun")
        if mpi and mpirun:
            env = {**os.environ, "OMP_NUM_THREADS": "1"}
            cmd = _mpirun_cmd(mpirun, mpi_procs, mpi, [str(csv_path), *extra_model])
            alvos.append((f"MPI ({mpi_procs} ranks)", cmd, env))

        cuda = _exe("meteo_cuda")
        if cuda:
            alvos.append(("CUDA (global)", [str(cuda), str(csv_path), "--kernel", "global", *extra_model], None))
            alvos.append(("CUDA (shared+SoA)", [str(cuda), str(csv_path), "--kernel", "shared", *extra_model], None))

        for nome, cmd, env in alvos:
            out_bin = tmp_dir / f"{nome.replace(' ', '_')}.bin"
            _rodar([*cmd, "--dump", str(out_bin)], env=env)
            got = _ler_bin(out_bin)
            assert len(got) == len(ref), f"{nome}: tamanho divergente ({len(got)} vs {len(ref)})"
            erro = float(np.max(np.abs(ref - got)))
            assert np.isfinite(got).all() and erro < 1e-5, f"{nome}: erro máximo {erro} >= 1e-5"
            print(f"[OK] {nome:<18s} -> erro absoluto máximo = {erro:.8g}")


def executar_benchmark(
    csv_path: Path,
    tamanhos: list[int],
    repeticoes: int,
    omp_threads: int,
    mpi_procs: int,
    model_path: Path | None,
) -> pd.DataFrame:
    seq = _exe("meteo")
    omp = _exe("meteo_omp")
    mpi = _exe("meteo_mpi")
    cuda = _exe("meteo_cuda")
    mpirun = shutil.which("mpirun")
    extra_model = ["--model", str(model_path)] if model_path and model_path.exists() else []

    linhas: list[dict[str, float | int]] = []
    for n in tamanhos:
        row: dict[str, float | int] = {"N": n}
        comuns = [str(csv_path), "--n", str(n), "--warmup", "1", "--repeat", str(repeticoes), *extra_model]

        if seq:
            out = _rodar([str(seq), *comuns])
            row["CPU_seq_ms"] = _extrair(RE_INF, out)

        if omp:
            env = {**os.environ, "OMP_NUM_THREADS": str(omp_threads)}
            out = _rodar([str(omp), *comuns], env=env)
            row[f"OpenMP_{omp_threads}t_ms"] = _extrair(RE_INF, out)

        if mpi and mpirun:
            env = {**os.environ, "OMP_NUM_THREADS": "1"}
            out = _rodar(_mpirun_cmd(mpirun, mpi_procs, mpi, comuns), env=env)
            row[f"MPI_{mpi_procs}p_comp_ms"] = _extrair(RE_MPI_C, out)
            row[f"MPI_{mpi_procs}p_total_ms"] = _extrair(RE_MPI_T, out)

        if cuda:
            out_g = _rodar([str(cuda), *comuns, "--kernel", "global"])
            row["CUDA_global_kernel_ms"] = _extrair(RE_CUDA_K, out_g)
            row["CUDA_global_total_ms"] = _extrair(RE_CUDA_T, out_g)

            out_s = _rodar([str(cuda), *comuns, "--kernel", "shared"])
            row["CUDA_shared_kernel_ms"] = _extrair(RE_CUDA_K, out_s)
            row["CUDA_shared_total_ms"] = _extrair(RE_CUDA_T, out_s)

        linhas.append(row)

    df = pd.DataFrame(linhas)
    if "CPU_seq_ms" in df.columns:
        base_col = f"OpenMP_{omp_threads}t_ms" if f"OpenMP_{omp_threads}t_ms" in df.columns else "CPU_seq_ms"
        if "CUDA_global_total_ms" in df.columns:
            df["speedup_CUDA_global"] = df[base_col] / df["CUDA_global_total_ms"]
        if "CUDA_shared_total_ms" in df.columns:
            df["speedup_CUDA_shared"] = df[base_col] / df["CUDA_shared_total_ms"]
    return df


def gerar_grafico(df: pd.DataFrame, destino: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for col in df.columns:
        if col.endswith("_ms"):
            ax.plot(df["N"], df[col], "o-", label=col.replace("_ms", ""))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Número de amostras (N)")
    ax.set_ylabel("Tempo mediano (ms) [escala log]")
    ax.set_title("MeteoPel — Comparativo de Desempenho (Seq vs. OpenMP vs. MPI vs. CUDA)")
    ax.grid(True, which="both", linestyle=":", alpha=0.6)
    ax.legend()
    plt.tight_layout()
    plt.savefig(destino, dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validação e benchmark dos binários MeteoPel.")
    parser.add_argument("--csv", type=Path, default=Path("dados_pelotas.csv"))
    parser.add_argument("--model", type=Path, default=None, help="Opcional: models/floresta_pelotas.rf")
    parser.add_argument("--threads", type=int, default=min(4, os.cpu_count() or 2))
    parser.add_argument("--mpi-procs", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--output-csv", type=Path, default=Path("resultados_benchmark_completo.csv"))
    parser.add_argument("--output-png", type=Path, default=Path("desempenho_completo.png"))
    args = parser.parse_args()

    if not args.csv.exists():
        print(f"Erro: arquivo '{args.csv}' não encontrado. Rode 'python get_meteo_data.py' antes.", file=sys.stderr)
        return 1

    print("=== 1. VALIDAÇÃO NUMÉRICA CRUZADA ===")
    validar_todos(args.csv, args.model, args.threads, args.mpi_procs)

    if args.validate_only:
        print("\nValidação concluída com sucesso.")
        return 0

    base_n = sum(1 for _ in args.csv.open("r", encoding="utf-8")) - 1
    tamanhos = sorted({base_n, 10_000, 100_000, 1_000_000})

    print("\n=== 2. BENCHMARK DE ESCALABILIDADE ===")
    df = executar_benchmark(args.csv, tamanhos, args.repeat, args.threads, args.mpi_procs, args.model)
    df.to_csv(args.output_csv, index=False)
    gerar_grafico(df, args.output_png)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.5g}"))
    print(f"\nTabela salva em '{args.output_csv}' e gráfico salvo em '{args.output_png}'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
