"""
Treinamento de Random Forest real (100 árvores x 15 nós) para previsão de chuva
na próxima hora em Pelotas, RS, exportando diretamente para a ABI binária de
30.400 bytes lida por meteo, meteo_omp, meteo_mpi e meteo_cuda.

Fontes suportadas:
  1. Estação Agroclimatológica de Pelotas / Capão do Leão (INMET A887 - Convênio
     UFPel / Embrapa Clima Temperado / INMET), via CSV, pasta ou ZIP anual:
       python train_forest_pelotas.py --inmet dados_inmet_a887/
  2. API Histórica Open-Meteo (padrão automático sem download manual):
       python train_forest_pelotas.py --years 3
"""

from __future__ import annotations

import argparse
import io
import struct
import sys
import unicodedata
import zipfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

NUM_FEATURES = 5
NUM_TREES = 100
MAX_NODES_PER_TREE = 15
FOREST_BYTES = NUM_TREES * MAX_NODES_PER_TREE * 20 + NUM_TREES * 4  # 30.400 bytes

FEATURE_COLS = [
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "delta_pressao",
    "wind_speed_10m",
]


def _norm(texto: str) -> str:
    sem_acento = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode("ascii")
    return " ".join(sem_acento.upper().split())


def _encontrar_coluna(colunas: list[str], termos: list[str]) -> str:
    for col in colunas:
        norm_col = _norm(col)
        if all(t in norm_col for t in termos):
            return col
    raise KeyError(f"Coluna não encontrada para os termos {termos}. Disponíveis: {colunas}")


def _ler_csv_inmet_stream(stream: io.TextIOBase) -> pd.DataFrame:
    df_raw = pd.read_csv(
        stream,
        sep=";",
        decimal=",",
        skiprows=8,
        na_values=["", " ", "null", "NULL", "-9999", "-9999.0", "9999", "9999.0"],
        low_memory=False,
    )
    cols = list(df_raw.columns)
    c_temp = _encontrar_coluna(cols, ["TEMPERATURA", "BULBO SECO"])
    c_umid = _encontrar_coluna(cols, ["UMIDADE RELATIVA", "HORARIA"])
    c_pres = _encontrar_coluna(cols, ["PRESSAO ATMOSFERICA", "NIVEL DA ESTACAO"])
    c_vent = _encontrar_coluna(cols, ["VENTO", "VELOCIDADE HORARIA"])
    c_prec = _encontrar_coluna(cols, ["PRECIPITACAO TOTAL", "HORARIO"])

    df = pd.DataFrame(
        {
            "temperature_2m": pd.to_numeric(df_raw[c_temp], errors="coerce"),
            "relative_humidity_2m": pd.to_numeric(df_raw[c_umid], errors="coerce"),
            "surface_pressure": pd.to_numeric(df_raw[c_pres], errors="coerce"),
            # INMET registra vento em m/s; projeto espera km/h (m/s * 3.6)
            "wind_speed_10m": pd.to_numeric(df_raw[c_vent], errors="coerce") * 3.6,
            "precipitation": pd.to_numeric(df_raw[c_prec], errors="coerce"),
        }
    )
    df.loc[df["precipitation"] < 0, "precipitation"] = np.nan
    df.loc[df["relative_humidity_2m"] < 0, "relative_humidity_2m"] = np.nan
    df.loc[df["surface_pressure"] < 800, "surface_pressure"] = np.nan
    return df


def carregar_dados_inmet(caminho: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    def processar_zip(zp: Path) -> None:
        with zipfile.ZipFile(zp) as zf:
            candidatos = [
                nome
                for nome in zf.namelist()
                if nome.lower().endswith(".csv") and ("a887" in nome.lower() or "pelotas" in nome.lower())
            ]
            if not candidatos:
                raise FileNotFoundError(
                    f"Nenhum CSV da estação A887 (Capão do Leão / Pelotas) encontrado em '{zp}'."
                )
            for nome in sorted(candidatos):
                with zf.open(nome) as raw:
                    texto = io.TextIOWrapper(raw, encoding="latin1", errors="replace")
                    frames.append(_ler_csv_inmet_stream(texto))

    if caminho.is_file():
        if caminho.suffix.lower() == ".zip":
            processar_zip(caminho)
        else:
            with caminho.open("r", encoding="latin1", errors="replace") as f:
                frames.append(_ler_csv_inmet_stream(f))
    elif caminho.is_dir():
        for arq in sorted(caminho.rglob("*")):
            if arq.suffix.lower() == ".zip":
                processar_zip(arq)
            elif arq.suffix.lower() == ".csv" and ("a887" in arq.name.lower() or "pelotas" in arq.name.lower()):
                with arq.open("r", encoding="latin1", errors="replace") as f:
                    frames.append(_ler_csv_inmet_stream(f))
    else:
        raise FileNotFoundError(f"Caminho INMET não encontrado: {caminho}")

    if not frames:
        raise RuntimeError(f"Nenhum dado válido da estação A887 carregado de '{caminho}'.")

    df = pd.concat(frames, ignore_index=True)
    df["delta_pressao"] = df["surface_pressure"].diff(3)
    df["target_next_hour_mm"] = df["precipitation"].shift(-1)
    return df[[*FEATURE_COLS, "target_next_hour_mm"]].dropna().reset_index(drop=True)


def carregar_dados_open_meteo_archive(years: int = 3) -> pd.DataFrame:
    fim = date.today() - timedelta(days=2)
    inicio = fim - timedelta(days=365 * years)
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": -31.765,
        "longitude": -52.337,
        "start_date": inicio.isoformat(),
        "end_date": fim.isoformat(),
        "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,precipitation",
        "timezone": "America/Sao_Paulo",
    }
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()["hourly"]
    df = pd.DataFrame(data)
    df["delta_pressao"] = df["surface_pressure"].diff(3)
    df["target_next_hour_mm"] = df["precipitation"].shift(-1)
    return df[[*FEATURE_COLS, "target_next_hour_mm"]].dropna().reset_index(drop=True)


def exportar_floresta_bin(modelo: RandomForestRegressor, destino: Path) -> bytes:
    """
    Converte cada árvore (max_depth=3) em uma árvore binária completa canônica
    de 15 nós (0..6 internos, 7..14 folhas) compatível com struct RandomForest.
    """
    assert len(modelo.estimators_) == NUM_TREES, f"Esperadas {NUM_TREES} árvores."

    nodes_buf = bytearray()
    roots: list[int] = []

    for t, est in enumerate(modelo.estimators_):
        tree = est.tree_
        tree_start = t * MAX_NODES_PER_TREE
        roots.append(tree_start)

        # Mapeia cada posição heap 0..14 para o id de nó correspondente no sklearn
        heap_to_sk = [-1] * MAX_NODES_PER_TREE
        heap_to_sk[0] = 0
        for h in range(7):
            sk_id = heap_to_sk[h]
            left_sk = int(tree.children_left[sk_id])
            right_sk = int(tree.children_right[sk_id])
            if left_sk == -1 or right_sk == -1:
                # Nó já é folha antes do nível 3: propaga a mesma folha para ambos os filhos
                heap_to_sk[2 * h + 1] = sk_id
                heap_to_sk[2 * h + 2] = sk_id
            else:
                heap_to_sk[2 * h + 1] = left_sk
                heap_to_sk[2 * h + 2] = right_sk

        for h in range(MAX_NODES_PER_TREE):
            sk_id = heap_to_sk[h]
            if h < 7:
                is_leaf = int(tree.children_left[sk_id]) == -1
                if is_leaf:
                    feat = 0
                    thr = 0.0
                else:
                    feat = int(tree.feature[sk_id])
                    thr = float(np.float32(tree.threshold[sk_id]))
                left = tree_start + (2 * h + 1)
                right = tree_start + (2 * h + 2)
                nodes_buf += struct.pack("<iffii", feat, thr, 0.0, left, right)
            else:
                val = float(np.float32(tree.value[sk_id][0, 0]))
                nodes_buf += struct.pack("<iffii", -1, 0.0, val, -1, -1)

    roots_buf = struct.pack(f"<{NUM_TREES}i", *roots)
    blob = bytes(nodes_buf + roots_buf)
    assert len(blob) == FOREST_BYTES, f"Tamanho inesperado: {len(blob)} != {FOREST_BYTES}"

    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_bytes(blob)
    return blob


def inferir_blob_python(blob: bytes, X: np.ndarray) -> np.ndarray:
    node_dt = np.dtype([("feat", "<i4"), ("thr", "<f4"), ("val", "<f4"), ("l", "<i4"), ("r", "<i4")])
    nodes = np.frombuffer(blob[:30000], dtype=node_dt)
    roots = np.frombuffer(blob[30000:], dtype="<i4")
    preds = np.zeros(len(X), dtype=np.float32)

    for t in range(NUM_TREES):
        k = np.full(len(X), roots[t], dtype=np.int32)
        for _ in range(3):
            feat = nodes["feat"][k]
            thr = nodes["thr"][k]
            vals = X[np.arange(len(X)), feat]
            k = np.where(vals <= thr, nodes["l"][k], nodes["r"][k])
        preds += nodes["val"][k]

    return preds / np.float32(NUM_TREES)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Treina Random Forest real para Pelotas (RS) e exporta para ABI C/CUDA (30.400 bytes)."
    )
    parser.add_argument(
        "--inmet",
        type=Path,
        default=None,
        help="CSV, ZIP anual ou pasta com dados da Estação A887 (UFPel/Embrapa/INMET)",
    )
    parser.add_argument(
        "--years",
        type=int,
        default=3,
        help="Anos de histórico via Open-Meteo Archive caso --inmet não seja informado (padrão: 3)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/floresta_pelotas.rf"),
        help="Arquivo binário de saída (padrão: models/floresta_pelotas.rf)",
    )
    args = parser.parse_args()

    try:
        if args.inmet is not None:
            print(f"Carregando série histórica da Estação A887 (UFPel/Embrapa/INMET) de: {args.inmet}")
            df = carregar_dados_inmet(args.inmet)
            origem = f"INMET Estação A887 ({args.inmet})"
        else:
            print(f"Consultando série histórica Open-Meteo Archive ({args.years} anos) para Pelotas, RS...")
            df = carregar_dados_open_meteo_archive(years=args.years)
            origem = f"Open-Meteo Archive ({args.years} anos, Pelotas/RS)"
    except Exception as exc:
        print(f"Erro ao carregar dados de treinamento: {exc}", file=sys.stderr)
        return 1

    X = np.ascontiguousarray(df[FEATURE_COLS].to_numpy(dtype=np.float32))
    y = df["target_next_hour_mm"].to_numpy(dtype=np.float32)

    # Divisão temporal (80% passado para treino, 20% mais recente para teste)
    corte = int(len(X) * 0.8)
    X_train, X_test = X[:corte], X[corte:]
    y_train, y_test = y[:corte], y[corte:]

    print(f"Origem: {origem}")
    print(f"Amostras válidas: {len(X)} (treino: {len(X_train)}, teste temporal: {len(X_test)})")
    print(f"Taxa de horas com chuva (>= 1.0 mm/h) no histórico: {(y >= 1.0).mean() * 100:.2f}%")

    modelo = RandomForestRegressor(
        n_estimators=NUM_TREES,
        max_depth=3,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1,
    )
    modelo.fit(X_train, y_train)

    y_pred_sk = modelo.predict(X_test).astype(np.float32)
    blob = exportar_floresta_bin(modelo, args.output)
    y_pred_bin = inferir_blob_python(blob, X_test)

    erro_export = float(np.max(np.abs(y_pred_sk - y_pred_bin)))
    mae = float(mean_absolute_error(y_test, y_pred_bin))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_bin)))

    print(f"\nModelo exportado com sucesso: {args.output} ({len(blob)} bytes)")
    print(f"Erro máximo de exportação (sklearn vs. ABI 15 nós): {erro_export:.8g}")
    print(f"Métricas no teste temporal -> MAE: {mae:.4f} mm/h | RMSE: {rmse:.4f} mm/h")
    print(f"Faixa prevista no teste: {y_pred_bin.min():.2f} a {y_pred_bin.max():.2f} mm/h")
    assert erro_export < 1e-4, "Divergência inesperada na exportação binária!"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
