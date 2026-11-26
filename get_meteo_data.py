import argparse
import sys
import pandas as pd
import requests

URL = "https://api.open-meteo.com/v1/forecast"
COLS = [
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "delta_pressao",
    "wind_speed_10m",
]


def coletar_dados(
    latitude: float = -31.765,
    longitude: float = -52.337,
    past_days: int = 14,
    forecast_days: int = 1,
    output_csv: str = "dados_pelotas.csv",
    save_timestamps: str | None = None,
) -> pd.DataFrame:
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m",
        "past_days": past_days,
        "forecast_days": forecast_days,
        "timezone": "America/Sao_Paulo",
    }

    response = requests.get(URL, params=params, timeout=20)
    response.raise_for_status()
    payload = response.json()
    if "hourly" not in payload:
        raise RuntimeError("Resposta da API Open-Meteo não contém o campo 'hourly'.")

    df = pd.DataFrame(payload["hourly"])
    df["delta_pressao"] = df["surface_pressure"].diff(3)

    df_valid = df.dropna(subset=COLS).reset_index(drop=True)
    df_c = df_valid[COLS]
    df_c.to_csv(output_csv, index=False, header=True)

    if save_timestamps:
        df_valid[["time", *COLS]].to_csv(save_timestamps, index=False, header=True)

    print(f"Arquivo '{output_csv}' gerado com sucesso contendo {len(df_c)} linhas!")
    return df_c


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Coleta dados meteorológicos horários de Pelotas (RS) via API Open-Meteo."
    )
    parser.add_argument("--lat", type=float, default=-31.765, help="Latitude (padrão: Pelotas -31.765)")
    parser.add_argument("--lon", type=float, default=-52.337, help="Longitude (padrão: Pelotas -52.337)")
    parser.add_argument("--past-days", type=int, default=14, help="Dias históricos (padrão: 14)")
    parser.add_argument("--forecast-days", type=int, default=1, help="Dias de previsão (padrão: 1)")
    parser.add_argument("--output", default="dados_pelotas.csv", help="Arquivo CSV de saída")
    parser.add_argument(
        "--save-timestamps",
        default=None,
        help="Opcional: salva CSV auxiliar incluindo a coluna 'time'",
    )
    args = parser.parse_args()

    try:
        coletar_dados(
            latitude=args.lat,
            longitude=args.lon,
            past_days=args.past_days,
            forecast_days=args.forecast_days,
            output_csv=args.output,
            save_timestamps=args.save_timestamps,
        )
    except requests.RequestException as exc:
        print(f"Erro ao consultar a API Open-Meteo: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Erro ao processar dados meteorológicos: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
