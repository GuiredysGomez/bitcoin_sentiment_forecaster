from datetime import datetime, timezone
import pandas as pd
from binance.client import Client
from binance.exceptions import BinanceAPIException

class BinanceMarketData:
    """
    Clase para interactuar con la API de Binance y obtener datos de mercado.
    """
    def __init__(self):
        self.client = Client()

    def obtener_datos_rango(self, simbolo: str, fecha_inicio: str, fecha_fin: str, intervalo: str):
        """
        Obtiene datos de mercado para un símbolo entre dos fechas usando el intervalo especificado.
        """
        try:
            klines = self.client.get_historical_klines(
                symbol=simbolo,
                interval=intervalo,
                start_str=fecha_inicio,
                end_str=fecha_fin
            )

            datos = []
            for kline in klines:
                fecha_utc = datetime.fromtimestamp(kline[0] / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
                datos.append({
                    'simbolo': simbolo,
                    'fecha': fecha_utc,
                    'precio_apertura': float(kline[1]),
                    'precio_maximo': float(kline[2]),
                    'precio_minimo': float(kline[3]),
                    'precio_cierre': float(kline[4]),
                    'volumen': float(kline[5])
                })
            return datos

        except BinanceAPIException as bae:
            print(f"ERROR API Binance: {bae}")
            return []
        except Exception as e:
            print(f"Error general al obtener datos de Binance: {e}")
            return []

def guardar_datos_en_csv(lista_datos: list, nombre_archivo: str):
    """
    Guarda una lista de diccionarios en un archivo CSV.
    """
    if not lista_datos:
        return

    df_nuevo = pd.DataFrame(lista_datos)

    try:
        df_existente = pd.read_csv(nombre_archivo)
        df_final = pd.concat([df_existente, df_nuevo], ignore_index=True)
        df_final.drop_duplicates(subset=['simbolo', 'fecha'], inplace=True)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        df_final = df_nuevo
    except Exception as e:
        print(f"Error al leer el CSV '{nombre_archivo}': {e}")
        df_final = df_nuevo

    df_final.to_csv(nombre_archivo, index=False)

if __name__ == "__main__":
    binance_data_fetcher = BinanceMarketData()
    nombre_archivo_csv = "../data/pre_cleaning/data_binance.csv"
    simbolo_btc = 'BTCUSDT'
    simbolo_btc = 'BTCUSDT'
    
    # Ajusta las fechas según tu necesidad
    fecha_inicio = '2020-01-01'
    fecha_fin = '2022-12-31'

    
    # Intervalos disponibles: mensual, semanal, diario
    intervalo = Client.KLINE_INTERVAL_1DAY
    
    datos_btc = binance_data_fetcher.obtener_datos_rango(simbolo_btc, fecha_inicio, fecha_fin, intervalo)
    guardar_datos_en_csv(datos_btc, nombre_archivo_csv)