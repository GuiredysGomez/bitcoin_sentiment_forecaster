import datetime
import pandas as pd
from binance.client import Client
from binance.exceptions import BinanceAPIException 

class BinanceMarketData:
    """
    Clase para interactuar con la API de Binance y obtener datos de mercado.
    """
    def __init__(self):
        """
        Inicializa el cliente de Binance.
        """
        self.client = Client()

    def _convertir_fecha_a_utc(self, fecha_str):
        """
        Convierte una cadena de fecha 'YYYY-MM-DD' a formato UTC (ISO 8601).
        """
        try:
            fecha_dt = datetime.datetime.strptime(fecha_str, '%Y-%m-%d')
            fecha_dt_utc = fecha_dt.replace(tzinfo=datetime.timezone.utc)
            return fecha_dt_utc.isoformat().replace('+00:00', 'Z')  
        except ValueError:
            raise ValueError(f"Formato de fecha inválido: '{fecha_str}'. Use 'YYYY-MM-DD'.")
        except Exception as e:
            raise Exception(f"Error al convertir la fecha '{fecha_str}' a UTC: {e}")

    def obtener_datos_diarios(self, simbolo: str, fecha_str: str) -> dict | None:
        """
        Obtiene los datos diarios de mercado (apertura, cierre, volumen)
        para un símbolo y fecha específicos.

        """
        try:
            start_timestamp = self._convertir_fecha_a_utc(fecha_str)

            klines = self.client.get_historical_klines(
                symbol=simbolo,
                interval=Client.KLINE_INTERVAL_1DAY,
                start_str=start_timestamp,
                limit=1
            )

            if klines:
                kline_data = klines[0]
                open_price = float(kline_data[1])
                max_price = float(kline_data[2])
                min_price = float(kline_data[3])
                close_price = float(kline_data[4])
                volume = float(kline_data[5])

                return {
                    'simbolo': simbolo,
                    'fecha': fecha_str,
                    'precio_apertura': open_price,
                    'precio_maximo': max_price,
                    'precio_minimo': min_price,
                    'precio_cierre': close_price,
                    'volumen': volume
                }
            else:
                return None
        except BinanceAPIException as bae:
            print(f"ERROR de la API para '{simbolo}' el '{fecha_str}': {bae}")
            return None
        except Exception as e:
            print(f"ERROR al obtener datos para '{simbolo}' el '{fecha_str}': {e}")
            return None


def guardar_datos_en_csv(data: dict, nombre_archivo: str):
    """
    Guarda los datos de un diccionario en un archivo CSV.
    Añade los datos como nueva fila, creando el archivo si no existe.
    """
    if not data:
        return

    df_nuevo = pd.DataFrame([data])

    try:
        df_existente = pd.read_csv(nombre_archivo)
        df_final = pd.concat([df_existente, df_nuevo], ignore_index=True)
        df_final.drop_duplicates(subset=['simbolo', 'fecha'], inplace=True)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        df_final = df_nuevo
    except Exception as e:
        print(f"Error al leer el archivo CSV '{nombre_archivo}': {e}")
        df_final = df_nuevo

    df_final.to_csv(nombre_archivo, index=False)


if __name__ == "__main__":
    binance_data_fetcher = BinanceMarketData()
    nombre_archivo_csv = "datos_mercado_binance.csv" 
    simbolo_btc = 'BTCUSDT'
    fecha_busqueda_btc = '2023-01-01'
    datos_mercado_btc = binance_data_fetcher.obtener_datos_diarios(simbolo_btc, fecha_busqueda_btc)
    if datos_mercado_btc:
        guardar_datos_en_csv(datos_mercado_btc, nombre_archivo_csv)
    else:
        print(f"No se pudieron obtener los datos para {simbolo_btc} el {fecha_busqueda_btc}.")




