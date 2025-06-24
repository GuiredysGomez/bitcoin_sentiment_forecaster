from twikit import Client, TooManyRequests
from datetime import datetime
import csv
from configparser import ConfigParser
from random import randint
import asyncio
import os
import tracemalloc

tracemalloc.start()

MINIMUM_TWEETS = 1000
QUERY = 'bitcoin (from:KuCoinUpdates) until:2025-05-10 since:2025-01-20'
CSV_FILENAME = 'tweets.csv'

async def main():
    """Función principal asíncrona para ejecutar el proceso."""

    client = Client(language='en-US')
    try:
        client.load_cookies('cookies.json')
        print(f'{datetime.now()} - Cookies cargadas exitosamente.')
    except Exception as e:
        print(f'{datetime.now()} - Error al cargar cookies: {e}')

    # crear o abrir el archivo csv
    write_header = not os.path.exists(CSV_FILENAME) or os.path.getsize(CSV_FILENAME) == 0
    with open(CSV_FILENAME, 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if write_header:
            writer.writerow(['Tweet_count', 'Username', 'Text', 'Created At', 'Retweets', 'Likes'])
            print(f'{datetime.now()} - Archivo CSV "{CSV_FILENAME}" creado con encabezado.')
        else:
             print(f'{datetime.now()} - Añadiendo al archivo CSV existente "{CSV_FILENAME}".')

        tweet_count = 0
        tweets_result = None 

        print(f'{datetime.now()} - Iniciando la recolección de tweets (objetivo al menos {MINIMUM_TWEETS} tweets para "{QUERY}")...')

        # Bucle principal para obtener tweets hasta alcanzar el mínimo o agotar resultados
        while tweet_count < MINIMUM_TWEETS:
            try:
                if tweets_result is None:
                    # Búsqueda inicial
                    print(f'{datetime.now()} - Realizando búsqueda inicial para "{QUERY}"...')
                    tweets_result = await client.search_tweet(QUERY, product='Top')
                else:
                    # Obtener la siguiente página
                    wait_time = randint(5, 10)
                    print(f'{datetime.now()} - Esperando {wait_time} segundos antes de obtener la siguiente página...')
                    await asyncio.sleep(wait_time)
                    print(f'{datetime.now()} - Obteniendo la siguiente página de tweets...')
                    tweets_result = await tweets_result.next()

                if not tweets_result:
                    print(f'{datetime.now()} - La búsqueda no devolvió más resultados (objeto de resultado vacío o None).')
                    break 
            except TooManyRequests as e:
                rate_limit_reset = datetime.fromtimestamp(e.rate_limit_reset)
                print(f'{datetime.now()} - Límite de peticiones alcanzado. Esperando hasta {rate_limit_reset.strftime("%Y-%m-%d %H:%M:%S")}')
                wait_time_seconds = (rate_limit_reset - datetime.now()).total_seconds()
                if wait_time_seconds < 1:
                    wait_time_seconds = 60 # Esperar al menos un minuto si el reinicio es inmediato
                print(f'{datetime.now()} - Esperando {wait_time_seconds:.2f} segundos...')
                await asyncio.sleep(wait_time_seconds)
                continue # Continúa el bucle while para intentar obtener de nuevo
            except Exception as e:
                 print(f'{datetime.now()} - Ocurrió un error durante la obtención de tweets: {e}')
                 break 

            # Procesar tweets de la página actual
            processed_in_batch = 0
            for tweet in tweets_result:
                tweet_count += 1
                processed_in_batch += 1

                tweet_data = [
                    tweet_count,
                    getattr(tweet.user, 'name', 'N/A') if getattr(tweet, 'user', None) else 'N/A',
                    getattr(tweet, 'text', 'N/A'),
                    getattr(tweet, 'created_at', 'N/A'),
                    getattr(tweet, 'retweet_count', 0),
                    getattr(tweet, 'favorite_count', 0)
                ]

                writer.writerow(tweet_data)

                # Detener el procesamiento si se alcanza el objetivo mínimo dentro de este lote
                if tweet_count >= MINIMUM_TWEETS:
                    break 
            # Imprimir progreso solo si se procesaron tweets en este lote
            if processed_in_batch > 0:
                print(f'{datetime.now()} - Procesados {processed_in_batch} tweets de este lote. Total recolectados: {tweet_count}')
            else:
                print(f'{datetime.now()} - El lote no devolvió tweets nuevos, pero se recibió un objeto de resultado válido.')


        print(f'{datetime.now()} - Bucle de recolección de tweets finalizado. Total de tweets recolectados: {tweet_count}')


# Ejecutar la función principal asíncrona
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"Ocurrió un error durante la ejecución: {e}")



