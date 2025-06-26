from twikit import Client, TooManyRequests
from datetime import datetime
import csv
from random import randint
import asyncio
import os
import tracemalloc

tracemalloc.start()
MINIMUM_TWEETS = 10000
año_objetivo = 2024 # Año en que se realizará la busqueda
USER = 'APompliano'
CSV_FILENAME = 'tweets_ordenado.csv'

# Rango trimestral
RANGOS = [
    (f"{año_objetivo}-01-01", f"{año_objetivo}-03-31"),
    (f"{año_objetivo}-04-01", f"{año_objetivo}-06-30"),
    (f"{año_objetivo}-07-01", f"{año_objetivo}-09-30"),
    (f"{año_objetivo}-10-01", f"{año_objetivo}-12-31"),
]

async def recolectar_tweets(desde, hasta, writer, tweet_start_count):
    query = f"(bitcoin OR BTC) (from:{USER}) until:{hasta} since:{desde}"
    client = Client(language='en-US')
    try:
        client.load_cookies('cookies.json')
        print(f'{datetime.now()} - Cookies cargadas exitosamente.')
    except Exception as e:
        print(f'{datetime.now()} - Error al cargar cookies: {e}')

    tweet_count = tweet_start_count
    tweets_result = None
    print(f'{datetime.now()} - Iniciando búsqueda para: {query}')

    while True:
        try:
            if tweets_result is None:
                tweets_result = await client.search_tweet(query, product='Top')
            else:
                wait_time = randint(5, 10)
                print(f'{datetime.now()} - Esperando {wait_time}s para siguiente página...')
                await asyncio.sleep(wait_time)
                tweets_result = await tweets_result.next()

            if not tweets_result:
                print(f'{datetime.now()} - No hay más resultados para este rango.')
                break

        except TooManyRequests as e:
            wait_time_seconds = max(60, (datetime.fromtimestamp(e.rate_limit_reset) - datetime.now()).total_seconds())
            print(f'{datetime.now()} - Límite alcanzado, esperando {wait_time_seconds:.0f}s...')
            await asyncio.sleep(wait_time_seconds)
            continue
        except Exception as e:
            print(f'{datetime.now()} - Error inesperado: {e}')
            break

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

        if processed_in_batch > 0:
            print(f'{datetime.now()} - Procesados {processed_in_batch} tweets. Total acumulado: {tweet_count}')
        else:
            break

    return tweet_count

async def main():
    write_header = not os.path.exists(CSV_FILENAME) or os.path.getsize(CSV_FILENAME) == 0
    tweet_counter = 0

    with open(CSV_FILENAME, 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if write_header:
            writer.writerow(['Tweet_count', 'Username', 'Text', 'Created At', 'Retweets', 'Likes'])
            print(f'{datetime.now()} - Archivo CSV "{CSV_FILENAME}" creado con encabezado.')
        else:
            print(f'{datetime.now()} - Añadiendo al archivo CSV existente "{CSV_FILENAME}".')

        for desde, hasta in RANGOS:
            tweet_counter = await recolectar_tweets(desde, hasta, writer, tweet_counter)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"Ocurrió un error durante la ejecución: {e}")