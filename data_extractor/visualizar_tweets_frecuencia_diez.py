import pandas as pd
import matplotlib.pyplot as plt

# Parámetro: año que quieres analizar
año_objetivo = 2025  # Cambia este valor si quieres otro año

# Cargar el CSV
df = pd.read_csv("tweets_ordenado.csv")

# Asegurarse de que la columna de fecha esté en formato datetime
df['Created At'] = pd.to_datetime(df['Created At'], errors='coerce')

# Filtrar por el año deseado
df_filtrado = df[df['Created At'].dt.year == año_objetivo].copy()

# Establecer la fecha como índice
df_filtrado.set_index('Created At', inplace=True)

# Agrupar cada 10 días y contar tweets
tweets_10d = df_filtrado.resample('10D').size()

# Graficar
tweets_10d.plot(kind='bar', figsize=(12, 6), color='darkorange')
plt.title(f'Tweets cada 10 días en {año_objetivo}')
plt.xlabel('Fecha de inicio del intervalo')
plt.ylabel('Número de tweets')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.show()