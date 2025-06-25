import pandas as pd
import matplotlib.pyplot as plt

# Cargar el CSV
df = pd.read_csv("tweets_ordenado.csv")

# Asegurarse de que la columna de fecha esté en formato datetime
df['Created At'] = pd.to_datetime(df['Created At'], errors='coerce')

# Crear una columna con el año y mes (formato AAAA-MM)
df['año_mes'] = df['Created At'].dt.to_period('M').astype(str)

# Contar tweets por mes
tweets_por_mes = df['año_mes'].value_counts().sort_index()

# Graficar
tweets_por_mes.plot(kind='bar', figsize=(12, 6), color='mediumseagreen')
plt.title('Cantidad de tweets por mes')
plt.xlabel('Mes')
plt.ylabel('Número de tweets')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.show()