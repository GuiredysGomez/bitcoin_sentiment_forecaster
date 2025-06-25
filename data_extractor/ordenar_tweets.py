import pandas as pd

# Cargar el archivo CSV
df = pd.read_csv("tweets_ordenado.csv")

# Limpiar nombres de columnas
df.columns = df.columns.str.strip()

# Mostrar cuántos duplicados hay
print("Duplicados antes:", df.duplicated().sum())

# Eliminar duplicados
df_sin_duplicados = df.drop_duplicates()

# Ordenar por la columna 'Username'
df_ordenado = df_sin_duplicados.sort_values(by="Username", ascending=True)

# Guardar el DataFrame limpio y ordenado
df_ordenado.to_csv("tweets_ordenado.csv", index=False)

# Verificar si quedan duplicados
print("Duplicados después:", df_ordenado.duplicated().sum())
