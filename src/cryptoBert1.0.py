import torch
import time
import pandas as pd
from transformers import TextClassificationPipeline, AutoModelForSequenceClassification, AutoTokenizer

start_time = time.time()  # 🔹 Inicia el temporizador
# 🔹 Verificar si CUDA está disponible
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando: {device}")

# 🔹 Cargar modelo CryptoBERT desde Hugging Face y asignarlo a la GPU
model_name = "ElKulako/cryptobert"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3).to(device)  # 🔹 Mueve el modelo a GPU

# 🔹 Crear pipeline de clasificación con CUDA
pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, device=0 if device == "cuda" else -1, max_length=64, truncation=True, padding="max_length")

class SentimentAnalyzer:
     def analyze(self, text):
        """Analiza el sentimiento usando CryptoBERT"""
        response = pipe(text)[0]["label"]
        return self._parse_response(response)

     def _parse_response(self, response):
        """Convierte etiquetas de CryptoBERT a valores numéricos"""
        if "Bullish" in response: return 1
        elif "Bearish" in response: return -1
        else: return 0

# 🔹 Cargar CSV con tweets
csv_path = "data/tweets.csv"  # Ruta del archivo CSV
df = pd.read_csv(csv_path)

# 🔹 Procesar tweets con análisis de sentimiento en GPU
analyzer = SentimentAnalyzer()
df["Sentiment"] = df["Text"].apply(analyzer.analyze)

# 🔹 Guardar resultados en un nuevo archivo
df.to_csv("tweets_with_sentiment_cryptobert.csv", index=False)

print("✅ Procesamiento completado con CryptoBERT en GPU. Datos guardados.")
end_time = time.time()
print(f"⏱️ Execution time: {end_time - start_time:.4f} seconds")
