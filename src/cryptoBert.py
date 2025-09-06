import torch
import time
import pandas as pd
from transformers import TextClassificationPipeline, AutoModelForSequenceClassification, AutoTokenizer

start_time = time.time()  # Inicia el temporizador
# Verificar si CUDA está disponible
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using: {device}")

# 🔹 Cargar modelo CryptoBERT desde Hugging Face y asignarlo a la GPU
model_name = "ElKulako/cryptobert"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
 # 🔹 Mueve el modelo a GPU
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3).to(device)

# 🔹 Crear pipeline de clasificación optimizado con CUDA
pipe = TextClassificationPipeline(
    model=model, 
    tokenizer=tokenizer, 
    device=0 if device == "cuda" else -1, 
    max_length=64, 
    truncation=True, 
    padding="max_length"
)

class SentimentAnalyzer:
    def batch_analyze(self, texts):
        """Analiza sentimiento en GPU con procesamiento en lotes (batch processing)"""
        responses = pipe(texts, batch_size=32)  # Procesa 32 tweets en cada iteración
        return [self._parse_response(res) for res in responses]

    def _parse_response(self, response):
        """Convierte etiquetas de CryptoBERT a valores numéricos"""
        if "Bullish" in response["label"]: return 1
        elif "Bearish" in response["label"]: return -1
        else: return 0

# 🔹 Cargar CSV con tweets
csv_path = "../data_extractor/raw_tweets.csv"
df = pd.read_csv(csv_path)

# 🔹 Procesar tweets con batch processing en GPU
analyzer = SentimentAnalyzer()
df["Sentiment"] = analyzer.batch_analyze(df["Text"].tolist())  # 🔹 Se procesa por lotes

# 🔹 Guardar resultados en un nuevo archivo
df.to_csv("../data/pre_cleaning/tweets_news_with_sentiment_cryptobert.csv", index=False)

print("✅ Processing completed with CryptoBERT on GPU. Data saved.")
end_time = time.time()
print(f"⏱️ Execution time: {end_time - start_time:.4f} seconds")