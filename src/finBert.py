import torch
import time
import pandas as pd
from transformers import TextClassificationPipeline, AutoModelForSequenceClassification, AutoTokenizer

start_time = time.time()  # 🔹 Inicia el temporizador

# 🔹 Verificar si CUDA está disponible
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using: {device}")

# 🔹 Cargar modelo FinBERT desde Hugging Face y asignarlo a la GPU
model_name = "yiyanghkust/finbert-tone"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3).to(device)

# 🔹 Crear pipeline de clasificación optimizado con CUDA
pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, device=0 if device == "cuda" else -1, max_length=128, truncation=True, padding="max_length")

class SentimentAnalyzer:
    def batch_analyze(self, texts):
        """Analyze sentiment using Few-Shot Prompting with FinBERT in batch mode"""
        
        # 🔹 Few-Shot Prompt Structure optimized for financial news
        prompts = [
            f"""
            With the following examples I want you to help in the sentiment analysis of the following tweets

            Example 1:
            Text: "Bitcoin reached an all-time high of $70K, boosting investor confidence."
            Sentiment: Positive

            Example 2:
            Text: "Ethereum dropped by 15% amid market uncertainty."
            Sentiment: Negative

            Example 3:
            Text: "The cryptocurrency market remained stable with minimal fluctuations."
            Sentiment: Neutral

            Analyze the sentiment of the following tweet
            Text: "{text}"
            Sentiment:
            """ 
            for text in texts
        ]

        responses = pipe(prompts, batch_size=32)  # 🔹 Procesa 32 textos por lote
        return [self._parse_response(res) for res in responses]

    def _parse_response(self, response):
        """Convert FinBERT sentiment labels into numerical values"""
        if "positive" in response["label"]: return 1
        elif "negative" in response["label"]: return -1
        else: return 0

# 🔹 Cargar CSV con noticias financieras
csv_path = "data/tweets.csv"
df = pd.read_csv(csv_path)

# 🔹 Procesar noticias con batch processing en GPU
analyzer = SentimentAnalyzer()
df["Sentiment"] = analyzer.batch_analyze(df["Text"].tolist())  # 🔹 Se procesa por lotes

# 🔹 Guardar resultados en un nuevo archivo
df.to_csv("tweets_with_sentiment_finbert.csv", index=False)

end_time = time.time()
print("✅ Processing completed with FinBERT & Few-Shot Prompting. Data saved.")
print(f"⏱️ Execution time: {end_time - start_time:.4f} seconds")