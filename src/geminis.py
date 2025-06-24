import time
import pandas as pd
import google.generativeai as genai  # API de Gemini

class SentimentAnalyzer:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)  # Configurar API Key
        self.llm = genai.GenerativeModel("gemini-1.5-flash")  # Modelo de Gemini
        
        # 🔹 Few-Shot Prompting con ejemplos detallados
        self.prompt_template = """ 
        You are a cryptocurrency market analyst. Classify the sentiment of the following tweet as:
        [POSITIVE/NEGATIVE/NEUTRAL]. Consider:
        1. Terms like "FOMO", "bankruptcy", "rally", "regulation".
        2. Key events: ETFs, halving, hacks.
        3. Justify in 10 words maximum.

        Examples:
        - "Binance hacked: $200M lost" → NEGATIVE (hack)
        - "SEC approves Bitcoin ETF" → POSITIVE (institutional adoption)
        - "Bitcoin struggles under regulatory scrutiny" → NEUTRAL (uncertainty)

        Tweet: {text}
        """

    def analyze(self, text):
        prompt = self.prompt_template.format(text=text)
        response = self.llm.generate_content(prompt)  # Gemini genera respuesta
        return self._parse_response(response.text)

    def _parse_response(self, response):
        """Extrae la clasificación POS/NEG/NEUT de la respuesta del LLM"""
        if "POSITIVE" in response: return 1
        elif "NEGATIVE" in response: return -1
        else: return 0

# 🔹 Cargar CSV con tweets
csv_path = "data\\tweets.csv"  # Ruta del archivo CSV
df = pd.read_csv(csv_path)
# print(df.head()) 

# 🔹 Procesar tweets con análisis de sentimiento
api_key = "AIzaSyC-vXWZiouZ-PV91bf2AtUoGl62RSibXa8"  # Sustituye con tu API Key de Gemini
analyzer = SentimentAnalyzer(api_key)
for index, row in df.iterrows():
    df.loc[index, "Sentiment"] = analyzer.analyze(row["Text"])
    time.sleep(3)  # Espera 3 segundos entre cada solicitud


#df["Sentiment"] = df["Text"].apply(analyzer.analyze)

# 🔹 Guardar resultados en un nuevo archivo
df.to_csv("tweets_with_sentiment.csv", index=False)
