"""
sentiment-radar/src/pipeline.py
Pipeline: notícias (yfinance) -> classificação FinBERT -> agregação diária -> histórico
"""
import os
import time
import pandas as pd
import yfinance as yf
from transformers import pipeline

def carregar_modelo():
    return pipeline("sentiment-analysis", model="ProsusAI/finbert", tokenizer="ProsusAI/finbert")

def get_headlines(ticker_symbol, max_results=10):
    ticker = yf.Ticker(ticker_symbol)
    try:
        news_items = ticker.news
    except Exception as e:
        print(f"Erro ao obter notícias de {ticker_symbol}: {e}")
        return []
    headlines = []
    for item in news_items[:max_results]:
        content = item.get("content", {})
        title = content.get("title")
        if title:
            headlines.append({
                "ticker": ticker_symbol,
                "title": title,
                "publisher": content.get("provider", {}).get("displayName", "desconhecido"),
                "published": content.get("pubDate")
            })
    return headlines

def classificar_noticias(headlines, modelo):
    if not headlines:
        return pd.DataFrame()
    titulos = [h["title"] for h in headlines]
    resultados = modelo(titulos)
    for h, r in zip(headlines, resultados):
        h["sentimento"] = r["label"]
        h["score"] = r["score"]
    return pd.DataFrame(headlines)

def score_numerico(row):
    if row["sentimento"] == "positive":
        return row["score"]
    elif row["sentimento"] == "negative":
        return -row["score"]
    return 0.0

def coletar_e_classificar(tickers, modelo, pausa=2):
    todas_noticias = []
    for t in tickers:
        resultado = get_headlines(t)
        todas_noticias.extend(resultado)
        time.sleep(pausa)
    df = classificar_noticias(todas_noticias, modelo)
    if not df.empty:
        df["score_sinal"] = df.apply(score_numerico, axis=1)
        df["published"] = pd.to_datetime(df["published"])
        df["data"] = df["published"].dt.date
    return df

def guardar_historico(df_novo, caminho):
    if os.path.exists(caminho):
        df_existente = pd.read_csv(caminho)
        df_combinado = pd.concat([df_existente, df_novo], ignore_index=True)
        df_combinado.drop_duplicates(subset=["ticker", "title", "published"], inplace=True)
    else:
        df_combinado = df_novo
    df_combinado.to_csv(caminho, index=False)
    return df_combinado
