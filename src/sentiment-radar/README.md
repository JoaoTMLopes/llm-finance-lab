# Sentiment Radar

Pipeline em Python que recolhe notícias financeiras em tempo real (via Yahoo Finance),
classifica o sentimento com o modelo FinBERT (ProsusAI/finbert) e cruza os resultados
com o preço de mercado das ações correspondentes.

## Como funciona
1. `get_headlines()` recolhe as últimas notícias de cada ticker.
2. `classificar_noticias()` aplica o FinBERT (positive/negative/neutral) a cada título.
3. Os scores são agregados por dia e ticker, e acumulados num histórico local.
4. `matplotlib` gera gráficos comparando sentimento médio diário com o preço de fecho.

## Stack
Python, transformers (Hugging Face), yfinance, pandas, matplotlib.

## Como correr
Ver `notebooks/sentiment_radar_demo.ipynb` para um exemplo completo, do zero ao gráfico.

## Limitações conhecidas
- O `.news` do yfinance é uma API não-oficial e pode sofrer rate-limiting ou instabilidade.
- FinBERT não capta ironia/sarcasmo e pode ser enganado por manchetes ambíguas.
- Amostra pequena de notícias por dia (10 por ticker) — não é estatisticamente robusto, é uma prova de conceito.

## Próximos passos
- Explicabilidade com SHAP (mostrar que palavras pesaram na classificação).
- Expandir para mais tickers e maior histórico.
