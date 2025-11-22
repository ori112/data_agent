# BigQuery Time-Series Agent

Course prototype:
- Streamlit conversational UI
- BigQuery time-series mart as data source
- Gemini on Vertex AI for interpretation + tool planning
- Time-series EDA + SARIMA modeling

## Features
Two intents only:
1. **EDA**:
   - log transform
   - time-series plot
   - ACF / PACF
   - seasonal decomposition
   - ADF stationarity test
   - year comparisons (AVG/MIN/MAX)

2. **MODEL**:
   - SARIMA(1,0,0)x(0,1,1,12) on log(y)
   - residual diagnostics
   - forecast plots (log + original scale)

## Run locally
```bash
uv sync
uv run streamlit run app.py
