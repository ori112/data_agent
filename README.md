# data_agent

Prototype data agent for my university project, built with:

- **Python 3.12**
- **Streamlit** for the conversational UI
- **BigQuery** for data access
- **ARIMA** (statsmodels) for time-series forecasting
- **Vertex AI** (planned) for a smarter conversational agent

## Quick start (local)

```bash
# create & activate venv (uv)
uv venv
.\.venv\Scripts\activate   # on Windows

# install dependencies
uv sync

# run the app
streamlit run app.py
