# app.py

from __future__ import annotations
import os

import streamlit as st
from dotenv import load_dotenv

from bigquery_client import BigQueryClient
from utils import (
    AppConfig,
    fit_arima_and_forecast,
    forecast_to_dataframe,
    handle_user_message,
)

# Load .env file
load_dotenv()


def init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state["messages"] = []


def main() -> None:
    st.set_page_config(
        page_title="BigQuery Time-Series Agent",
        page_icon="📈",
        layout="wide",
    )

    st.title("📈 BigQuery Time-Series Agent (Prototype)")
    st.caption(
        "This app uses Streamlit, BigQuery, ARIMA, and (later) Vertex AI.\n"
        "Currently running in local development mode using .env variables."
    )

    init_session_state()
    config = AppConfig.from_env()

    # -----------------------------
    # SIDEBAR CONFIG
    # -----------------------------
    st.sidebar.header("BigQuery Configuration")

    project_id = st.sidebar.text_input("GCP Project ID", value=config.project_id)
    dataset_id = st.sidebar.text_input("Dataset ID", value=config.dataset_id)
    table_id = st.sidebar.text_input("Table ID", value=config.table_id)

    date_column = st.sidebar.text_input("Date column", value=config.date_column)
    target_column = st.sidebar.text_input("Target column", value=config.target_column)

    horizon = st.sidebar.slider(
        "Forecast horizon",
        min_value=1,
        max_value=60,
        value=config.default_horizon,
    )

    st.sidebar.info(
        "These values are pre-filled from your .env.\n"
        "Once your GCP project is ready, update:\n\n"
        "**GCP_PROJECT**\n"
        "**BQ_DATASET**\n"
        **BQ_TABLE**\n"
        "**GOOGLE_APPLICATION_CREDENTIALS**"
    )

    # -----------------------------
    # CONVERSATIONAL UI
    # -----------------------------
    st.subheader("💬 Chat with the proto-agent")

    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Type something (try 'help' or 'forecast')")

    if user_input:
        st.session_state["messages"].append({"role": "user", "content": user_input})

        cfg = AppConfig(
            project_id=project_id,
            dataset_id=dataset_id,
            table_id=table_id,
            date_column=date_column,
            target_column=target_column,
            default_horizon=horizon,
        )

        reply = handle_user_message(user_input, config=cfg)
        st.session_state["messages"].append({"role": "assistant", "content": reply})

        st.experimental_rerun()

    st.markdown("---")

    # -----------------------------
    # RUN DEMO FORECAST
    # -----------------------------
    st.subheader("🔮 Demo: ARIMA Forecast from BigQuery")

    if st.button("Run demo forecast"):
        try:
            client = BigQueryClient.from_env()
            client.project_id = project_id  # override if different
        except Exception as e:
            st.error(f"Failed to create BigQuery client:\n{e}")
            st.stop()

        with st.spinner("Querying BigQuery and fitting ARIMA..."):
            df = client.get_time_series(
                dataset=dataset_id,
                table=table_id,
                date_column=date_column,
                target_column=target_column,
                limit=365,
            )

            if df.empty:
                st.warning("No rows returned from BigQuery.")
                st.stop()

            result = fit_arima_and_forecast(
                df=df,
                ts_col="ts",
                target_col="y",
                periods=horizon,
            )

            df_plot = forecast_to_dataframe(result)

        st.success("Forecast generated!")

        col1, col2 = st.columns([2, 1])

        with col1:
            chart_data = df_plot.reset_index(names="ts").pivot_table(
                index="ts", columns="type", values="y"
            )
            st.line_chart(chart_data)

        with col2:
            st.write("Raw BigQuery sample:")
            st.dataframe(df.head())

            st.write("Forecast head:")
            st.dataframe(df_plot[df_plot["type"] == "forecast"].head())


if __name__ == "__main__":
    main()
