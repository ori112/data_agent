"""
app.py

Streamlit front-end for the data agent.

Run locally with:
    streamlit run app.py
"""

from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from bigquery_client import BigQueryClient
from utils import (
    AppConfig,
    fit_arima_and_forecast,
    forecast_to_dataframe,
    handle_user_message,
)


def init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state["messages"] = []  # list[dict(role, content)]


def main() -> None:
    st.set_page_config(
        page_title="BigQuery Time-Series Agent",
        page_icon="📈",
        layout="wide",
    )

    st.title("📈 BigQuery Time-Series Agent (Prototype)")
    st.caption(
        "GCP + BigQuery + ARIMA + Streamlit. "
        "We’ll later plug in Vertex AI for a full conversational agent."
    )

    init_session_state()
    config = AppConfig.from_env()

    # ------------------------------------------------------------------
    # Sidebar configuration
    # ------------------------------------------------------------------
    st.sidebar.header("BigQuery configuration")

    project_id = st.sidebar.text_input("GCP Project ID", value=config.project_id)
    dataset_id = st.sidebar.text_input("Dataset ID", value=config.dataset_id)
    table_id = st.sidebar.text_input("Table ID", value=config.table_id)

    date_column = st.sidebar.text_input("Date column", value=config.date_column)
    target_column = st.sidebar.text_input("Target column", value=config.target_column)

    horizon = st.sidebar.slider(
        "Forecast horizon (steps)",
        min_value=1,
        max_value=60,
        value=config.default_horizon,
    )

    st.sidebar.info(
        "Later, once your GCP project is ready:\n"
        "1. Set `GCP_PROJECT` env var.\n"
        "2. Point `GOOGLE_APPLICATION_CREDENTIALS` to your service-account JSON.\n"
        "3. (Optional) Set `BQ_DATASET`, `BQ_TABLE`.\n"
    )

    # ------------------------------------------------------------------
    # Chat area (very simple for now)
    # ------------------------------------------------------------------
    st.subheader("💬 Conversational interface (stub)")

    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask the agent something (try 'help' or 'forecast')")

    if user_input:
        st.session_state["messages"].append({"role": "user", "content": user_input})

        cfg_for_agent = AppConfig(
            project_id=project_id,
            dataset_id=dataset_id,
            table_id=table_id,
            date_column=date_column,
            target_column=target_column,
            default_horizon=horizon,
        )

        reply = handle_user_message(user_input, config=cfg_for_agent)
        st.session_state["messages"].append({"role": "assistant", "content": reply})
        st.experimental_rerun()

    st.markdown("---")

    # ------------------------------------------------------------------
    # Demo: load data from BigQuery & run ARIMA forecast
    # ------------------------------------------------------------------
    st.subheader("🔮 Demo: ARIMA forecast from BigQuery")

    if st.button("Run demo forecast"):
        if not project_id or not dataset_id or not table_id:
            st.error("Please fill in project, dataset and table IDs in the sidebar.")
            st.stop()

        try:
            # prefer env-based creds, but override project with UI value
            client = BigQueryClient.from_env()
            client.project_id = project_id  # override if user typed different project
        except Exception as e:  # noqa: BLE001
            st.error(
                f"Failed to initialize BigQuery client.\n\n"
                f"Details: {e}\n\n"
                "Make sure:\n"
                "- Your GCP project exists\n"
                "- GOOGLE_APPLICATION_CREDENTIALS points to a valid service-account JSON\n"
                "- GCP_PROJECT is set correctly"
            )
            st.stop()

        with st.spinner("Querying BigQuery and fitting ARIMA model..."):
            df = client.get_time_series(
                dataset=dataset_id,
                table=table_id,
                date_column=date_column,
                target_column=target_column,
                limit=365,
            )

            if df.empty:
                st.warning("BigQuery query returned no rows.")
                st.stop()

            result = fit_arima_and_forecast(
                df=df,
                ts_col="ts",
                target_col="y",
                periods=horizon,
            )
            df_plot = forecast_to_dataframe(result)

        st.success("Done!")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.line_chart(
                df_plot.reset_index(names="ts").pivot_table(
                    index="ts", columns="type", values="y"
                )
            )

        with col2:
            st.write("Sample of data from BigQuery:")
            st.dataframe(df.head())

            st.write("Forecast head:")
            st.dataframe(df_plot[df_plot["type"] == "forecast"].head())


if __name__ == "__main__":
    main()
