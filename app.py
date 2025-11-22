# app.py

from __future__ import annotations

import streamlit as st
from dotenv import load_dotenv

from bigquery_client import BigQueryClient
from utils import (
    AppConfig,
    fit_arima_and_forecast,
    forecast_to_dataframe,
    handle_user_message,
)

# Load environment variables from .env
load_dotenv()


def init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "last_figures" not in st.session_state:
        st.session_state["last_figures"] = []
    if "last_tables" not in st.session_state:
        st.session_state["last_tables"] = {}


def main() -> None:
    st.set_page_config(
        page_title="BigQuery Time-Series Agent",
        page_icon="📈",
        layout="wide",
    )

    st.title("📈 BigQuery Time-Series Agent")
    st.caption(
        "Live setup: Streamlit + BigQuery mart + Gemini on Vertex AI + "
        "time-series tools (ADF, ACF/PACF, decomposition, SARIMA)."
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

    cfg = AppConfig(
        project_id=project_id,
        dataset_id=dataset_id,
        table_id=table_id,
        date_column=date_column,
        target_column=target_column,
        default_horizon=horizon,
    )

    st.sidebar.info(
        "Defaults are loaded from your .env.\nEdit here to override per-session."
    )

    # -----------------------------
    # OPTIONAL: USE VERTEX AGENT
    # -----------------------------
    use_vertex = st.sidebar.checkbox(
        "Use Vertex AI agent (live)",
        value=True,
        help="ON = Gemini + BigQuery + TS tools. OFF = local stub agent.",
    )

    agent = None
    if use_vertex:
        try:
            from vertex_agent import VertexConfig, DataAgent

            agent = DataAgent(cfg, VertexConfig.from_env())
        except Exception as e:  # noqa: BLE001
            st.sidebar.error(f"Vertex agent failed to init:\n{e}")
            use_vertex = False
            agent = None

    # -----------------------------
    # CONVERSATIONAL UI
    # -----------------------------
    st.subheader("💬 Chat with your Data Agent")

    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask about the dataset, diagnostics, or forecasting...")

    if user_input:
        st.session_state["messages"].append({"role": "user", "content": user_input})

        if use_vertex and agent:
            result = agent.handle_message(user_input)

            st.session_state["messages"].append(
                {"role": "assistant", "content": result["text"]}
            )

            st.session_state["last_figures"] = result.get("figures", []) or []
            st.session_state["last_tables"] = result.get("tables", {}) or {}
        else:
            reply = handle_user_message(user_input, config=cfg)
            st.session_state["messages"].append({"role": "assistant", "content": reply})
            st.session_state["last_figures"] = []
            st.session_state["last_tables"] = {}

        st.experimental_rerun()

    # -----------------------------
    # SHOW AGENT ARTIFACTS
    # -----------------------------
    if st.session_state.get("last_figures"):
        st.subheader("📊 Agent plots")
        for fig in st.session_state["last_figures"]:
            st.pyplot(fig)

    if st.session_state.get("last_tables"):
        st.subheader("🧾 Agent tables")
        for name, df_tbl in st.session_state["last_tables"].items():
            st.write(f"**{name}**")
            st.dataframe(df_tbl)

    st.markdown("---")

    # -----------------------------
    # BASIC DEMO FORECAST BUTTON
    # (still useful as a quick sanity check)
    # -----------------------------
    st.subheader("🔮 Quick demo: ARIMA forecast from BigQuery")

    if st.button("Run demo forecast"):
        try:
            client = BigQueryClient.from_env()
            client.project_id = project_id  # override if user changed it
        except Exception as e:  # noqa: BLE001
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
                index="ts",
                columns="type",
                values="y",
            )
            st.line_chart(chart_data)

        with col2:
            st.write("Raw BigQuery sample:")
            st.dataframe(df.head())

            st.write("Forecast head:")
            st.dataframe(df_plot[df_plot["type"] == "forecast"].head())


if __name__ == "__main__":
    main()
