# app.py
"""
Streamlit App
-------------
This is the UI layer.

Responsibilities:
1) Read config from sidebar / .env.
2) Initialize Vertex agent if enabled.
3) Maintain chat state in st.session_state.
4) Display:
   - chat history
   - plots (figures)
   - tables
5) Show loaders while the agent runs.
6) Provide an optional deterministic ARIMA demo.

Important fix:
- We do NOT print summary twice.
  Only chat displays agent text.
"""

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

load_dotenv()


def init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "last_figures" not in st.session_state:
        st.session_state["last_figures"] = []
    if "last_tables" not in st.session_state:
        st.session_state["last_tables"] = {}
    if "last_summary" not in st.session_state:
        st.session_state["last_summary"] = ""


def main() -> None:
    st.set_page_config(
        page_title="BigQuery Time-Series Agent", page_icon="📈", layout="wide"
    )

    st.title("📈 Time-Series Agent")
    st.caption(
        "EDA includes diagnostics + comparisons. MODEL includes SARIMA forecasting."
    )

    init_session_state()
    config = AppConfig.from_env()

    # ---------------- Sidebar ----------------
    st.sidebar.header("BigQuery Configuration")

    project_id = st.sidebar.text_input("GCP Project ID", value=config.project_id)
    dataset_id = st.sidebar.text_input("Dataset ID", value=config.dataset_id)
    table_id = st.sidebar.text_input("Table ID", value=config.table_id)

    date_column = st.sidebar.text_input("Date column", value=config.date_column)
    target_column = st.sidebar.text_input("Target column", value=config.target_column)

    horizon = st.sidebar.slider("Forecast horizon", 1, 60, value=config.default_horizon)

    cfg = AppConfig(
        project_id=project_id,
        dataset_id=dataset_id,
        table_id=table_id,
        date_column=date_column,
        target_column=target_column,
        default_horizon=horizon,
    )

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
        except Exception as e:
            st.sidebar.error(f"Vertex agent failed to init:\n{e}")
            use_vertex = False
            agent = None

    # ---------------- Chat UI ----------------
    st.subheader("💬 Chat with your Data Agent")
    st.caption("You can ask me everything from data exploration up to data modeling")

    # render prior messages
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask for EDA or modeling (forecasting)...")

    if user_input:
        st.session_state["messages"].append({"role": "user", "content": user_input})

        # ---- LOADER START ----
        with st.status("Working on it...", expanded=True) as status:
            status.write("Understanding your request...")

            if use_vertex and agent:
                status.write("Calling Vertex AI agent...")
                try:
                    with st.spinner("Agent is thinking and running tools..."):
                        result = agent.handle_message(user_input)
                except Exception as e:
                    status.update(label="Agent failed", state="error", expanded=True)
                    st.session_state["messages"].append(
                        {"role": "assistant", "content": f"Agent error:\n\n{e}"}
                    )
                    st.session_state["last_figures"] = []
                    st.session_state["last_tables"] = {}
                    st.session_state["last_summary"] = f"Agent error: {e}"
                    st.rerun()

                status.write("Preparing response + artifacts...")
                st.session_state["messages"].append(
                    {"role": "assistant", "content": result["text"]}
                )
                st.session_state["last_figures"] = result.get("figures", []) or []
                st.session_state["last_tables"] = result.get("tables", {}) or {}
                st.session_state["last_summary"] = result["text"]

            else:
                status.write("Running local prototype agent...")
                with st.spinner("Processing locally..."):
                    reply = handle_user_message(user_input, config=cfg)

                st.session_state["messages"].append(
                    {"role": "assistant", "content": reply}
                )
                st.session_state["last_figures"] = []
                st.session_state["last_tables"] = {}
                st.session_state["last_summary"] = reply

            status.update(label="Done ✅", state="complete", expanded=False)
        # ---- LOADER END ----

        st.rerun()

    # -------- Show artifacts (plots/tables only) --------
    if st.session_state.get("last_figures"):
        st.subheader("📊 Agent plots")
        for fig in st.session_state["last_figures"]:
            st.pyplot(fig)

    if st.session_state.get("last_tables"):
        st.subheader("🧾 Agent tables")
        for name, df_tbl in st.session_state["last_tables"].items():
            st.write(f"**{name}**")
            st.dataframe(df_tbl, use_container_width=True)

    st.markdown("---")


if __name__ == "__main__":
    main()
