"""
utils.py

Utility functions for:
- Configuration handling
- Time series + ARIMA modeling
- Simple placeholder conversational "agent" logic
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


# -------------------------------------------------------------------
# Config model
# -------------------------------------------------------------------


@dataclass
class AppConfig:
    project_id: str
    dataset_id: str
    table_id: str
    date_column: str = "date"
    target_column: str = "value"
    default_horizon: int = 14

    @classmethod
    def from_env(cls) -> "AppConfig":
        """
        Load default config from environment variables (if present).
        This keeps Streamlit UI defaults in sync with the backend.
        """
        return cls(
            project_id=os.getenv("GCP_PROJECT", ""),
            dataset_id=os.getenv("BQ_DATASET", ""),
            table_id=os.getenv("BQ_TABLE", ""),
            date_column=os.getenv("TS_DATE_COL", "date"),
            target_column=os.getenv("TS_TARGET_COL", "value"),
            default_horizon=int(os.getenv("FORECAST_HORIZON", "14")),
        )


# -------------------------------------------------------------------
# ARIMA helpers
# -------------------------------------------------------------------


def fit_arima_and_forecast(
    df: pd.DataFrame,
    ts_col: str,
    target_col: str,
    periods: int = 14,
    order: Tuple[int, int, int] = (1, 1, 1),
) -> Dict[str, Any]:
    """
    Fit a basic ARIMA model and return forecast results.

    Returns a dict so it's easy to extend later (e.g. add confidence intervals).
    """
    if df.empty:
        raise ValueError("Received empty DataFrame for ARIMA model.")

    # Ensure sorted and set index
    df = df[[ts_col, target_col]].dropna().sort_values(ts_col)
    ts = df.set_index(ts_col)[target_col].asfreq("D")  # assume daily for now

    model = ARIMA(ts, order=order)
    fitted = model.fit()

    forecast = fitted.forecast(steps=periods)

    return {
        "model": fitted,
        "history": ts,
        "forecast": forecast,
    }


def forecast_to_dataframe(result: Dict[str, Any]) -> pd.DataFrame:
    """
    Convert ARIMA results dict into a tidy DataFrame for plotting/display.
    """
    history = result["history"]
    forecast = result["forecast"]

    df_hist = history.to_frame(name="y")
    df_hist["type"] = "history"

    df_fc = forecast.to_frame(name="y")
    df_fc["type"] = "forecast"

    return pd.concat([df_hist, df_fc])


# -------------------------------------------------------------------
# Simple placeholder conversational "agent"
# -------------------------------------------------------------------


def handle_user_message(
    message: str,
    *,
    config: AppConfig,
) -> str:
    """
    Placeholder conversational logic.

    For now this is just rule-based and does NOT call Vertex AI yet.
    Later we'll replace this with a real Vertex AI / tools-based agent.

    The goal is to wire the Streamlit UI and overall flow first.
    """
    text = message.lower()

    if "help" in text:
        return (
            "Hi! I'm your time-series data agent.\n\n"
            "Right now I can:\n"
            "- Load a time series from BigQuery (once credentials are configured)\n"
            "- Fit a basic ARIMA model and show a forecast\n\n"
            "Use the sidebar to set dataset/table, then click 'Run demo forecast'.\n"
            "Later we will upgrade me to a full Vertex AI agent that can answer free-text questions."
        )

    if "forecast" in text:
        return (
            f"Great, we will forecast `{config.target_column}` from "
            f"`{config.dataset_id}.{config.table_id}` for about {config.default_horizon} steps. "
            "Use the 'Forecast horizon' slider and the button below."
        )

    # default
    return (
        "I'm still a simple prototype 🤖.\n"
        "Right now you can:\n"
        "- Ask for 'help'\n"
        "- Mention 'forecast'\n\n"
        "We will later connect me to Vertex AI so I can understand arbitrary questions "
        "and automatically decide whether to query BigQuery or run ARIMA."
    )
