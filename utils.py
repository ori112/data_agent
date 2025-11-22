# utils.py
"""
utils.py
--------
Stable utilities that should not change when we add more agent logic.

Includes:
1) AppConfig - central config object loaded from .env.
2) fit_arima_and_forecast - a simple ARIMA helper used in "Quick demo".
3) forecast_to_dataframe - unify history/forecast for plotting.
4) handle_user_message - a *local stub* agent for fallback mode.

This file is still useful even with Vertex AI,
because it gives the app deterministic utilities.

"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


@dataclass
class AppConfig:
    """
    Central configuration for the app.

    Fields
    ------
    project_id, dataset_id, table_id:
        Identify the BigQuery table.
    date_column, target_column:
        Column names in the BigQuery table.
    default_horizon:
        Forecast horizon shown in UI.
    """

    project_id: str
    dataset_id: str
    table_id: str
    date_column: str = "date"
    target_column: str = "value"
    default_horizon: int = 14

    @classmethod
    def from_env(cls) -> "AppConfig":
        """
        Load config from .env.
        """
        return cls(
            project_id=os.getenv("GCP_PROJECT", ""),
            dataset_id=os.getenv("BQ_DATASET", ""),
            table_id=os.getenv("BQ_TABLE", ""),
            date_column=os.getenv("TS_DATE_COL", "date"),
            target_column=os.getenv("TS_TARGET_COL", "value"),
            default_horizon=int(os.getenv("FORECAST_HORIZON", "14")),
        )


def fit_arima_and_forecast(
    df: pd.DataFrame,
    ts_col: str,
    target_col: str,
    periods: int = 14,
    order: Tuple[int, int, int] = (1, 1, 1),
) -> Dict[str, Any]:
    """
    Simple ARIMA fit for the "quick demo".

    Why still keep it?
    - deterministic baseline for course.
    - good sanity check even when Vertex agent is ON.

    Parameters
    ----------
    df:
        dataframe containing a time series.
    ts_col:
        timestamp column name.
    target_col:
        numeric target column name.
    periods:
        number of steps to forecast.
    order:
        (p,d,q)

    Returns dict with:
        model, history, forecast
    """
    if df.empty:
        raise ValueError("ARIMA received an empty dataframe.")

    df = df[[ts_col, target_col]].dropna().sort_values(ts_col)

    ts = df.set_index(ts_col)[target_col]

    # ARIMA expects a regular index — if daily, use asfreq("D").
    # Your data is monthly, so we keep natural frequency.
    model = ARIMA(ts, order=order)
    fitted = model.fit()
    forecast = fitted.forecast(steps=periods)

    return {"model": fitted, "history": ts, "forecast": forecast}


def forecast_to_dataframe(result: Dict[str, Any]) -> pd.DataFrame:
    """
    Convert fit_arima_and_forecast output into a single dataframe.

    output:
        index = ts
        columns: y, type in {"history", "forecast"}
    """
    history = result["history"]
    forecast = result["forecast"]

    hist_df = history.to_frame(name="y")
    hist_df["type"] = "history"

    fc_df = forecast.to_frame(name="y")
    fc_df["type"] = "forecast"

    return pd.concat([hist_df, fc_df])


def handle_user_message(message: str, config: AppConfig) -> str:
    """
    Local / no-Vertex fallback agent.
    Keeps UX usable if Vertex isn't enabled.

    It doesn't do real work; just guides user.
    """
    msg = message.lower()

    if "help" in msg:
        return (
            "Hi! I'm your local prototype agent.\n\n"
            "When Vertex AI is OFF I can only:\n"
            "• Show config from .env\n"
            "• Run quick ARIMA demo\n\n"
            "Turn Vertex AI ON for full EDA + SARIMA."
        )

    if "forecast" in msg or "model" in msg:
        return (
            f"I will forecast `{config.target_column}` from "
            f"{config.dataset_id}.{config.table_id} "
            f"for {config.default_horizon} steps.\n"
            "Use the button below."
        )

    return "Try asking:\n• 'help'\n• 'forecast'\n• 'run EDA'"
