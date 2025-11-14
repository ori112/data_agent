# utils.py

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


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
    if df.empty:
        raise ValueError("ARIMA received an empty dataframe.")

    df = df[[ts_col, target_col]].dropna().sort_values(ts_col)
    ts = df.set_index(ts_col)[target_col].asfreq("D")

    model = ARIMA(ts, order=order)
    fitted = model.fit()

    forecast = fitted.forecast(steps=periods)

    return {
        "model": fitted,
        "history": ts,
        "forecast": forecast,
    }


def forecast_to_dataframe(result: Dict[str, Any]) -> pd.DataFrame:
    history = result["history"]
    forecast = result["forecast"]

    hist_df = history.to_frame(name="y")
    hist_df["type"] = "history"

    fc_df = forecast.to_frame(name="y")
    fc_df["type"] = "forecast"

    return pd.concat([hist_df, fc_df])


def handle_user_message(message: str, config: AppConfig) -> str:
    msg = message.lower()

    if "help" in msg:
        return (
            "Hi! I'm your prototype data agent.\n\n"
            "Right now I can:\n"
            "• Read config from your .env file\n"
            "• Query BigQuery (once configured)\n"
            "• Run ARIMA forecasting\n\n"
            "Later: full Vertex AI agent with tool-calling."
        )

    if "forecast" in msg:
        return (
            f"Preparing to forecast `{config.target_column}` from "
            f"{config.dataset_id}.{config.table_id} "
            f"for {config.default_horizon} steps.\n"
            "Use the button below."
        )

    return "I'm still a simple prototype 🤖.\nTry asking:\n• 'help'\n• 'forecast'"
