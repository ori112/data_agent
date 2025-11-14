"""
vertex_agent.py

Vertex AI–backed conversational data agent.

Capabilities (designed for your course project):
- Understand questions about the dataset.
- Decide whether to:
  * answer conceptually,
  * run time-series diagnostics (ACF/PACF, decomposition, ADF),
  * run a full SARIMA workflow similar to your Colab notebook,
  * run a basic ARIMA-style forecast (simpler path).

The agent returns:
- text  -> explanation/answer for the user (LLM generated)
- plan  -> JSON "plan" the model proposed (for debugging / logging)
- figures -> list of matplotlib Figure objects (for Streamlit plotting)
- tables  -> dict of name -> DataFrame (for Streamlit tables)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig

from bigquery_client import BigQueryClient
from utils import AppConfig, fit_arima_and_forecast, forecast_to_dataframe
from timeseries_tools import (
    plot_time_series,
    log_transform,
    plot_acf_pacf_series,
    seasonal_decompose_plot,
    run_adf_test,
    train_test_split_series,
    fit_sarima,
    residual_diagnostics,
    sarima_forecast_plots,
)


# ---------------------------------------------------------------------------
# Vertex AI configuration dataclass
# ---------------------------------------------------------------------------


@dataclass
class VertexConfig:
    project_id: str
    location: str
    model_name: str = "gemini-1.5-pro-001"

    @classmethod
    def from_env(cls) -> "VertexConfig":
        """
        Load Vertex AI config from environment variables (.env).

        Required:
          - GCP_PROJECT
        Optional:
          - VERTEX_LOCATION  (default: us-central1)
          - VERTEX_MODEL     (default: gemini-1.5-pro-001)
        """
        project_id = os.getenv("GCP_PROJECT")
        if not project_id:
            raise ValueError(
                "GCP_PROJECT is not set. Vertex AI needs the same project ID "
                "you use for BigQuery."
            )

        location = os.getenv("VERTEX_LOCATION", "us-central1")
        model_name = os.getenv("VERTEX_MODEL", "gemini-1.5-pro-001")

        return cls(
            project_id=project_id,
            location=location,
            model_name=model_name,
        )


# ---------------------------------------------------------------------------
# DataAgent orchestrating Gemini + BigQuery + TS tools
# ---------------------------------------------------------------------------


class DataAgent:
    """
    High-level agent that:
      - Calls Gemini (Vertex AI) to understand the question.
      - Optionally loads time series from BigQuery.
      - Optionally runs diagnostics or SARIMA/ARIMA workflows.

    NOTE: This is not wired into Streamlit yet. Once your GCP project and
    Vertex AI are configured, you'll:
      - create a single DataAgent instance in app.py
      - call agent.handle_message(user_input) per chat turn
      - display result["text"], result["figures"], result["tables"].
    """

    def __init__(self, app_config: AppConfig, vertex_config: VertexConfig) -> None:
        self.app_config = app_config
        self.vertex_config = vertex_config

        # Init Vertex AI
        vertexai.init(
            project=self.vertex_config.project_id,
            location=self.vertex_config.location,
        )
        self.model = GenerativeModel(self.vertex_config.model_name)

        # BigQuery client (same project / credentials)
        self.bq_client = BigQueryClient.from_env()

    # -----------------------------
    # Public entry point
    # -----------------------------
    def handle_message(self, user_message: str) -> Dict[str, Any]:
        """
        Main method called from the UI.

        Returns a dict with:
          - "text": natural-language answer
          - "plan": JSON action spec from Gemini
          - "figures": list[matplotlib.figure.Figure] (for plotting in Streamlit)
          - "tables": dict[str, pd.DataFrame] (for showing in Streamlit)
        """
        plan = self._plan_with_model(user_message)
        action = plan.get("action", "answer_only")

        answer: str
        figures: List[plt.Figure] = []
        tables: Dict[str, pd.DataFrame] = {}

        if action == "answer_only":
            answer = plan.get("answer", plan.get("explanation", ""))

        elif action == "describe_dataset":
            answer = self._answer_about_dataset(user_message)

        elif action == "forecast":
            periods = int(plan.get("forecast_periods", self.app_config.default_horizon))
            answer, df_fc = self._run_simple_arima_forecast(periods=periods)
            if df_fc is not None:
                tables["forecast"] = df_fc

        elif action == "diagnostics_plots":
            answer, figures = self._run_ts_diagnostics()

        elif action == "adf_test":
            answer, tables = self._run_adf_workflow()

        elif action == "sarima_workflow":
            answer, figures, tables = self._run_sarima_workflow()

        else:
            answer = (
                "I couldn't confidently decide which time-series tool to use.\n\n"
                "Try something like:\n"
                "- 'Plot and diagnose the time series'\n"
                "- 'Run ADF test on the series'\n"
                "- 'Run the full SARIMA workflow like in the notebook'\n"
                "- 'Forecast the metric for the next 12 months'\n"
            )

        return {
            "text": answer,
            "plan": plan,
            "figures": figures,
            "tables": tables,
        }

    # -----------------------------
    # LLM planning
    # -----------------------------

    def _plan_with_model(self, user_message: str) -> Dict[str, Any]:
        """
        Ask Gemini to output a small JSON "plan" telling us what to do.

        Supported actions:

        - "answer_only":
            Conceptual / theoretical questions; answer directly, no tools.

        - "describe_dataset":
            User asks about what the dataset is, how it might be used, etc.

        - "forecast":
            User explicitly asks to forecast/predict the target variable but
            does not mention SARIMA diagnostics. Use simple ARIMA pipeline.

        - "diagnostics_plots":
            User asks to inspect / visualize the series, e.g.:
            'plot the series', 'acf', 'pacf', 'decompose', 'check seasonality'.

        - "adf_test":
            User mentions stationarity, unit root, ADF, Augmented Dickey-Fuller, etc.

        - "sarima_workflow":
            User wants the full modeling pipeline like the notebook:
            log transform, train/test split, SARIMA fit, residual diagnostics,
            and forecast with confidence intervals.

        You must respond ONLY with JSON (no markdown, no extra text).
        """

        system_prompt = f"""
You are a time-series data analyst assistant working with a BigQuery dataset.

Dataset configuration:
  - project: {self.app_config.project_id or "[GCP_PROJECT env]"}
  - dataset: {self.app_config.dataset_id or "[BQ_DATASET env]"}
  - table:   {self.app_config.table_id or "[BQ_TABLE env]"}
  - date column: {self.app_config.date_column}
  - target column: {self.app_config.target_column}

Respond STRICTLY in JSON with keys:

- "action": one of:
    "answer_only",
    "describe_dataset",
    "forecast",
    "diagnostics_plots",
    "adf_test",
    "sarima_workflow"

- "forecast_periods": integer, only used when action == "forecast".
  Choose a reasonable horizon (e.g. 7, 14, 30, or 12 for months).

- "explanation": short natural language explanation WHY you chose this action.

- "answer": direct natural language answer, ONLY when action == "answer_only".

If the user wants a deep modeling workflow, including diagnostics like ACF/PACF,
seasonal decomposition, residual plots, and SARIMA, choose "sarima_workflow".
If they explicitly ask about stationarity or ADF, choose "adf_test".
If they simply ask for basic inspection of the series, choose "diagnostics_plots".
"""

        cfg = GenerationConfig(
            response_mime_type="application/json",
            temperature=0.2,
        )

        response = self.model.generate_content(
            [system_prompt, user_message],
            generation_config=cfg,
        )

        raw = response.text or "{}"
        try:
            plan = json.loads(raw)
        except json.JSONDecodeError:
            # Fallback: treat as a direct answer
            plan = {
                "action": "answer_only",
                "answer": raw,
                "explanation": "Model did not return valid JSON; treated as direct answer.",
            }

        plan.setdefault("action", "answer_only")
        plan.setdefault("explanation", "")
        return plan

    # -----------------------------
    # Concrete actions
    # -----------------------------

    def _load_series(self) -> pd.DataFrame:
        """Load the base time series from BigQuery as a DataFrame."""
        df = self.bq_client.get_time_series(
            dataset=self.app_config.dataset_id,
            table=self.app_config.table_id,
            date_column=self.app_config.date_column,
            target_column=self.app_config.target_column,
            limit=365,
        )
        if df.empty:
            raise ValueError(
                "BigQuery returned no data. Check dataset, table, and column names."
            )
        return df

    def _run_simple_arima_forecast(
        self,
        periods: int,
    ) -> tuple[str, Optional[pd.DataFrame]]:
        """Use existing ARIMA helper for a simpler forecast."""
        df = self._load_series()
        result = fit_arima_and_forecast(
            df=df,
            ts_col="ts",
            target_col="y",
            periods=periods,
        )
        df_fc = forecast_to_dataframe(result)

        text = (
            f"I ran a basic ARIMA model on `{self.app_config.target_column}` and "
            f"forecasted the next {periods} time steps. The table 'forecast' "
            "contains both history and forecast values."
        )
        return text, df_fc

    def _run_ts_diagnostics(self) -> tuple[str, List[plt.Figure]]:
        """
        Run notebook-style diagnostics:
          - plot of series (log transformed)
          - ACF and PACF
          - seasonal decomposition
        """
        df = self._load_series()
        df_log = log_transform(df, "y")

        figs: List[plt.Figure] = []

        # 1. Time series plot (log)
        fig_ts = plot_time_series(
            df_log, date_col="ts", value_col="y", title="Log-transformed series"
        )
        figs.append(fig_ts)

        # 2. ACF & PACF
        fig_acf, fig_pacf = plot_acf_pacf_series(df_log["y"])
        figs.extend([fig_acf, fig_pacf])

        # 3. Seasonal decomposition
        fig_decomp = seasonal_decompose_plot(df_log["y"], period=12, model="additive")
        figs.append(fig_decomp)

        text = (
            "I logged the series to stabilize variance, then produced:\n"
            "- A line plot of the log-transformed series\n"
            "- ACF and PACF plots to inspect autocorrelation\n"
            "- An additive seasonal decomposition with period=12\n\n"
            "You can use these plots to reason about trend, seasonality, and "
            "appropriate SARIMA orders."
        )

        return text, figs

    def _run_adf_workflow(self) -> tuple[str, Dict[str, pd.DataFrame]]:
        """
        Run ADF test on (log-transformed) series and return a summary table.
        """
        df = self._load_series()
        df_log = log_transform(df, "y")
        stats = run_adf_test(df_log["y"])

        summary_df = pd.DataFrame(
            {
                "ADF Statistic": [stats["adf_statistic"]],
                "p-value": [stats["p_value"]],
                "used_lag": [stats["used_lag"]],
                "n_obs": [stats["n_obs"]],
                "is_stationary": [stats["is_stationary"]],
            }
        )

        crit_df = pd.DataFrame(
            list(stats["critical_values"].items()),
            columns=["Significance level", "Critical value"],
        )

        text = (
            "I ran the Augmented Dickey-Fuller (ADF) test on the log-transformed series.\n\n"
            f"{stats['interpretation']}\n\n"
            "You can inspect the exact statistic, p-value, and critical values "
            "in the returned tables."
        )

        tables = {
            "adf_summary": summary_df,
            "adf_critical_values": crit_df,
        }

        return text, tables

    def _run_sarima_workflow(
        self,
    ) -> tuple[str, List[plt.Figure], Dict[str, pd.DataFrame]]:
        """
        Reproduce your notebook-style workflow:

        - log transform series
        - train/test split (80/20)
        - fit SARIMA(1,0,0)x(0,1,1,12)
        - residual diagnostics (residuals + ACF/PACF)
        - forecast in log scale and original scale
        """
        df = self._load_series()
        df_log = log_transform(df, "y")
        series = df_log.set_index("ts")["y"]

        train, test = train_test_split_series(series, train_ratio=0.8)
        results = fit_sarima(
            train,
            order=(1, 0, 0),
            seasonal_order=(0, 1, 1, 12),
        )

        figs: List[plt.Figure] = []
        tables: Dict[str, pd.DataFrame] = {}

        # Residual diagnostics
        diag_figs, residuals = residual_diagnostics(results, lags=30)
        figs.extend(diag_figs)

        # Forecast plots (log scale)
        fig_log, forecast_mean_log, conf_int_log = sarima_forecast_plots(
            train, test, results, back_transform=False
        )
        figs.append(fig_log)

        # Forecast plots (original scale)
        fig_orig, forecast_mean_orig, conf_int_orig = sarima_forecast_plots(
            train, test, results, back_transform=True
        )
        figs.append(fig_orig)

        # Tables with numeric results (optional, nice for inspection)
        tables["sarima_residuals"] = residuals.to_frame(name="residuals")
        tables["sarima_forecast_log"] = pd.DataFrame(
            {
                "forecast_mean": forecast_mean_log,
                "ci_lower": conf_int_log.iloc[:, 0],
                "ci_upper": conf_int_log.iloc[:, 1],
            }
        )
        tables["sarima_forecast_original"] = pd.DataFrame(
            {
                "forecast_mean": np.exp(forecast_mean_orig),
                "ci_lower": np.exp(conf_int_orig.iloc[:, 0]),
                "ci_upper": np.exp(conf_int_orig.iloc[:, 1]),
            },
            index=forecast_mean_orig.index,
        )

        text = (
            "I ran the full SARIMA workflow similar to your notebook:\n"
            "- Log-transformed the series to reduce variance\n"
            "- Split the data into 80% train / 20% test\n"
            "- Fitted SARIMA(1,0,0)x(0,1,1,12)\n"
            "- Produced residual plots plus ACF and PACF of residuals\n"
            "- Generated forecasts in both log scale and back-transformed "
            "original scale, with confidence intervals.\n\n"
            "Use the residual diagnostics to check model adequacy and the "
            "forecast plots to interpret future behaviour."
        )

        return text, figs, tables

    def _answer_about_dataset(self, user_message: str) -> str:
        """
        Let Gemini explain the dataset conceptually using known configuration.
        """
        prompt = f"""
You are a data analyst explaining a BigQuery time-series dataset to a student.

Dataset info:
  - Project: {self.app_config.project_id or "[GCP_PROJECT env]"}
  - Dataset: {self.app_config.dataset_id or "[BQ_DATASET env]"}
  - Table:   {self.app_config.table_id or "[BQ_TABLE env]"}
  - Date column: {self.app_config.date_column}
  - Target column: {self.app_config.target_column}

User question:
{user_message}

Explain clearly, in a few short paragraphs, what the dataset likely represents,
how it could be used, and how forecasts on the target column would be interpreted.
"""
        response = self.model.generate_content(prompt)
        return response.text


# ---------------------------------------------------------------------------
# Simple CLI test (once GCP + Vertex are configured)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Example:

        uv run python vertex_agent.py

    Then type in some questions when you wire this into a simple CLI,
    or just inspect the plan for a single test question as below.
    """
    app_cfg = AppConfig.from_env()
    vx_cfg = VertexConfig.from_env()
    agent = DataAgent(app_cfg, vx_cfg)

    question = "Run the full SARIMA workflow like in my notebook."
    result = agent.handle_message(question)

    print("PLAN:", json.dumps(result["plan"], indent=2))
    print("\nTEXT:\n", result["text"])
    print("\nFigures:", len(result["figures"]))
    print("Tables:", list(result["tables"].keys()))
