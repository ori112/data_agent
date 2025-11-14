"""
vertex_agent.py

Vertex AI–backed conversational data agent.

Responsibilities (design for the course):
- Use Gemini on Vertex AI to understand natural-language questions.
- Decide whether to:
  * Just answer conceptually ("what is ARIMA", "how many rows do we have?", etc.)
  * Explain or describe the dataset (columns, metrics, etc.)
  * Trigger a time-series forecast (ARIMA) on the configured dataset.

Implementation notes:
- Uses a simple JSON "plan" protocol (action + parameters) instead of full
  function-calling to keep the code readable for a university project.
- Later, you can upgrade this to real Vertex AI function calling if needed.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig

from bigquery_client import BigQueryClient
from utils import AppConfig, fit_arima_and_forecast, forecast_to_dataframe


# ---------------------------------------------------------------------------
# Vertex AI configuration
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
          - VERTEX_LOCATION (default: us-central1)
          - VERTEX_MODEL    (default: gemini-1.5-pro-001)
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
# DataAgent: orchestrates Gemini + BigQuery + ARIMA
# ---------------------------------------------------------------------------


class DataAgent:
    """
    High-level agent that:
      - talks to Gemini (Vertex AI)
      - optionally queries BigQuery
      - optionally trains/runs ARIMA forecast

    This is NOT connected to Streamlit yet – app.py still uses the simple
    handle_user_message() stub. Later, you will:
      - create one DataAgent instance in app.py
      - call agent.handle_message(user_input) for each chat turn
      - display agent_response["text"] + optional tables/plots
    """

    def __init__(self, app_config: AppConfig, vertex_config: VertexConfig) -> None:
        self.app_config = app_config
        self.vertex_config = vertex_config

        # Init Vertex AI client & model
        vertexai.init(
            project=self.vertex_config.project_id,
            location=self.vertex_config.location,
        )
        self.model = GenerativeModel(self.vertex_config.model_name)

        # BigQuery client (will use same GCP project + service account)
        self.bq_client = BigQueryClient.from_env()

    # -----------------------------
    # Public entry point
    # -----------------------------
    def handle_message(self, user_message: str) -> Dict[str, Any]:
        """
        Main method called from the UI.

        Returns a dict with:
          - "text": final natural-language answer for the user
          - "plan": JSON action the model proposed
          - "data": optional DataFrame (for UI tables) – currently None
          - "forecast_df": optional forecast DataFrame – currently None
        """
        plan = self._plan_with_model(user_message)

        action = plan.get("action", "answer_only")
        explanation = plan.get("explanation", "")

        data = None
        forecast_df = None

        if action == "answer_only":
            # Pure LLM answer – no BigQuery/ARIMA
            answer = plan.get("answer", explanation)

        elif action == "describe_dataset":
            # Ask Gemini to describe dataset based on columns, metric, etc.
            answer = self._answer_about_dataset(user_message)

        elif action == "forecast":
            periods = int(plan.get("forecast_periods", self.app_config.default_horizon))
            answer, forecast_df = self._run_forecast(periods=periods)

        else:
            # Fallback
            answer = (
                "I couldn't confidently decide what to do with that question.\n\n"
                "Try asking something like:\n"
                "- 'Describe the dataset'\n"
                "- 'Forecast the metric for the next 14 days'\n"
            )

        return {
            "text": answer,
            "plan": plan,
            "data": data,
            "forecast_df": forecast_df,
        }

    # -----------------------------
    # Internal helpers
    # -----------------------------

    def _plan_with_model(self, user_message: str) -> Dict[str, Any]:
        """
        Ask Gemini to output a small JSON "plan" telling us what to do.

        Expected JSON shape:
        {
          "action": "answer_only" | "describe_dataset" | "forecast",
          "forecast_periods": 14,
          "explanation": "why this action makes sense",
          "answer": "direct answer if no tools are needed"
        }
        """
        system_prompt = f"""
You are a data analyst assistant working with a BigQuery time-series dataset.

The main dataset is:
  project: {self.app_config.project_id or "[GCP_PROJECT env]"}
  dataset: {self.app_config.dataset_id or "[BQ_DATASET env]"}
  table:   {self.app_config.table_id or "[BQ_TABLE env]"}
  date column: {self.app_config.date_column}
  target column: {self.app_config.target_column}

You must respond STRICTLY in JSON (no markdown, no extra text) with the following keys:

- "action": one of:
    - "answer_only"       -> question is conceptual; answer directly, no tools
    - "describe_dataset"  -> user is asking about columns, metric, time grain, etc.
    - "forecast"          -> user explicitly asks to forecast / predict the target

- "forecast_periods": integer steps for forecasting (only used when action="forecast").
- "explanation": short natural language explanation of why you chose this action.
- "answer": direct natural language answer if action="answer_only".

If the user wants a prediction, use "forecast", and choose a reasonable horizon
(e.g. 7, 14, or 30) in "forecast_periods".
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
            # Fallback: treat as a plain answer
            plan = {
                "action": "answer_only",
                "answer": raw,
                "explanation": "Model did not return valid JSON; treated as direct answer.",
            }

        # Ensure mandatory keys exist
        plan.setdefault("action", "answer_only")
        plan.setdefault("explanation", "")
        return plan

    def _answer_about_dataset(self, user_message: str) -> str:
        """
        For now, this just lets Gemini explain the dataset conceptually using
        the known column names. You can later enrich this by querying
        BigQuery INFORMATION_SCHEMA for real metadata and row counts.
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

Explain clearly, in a few short paragraphs, what this dataset likely represents,
how it could be used, and how forecasts on the target column would be interpreted.
"""
        response = self.model.generate_content(prompt)
        return response.text

    def _run_forecast(self, periods: int) -> (str, Optional[Any]):
        """
        Runs the ARIMA forecast on the configured dataset/table using
        BigQueryClient.get_time_series() and utils.fit_arima_and_forecast().
        """
        df = self.bq_client.get_time_series(
            dataset=self.app_config.dataset_id,
            table=self.app_config.table_id,
            date_column=self.app_config.date_column,
            target_column=self.app_config.target_column,
            limit=365,
        )

        if df.empty:
            return (
                "I tried to query the time series from BigQuery but received no data. "
                "Please check that the dataset, table, and column names are correct.",
                None,
            )

        result = fit_arima_and_forecast(
            df=df,
            ts_col="ts",
            target_col="y",
            periods=periods,
        )
        forecast_df = forecast_to_dataframe(result)

        # Let Gemini generate a short explanation of the forecast
        # (purely optional, a nice touch for the course).
        summary_prompt = f"""
You are a data scientist explaining an ARIMA forecast to a marketing / business audience.

We have a time series of `{self.app_config.target_column}` and we ran an ARIMA model
to forecast the next {periods} steps (e.g. days).

Explain in simple language:
- what the model is doing,
- how to interpret "history" vs "forecast",
- one or two potential use cases.

Keep it under 2 short paragraphs.
"""
        response = self.model.generate_content(summary_prompt)
        explanation = response.text

        return explanation, forecast_df


# ---------------------------------------------------------------------------
# Simple CLI test (optional)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    This block lets you quickly test the planning behaviour from the terminal,
    once your GCP + Vertex setup is ready:

        uv run python vertex_agent.py
    """
    app_cfg = AppConfig.from_env()
    vx_cfg = VertexConfig.from_env()
    agent = DataAgent(app_cfg, vx_cfg)

    question = "Can you forecast the metric for the next 14 days?"
    result = agent.handle_message(question)

    print("PLAN:", json.dumps(result["plan"], indent=2))
    print("\nRESPONSE TEXT:\n", result["text"])
