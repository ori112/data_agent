# vertex_agent.py
"""
vertex_agent.py
---------------
The LLM-powered brain.

Key guarantees for your course:
1) ONLY TWO intents:
   - "eda": run diagnostics + comparisons + ADF + interpretation.
   - "model": run SARIMA workflow + interpretation.

2) FULL HISTORY ALWAYS:
   _load_series() pulls the entire BigQuery table in order.

3) FLEXIBLE routing:
   You don't need perfect prompts —
   heuristic routing happens before Gemini.

Gemini's role:
- Interpret deterministic results and explain to the user.
- Not decide *whether* to model unless you ask.

"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import vertexai
from vertexai.generative_models import GenerativeModel

from bigquery_client import BigQueryClient
from utils import AppConfig
from timeseries_tools import (
    diagnostics_bundle,
    log_transform,
    run_adf_test,
    train_test_split_series,
    fit_sarima,
    residual_diagnostics,
    sarima_forecast_plots,
)


@dataclass
class VertexConfig:
    """
    Configuration for Vertex AI / Gemini.

    project_id:
        GCP project hosting Vertex.
    location:
        Region (us-central1 recommended).
    model_name:
        Gemini publisher model id.
    """

    project_id: str
    location: str
    model_name: str = "gemini-1.5-pro-001"

    @classmethod
    def from_env(cls) -> "VertexConfig":
        project_id = os.getenv("GCP_PROJECT")
        if not project_id:
            raise ValueError("GCP_PROJECT must be set for Vertex AI.")
        location = os.getenv("VERTEX_LOCATION", "us-central1")
        model_name = os.getenv("VERTEX_MODEL", "gemini-1.5-pro-001")
        return cls(project_id=project_id, location=location, model_name=model_name)


class DataAgent:
    """
    Two-use-case agent:
      - EDA
      - MODEL
    """

    def __init__(self, app_config: AppConfig, vertex_config: VertexConfig) -> None:
        self.app_config = app_config
        self.vertex_config = vertex_config

        # Initialize Vertex runtime
        vertexai.init(project=vertex_config.project_id, location=vertex_config.location)
        self.model = GenerativeModel(vertex_config.model_name)

        # BigQuery client
        self.bq_client = BigQueryClient.from_env()

    # ---------------- Intent routing ----------------

    def _route_intent(self, msg: str) -> str:
        """
        Determine if user wants EDA or MODEL.
        Defaults to EDA when unsure (as you requested).
        """
        m = msg.lower()

        modeling_keywords = [
            "forecast",
            "predict",
            "model",
            "train",
            "fit",
            "arima",
            "sarima",
            "seasonal arima",
            "future",
            "projection",
        ]
        eda_keywords = [
            "eda",
            "explore",
            "diagnose",
            "plot",
            "visual",
            "acf",
            "pacf",
            "adf",
            "stationar",
            "seasonal",
            "decompose",
            "trend",
            "compare",
            "comparison",
            "vs",
            "versus",
            "between",
            "average",
            "mean",
            "min",
            "max",
            "summary",
            "describe",
        ]

        if any(k in m for k in modeling_keywords):
            return "model"

        if any(k in m for k in eda_keywords):
            return "eda"

        return "eda"

    # ---------------- Public API ----------------

    def handle_message(self, user_message: str) -> Dict[str, Any]:
        """
        Main entry point from app.py.
        Returns:
          text: narrative/interpretation
          figures: list of matplotlib figures
          tables: dict of dataframes
        """
        intent = self._route_intent(user_message)

        if intent == "eda":
            base_text, figures, tables, stats = self._run_eda(user_message)
            final_text = self._narrate_with_model(
                user_message=user_message,
                mode="EDA",
                stats=stats,
                base_text=base_text,
            )
            return {
                "text": final_text,
                "plan": {"intent": "eda"},
                "figures": figures,
                "tables": tables,
            }

        base_text, figures, tables, stats = self._run_model(user_message)
        final_text = self._narrate_with_model(
            user_message=user_message,
            mode="MODEL",
            stats=stats,
            base_text=base_text,
        )
        return {
            "text": final_text,
            "plan": {"intent": "model"},
            "figures": figures,
            "tables": tables,
        }

    # ---------------- Data loading ----------------

    def _load_series(self) -> pd.DataFrame:
        """
        always full history from your mart table.
        """
        df = self.bq_client.get_time_series(
            dataset=self.app_config.dataset_id,
            table=self.app_config.table_id,
            date_column=self.app_config.date_column,
            target_column=self.app_config.target_column,
            limit=None,
        )
        if df.empty:
            raise ValueError("BigQuery returned no data.")
        return df

    # ---------------- EDA mode ----------------

    def _run_eda(
        self, user_message: str
    ) -> tuple[str, List[plt.Figure], Dict[str, pd.DataFrame], Dict[str, Any]]:
        """
        EDA includes:
          1) optional year comparison
          2) diagnostics bundle (log plot + ACF/PACF + decomposition)
          3) ADF test
        """
        df = self._load_series()

        figures: List[plt.Figure] = []
        tables: Dict[str, pd.DataFrame] = {}
        stats: Dict[str, Any] = {"eda_performed": []}

        # 1) Comparisons / summaries if asked
        if self._looks_like_comparison(user_message):
            sql = self._build_comparison_sql(user_message)
            agg_df = self.bq_client.query_to_dataframe(sql)
            tables["comparison"] = agg_df
            stats["eda_performed"].append("comparison_query")
            stats["comparison_sql"] = sql
            stats["comparison_result_head"] = agg_df.head(10).to_dict()

        # 2) Diagnostics bundle
        bundle = diagnostics_bundle(
            df, ts_col="ts", target_col="y", seasonal_period=12, lags=40
        )
        figures.extend(bundle["figures"])
        stats["eda_performed"].append("diagnostics_bundle")
        stats["diagnostics_stats"] = bundle["stats"]

        # 3) ADF test on log series
        df_log = log_transform(df, "y")
        series_log = df_log.set_index("ts")["y"]
        adf_stats = run_adf_test(series_log)
        tables["adf_summary"] = pd.DataFrame(
            {
                "ADF Statistic": [adf_stats["adf_statistic"]],
                "p-value": [adf_stats["p_value"]],
                "used_lag": [adf_stats["used_lag"]],
                "n_obs": [adf_stats["n_obs"]],
                "is_stationary": [adf_stats["is_stationary"]],
            }
        )
        stats["eda_performed"].append("adf_test")
        stats["adf"] = adf_stats

        base_text = (
            "EDA complete on FULL HISTORY.\n"
            "- log(y) series plot\n"
            "- ACF/PACF\n"
            "- seasonal decomposition\n"
            "- ADF stationarity test\n"
        )
        if "comparison" in tables:
            base_text += "- comparison table from BigQuery\n"

        return base_text, figures, tables, stats

    def _looks_like_comparison(self, msg: str) -> bool:
        """
        comparison detection:
        - if user uses words: compare/vs/between/average/mean
        - and includes years.
        """
        m = msg.lower()
        if any(
            k in m
            for k in [
                "compare",
                "comparison",
                "vs",
                "versus",
                "between",
                "average",
                "mean",
            ]
        ):
            if re.search(r"\b(19|20)\d{2}\b", m):
                return True
        return False

    def _build_comparison_sql(self, msg: str) -> str:
        """
        Extract years from prompt and build AVG/MIN/MAX comparison.
        AVG is correct for an index-series.
        """
        years = sorted(set(int(y) for y in re.findall(r"\b(19|20)\d{2}\b", msg)))
        if not years:
            years = [2023, 2024, 2025]

        years_sql = ", ".join(str(y) for y in years)

        return f"""
        SELECT
          EXTRACT(YEAR FROM {self.app_config.date_column}) AS year,
          AVG({self.app_config.target_column}) AS avg_index,
          MIN({self.app_config.target_column}) AS min_index,
          MAX({self.app_config.target_column}) AS max_index,
          COUNT(1) AS n_months
        FROM `{self.app_config.project_id}.{self.app_config.dataset_id}.{self.app_config.table_id}`
        WHERE EXTRACT(YEAR FROM {self.app_config.date_column}) IN ({years_sql})
        GROUP BY year
        ORDER BY year
        """

    # ---------------- MODEL mode ----------------

    def _run_model(
        self, user_message: str
    ) -> tuple[str, List[plt.Figure], Dict[str, pd.DataFrame], Dict[str, Any]]:
        """
        Full SARIMA workflow:
          - log transform
          - train/test split (80/20)
          - SARIMA(1,0,0)x(0,1,1,12)
          - residual diagnostics
          - forecast plots (log + original scale)
        """
        df = self._load_series()
        df_log = log_transform(df, "y")
        series_log = df_log.set_index("ts")["y"]

        train, test = train_test_split_series(series_log, 0.8)

        order = (1, 0, 0)
        seasonal_order = (0, 1, 1, 12)
        results = fit_sarima(train, order, seasonal_order)

        figures: List[plt.Figure] = []
        tables: Dict[str, pd.DataFrame] = {}

        diag_figs, residuals, resid_stats = residual_diagnostics(results, 30)
        figures.extend(diag_figs)

        fig_log, fc_mean_log, ci_log = sarima_forecast_plots(
            train, test, results, back_transform=False
        )
        figures.append(fig_log)

        fig_orig, fc_mean_orig, ci_orig = sarima_forecast_plots(
            train, test, results, back_transform=True
        )
        figures.append(fig_orig)

        tables["sarima_residuals"] = residuals.to_frame("residuals")
        tables["sarima_forecast_log"] = pd.DataFrame(
            {
                "forecast_mean": fc_mean_log,
                "ci_lower": ci_log.iloc[:, 0],
                "ci_upper": ci_log.iloc[:, 1],
            }
        )
        tables["sarima_forecast_original"] = pd.DataFrame(
            {
                "forecast_mean": np.exp(fc_mean_orig),
                "ci_lower": np.exp(ci_orig.iloc[:, 0]),
                "ci_upper": np.exp(ci_orig.iloc[:, 1]),
            },
            index=fc_mean_orig.index,
        )

        stats = {
            "mode": "model",
            "order": order,
            "seasonal_order": seasonal_order,
            "seasonal_period": 12,
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "aic": float(results.aic),
            "bic": float(results.bic),
            "residual_stats": resid_stats,
            "note": "Model fit on log(y). Forecast shown in log + original scales.",
        }

        base_text = (
            "MODEL run complete on FULL HISTORY.\n"
            f"- SARIMA{order} x {seasonal_order}\n"
            "- residual diagnostics\n"
            "- forecast (log + original scale)\n"
        )
        return base_text, figures, tables, stats

    # ---------------- Gemini narrator ----------------

    def _narrate_with_model(
        self,
        user_message: str,
        mode: str,
        stats: Dict[str, Any],
        base_text: str,
    ) -> str:
        """
        Ask Gemini to interpret deterministic stats.

        This is the only LLM call:
        - It does NOT choose intent
        - It only writes a student-friendly explanation.
        """
        prompt = f"""
You are an expert time-series analyst writing a short, course-level interpretation.

Mode: {mode}

User request:
{user_message}

Deterministic results (JSON):
{json.dumps(stats, indent=2)}

Write:
1) Interpretation of results (trend/seasonality/stationarity/ACF-PACF or model fit).
2) Clear modeling guidance/recommendation.
3) If EDA includes year comparison, explain what the averages mean for an INDEX.
Be flexible, answer the user’s intent first.
Keep it concise but useful.

Start with "Summary:" (one line), then 1-2 short sections.
"""
        try:
            resp = self.model.generate_content(prompt)
            narrative = (resp.text or "").strip()
            if narrative:
                return f"{base_text}\n\n{narrative}"
        except Exception:
            pass

        return base_text + "\n\n(Interpretation unavailable — model call failed.)"
