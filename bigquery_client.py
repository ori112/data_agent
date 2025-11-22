# bigquery_client.py
"""
BigQueryClient
--------------
A tiny wrapper around google-cloud-bigquery that:
1) Loads config from .env via AppConfig-like env vars.
2) Creates a BigQuery client using ADC (Application Default Credentials).
3) Provides:
   - query_to_dataframe(sql)
   - get_time_series(...) returning a clean df with columns: ts, y

Design goals:
- KEEP SIMPLE for course submission.
- FULL HISTORY by default.
- Safe parameterization of dataset/table identifiers.

Expected .env variables:
- GCP_PROJECT
- BQ_LOCATION (optional)
- GOOGLE_APPLICATION_CREDENTIALS (optional local dev)

"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import pandas as pd
from google.cloud import bigquery


@dataclass
class BigQueryClient:
    """
    Holds a BigQuery client + minimal config.

    project_id:
        GCP project to bill queries to.
    location:
        BigQuery processing location (US/EU/etc).
    client:
        google.cloud.bigquery.Client instance.
    """

    project_id: str
    location: str = "US"
    client: Optional[bigquery.Client] = None

    # ---------- constructors ----------

    @classmethod
    def from_env(cls) -> "BigQueryClient":
        """
        Create client from environment variables.
        Uses Application Default Credentials (ADC):
          - If GOOGLE_APPLICATION_CREDENTIALS points to a service-account json => used.
          - Otherwise uses gcloud / Vertex runtime identity.
        """
        project_id = os.getenv("GCP_PROJECT", "")
        if not project_id:
            raise ValueError("GCP_PROJECT is not set in .env")

        location = os.getenv("BQ_LOCATION", "US")

        # Instantiate the BigQuery client.
        # ADC automatically picks up service account json or gcloud auth.
        client = bigquery.Client(project=project_id, location=location)

        return cls(project_id=project_id, location=location, client=client)

    # ---------- core methods ----------

    def query_to_dataframe(self, sql: str) -> pd.DataFrame:
        """
        Run a SQL query and return results as a pandas DataFrame.

        Note:
        BigQuery's to_dataframe() requires db-dtypes for some types.
        If you get:
            "Please install the 'db-dtypes' package"
        then run:
            uv add db-dtypes
        """
        if not self.client:
            raise RuntimeError("BigQuery client not initialized")

        job = self.client.query(sql)
        result = job.result()
        return result.to_dataframe(create_bqstorage_client=False)

    def get_time_series(
        self,
        dataset: str,
        table: str,
        date_column: str,
        target_column: str,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Pull a time series from a dataset.table.

        Returns df with:
          ts: datetime64[ns]
          y : float

        Parameters
        ----------
        dataset, table:
            identifiers in BigQuery.
        date_column:
            column holding timestamps (DATE/TIMESTAMP).
        target_column:
            numeric series column.
        limit:
            if None => pull full history.
        """
        if not self.client:
            raise RuntimeError("BigQuery client not initialized")

        limit_clause = f"LIMIT {limit}" if limit else ""

        sql = f"""
        SELECT
          CAST({date_column} AS DATE) AS ts,
          CAST({target_column} AS FLOAT64) AS y
        FROM `{self.project_id}.{dataset}.{table}`
        WHERE {date_column} IS NOT NULL
          AND {target_column} IS NOT NULL
        ORDER BY ts
        {limit_clause}
        """

        df = self.query_to_dataframe(sql)

        # Ensure correct types
        df["ts"] = pd.to_datetime(df["ts"])
        df["y"] = pd.to_numeric(df["y"], errors="coerce")

        df = df.dropna(subset=["ts", "y"]).sort_values("ts").reset_index(drop=True)
        return df
