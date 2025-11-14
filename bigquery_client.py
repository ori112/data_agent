"""
bigquery_client.py

Thin wrapper around the Google BigQuery Python client.
Handles auth/config via environment variables so we can plug in
the real GCP project & service account later.
"""

from __future__ import annotations

import os
from typing import Optional, Sequence

import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account


class BigQueryClient:
    def __init__(
        self,
        project_id: str,
        location: str = "US",
        credentials: Optional[service_account.Credentials] = None,
    ) -> None:
        self.project_id = project_id
        self.location = location

        self._client = bigquery.Client(
            project=project_id,
            credentials=credentials,
            location=location,
        )

    @classmethod
    def from_env(cls) -> "BigQueryClient":
        """
        Factory method that reads basic config from env vars.

        Required (later):
        - GCP_PROJECT  -> your GCP project ID

        Optional:
        - BQ_LOCATION  -> e.g. 'US', 'EU'
        - GOOGLE_APPLICATION_CREDENTIALS -> path to service-account JSON
        """
        project_id = os.getenv("GCP_PROJECT")
        if not project_id:
            raise ValueError(
                "GCP_PROJECT env var is not set. "
                "Set it to your GCP project ID before running the app."
            )

        location = os.getenv("BQ_LOCATION", "US")

        cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        credentials = None
        if cred_path:
            credentials = service_account.Credentials.from_service_account_file(
                cred_path
            )

        return cls(project_id=project_id, location=location, credentials=credentials)

    def query_to_dataframe(
        self,
        sql: str,
        params: Optional[Sequence[bigquery.ScalarQueryParameter]] = None,
    ) -> pd.DataFrame:
        """
        Execute a SQL query and return results as a pandas DataFrame.
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=list(params) if params else None
        )

        query_job = self._client.query(sql, job_config=job_config)
        result = query_job.result()
        return result.to_dataframe(create_bqstorage_client=False)

    def get_time_series(
        self,
        dataset: str,
        table: str,
        date_column: str,
        target_column: str,
        limit: int = 365,
        where_clause: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Convenience method for loading a simple (date, target) time series.
        """
        table_ref = f"`{self.project_id}.{dataset}.{table}`"

        where_sql = f"WHERE {where_clause}" if where_clause else ""
        sql = f"""
        SELECT
          {date_column} AS ts,
          {target_column} AS y
        FROM {table_ref}
        {where_sql}
        ORDER BY ts
        LIMIT {int(limit)}
        """

        return self.query_to_dataframe(sql)
