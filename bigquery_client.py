# bigquery_client.py

from __future__ import annotations

import os
from typing import Optional, Sequence

import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account
from dotenv import load_dotenv

load_dotenv()


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
        Creates a BigQuery client using environment variables (.env).
        """

        project_id = os.getenv("GCP_PROJECT")
        if not project_id:
            raise ValueError("GCP_PROJECT is not set in the environment.")

        location = os.getenv("BQ_LOCATION", "US")
        cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

        credentials = None
        if cred_path:
            if not os.path.exists(cred_path):
                raise FileNotFoundError(
                    f"Service account JSON not found at: {cred_path}"
                )
            credentials = service_account.Credentials.from_service_account_file(
                cred_path
            )

        return cls(
            project_id=project_id,
            location=location,
            credentials=credentials,
        )

    def query_to_dataframe(self, sql: str) -> pd.DataFrame:
        query_job = self._client.query(sql)
        result = query_job.result()
        return result.to_dataframe(create_bqstorage_client=False)

    def get_time_series(
        self,
        dataset: str,
        table: str,
        date_column: str,
        target_column: str,
        limit: int = 365,
    ) -> pd.DataFrame:
        table_ref = f"`{self.project_id}.{dataset}.{table}`"

        sql = f"""
        SELECT
          {date_column} AS ts,
          {target_column} AS y
        FROM {table_ref}
        ORDER BY ts
        LIMIT {limit}
        """

        return self.query_to_dataframe(sql)
