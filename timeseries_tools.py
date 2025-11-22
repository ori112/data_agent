# timeseries_tools.py
"""
timeseries_tools.py
-------------------
Pure statistical/plotting tools for EDA + modeling.

This module is intentionally *deterministic*:
- No LLM calls here.
- Only math, statsmodels, seaborn/matplotlib.

The Vertex agent calls these tools and then asks Gemini
to *interpret* the deterministic results.

Main functions:
1) diagnostics_bundle(df):
   - log transform
   - line plot
   - ACF/PACF
   - seasonal decomposition
   - returns figures + computed stats for interpretation

2) Modeling helpers:
   - train_test_split_series
   - fit_sarima
   - residual_diagnostics
   - sarima_forecast_plots

"""

from __future__ import annotations

from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose


# ---------------- Basic visual helpers ----------------


def plot_time_series(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    title: str | None = None,
) -> plt.Figure:
    """
    Line plot of a time series dataframe.
    Returns a matplotlib Figure for Streamlit to render.
    """
    fig, ax = plt.subplots(figsize=(16, 6))
    sns.lineplot(data=df, x=date_col, y=value_col, ax=ax)
    ax.set_title(title or f"{value_col} over time")
    ax.tick_params(axis="x", rotation=90)
    fig.tight_layout()
    return fig


def log_transform(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """
    Take natural log of a series column.
    Used to stabilize variance (heteroscedasticity).
    """
    df_log = df.copy()
    df_log[value_col] = np.log(df_log[value_col])
    return df_log


def plot_acf_pacf_series(
    series: pd.Series,
    lags: int = 40,
) -> Tuple[plt.Figure, plt.Figure]:
    """
    Plot ACF and PACF for a given univariate series.
    Returns (acf_fig, pacf_fig).
    """
    fig_acf, ax_acf = plt.subplots(figsize=(12, 6))
    plot_acf(series, lags=lags, ax=ax_acf)
    ax_acf.set_title("ACF")
    fig_acf.tight_layout()

    fig_pacf, ax_pacf = plt.subplots(figsize=(12, 6))
    plot_pacf(series, lags=lags, ax=ax_pacf)
    ax_pacf.set_title("PACF")
    fig_pacf.tight_layout()

    return fig_acf, fig_pacf


def seasonal_decompose_plot(
    series: pd.Series,
    period: int = 12,
    model: str = "additive",
) -> plt.Figure:
    """
    Seasonal decomposition (trend/seasonal/residual).
    period=12 matches monthly seasonality.
    """
    decompose = seasonal_decompose(series, model=model, period=period)
    fig = decompose.plot()
    fig.set_size_inches(12, 8)
    fig.tight_layout()
    return fig


# ---------------- Stats helpers ----------------


def run_adf_test(series: pd.Series) -> Dict[str, Any]:
    """
    Augmented Dickey-Fuller test for stationarity.
    Returns statistics + human-readable interpretation.
    """
    result = adfuller(series.dropna())
    stat, pvalue, usedlag, nobs, crit_vals, icbest = result

    is_stationary = pvalue < 0.05
    interpretation = (
        "Series appears STATIONARY (p < 0.05, reject H0 of unit root)."
        if is_stationary
        else "Series appears NON-STATIONARY (p >= 0.05, cannot reject H0 of unit root)."
    )

    return {
        "adf_statistic": float(stat),
        "p_value": float(pvalue),
        "used_lag": int(usedlag),
        "n_obs": int(nobs),
        "critical_values": {k: float(v) for k, v in crit_vals.items()},
        "ic_best": float(icbest),
        "is_stationary": bool(is_stationary),
        "interpretation": interpretation,
    }


def train_test_split_series(
    series: pd.Series,
    train_ratio: float = 0.8,
) -> Tuple[pd.Series, pd.Series]:
    """
    Chronological split for time-series.
    (No shuffle!)
    """
    n = len(series)
    train_size = int(n * train_ratio)
    train = series.iloc[:train_size]
    test = series.iloc[train_size:]
    return train, test


def fit_sarima(
    train: pd.Series,
    order: tuple[int, int, int] = (1, 0, 0),
    seasonal_order: tuple[int, int, int, int] = (0, 1, 1, 12),
):
    """
    Fit SARIMA model on train series.
    We turn off enforce_stationarity/invertibility because real data is messy.
    """
    model = SARIMAX(
        train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit()


def residual_diagnostics(
    results,
    lags: int = 30,
) -> Tuple[List[plt.Figure], pd.Series, Dict[str, Any]]:
    """
    Residual diagnostics:
      - residual time plot
      - residual ACF
      - residual PACF

    Returns:
      figs, residuals_series, residual_stats
    """
    residuals = results.resid
    figs: List[plt.Figure] = []

    fig_res, ax_res = plt.subplots(figsize=(10, 4))
    ax_res.plot(residuals)
    ax_res.set_title("Residuals of SARIMA Model")
    fig_res.tight_layout()
    figs.append(fig_res)

    fig_acf, ax_acf = plt.subplots(figsize=(10, 4))
    plot_acf(residuals, lags=lags, ax=ax_acf)
    ax_acf.set_title("Residuals ACF")
    fig_acf.tight_layout()
    figs.append(fig_acf)

    fig_pacf, ax_pacf = plt.subplots(figsize=(10, 4))
    plot_pacf(residuals, lags=lags, ax=ax_pacf)
    ax_pacf.set_title("Residuals PACF")
    fig_pacf.tight_layout()
    figs.append(fig_pacf)

    resid_stats = {
        "resid_mean": float(np.mean(residuals)),
        "resid_std": float(np.std(residuals)),
        "resid_min": float(np.min(residuals)),
        "resid_max": float(np.max(residuals)),
    }

    return figs, residuals, resid_stats


def sarima_forecast_plots(
    train: pd.Series,
    test: pd.Series,
    results,
    back_transform: bool = False,
):
    """
    Create forecast plot on train/test split.

    back_transform=True:
      assumes series is log-scale and exp() back to original.
    """
    steps = len(test)
    forecast = results.get_forecast(steps=steps)
    forecast_mean = forecast.predicted_mean
    conf_int = forecast.conf_int()

    if back_transform:
        train_plot = np.exp(train)
        test_plot = np.exp(test)
        forecast_plot = np.exp(forecast_mean)
        conf_plot = np.exp(conf_int)
        title = "SARIMA Forecast (original scale)"
    else:
        train_plot = train
        test_plot = test
        forecast_plot = forecast_mean
        conf_plot = conf_int
        title = "SARIMA Forecast (log scale)"

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(train.index, train_plot, label="Train")
    ax.plot(test.index, test_plot, label="Test", color="gray")
    ax.plot(test.index, forecast_plot, label="Forecast", color="green")
    ax.fill_between(
        test.index,
        conf_plot.iloc[:, 0],
        conf_plot.iloc[:, 1],
        color="lightgreen",
        alpha=0.3,
    )
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    return fig, forecast_mean, conf_int


# ---------------- EDA bundle ----------------


def diagnostics_bundle(
    df: pd.DataFrame,
    ts_col: str = "ts",
    target_col: str = "y",
    seasonal_period: int = 12,
    lags: int = 40,
) -> Dict[str, Any]:
    """
    Full deterministic EDA bundle.

    Steps:
    1) sort + clean data
    2) log transform y
    3) plot log series
    4) plot ACF/PACF on log series
    5) seasonal decomposition on log series
    6) compute summary stats for LLM interpretation
    """
    df = df[[ts_col, target_col]].dropna().sort_values(ts_col)
    series = df.set_index(ts_col)[target_col]

    df_log = df.copy()
    df_log[target_col] = np.log(df_log[target_col])
    series_log = df_log.set_index(ts_col)[target_col]

    figures: List[plt.Figure] = []
    figures.append(
        plot_time_series(df_log, ts_col, target_col, "Log-transformed series")
    )

    fig_acf, fig_pacf = plot_acf_pacf_series(series_log, lags=lags)
    figures.extend([fig_acf, fig_pacf])

    figures.append(seasonal_decompose_plot(series_log, period=seasonal_period))

    acf_vals = acf(series_log.dropna(), nlags=lags)
    pacf_vals = pacf(series_log.dropna(), nlags=lags)

    stats = {
        "n_rows": int(len(series_log)),
        "date_min": str(df[ts_col].min().date()),
        "date_max": str(df[ts_col].max().date()),
        "mean_log": float(series_log.mean()),
        "std_log": float(series_log.std()),
        "acf_first_5": [float(x) for x in acf_vals[1:6]],
        "pacf_first_5": [float(x) for x in pacf_vals[1:6]],
        "acf_lag12": float(acf_vals[12]) if len(acf_vals) > 12 else None,
        "acf_lag24": float(acf_vals[24]) if len(acf_vals) > 24 else None,
        "pacf_lag12": float(pacf_vals[12]) if len(pacf_vals) > 12 else None,
        "pacf_lag24": float(pacf_vals[24]) if len(pacf_vals) > 24 else None,
        "seasonal_period_assumed": seasonal_period,
        "notes": "Computed on log(y).",
    }

    return {"figures": figures, "stats": stats}
