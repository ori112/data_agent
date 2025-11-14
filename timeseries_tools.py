# timeseries_tools.py

from __future__ import annotations

from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose


# -------------------------------------------------------------------
# Basic visualizations and transforms
# -------------------------------------------------------------------


def plot_time_series(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    title: str | None = None,
) -> plt.Figure:
    """Line plot of the time series."""
    fig, ax = plt.subplots(figsize=(16, 6))
    sns.lineplot(data=df, x=date_col, y=value_col, ax=ax)
    ax.set_title(title or f"{value_col} over time")
    ax.tick_params(axis="x", rotation=90)
    fig.tight_layout()
    return fig


def log_transform(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Return a copy of df with log transform applied to value_col."""
    df_log = df.copy()
    df_log[value_col] = np.log(df_log[value_col])
    return df_log


# -------------------------------------------------------------------
# Diagnostics: ACF, PACF, decomposition, ADF
# -------------------------------------------------------------------


def plot_acf_pacf_series(
    series: pd.Series,
    lags: int = 40,
) -> Tuple[plt.Figure, plt.Figure]:
    """Return ACF and PACF plots for a series."""
    fig_acf, ax_acf = plt.subplots(figsize=(12, 6))
    plot_acf(series, lags=lags, ax=ax_acf)
    ax_acf.set_title("ACF")

    fig_pacf, ax_pacf = plt.subplots(figsize=(12, 6))
    plot_pacf(series, lags=lags, ax=ax_pacf)
    ax_pacf.set_title("PACF")

    fig_acf.tight_layout()
    fig_pacf.tight_layout()
    return fig_acf, fig_pacf


def seasonal_decompose_plot(
    series: pd.Series,
    period: int = 12,
    model: str = "additive",
) -> plt.Figure:
    """Run seasonal decomposition and return the figure."""
    decompose = seasonal_decompose(series, model=model, period=period)
    fig = decompose.plot()
    fig.set_size_inches(12, 8)
    fig.tight_layout()
    return fig


def run_adf_test(series: pd.Series) -> Dict[str, Any]:
    """Run Augmented Dickey-Fuller test and return stats + interpretation."""
    result = adfuller(series.dropna())
    stat, pvalue, usedlag, nobs, crit_vals, icbest = result

    is_stationary = pvalue < 0.05
    interpretation = (
        "Series appears STATIONARY (p < 0.05, reject H0 of unit root)."
        if is_stationary
        else "Series appears NON-STATIONARY (p >= 0.05, cannot reject H0 of unit root)."
    )

    return {
        "adf_statistic": stat,
        "p_value": pvalue,
        "used_lag": usedlag,
        "n_obs": nobs,
        "critical_values": crit_vals,
        "ic_best": icbest,
        "is_stationary": is_stationary,
        "interpretation": interpretation,
    }


# -------------------------------------------------------------------
# Train / test split and SARIMA modeling
# -------------------------------------------------------------------


def train_test_split_series(
    series: pd.Series,
    train_ratio: float = 0.8,
) -> Tuple[pd.Series, pd.Series]:
    """Simple chronological split of a series into train and test parts."""
    n = len(series)
    train_size = int(n * train_ratio)
    train = series.iloc[:train_size]
    test = series.iloc[train_size:]
    return train, test


def fit_sarima(
    train: pd.Series,
    order: tuple[int, int, int] = (1, 0, 0),
    seasonal_order: tuple[int, int, int, int] = (0, 1, 1, 12),
) -> Any:
    """Fit SARIMAX model and return fitted results object."""
    model = SARIMAX(
        train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    results = model.fit()
    return results


def residual_diagnostics(
    results: Any,
    lags: int = 30,
) -> Tuple[List[plt.Figure], pd.Series]:
    """Plot residuals + ACF/PACF of residuals; return figures and residual series."""
    residuals = results.resid

    figs: List[plt.Figure] = []

    # Residuals over time
    fig_res, ax_res = plt.subplots(figsize=(10, 4))
    ax_res.plot(residuals)
    ax_res.set_title("Residuals of SARIMA Model")
    fig_res.tight_layout()
    figs.append(fig_res)

    # ACF of residuals
    fig_acf, ax_acf = plt.subplots(figsize=(10, 4))
    plot_acf(residuals, lags=lags, ax=ax_acf)
    ax_acf.set_title("Residuals ACF")
    fig_acf.tight_layout()
    figs.append(fig_acf)

    # PACF of residuals
    fig_pacf, ax_pacf = plt.subplots(figsize=(10, 4))
    plot_pacf(residuals, lags=lags, ax=ax_pacf)
    ax_pacf.set_title("Residuals PACF")
    fig_pacf.tight_layout()
    figs.append(fig_pacf)

    return figs, residuals


def sarima_forecast_plots(
    train: pd.Series,
    test: pd.Series,
    results: Any,
    back_transform: bool = False,
) -> Tuple[plt.Figure, pd.Series, pd.DataFrame]:
    """
    Forecast the length of `test` and produce a plot.
    If back_transform=True, exponentiate values to original scale.
    Returns (figure, forecast_mean, conf_int_df).
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
