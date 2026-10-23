"""Baseline forecasting models."""
from __future__ import annotations

import pandas as pd


def persistence_forecast(train: pd.DataFrame, template: pd.DataFrame, target: str) -> pd.Series:
    """Predict the last observed historical target value for all horizon steps."""
    if train.empty:
        raise ValueError("train must not be empty")
    last_value = float(train.iloc[-1][target])
    return pd.Series(last_value, index=template.index, dtype=float)


def moving_average_forecast(train: pd.DataFrame, template: pd.DataFrame, target: str, window: int = 6) -> pd.Series:
    """Predict the mean of the trailing `window` historical observations."""
    if train.empty:
        raise ValueError("train must not be empty")
    if window <= 0:
        raise ValueError("window must be positive")
    values = train[target].astype(float).to_numpy()
    if len(values) < window:
        mean_val = float(values.mean())
    else:
        mean_val = float(values[-window:].mean())
    return pd.Series(mean_val, index=template.index, dtype=float)


def seasonal_naive_forecast(train: pd.DataFrame, template: pd.DataFrame, target: str, season_length: int) -> pd.Series:
    """Predict the historical value from `season_length` steps prior."""
    if season_length <= 0:
        raise ValueError("season_length must be positive")
    values = train[target].astype(float).to_numpy()
    if len(values) < season_length:
        raise ValueError("not enough history for seasonal-naive forecast")
    forecast = [float(values[-season_length + (i % season_length)]) for i in range(len(template))]
    return pd.Series(forecast, index=template.index, dtype=float)
