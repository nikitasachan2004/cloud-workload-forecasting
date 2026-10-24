"""Leakage-safe causal feature engineering for workload forecasting."""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_lag_features(
    series: pd.Series,
    lags: list[int] = [1, 2, 3, 4, 6],
    rolling_windows: list[int] = [3, 6, 12],
    prefix: str = "target",
) -> pd.DataFrame:
    """Construct strictly causal lag and rolling features from historical target values.

    Leakage Prevention Guarantee:
    All features at row index `t` depend strictly on observations at or before `t - 1`.
    The current observation `t` is shifted out before computing rolling statistics.

    Parameters
    ----------
    series : pd.Series
        Historical target series in chronological order.
    lags : list[int]
        Discrete lag offsets (in time steps). Each lag `k` corresponds to series.shift(k).
    rolling_windows : list[int]
        Window lengths for rolling statistics computed on the 1-step shifted series.
    prefix : str
        Prefix for generated feature column names.

    Returns
    -------
    pd.DataFrame
        DataFrame with index aligned to `series`, containing causal features.
    """
    df_feat = pd.DataFrame(index=series.index)

    # 1. Discrete Lags (shift >= 1)
    for lag in lags:
        if lag < 1:
            raise ValueError(f"Lag must be >= 1 to prevent leakage, got {lag}")
        df_feat[f"{prefix}_lag_{lag}"] = series.shift(lag)

    # 2. Shifted series for rolling calculations (so current target is excluded)
    shifted = series.shift(1)

    for w in rolling_windows:
        if w < 2:
            raise ValueError(f"Rolling window must be >= 2, got {w}")
        df_feat[f"{prefix}_roll_mean_{w}"] = shifted.rolling(window=w, min_periods=1).mean()
        df_feat[f"{prefix}_roll_std_{w}"] = shifted.rolling(window=w, min_periods=2).std().fillna(0.0)
        df_feat[f"{prefix}_roll_min_{w}"] = shifted.rolling(window=w, min_periods=1).min()
        df_feat[f"{prefix}_roll_max_{w}"] = shifted.rolling(window=w, min_periods=1).max()

    # 3. Recent Momentum / Velocity (changes over last 5-15 min)
    df_feat[f"{prefix}_diff_1_2"] = series.shift(1) - series.shift(2)
    df_feat[f"{prefix}_diff_1_4"] = series.shift(1) - series.shift(4)

    return df_feat
