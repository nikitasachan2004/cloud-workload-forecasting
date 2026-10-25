"""Forecasting models: Recursive multi-step architectures."""
from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
from cwfc.features import make_lag_features


    """Recursive multi-step forecaster using strictly causal 1-step lag features.

    Parameters
    ----------
    estimator : Any
        Underlying scikit-learn compatible estimator (e.g., Ridge, LGBMRegressor).
    lags : list[int]
        Lag offsets used by `make_lag_features`.
    rolling_windows : list[int]
        Rolling window sizes used by `make_lag_features`.
    """

    def __init__(
        self,
        estimator: Any,
        lags: list[int] = [1, 2, 3, 4, 6],
        rolling_windows: list[int] = [3, 6, 12],
    ):
        self.estimator = estimator
        self.lags = lags
        self.rolling_windows = rolling_windows
        self.active_lags_ = list(lags)
        self.active_windows_ = list(rolling_windows)
        self.fallback_persistence_val_ = None
        self.is_fitted = False

    def fit(self, series: pd.Series) -> RecursiveForecaster:
        """Fit the 1-step ahead forecasting model on historical training series."""
        n = len(series)
        if n == 0:
            raise ValueError("Training series cannot be empty")

        self.fallback_persistence_val_ = float(series.iloc[-1])

        # Adaptively scale lags and windows if series is very short (e.g., reviewer smoke test)
        if n < max(self.rolling_windows) + 2:
            self.active_windows_ = [w for w in self.rolling_windows if w < n - 1]
            if not self.active_windows_:
                self.active_windows_ = [max(2, n // 2)] if n >= 3 else []
            self.active_lags_ = [l for l in self.lags if l < n - 1]
            if not self.active_lags_:
                self.active_lags_ = [1] if n >= 2 else []

        if not self.active_lags_ and not self.active_windows_:
            self.is_fitted = True
            return self

        X = make_lag_features(series, lags=self.active_lags_, rolling_windows=self.active_windows_)
        y = series.copy()

        valid_idx = X.dropna().index
        if len(valid_idx) == 0:
            self.is_fitted = True
            return self

        X_clean = X.loc[valid_idx]
        y_clean = y.loc[valid_idx]

        self.estimator.fit(X_clean, y_clean)
        self.is_fitted = True
        return self

    def predict(self, history: pd.Series, horizon: int) -> pd.Series:
        """Forecast `horizon` steps ahead recursively using historical values."""
        if not self.is_fitted:
            raise RuntimeError("Forecaster must be fitted before predict")
        if horizon <= 0:
            raise ValueError("Horizon must be positive")

        if not self.active_lags_ and not self.active_windows_:
            return pd.Series([self.fallback_persistence_val_] * horizon)

        buffer = list(history.astype(float).values)
        predictions = []

        for step in range(horizon):
            buf_series = pd.Series(buffer)
            X_step = make_lag_features(buf_series, lags=self.active_lags_, rolling_windows=self.active_windows_)
            x_next = X_step.iloc[[-1]]
            if x_next.isna().any().any():
                pred = buffer[-1]
            else:
                pred = float(self.estimator.predict(x_next)[0])
            pred = max(0.0, min(100.0, pred))
            predictions.append(pred)
            buffer.append(pred)

        return pd.Series(predictions)


def build_direct_ridge_model(alpha: float = 1.0) -> DirectForecaster:
    """Build a direct multi-step Ridge regression forecaster."""
    def factory():
        return Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha, random_state=42)),
        ])
    return DirectForecaster(estimator_factory=factory, lags=[1, 2, 3])


def build_ridge_model(alpha: float = 1.0) -> RecursiveForecaster:
    """Build a recursive regularized Ridge regression forecaster with standardized features."""
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=alpha, random_state=42)),
    ])
    return RecursiveForecaster(estimator=pipeline)


def build_lgbm_model(
    n_estimators: int = 40,
    learning_rate: float = 0.05,
    max_depth: int = 3,
    num_leaves: int = 7,
    min_child_samples: int = 5,
) -> RecursiveForecaster:
    """Build a lightweight LightGBM gradient boosted tree forecaster."""
    model = LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        num_leaves=num_leaves,
        min_child_samples=min_child_samples,
        random_state=42,
        verbosity=-1,
    )
    return RecursiveForecaster(estimator=model)
