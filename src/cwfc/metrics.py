"""Evaluation metrics for time-series forecasting."""
from __future__ import annotations

import numpy as np


def mae(y_true, y_pred) -> float:
    """Mean Absolute Error."""
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    if a.shape != b.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if len(a) == 0:
        return 0.0
    return float(np.mean(np.abs(a - b)))


def rmse(y_true, y_pred) -> float:
    """Root Mean Squared Error."""
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    if a.shape != b.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if len(a) == 0:
        return 0.0
    return float(np.sqrt(np.mean((a - b) ** 2)))


def smape(y_true, y_pred, eps: float = 1e-8) -> float:
    """Symmetric Mean Absolute Percentage Error (sMAPE) in [0, 200] percent."""
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    if a.shape != b.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if len(a) == 0:
        return 0.0
    denom = np.maximum(np.abs(a) + np.abs(b), eps)
    return float(np.mean(2.0 * np.abs(a - b) / denom) * 100.0)


def normalized_mae(y_true, y_pred, baseline_pred) -> float:
    """Ratio of model MAE to naive/baseline MAE."""
    model_mae = mae(y_true, y_pred)
    baseline_mae = mae(y_true, baseline_pred)
    if baseline_mae == 0:
        return 0.0 if model_mae == 0 else float("inf")
    return float(model_mae / baseline_mae)


def mase(y_true, y_pred, y_train, seasonality: int = 1, eps: float = 1e-8) -> float:
    """Mean Absolute Scaled Error (MASE) relative to naive forecast on training history.

    Parameters
    ----------
    y_true : array-like
        Ground truth test values.
    y_pred : array-like
        Predicted values.
    y_train : array-like
        Historical training values used to compute the in-sample scale denominator.
    seasonality : int, default=1
        Lag for naive differencing. 1 for non-seasonal persistence.
    eps : float
        Numerical epsilon to avoid divide-by-zero.
    """
    model_mae = mae(y_true, y_pred)
    train_arr = np.asarray(y_train, dtype=float)
    if len(train_arr) <= seasonality:
        raise ValueError("y_train length must be greater than seasonality")
    scale = np.mean(np.abs(train_arr[seasonality:] - train_arr[:-seasonality]))
    denom = max(float(scale), eps)
    return float(model_mae / denom)


def interval_coverage(y_true, lower_bound, upper_bound) -> float:
    """Empirical coverage proportion of prediction intervals."""
    y = np.asarray(y_true, dtype=float)
    lb = np.asarray(lower_bound, dtype=float)
    ub = np.asarray(upper_bound, dtype=float)
    if y.shape != lb.shape or y.shape != ub.shape:
        raise ValueError("All inputs must have the same shape")
    if len(y) == 0:
        return 0.0
    in_interval = (y >= lb) & (y <= ub)
    return float(np.mean(in_interval))


def interval_width(lower_bound, upper_bound) -> float:
    """Mean width of prediction intervals."""
    lb = np.asarray(lower_bound, dtype=float)
    ub = np.asarray(upper_bound, dtype=float)
    if lb.shape != ub.shape:
        raise ValueError("lower_bound and upper_bound must have the same shape")
    if len(lb) == 0:
        return 0.0
    return float(np.mean(ub - lb))
