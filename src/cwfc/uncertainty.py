"""Uncertainty estimation and capacity headroom calibration."""
from __future__ import annotations

import numpy as np
import pandas as pd


class ResidualIntervalCalibrator:
    """Calibrate prediction intervals and capacity buffers using historical out-of-fold residuals.

    Conventions
    -----------
    Signed residual definition:
        e = y_actual - y_predicted

    - Positive residual (e > 0): Actual workload exceeded forecast (under-forecasting).
    - Negative residual (e < 0): Actual workload was below forecast (over-forecasting).

    For capacity planning, an operator sets:
        Capacity(t) = y_predicted(t) + Buffer

    Under-provisioning occurs if y_actual > Capacity, which is e > Buffer.
    A one-sided (1 - alpha) buffer is calibrated as the (1 - alpha) percentile of `e`:
        Buffer_{1-alpha} = Quantile_{1-alpha}(e)

    Parameters
    ----------
    coverage : float, default=0.80
        Nominal target coverage level for two-sided intervals.
    one_sided_level : float, default=0.95
        Nominal target coverage for one-sided upper capacity buffer.
    """

    def __init__(self, coverage: float = 0.80, one_sided_level: float = 0.95):
        if not (0.0 < coverage < 1.0):
            raise ValueError("coverage must be between 0 and 1")
        if not (0.0 < one_sided_level < 1.0):
            raise ValueError("one_sided_level must be between 0 and 1")
        self.coverage = coverage
        self.one_sided_level = one_sided_level
        self.lower_quantile_ = None
        self.upper_quantile_ = None
        self.one_sided_buffer_ = None
        self.n_samples_ = 0

    def fit(self, y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> ResidualIntervalCalibrator:
        """Fit empirical residual quantiles from historical out-of-fold predictions."""
        residuals = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
        self.n_samples_ = len(residuals)
        if self.n_samples_ == 0:
            raise ValueError("Residual sample cannot be empty")

        alpha = 1.0 - self.coverage
        # Two-sided empirical interval bounds
        self.lower_quantile_ = float(np.percentile(residuals, 100.0 * (alpha / 2.0)))
        self.upper_quantile_ = float(np.percentile(residuals, 100.0 * (1.0 - alpha / 2.0)))

        # One-sided upper capacity buffer (covers 1 - alpha of actual demands)
        self.one_sided_buffer_ = float(np.percentile(residuals, 100.0 * self.one_sided_level))
        return self

    def predict_intervals(
        self, y_pred: np.ndarray | pd.Series, clip_bounds: tuple[float, float] = (0.0, 100.0)
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute two-sided lower and upper prediction interval bounds."""
        if self.lower_quantile_ is None or self.upper_quantile_ is None:
            raise RuntimeError("Calibrator must be fitted before predict_intervals")
        preds = np.asarray(y_pred, dtype=float)
        lower = preds + self.lower_quantile_
        upper = preds + self.upper_quantile_
        if clip_bounds:
            lower = np.clip(lower, clip_bounds[0], clip_bounds[1])
            upper = np.clip(upper, clip_bounds[0], clip_bounds[1])
        return lower, upper

    def compute_headroom_capacity(
        self, y_pred: np.ndarray | pd.Series, clip_bounds: tuple[float, float] = (0.0, 100.0)
    ) -> np.ndarray:
        """Compute one-sided capacity allocation with the calibrated headroom buffer."""
        if self.one_sided_buffer_ is None:
            raise RuntimeError("Calibrator must be fitted before compute_headroom_capacity")
        preds = np.asarray(y_pred, dtype=float)
        capacity = preds + self.one_sided_buffer_
        if clip_bounds:
            capacity = np.clip(capacity, clip_bounds[0], clip_bounds[1])
        return capacity
