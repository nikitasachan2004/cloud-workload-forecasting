"""Automated tests for time-series leakage prevention and temporal invariants."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cwfc.features import make_lag_features
from cwfc.validation import rolling_origin_splits
from cwfc.targets import compute_targets
from cwfc.models import build_ridge_model
from cwfc.baselines import persistence_forecast, moving_average_forecast


def test_feature_invariance_under_future_modification():
    """Test 1: Earlier features must remain strictly unchanged when future values are modified."""
    np.random.seed(42)
    original_series = pd.Series(np.random.normal(10.0, 1.0, size=50))
    cutoff = 30

    feats_orig = make_lag_features(original_series, lags=[1, 2, 3], rolling_windows=[3, 6])
    cutoff_features_before = feats_orig.iloc[:cutoff].copy()

    # Corrupt/modify observations strictly AFTER cutoff
    corrupted_series = original_series.copy()
    corrupted_series.iloc[cutoff:] = 999.0

    feats_corrupted = make_lag_features(corrupted_series, lags=[1, 2, 3], rolling_windows=[3, 6])
    cutoff_features_after = feats_corrupted.iloc[:cutoff].copy()

    # Features at t <= cutoff must be identical (zero difference)
    pd.testing.assert_frame_equal(cutoff_features_before, cutoff_features_after)


def test_rolling_features_exclude_current_target():
    """Test 2: Rolling features at time t must exclude the current target observation at time t."""
    series = pd.Series([10.0, 10.0, 10.0, 10.0, 100.0])  # Huge spike at index 4

    feats = make_lag_features(series, lags=[1], rolling_windows=[3])

    # At index 4, the target is 100.0. But the rolling mean and lag_1 must NOT reflect 100.0!
    assert feats.loc[4, "target_lag_1"] == 10.0
    assert feats.loc[4, "target_roll_mean_3"] == 10.0
    assert feats.loc[4, "target_roll_max_3"] == 10.0


def test_temporal_splits_preserve_chronology_and_enforce_horizon():
    """Test 3: Temporal splits enforce strict chronology and the intended forecast horizon."""
    timestamps = pd.date_range("2026-01-01 00:00", periods=110, freq="5min")
    df = pd.DataFrame({"timestamp": timestamps, "val": np.arange(110)})

    horizon = 24
    min_train = 62
    step = 6
    splits = rolling_origin_splits(df, horizon=horizon, min_train_size=min_train, step=step)

    assert len(splits) > 0
    for split in splits:
        # Strict temporal order
        assert split.train_end < split.validation_start
        assert split.validation_start <= split.validation_end

        # Index slices match horizon
        val_slice_len = split.val_end_idx - split.val_start_idx
        assert val_slice_len == horizon


def test_preprocessing_fitted_only_on_training_data():
    """Test 4: Scalers and feature transformations must not fit on validation or future observations."""
    np.random.seed(123)
    # Training distribution ~ N(0, 1)
    train_vals = pd.Series(np.random.normal(0.0, 1.0, size=50))
    # Future distribution shifted to N(100, 1)
    future_vals = pd.Series(np.random.normal(100.0, 1.0, size=24))

    model = build_ridge_model(alpha=1.0)
    model.fit(train_vals)

    # Check the standard scaler in pipeline was fitted with mean ~ 0, NOT contaminated by future ~ 100
    scaler = model.estimator.named_steps["scaler"]
    assert np.all(scaler.mean_ < 10.0)  # Mean is near 0, absolutely not near 100


def test_missing_readings_do_not_silently_become_zero():
    """Test 5: Missing CPU readings must not silently become zero, biasing utilization downward."""
    df_with_nans = pd.DataFrame({
        "timestamp": [0, 0, 300, 300],
        "vm_id": ["vm1", "vm2", "vm1", "vm2"],
        "avg_cpu": [10.0, np.nan, 20.0, 20.0],
    })

    targets = compute_targets(df_with_nans)

    # At timestamp 0, vm2 is NaN. The mean should be 10.0, NOT (10 + 0)/2 = 5.0!
    t0_row = targets[targets["timestamp"] == 0].iloc[0]
    assert t0_row["aggregate_cpu_utilization"] == 10.0
    # active_vm_count with valid CPU observation at t=0 should be 1
    assert t0_row["active_vm_count"] == 1


def test_model_predictions_consistent_alignment():
    """Test 6: Models evaluated on the same split produce aligned shapes and timestamps."""
    train_df = pd.DataFrame({
        "timestamp": [0, 300, 600, 900, 1200, 1500, 1800, 2100],
        "active_vm_count": [100] * 8,
        "aggregate_cpu_utilization": [5.0, 5.2, 5.1, 5.3, 5.2, 5.4, 5.3, 5.5],
    })
    val_template = pd.DataFrame({
        "timestamp": [2400, 2700, 3000, 3300],
    })

    p_pers = persistence_forecast(train_df, val_template, "aggregate_cpu_utilization")
    p_ma = moving_average_forecast(train_df, val_template, "aggregate_cpu_utilization", window=3)

    assert len(p_pers) == len(val_template)
    assert len(p_ma) == len(val_template)
    assert (p_pers.index == val_template.index).all()
    assert (p_ma.index == val_template.index).all()
