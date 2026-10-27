"""Automated verification of submission contract and mathematical invariants."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from cwfc.data import load_dataset
from cwfc.predict import generate_predictions


def test_predictions_csv_contract():
    """Verify predictions.csv satisfies every requirement of SUBMISSION.md."""
    pred_path = Path("predictions.csv")
    assert pred_path.exists(), "predictions.csv must exist"

    df = pd.read_csv(pred_path)

    # 1. Row count
    assert len(df) == 24, f"Expected exactly 24 rows, got {len(df)}"

    # 2. Columns and order
    expected_cols = ["timestamp", "active_vm_count", "aggregate_cpu_utilization"]
    assert list(df.columns) == expected_cols, f"Columns must be {expected_cols}, got {list(df.columns)}"

    # 3. No NaN / infinite values
    assert not df.isna().any().any(), "predictions.csv must not contain any NaNs"
    assert np.isfinite(df.values).all(), "predictions.csv values must be finite numeric"

    # 4. Chronological order & no duplicates
    assert df["timestamp"].is_monotonic_increasing, "Timestamps must be monotonically increasing"
    assert df["timestamp"].nunique() == 24, "Timestamps must have no duplicates"

    # 5. Domain boundaries
    assert (df["active_vm_count"] >= 0).all(), "active_vm_count must be non-negative"
    assert ((df["aggregate_cpu_utilization"] >= 0.0) & (df["aggregate_cpu_utilization"] <= 100.0)).all(), \
        "aggregate_cpu_utilization must be in [0, 100]"


def test_variance_decomposition_law_of_total_variance():
    """Verify that the variance decomposition satisfies the Law of Total Variance exactly."""
    df_cpu, _ = load_dataset("challenge_data")

    total_var = float(df_cpu["avg_cpu"].var(ddof=0))
    vm_means = df_cpu.groupby("vm_id")["avg_cpu"].mean().values
    vm_vars = df_cpu.groupby("vm_id")["avg_cpu"].var(ddof=0).values

    between_var = float(np.var(vm_means, ddof=0))
    within_var = float(np.mean(vm_vars))

    assert np.isclose(total_var, between_var + within_var, rtol=1e-7), \
        f"Total variance {total_var} must equal between {between_var} + within {within_var}"

    between_pct = (between_var / total_var) * 100.0
    within_pct = (within_var / total_var) * 100.0
    assert np.isclose(between_pct + within_pct, 100.0, rtol=1e-7), \
        f"Decomposition percentages must sum to 100%, got {between_pct + within_pct}"


def test_direct_forecaster_adaptive_smoke(tmp_path):
    """Verify DirectForecaster runs cleanly on mock data from CLI."""
    tiny_cpu = pd.DataFrame({
        "timestamp": [0, 0, 300, 300, 600, 600, 900, 900, 1200, 1200],
        "vm_id": ["v1", "v2"] * 5,
        "avg_cpu": [5.0, 15.0, 6.0, 14.0, 5.5, 14.5, 6.5, 13.5, 7.0, 13.0],
        "min_cpu": [4.0, 14.0] * 5,
        "max_cpu": [6.0, 16.0] * 5,
    })
    mock_dir = tmp_path / "mock_dataset"
    mock_dir.mkdir()
    tiny_cpu.to_csv(mock_dir / "vm_cpu_readings.csv.gz", index=False)

    out_file = tmp_path / "test_out.csv"
    res = generate_predictions(mock_dir, out_file, holdout_steps=2)
    assert len(res) == 2
    assert list(res.columns) == ["timestamp", "active_vm_count", "aggregate_cpu_utilization"]
