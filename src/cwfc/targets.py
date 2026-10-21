"""Target computation utilities for the Cloud Workload Forecasting Challenge."""
from __future__ import annotations

import pandas as pd


def compute_targets(df_cpu: pd.DataFrame) -> pd.DataFrame:
    """Compute required aggregate time-series targets from VM-level CPU readings.

    Targets defined by the challenge contract:
      1. active_vm_count: number of distinct VM IDs with valid CPU readings at t.
      2. aggregate_cpu_utilization: mean of VM-level avg_cpu readings at t, in [0, 100].

    Parameters
    ----------
    df_cpu : pd.DataFrame
        VM-level readings containing at least ['timestamp', 'vm_id', 'avg_cpu'].

    Returns
    -------
    pd.DataFrame
        DataFrame sorted by timestamp containing:
        - timestamp (int or datetime)
        - active_vm_count (int)
        - aggregate_cpu_utilization (float)
    """
    required_cols = {"timestamp", "vm_id", "avg_cpu"}
    if not required_cols.issubset(df_cpu.columns):
        missing = required_cols - set(df_cpu.columns)
        raise ValueError(f"Input dataframe missing required columns: {missing}")

    if df_cpu.empty:
        return pd.DataFrame(columns=["timestamp", "active_vm_count", "aggregate_cpu_utilization"])

    # Drop any null records in avg_cpu to compute valid count and mean
    valid_cpu = df_cpu.dropna(subset=["avg_cpu"])

    agg = valid_cpu.groupby("timestamp").agg(
        active_vm_count=("vm_id", "nunique"),
        aggregate_cpu_utilization=("avg_cpu", "mean"),
    ).reset_index().sort_values("timestamp").reset_index(drop=True)

    # Invariant assertions
    if (agg["active_vm_count"] < 0).any():
        raise ValueError("active_vm_count must be non-negative")
    if ((agg["aggregate_cpu_utilization"] < 0) | (agg["aggregate_cpu_utilization"] > 100)).any():
        raise ValueError("aggregate_cpu_utilization must be within [0, 100]")

    return agg
