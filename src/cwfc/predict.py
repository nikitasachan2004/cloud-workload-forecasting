"""Prediction entrypoint conforming to SUBMISSION.md reproduction contract."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from cwfc.data import load_dataset
from cwfc.targets import compute_targets
from cwfc.models import build_direct_ridge_model


def generate_predictions(data_dir: Path | str, output_path: Path | str, holdout_steps: int = 24) -> pd.DataFrame:
    """Generate holdout predictions conforming to the submission contract.

    Parameters
    ----------
    data_dir : Path | str
        Path to directory containing dataset files (challenge_data).
    output_path : Path | str
        Output CSV filepath for predictions.csv.
    holdout_steps : int, default=24
        Number of timestamps in the holdout evaluation period.
    """
    df_cpu, _ = load_dataset(data_dir)
    targets_df = compute_targets(df_cpu)

    n_total = len(targets_df)
    if n_total <= holdout_steps:
        train_steps = max(1, n_total // 2)
        holdout_steps = n_total - train_steps
    else:
        train_steps = n_total - holdout_steps

    train_df = targets_df.iloc[:train_steps].copy().reset_index(drop=True)
    holdout_df = targets_df.iloc[train_steps:].copy().reset_index(drop=True)

    target_col = "aggregate_cpu_utilization"

    # Train selected Direct Ridge model strictly on training window (excluding holdout)
    model = build_direct_ridge_model(alpha=1.0)
    model.fit(train_df[target_col], max_horizon=len(holdout_df))

    # Forecast holdout steps directly
    preds = model.predict(train_df[target_col], horizon=len(holdout_df)).values

    # Active VM count is persistent from training population
    last_active_vms = int(train_df["active_vm_count"].iloc[-1])

    submission_df = pd.DataFrame({
        "timestamp": holdout_df["timestamp"].values,
        "active_vm_count": last_active_vms,
        "aggregate_cpu_utilization": np.round(preds, 4),
    })

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(out_file, index=False)
    print(f"Generated {len(submission_df)} predictions saved to {out_file}")
    return submission_df


def main():
    parser = argparse.ArgumentParser(description="Generate workload forecast predictions.")
    parser.add_argument(
        "--data",
        type=str,
        default="challenge_data",
        help="Path to dataset directory (default: challenge_data)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="predictions.csv",
        help="Output CSV path (default: predictions.csv)",
    )
    args = parser.parse_args()
    generate_predictions(data_dir=args.data, output_path=args.output)


if __name__ == "__main__":
    main()
