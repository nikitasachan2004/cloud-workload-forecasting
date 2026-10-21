"""Data loading and validation utilities."""
from __future__ import annotations

from pathlib import Path
import pandas as pd


def load_dataset(data_dir: Path | str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load CPU readings and VM metadata from the given data directory.

    Supports both compressed (.csv.gz) and uncompressed (.csv) formats.

    Parameters
    ----------
    data_dir : Path | str
        Path to directory containing dataset files.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Tuple of (df_cpu, df_metadata).
    """
    path = Path(data_dir)
    if not path.exists():
        raise FileNotFoundError(f"Data directory does not exist: {path}")

    # Look for CPU readings
    cpu_candidates = [
        path / "vm_cpu_readings.csv.gz",
        path / "vm_cpu_readings.csv",
    ]
    cpu_file = next((f for f in cpu_candidates if f.exists()), None)
    if cpu_file is None:
        raise FileNotFoundError(f"Could not find vm_cpu_readings file in {path}")

    # Look for metadata (optional for inference if CPU readings contain vm_id and avg_cpu)
    meta_candidates = [
        path / "vm_metadata.csv.gz",
        path / "vm_metadata.csv",
    ]
    meta_file = next((f for f in meta_candidates if f.exists()), None)

    df_cpu = pd.read_csv(cpu_file)
    df_meta = pd.read_csv(meta_file) if meta_file is not None else pd.DataFrame()

    return df_cpu, df_meta
