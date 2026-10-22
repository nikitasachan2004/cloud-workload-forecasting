"""Exploratory Data Analysis script for Cloud Workload Forecasting Challenge."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf, adfuller

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({"font.size": 11, "figure.autolayout": True})

DATA_DIR = Path("challenge_data")
OUTPUT_FIG_DIR = Path("reports/figures")
OUTPUT_FIG_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 70)
    print("PHASE 2: EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 70)

    # 1. Load Data
    cpu_path = DATA_DIR / "vm_cpu_readings.csv.gz"
    meta_path = DATA_DIR / "vm_metadata.csv.gz"

    print(f"Loading CPU readings from {cpu_path}...")
    df_cpu = pd.read_csv(cpu_path)
    print(f"Loading VM metadata from {meta_path}...")
    df_meta = pd.read_csv(meta_path)

    print("\n--- 1. SCHEMA AND ROW COUNTS ---")
    print(f"df_cpu shape: {df_cpu.shape}")
    print(f"df_cpu columns & dtypes:\n{df_cpu.dtypes}")
    print(f"df_meta shape: {df_meta.shape}")
    print(f"df_meta columns & dtypes:\n{df_meta.dtypes}")

    # 2. Integrity & Quality Checks
    print("\n--- 2. DATA INTEGRITY & VALIDATION ---")
    n_unique_vms_cpu = df_cpu["vm_id"].nunique()
    n_unique_vms_meta = df_meta["vm_id"].nunique()
    print(f"Unique VM IDs in CPU data: {n_unique_vms_cpu}")
    print(f"Unique VM IDs in Metadata: {n_unique_vms_meta}")

    vm_set_cpu = set(df_cpu["vm_id"].unique())
    vm_set_meta = set(df_meta["vm_id"].unique())
    assert vm_set_cpu == vm_set_meta, "Mismatch between CPU VM set and Metadata VM set!"
    print("CPU and Metadata VM sets match exactly: True")

    # Observations per VM
    obs_per_vm = df_cpu.groupby("vm_id")["timestamp"].count()
    print(f"Observations per VM: min={obs_per_vm.min()}, max={obs_per_vm.max()}, median={obs_per_vm.median()}")
    assert (obs_per_vm == 134).all(), "Not all VMs have 134 observations!"
    print("Every VM has exactly 134 observations: True (balanced panel)")

    # Duplicates
    dup_pairs = df_cpu.duplicated(subset=["vm_id", "timestamp"]).sum()
    print(f"Duplicate (vm_id, timestamp) rows: {dup_pairs}")
    assert dup_pairs == 0, "Duplicate (vm_id, timestamp) rows found!"

    # Missing values
    missing_cpu = df_cpu.isna().sum().to_dict()
    missing_meta = df_meta.isna().sum().to_dict()
    print(f"Missing values in CPU data: {missing_cpu}")
    print(f"Missing values in Metadata: {missing_meta}")

    # Timestamp cadence & ordering
    unique_ts = np.sort(df_cpu["timestamp"].unique())
    ts_diffs = np.diff(unique_ts)
    print(f"Total distinct timestamps: {len(unique_ts)}")
    print(f"Timestamp range: min={unique_ts.min()}, max={unique_ts.max()}, span={unique_ts.max() - unique_ts.min()}s ({(unique_ts.max() - unique_ts.min()) / 3600:.2f} hours)")
    print(f"Timestamp step values: unique differences = {np.unique(ts_diffs)}")
    assert len(unique_ts) == 134, "Expected 134 distinct timestamps"
    assert np.all(ts_diffs == 300), "Cadence is not strictly 300 seconds"
    print("Cadence is strictly 300 seconds (5 minutes): True")

    is_sorted_per_vm = df_cpu.groupby("vm_id")["timestamp"].apply(lambda s: s.is_monotonic_increasing).all()
    print(f"All VM traces monotonically increasing in time: {is_sorted_per_vm}")

    # Value ranges
    print("\n--- 3. CPU VALUE RANGES & CONSISTENCY ---")
    min_cpu_min = df_cpu["min_cpu"].min()
    min_cpu_max = df_cpu["min_cpu"].max()
    avg_cpu_min = df_cpu["avg_cpu"].min()
    avg_cpu_max = df_cpu["avg_cpu"].max()
    max_cpu_min = df_cpu["max_cpu"].min()
    max_cpu_max = df_cpu["max_cpu"].max()
    print(f"min_cpu range: [{min_cpu_min:.4f}, {min_cpu_max:.4f}]")
    print(f"avg_cpu range: [{avg_cpu_min:.4f}, {avg_cpu_max:.4f}]")
    print(f"max_cpu range: [{max_cpu_min:.4f}, {max_cpu_max:.4f}]")

    violation_min_avg = (df_cpu["min_cpu"] > df_cpu["avg_cpu"] + 1e-6).sum()
    violation_avg_max = (df_cpu["avg_cpu"] > df_cpu["max_cpu"] + 1e-6).sum()
    print(f"Invariance violations (min_cpu > avg_cpu): {violation_min_avg}")
    print(f"Invariance violations (avg_cpu > max_cpu): {violation_avg_max}")
    assert violation_min_avg == 0 and violation_avg_max == 0, "CPU ordering invariant violated!"

    # Quantiles of CPU utilization
    quantiles = [0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0]
    q_vals = df_cpu["avg_cpu"].quantile(quantiles)
    print("\nPercentiles of VM-level avg_cpu (%):")
    for q, v in zip(quantiles, q_vals):
        print(f"  P{int(q * 100):02d}: {v:.4f}%")

    mean_avg_cpu = df_cpu["avg_cpu"].mean()
    std_avg_cpu = df_cpu["avg_cpu"].std()
    print(f"Mean avg_cpu: {mean_avg_cpu:.4f}%, Std: {std_avg_cpu:.4f}%")

    # 4. Rigorous Variance Decomposition (Law of Total Variance, ddof=0)
    print("\n--- 4. VM HETEROGENEITY & VARIANCE DECOMPOSITION ---")
    vm_means = df_cpu.groupby("vm_id")["avg_cpu"].mean().values
    vm_vars = df_cpu.groupby("vm_id")["avg_cpu"].var(ddof=0).values

    total_var = float(df_cpu["avg_cpu"].var(ddof=0))
    between_vm_var = float(np.var(vm_means, ddof=0))
    within_vm_var = float(np.mean(vm_vars))

    between_pct = (between_vm_var / total_var) * 100.0
    within_pct = (within_vm_var / total_var) * 100.0

    print(f"Total variance (ddof=0): {total_var:.6f}")
    print(f"Between-VM variance:    {between_vm_var:.6f} ({between_pct:.4f}%)")
    print(f"Within-VM variance:     {within_vm_var:.6f} ({within_pct:.4f}%)")
    print(f"Sum of components:      {between_vm_var + within_vm_var:.6f} ({(between_pct + within_pct):.4f}%)")
    assert np.isclose(total_var, between_vm_var + within_vm_var), "Law of total variance violated!"

    # Workload Clusters
    vm_stats = df_cpu.groupby("vm_id")["avg_cpu"].agg(["mean", "std", "min", "max", "median"])
    vm_stats["std"] = vm_stats["std"].fillna(0.0)

    idle_vms = (vm_stats["mean"] < 1.0).sum()
    low_vms = ((vm_stats["mean"] >= 1.0) & (vm_stats["mean"] < 10.0)).sum()
    med_vms = ((vm_stats["mean"] >= 10.0) & (vm_stats["mean"] < 50.0)).sum()
    high_vms = (vm_stats["mean"] >= 50.0).sum()
    volatile_vms = (vm_stats["std"] > 15.0).sum()
    print(f"Idle VMs (mean avg_cpu < 1%): {idle_vms} ({idle_vms / 100:.1f}%)")
    print(f"Low-utilization VMs (1% <= mean < 10%): {low_vms} ({low_vms / 100:.1f}%)")
    print(f"Medium-utilization VMs (10% <= mean < 50%): {med_vms} ({med_vms / 100:.1f}%)")
    print(f"High-utilization VMs (mean >= 50%): {high_vms} ({high_vms / 100:.1f}%)")
    print(f"Volatile VMs (std > 15%): {volatile_vms} ({volatile_vms / 100:.1f}%)")

    # Metadata distributions
    print("\n--- 5. VM METADATA DISTRIBUTIONS ---")
    cat_counts = df_meta["vm_category"].value_counts(dropna=False)
    print(f"vm_category distribution:\n{cat_counts}")
    core_counts = df_meta["virtual_core_bucket"].value_counts(dropna=False)
    print(f"\nvirtual_core_bucket distribution:\n{core_counts}")
    mem_counts = df_meta["memory_gb_bucket"].value_counts(dropna=False)
    print(f"\nmemory_gb_bucket distribution:\n{mem_counts}")

    vm_full = vm_stats.merge(df_meta[["vm_id", "vm_category", "virtual_core_bucket", "memory_gb_bucket"]], on="vm_id")
    cat_means = vm_full.groupby("vm_category")["mean"].agg(["count", "mean", "median", "std"])
    print(f"\nMean CPU by VM category:\n{cat_means}")

    # 5. Aggregate Time-Series Construction & Behavior
    print("\n--- 6. AGGREGATE TIME SERIES ANALYSIS ---")
    agg_df = df_cpu.groupby("timestamp").agg(
        active_vm_count=("vm_id", "nunique"),
        aggregate_cpu_utilization=("avg_cpu", "mean"),
        total_cpu_utilization=("avg_cpu", "sum"),
        median_cpu_utilization=("avg_cpu", "median"),
        p90_cpu_utilization=("avg_cpu", lambda s: np.percentile(s, 90)),
        std_cpu_utilization=("avg_cpu", "std"),
    ).reset_index().sort_values("timestamp")

    print(f"Aggregate series length: {len(agg_df)}")
    print(f"active_vm_count: min={agg_df['active_vm_count'].min()}, max={agg_df['active_vm_count'].max()}")
    print(f"aggregate_cpu_utilization: min={agg_df['aggregate_cpu_utilization'].min():.4f}%, "
          f"max={agg_df['aggregate_cpu_utilization'].max():.4f}%, "
          f"mean={agg_df['aggregate_cpu_utilization'].mean():.4f}%, "
          f"std={agg_df['aggregate_cpu_utilization'].std():.4f}%")

    train_agg = agg_df.iloc[:110]
    holdout_agg = agg_df.iloc[110:]
    print(f"Train window (first 110 steps): mean={train_agg['aggregate_cpu_utilization'].mean():.4f}%, std={train_agg['aggregate_cpu_utilization'].std():.4f}%, min={train_agg['aggregate_cpu_utilization'].min():.4f}%, max={train_agg['aggregate_cpu_utilization'].max():.4f}%")
    print(f"Holdout window (last 24 steps): mean={holdout_agg['aggregate_cpu_utilization'].mean():.4f}%, std={holdout_agg['aggregate_cpu_utilization'].std():.4f}%, min={holdout_agg['aggregate_cpu_utilization'].min():.4f}%, max={holdout_agg['aggregate_cpu_utilization'].max():.4f}%")

    # Time series tests: Stationarity (ADF on train)
    adf_res = adfuller(train_agg["aggregate_cpu_utilization"], autolag="AIC")
    print(f"ADF Statistic on Train aggregate CPU: {adf_res[0]:.4f}, p-value: {adf_res[1]:.4e}")

    # Autocorrelation & Partial Autocorrelation
    acf_vals = acf(train_agg["aggregate_cpu_utilization"], nlags=24)
    pacf_vals = pacf(train_agg["aggregate_cpu_utilization"], nlags=24)
    print("\nAutocorrelation (ACF) on Train (first 6 lags = 30 min):")
    for lag, (a, p) in enumerate(zip(acf_vals[:7], pacf_vals[:7])):
        print(f"  Lag {lag} ({lag * 5} min): ACF={a:.4f}, PACF={p:.4f}")

    # Save summary metrics to json
    summary_stats = {
        "n_vms": int(n_unique_vms_cpu),
        "n_observations": int(len(df_cpu)),
        "n_timestamps": int(len(unique_ts)),
        "cadence_seconds": 300,
        "mean_cpu_overall": float(mean_avg_cpu),
        "std_cpu_overall": float(std_avg_cpu),
        "total_variance_ddof0": float(total_var),
        "between_vm_var_ddof0": float(between_vm_var),
        "between_vm_var_pct": float(between_pct),
        "within_vm_var_ddof0": float(within_vm_var),
        "within_vm_var_pct": float(within_pct),
        "train_agg_mean": float(train_agg["aggregate_cpu_utilization"].mean()),
        "train_agg_std": float(train_agg["aggregate_cpu_utilization"].std()),
        "holdout_agg_mean": float(holdout_agg["aggregate_cpu_utilization"].mean()),
        "holdout_agg_std": float(holdout_agg["aggregate_cpu_utilization"].std()),
        "adf_stat": float(adf_res[0]),
        "adf_pvalue": float(adf_res[1]),
        "lag1_acf": float(acf_vals[1]),
        "lag2_acf": float(acf_vals[2]),
        "lag3_acf": float(acf_vals[3]),
    }
    with open("reports/eda_summary.json", "w") as f:
        json.dump(summary_stats, f, indent=2)

    # 6. Generate Figures
    print("\n--- 7. GENERATING INFORMATIVE FIGURES ---")

    # Figure 1: Aggregate CPU Utilization & Active VM Count over Time
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    time_hours = (agg_df["timestamp"] - agg_df["timestamp"].iloc[0]) / 3600.0
    split_hour = time_hours.iloc[110]

    ax1.plot(time_hours, agg_df["aggregate_cpu_utilization"], color="#1f77b4", lw=2, label="Aggregate CPU Mean (%)")
    ax1.axvline(split_hour, color="#d62728", ls="--", lw=1.5, label=f"Train / Holdout Split (t=110, {split_hour:.1f}h)")
    ax1.set_ylabel("CPU Utilization (%)")
    ax1.set_title("Aggregate Cloud CPU Demand Over Time (11.1h Trace)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left")

    ax2.plot(time_hours, agg_df["active_vm_count"], color="#2ca02c", lw=2, label="Active VM Count")
    ax2.axvline(split_hour, color="#d62728", ls="--", lw=1.5)
    ax2.set_xlabel("Elapsed Time (Hours)")
    ax2.set_ylabel("Active VM Count")
    ax2.set_title("Active VM Count Stability Over Time (Constant 10,000 VMs)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper left")

    fig.tight_layout()
    fig1_path = OUTPUT_FIG_DIR / "fig1_aggregate_workload_timeseries.png"
    fig.savefig(fig1_path, dpi=300)
    plt.close(fig)
    print(f"Saved {fig1_path}")

    # Figure 2: VM-level CPU Distribution and Percentiles
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.hist(df_cpu["avg_cpu"], bins=50, color="#3470a3", edgecolor="black", alpha=0.8, density=True)
    ax1.axvline(mean_avg_cpu, color="#d62728", ls="--", label=f"Mean: {mean_avg_cpu:.1f}%")
    ax1.axvline(q_vals[0.5], color="#2ca02c", ls="-.", label=f"Median: {q_vals[0.5]:.1f}%")
    ax1.axvline(q_vals[0.9], color="#ff7f0e", ls=":", label=f"P90: {q_vals[0.9]:.1f}%")
    ax1.set_xlabel("VM-Level Average CPU Utilization (%)")
    ax1.set_ylabel("Density")
    ax1.set_title("VM-Level CPU Distribution (Severe Right Skew)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    cats = [c for c in ["Delay-insensitive", "Interactive", "Unknown"] if c in vm_full["vm_category"].values]
    cat_data = [vm_full[vm_full["vm_category"] == c]["mean"].values for c in cats]
    ax2.boxplot(cat_data, tick_labels=cats, showmeans=True)
    ax2.set_ylabel("Per-VM Mean CPU Utilization (%)")
    ax2.set_title("Per-VM Mean CPU by Workload Category")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig2_path = OUTPUT_FIG_DIR / "fig2_cpu_distribution_by_category.png"
    fig.savefig(fig2_path, dpi=300)
    plt.close(fig)
    print(f"Saved {fig2_path}")

    # Figure 3: Segment-level Aggregate CPU Over Time
    fig, ax = plt.subplots(figsize=(10, 4.5))
    df_merged = df_cpu.merge(df_meta[["vm_id", "vm_category"]], on="vm_id")
    cat_time = df_merged.groupby(["timestamp", "vm_category"])["avg_cpu"].mean().unstack()
    cat_time_hours = (cat_time.index - cat_time.index[0]) / 3600.0

    for col in cat_time.columns:
        ax.plot(cat_time_hours, cat_time[col], lw=2, label=f"Category: {col}")
    ax.axvline(split_hour, color="#d62728", ls="--", lw=1.5, label="Train / Holdout Split")
    ax.set_xlabel("Elapsed Time (Hours)")
    ax.set_ylabel("Mean CPU Utilization (%)")
    ax.set_title("Aggregate CPU Utilization Trajectories by VM Category")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")

    fig.tight_layout()
    fig3_path = OUTPUT_FIG_DIR / "fig3_category_trajectories.png"
    fig.savefig(fig3_path, dpi=300)
    plt.close(fig)
    print(f"Saved {fig3_path}")

    # Figure 4: Autocorrelation (ACF) and Partial Autocorrelation (PACF)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    lags = np.arange(len(acf_vals))
    ax1.stem(lags, acf_vals)
    ax1.axhline(0, color="black", lw=0.8)
    ax1.axhline(1.96 / np.sqrt(len(train_agg)), color="gray", ls="--")
    ax1.axhline(-1.96 / np.sqrt(len(train_agg)), color="gray", ls="--")
    ax1.set_xlabel("Lag (5-minute steps)")
    ax1.set_ylabel("Autocorrelation")
    ax1.set_title("ACF: Aggregate CPU Demand (Train History)")
    ax1.grid(True, alpha=0.3)

    ax2.stem(lags, pacf_vals)
    ax2.axhline(0, color="black", lw=0.8)
    ax2.axhline(1.96 / np.sqrt(len(train_agg)), color="gray", ls="--")
    ax2.axhline(-1.96 / np.sqrt(len(train_agg)), color="gray", ls="--")
    ax2.set_xlabel("Lag (5-minute steps)")
    ax2.set_ylabel("Partial Autocorrelation")
    ax2.set_title("PACF: Aggregate CPU Demand (Train History)")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig4_path = OUTPUT_FIG_DIR / "fig4_acf_pacf.png"
    fig.savefig(fig4_path, dpi=300)
    plt.close(fig)
    print(f"Saved {fig4_path}")

    # Figure 5: Within-VM Volatility vs Mean Utilization
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(vm_stats["mean"], vm_stats["std"], alpha=0.15, s=12, color="#1f77b4")
    ax.set_xlabel("Mean CPU Utilization per VM (%)")
    ax.set_ylabel("Standard Deviation of CPU per VM (%)")
    ax.set_title("Within-VM Volatility vs. Baseline Utilization Across 10,000 VMs")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig5_path = OUTPUT_FIG_DIR / "fig5_within_vm_volatility.png"
    fig.savefig(fig5_path, dpi=300)
    plt.close(fig)
    print(f"Saved {fig5_path}")

    print("\nEDA script completed successfully!")


if __name__ == "__main__":
    main()
