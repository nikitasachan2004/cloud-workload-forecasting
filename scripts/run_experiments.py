"""End-to-end experiment runner: cross-validation, model comparison, uncertainty, and holdout evaluation."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cwfc.data import load_dataset
from cwfc.targets import compute_targets
from cwfc.validation import rolling_origin_splits
from cwfc.metrics import mae, rmse, smape, mase, interval_coverage, interval_width
from cwfc.baselines import persistence_forecast, moving_average_forecast
from cwfc.models import build_direct_ridge_model, build_ridge_model, build_lgbm_model
from cwfc.uncertainty import ResidualIntervalCalibrator

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({"font.size": 11, "figure.autolayout": True})

REPORTS_DIR = Path("reports")
FIG_DIR = REPORTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def run_experiments():
    print("=" * 75)
    print("SYSTEMATIC TIME-SERIES MODELING & EVALUATION (AUDITED & REPAIRED)")
    print("=" * 75)

    # 1. Load dataset & compute targets
    print("Loading data...")
    df_cpu, _ = load_dataset("challenge_data")
    targets_df = compute_targets(df_cpu)
    print(f"Computed targets for {len(targets_df)} timestamps.")

    # 2. Chronological split: 110 training steps, 24 test holdout steps
    train_full = targets_df.iloc[:110].copy().reset_index(drop=True)
    test_holdout = targets_df.iloc[110:].copy().reset_index(drop=True)
    print(f"Training window: {len(train_full)} steps (indices 0..109)")
    print(f"Test holdout:    {len(test_holdout)} steps (indices 110..133)")

    target_col = "aggregate_cpu_utilization"

    # 3. Expanding-window validation setup on training set
    # horizon = 24 (2 hours), min_train_size = 62 (~5.2 hours), step = 6 (30 min)
    splits = rolling_origin_splits(train_full, horizon=24, min_train_size=62, step=6)
    print(f"\nGenerated {len(splits)} expanding-window validation folds on training period:")
    for i, s in enumerate(splits):
        print(f"  Fold {i}: Train [0..{s.train_end_idx - 1}] ({s.train_end_idx} steps) -> Val [{s.val_start_idx}..{s.val_end_idx - 1}] ({s.val_end_idx - s.val_start_idx} steps)")

    # Candidate models for primary benchmark
    # Note: Seasonal Naive is omitted from primary models because ACF/PACF and periodogram
    # show no genuine periodic cycle; lag-3 autocorrelation reflects short-term persistence.
    candidate_models = {
        "Persistence": lambda: None,
        "MovingAverage_w3": lambda: None,
        "MovingAverage_w6": lambda: None,
        "MovingAverage_w12": lambda: None,
        "Direct_Ridge": lambda: build_direct_ridge_model(alpha=1.0),
        "Recursive_Ridge": lambda: build_ridge_model(alpha=1.0),
        "Recursive_LightGBM": lambda: build_lgbm_model(max_depth=3, n_estimators=40),
    }

    fold_predictions = {m: [] for m in candidate_models}
    fold_actuals = []
    fold_train_histories = []

    print("\n--- RUNNING BACKTESTING FOLDS ---")
    for fold_idx, split in enumerate(splits):
        df_tr = train_full.iloc[:split.train_end_idx].copy()
        df_val = train_full.iloc[split.val_start_idx:split.val_end_idx].copy()
        y_val = df_val[target_col].values
        fold_actuals.append(y_val)
        fold_train_histories.append(df_tr[target_col].values)

        # Baseline: Persistence
        p_pers = persistence_forecast(df_tr, df_val, target_col).values
        fold_predictions["Persistence"].append(p_pers)

        # Baseline: Moving Average
        p_ma3 = moving_average_forecast(df_tr, df_val, target_col, window=3).values
        fold_predictions["MovingAverage_w3"].append(p_ma3)
        p_ma6 = moving_average_forecast(df_tr, df_val, target_col, window=6).values
        fold_predictions["MovingAverage_w6"].append(p_ma6)
        p_ma12 = moving_average_forecast(df_tr, df_val, target_col, window=12).values
        fold_predictions["MovingAverage_w12"].append(p_ma12)

        # ML: Direct Ridge
        direct_m = candidate_models["Direct_Ridge"]()
        direct_m.fit(df_tr[target_col], max_horizon=len(df_val))
        p_direct = direct_m.predict(df_tr[target_col], horizon=len(df_val)).values
        fold_predictions["Direct_Ridge"].append(p_direct)

        # ML: Recursive Ridge
        rec_ridge = candidate_models["Recursive_Ridge"]()
        rec_ridge.fit(df_tr[target_col])
        p_rec_ridge = rec_ridge.predict(df_tr[target_col], horizon=len(df_val)).values
        fold_predictions["Recursive_Ridge"].append(p_rec_ridge)

        # ML: Recursive LightGBM
        lgbm_m = candidate_models["Recursive_LightGBM"]()
        lgbm_m.fit(df_tr[target_col])
        p_lgbm = lgbm_m.predict(df_tr[target_col], horizon=len(df_val)).values
        fold_predictions["Recursive_LightGBM"].append(p_lgbm)

    # Calculate metrics across folds
    metrics_summary = {}
    print("\n--- VALIDATION METRICS SUMMARY (MEAN ACROSS 5 FOLDS ± STD) ---")
    print(f"{'Model':<20} | {'MAE':<18} | {'RMSE':<18} | {'sMAPE (%)':<18} | {'MASE':<14}")
    print("-" * 94)

    for m in candidate_models:
        mae_list, rmse_list, smape_list, mase_list = [], [], [], []
        for f_idx in range(len(splits)):
            y_t = fold_actuals[f_idx]
            y_p = fold_predictions[m][f_idx]
            y_tr = fold_train_histories[f_idx]

            mae_list.append(mae(y_t, y_p))
            rmse_list.append(rmse(y_t, y_p))
            smape_list.append(smape(y_t, y_p))
            mase_list.append(mase(y_t, y_p, y_tr))

        mean_mae, std_mae = np.mean(mae_list), np.std(mae_list)
        mean_rmse, std_rmse = np.mean(rmse_list), np.std(rmse_list)
        mean_smape, std_smape = np.mean(smape_list), np.std(smape_list)
        mean_mase, std_mase = np.mean(mase_list), np.std(mase_list)

        metrics_summary[m] = {
            "mae_mean": float(mean_mae), "mae_std": float(std_mae),
            "rmse_mean": float(mean_rmse), "rmse_std": float(std_rmse),
            "smape_mean": float(mean_smape), "smape_std": float(std_smape),
            "mase_mean": float(mean_mase), "mase_std": float(std_mase),
            "fold_maes": [float(x) for x in mae_list],
            "fold_rmses": [float(x) for x in rmse_list],
        }

        print(f"{m:<20} | {mean_mae:.4f} ± {std_mae:.4f}   | {mean_rmse:.4f} ± {std_rmse:.4f}   | {mean_smape:.4f} ± {std_smape:.4f}   | {mean_mase:.4f} ± {std_mase:.4f}")

    # Horizon breakdown (MAE at h=1, 6, 12, 24)
    horizon_mae = {m: [] for m in candidate_models}
    for h in range(1, 25):
        for m in candidate_models:
            errors = [abs(fold_actuals[f][h - 1] - fold_predictions[m][f][h - 1]) for f in range(len(splits))]
            horizon_mae[m].append(float(np.mean(errors)))

    print("\n--- MAE BY FORECAST HORIZON STEP ---")
    print(f"{'Model':<20} | {'h=1 (5m)':<10} | {'h=6 (30m)':<10} | {'h=12 (1h)':<10} | {'h=24 (2h)':<10}")
    print("-" * 68)
    for m in candidate_models:
        print(f"{m:<20} | {horizon_mae[m][0]:.4f}     | {horizon_mae[m][5]:.4f}      | {horizon_mae[m][11]:.4f}      | {horizon_mae[m][23]:.4f}")

    # Model Selection (strictly on development validation data)
    best_model_name = min(metrics_summary, key=lambda k: metrics_summary[k]["mae_mean"])
    print(f"\nWinning Model Selected on Validation MAE: {best_model_name} (MAE: {metrics_summary[best_model_name]['mae_mean']:.4f} ± {metrics_summary[best_model_name]['mae_std']:.4f})")

    # 4. Uncertainty Calibration (Signed Residuals on Selected Model)
    # Signed residuals: e = y_actual - y_predicted
    oof_y_true = np.concatenate(fold_actuals)
    oof_y_pred = np.concatenate(fold_predictions[best_model_name])
    oof_residuals = oof_y_true - oof_y_pred

    calibrator = ResidualIntervalCalibrator(coverage=0.80, one_sided_level=0.95).fit(oof_y_true, oof_y_pred)
    calibrator_95 = ResidualIntervalCalibrator(coverage=0.95, one_sided_level=0.95).fit(oof_y_true, oof_y_pred)

    lb_80, ub_80 = calibrator.predict_intervals(oof_y_pred)
    lb_95, ub_95 = calibrator_95.predict_intervals(oof_y_pred)
    one_sided_buffer = calibrator.one_sided_buffer_

    val_cov_80 = interval_coverage(oof_y_true, lb_80, ub_80)
    val_width_80 = interval_width(lb_80, ub_80)
    val_cov_95 = interval_coverage(oof_y_true, lb_95, ub_95)
    val_width_95 = interval_width(lb_95, ub_95)
    val_one_sided_cov = float(np.mean(oof_residuals <= one_sided_buffer))

    print(f"\nUncertainty Calibration on Out-of-Fold Development Residuals ({best_model_name}, N={len(oof_residuals)}):")
    print(f"  Residual Mean: {np.mean(oof_residuals):.4f}%, Std: {np.std(oof_residuals):.4f}%")
    print(f"  Calibrated 95% One-Sided Upper Buffer: +{one_sided_buffer:.4f}% CPU (Val Coverage: {val_one_sided_cov * 100:.1f}%)")
    print(f"  Calibrated 80% Two-Sided Interval: [{calibrator.lower_quantile_:.4f}%, +{calibrator.upper_quantile_:.4f}%] (Width: {val_width_80:.4f}%, Val Coverage: {val_cov_80 * 100:.1f}%)")
    print(f"  Calibrated 95% Two-Sided Interval: [{calibrator_95.lower_quantile_:.4f}%, +{calibrator_95.upper_quantile_:.4f}%] (Width: {val_width_95:.4f}%, Val Coverage: {val_cov_95 * 100:.1f}%)")

    # 5. Final Holdout Evaluation & Prediction Generation
    print("\n--- FINAL TEST HOLDOUT EVALUATION (LAST 24 STEPS, HELD-OUT PUBLIC EVALUATION) ---")
    y_holdout_true = test_holdout[target_col].values

    final_predictions = {}

    # Persistence
    final_predictions["Persistence"] = persistence_forecast(train_full, test_holdout, target_col).values

    # Moving Averages
    final_predictions["MovingAverage_w3"] = moving_average_forecast(train_full, test_holdout, target_col, window=3).values
    final_predictions["MovingAverage_w6"] = moving_average_forecast(train_full, test_holdout, target_col, window=6).values
    final_predictions["MovingAverage_w12"] = moving_average_forecast(train_full, test_holdout, target_col, window=12).values

    # Direct Ridge
    final_direct_ridge = build_direct_ridge_model(alpha=1.0).fit(train_full[target_col], max_horizon=24)
    final_predictions["Direct_Ridge"] = final_direct_ridge.predict(train_full[target_col], horizon=24).values

    # Recursive Ridge
    final_rec_ridge = build_ridge_model(alpha=1.0).fit(train_full[target_col])
    final_predictions["Recursive_Ridge"] = final_rec_ridge.predict(train_full[target_col], horizon=24).values

    # Recursive LightGBM
    final_lgbm = build_lgbm_model(max_depth=3, n_estimators=40).fit(train_full[target_col])
    final_predictions["Recursive_LightGBM"] = final_lgbm.predict(train_full[target_col], horizon=24).values

    holdout_metrics = {}
    print(f"{'Model':<20} | {'Holdout MAE':<14} | {'Holdout RMSE':<14} | {'sMAPE (%)':<14} | {'MASE':<10}")
    print("-" * 80)
    for m in candidate_models:
        y_p = final_predictions[m]
        h_mae = mae(y_holdout_true, y_p)
        h_rmse = rmse(y_holdout_true, y_p)
        h_smape = smape(y_holdout_true, y_p)
        h_mase = mase(y_holdout_true, y_p, train_full[target_col].values)
        holdout_metrics[m] = {
            "mae": float(h_mae), "rmse": float(h_rmse), "smape": float(h_smape), "mase": float(h_mase)
        }
        print(f"{m:<20} | {h_mae:.4f}         | {h_rmse:.4f}          | {h_smape:.4f}        | {h_mase:.4f}")

    # Holdout diagnostic coverage for selected model
    best_holdout_pred = final_predictions[best_model_name]
    holdout_residuals = y_holdout_true - best_holdout_pred
    holdout_lb_80, holdout_ub_80 = calibrator.predict_intervals(best_holdout_pred)
    holdout_lb_95, holdout_ub_95 = calibrator_95.predict_intervals(best_holdout_pred)

    holdout_cov_80 = interval_coverage(y_holdout_true, holdout_lb_80, holdout_ub_80)
    holdout_cov_95 = interval_coverage(y_holdout_true, holdout_lb_95, holdout_ub_95)
    holdout_upper_breaches = int(np.sum(holdout_residuals > one_sided_buffer))
    holdout_one_sided_cov = float(1.0 - holdout_upper_breaches / len(y_holdout_true))

    print(f"\nHoldout Diagnostic Coverage for {best_model_name}:")
    print(f"  One-Sided 95% Upper Buffer (+{one_sided_buffer:.4f}%): actual breaches = {holdout_upper_breaches}/24 (Coverage: {holdout_one_sided_cov * 100:.1f}%)")
    print(f"  Nominal 80% Two-Sided Interval: actual coverage = {holdout_cov_80 * 100:.1f}%")
    print(f"  Nominal 95% Two-Sided Interval: actual coverage = {holdout_cov_95 * 100:.1f}%")

    # 6. Save final predictions.csv adhering to SUBMISSION.md
    submission_df = pd.DataFrame({
        "timestamp": test_holdout["timestamp"].values,
        "active_vm_count": 10000,
        "aggregate_cpu_utilization": np.round(best_holdout_pred, 4),
    })
    submission_path = Path("predictions.csv")
    submission_df.to_csv(submission_path, index=False)
    print(f"\nSaved required submission file to {submission_path} ({len(submission_df)} rows).")

    # 7. Save complete experiment results JSON
    experiment_results = {
        "selected_model": best_model_name,
        "validation_metrics": metrics_summary,
        "horizon_mae": horizon_mae,
        "holdout_metrics": holdout_metrics,
        "uncertainty_calibration": {
            "n_calibration_residuals": len(oof_residuals),
            "one_sided_95_buffer": float(one_sided_buffer),
            "val_one_sided_coverage_95": float(val_one_sided_cov),
            "holdout_one_sided_coverage_95": float(holdout_one_sided_cov),
            "holdout_upper_breaches": holdout_upper_breaches,
            "two_sided_80_bounds": [calibrator.lower_quantile_, calibrator.upper_quantile_],
            "two_sided_95_bounds": [calibrator_95.lower_quantile_, calibrator_95.upper_quantile_],
            "val_coverage_80": float(val_cov_80),
            "val_width_80": float(val_width_80),
            "val_coverage_95": float(val_cov_95),
            "val_width_95": float(val_width_95),
            "holdout_coverage_80": float(holdout_cov_80),
            "holdout_coverage_95": float(holdout_cov_95),
        }
    }
    with open(REPORTS_DIR / "experiment_results.json", "w") as f:
        json.dump(experiment_results, f, indent=2)

    # 8. Generate Evaluation Figures
    print("\n--- GENERATING EVALUATION FIGURES ---")

    # Figure 6: Model Validation Comparison (MAE and RMSE)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    model_names = list(metrics_summary.keys())
    maes = [metrics_summary[m]["mae_mean"] for m in model_names]
    mae_errs = [metrics_summary[m]["mae_std"] for m in model_names]
    rmses = [metrics_summary[m]["rmse_mean"] for m in model_names]
    rmse_errs = [metrics_summary[m]["rmse_std"] for m in model_names]

    y_pos = np.arange(len(model_names))
    ax1.barh(y_pos, maes, xerr=mae_errs, color="#2b5c8f", alpha=0.85, capsize=4)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(model_names)
    ax1.set_xlabel("Mean Absolute Error (MAE, %)")
    ax1.set_title("Validation MAE Across 5 Folds")
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3)

    ax2.barh(y_pos, rmses, xerr=rmse_errs, color="#e66101", alpha=0.85, capsize=4)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([""] * len(model_names))
    ax2.set_xlabel("Root Mean Squared Error (RMSE, %)")
    ax2.set_title("Validation RMSE Across 5 Folds")
    ax2.invert_yaxis()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig6_path = FIG_DIR / "fig6_model_comparison_mae_rmse.png"
    fig.savefig(fig6_path, dpi=300)
    plt.close(fig)
    print(f"Saved {fig6_path}")

    # Figure 7: Error by Forecast Horizon Step
    fig, ax = plt.subplots(figsize=(10, 4.8))
    h_steps = np.arange(1, 25)
    mins = h_steps * 5
    for m in ["Persistence", "MovingAverage_w3", "Direct_Ridge", "Recursive_Ridge", "Recursive_LightGBM"]:
        ax.plot(mins, horizon_mae[m], lw=2, marker="o", markersize=3, label=m)
    ax.set_xlabel("Forecast Horizon Lookahead (Minutes)")
    ax.set_ylabel("Mean Absolute Error (%)")
    ax.set_title("Forecast Error Accumulation by Lookahead Horizon (1 to 24 Steps)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")

    fig.tight_layout()
    fig7_path = FIG_DIR / "fig7_error_by_horizon.png"
    fig.savefig(fig7_path, dpi=300)
    plt.close(fig)
    print(f"Saved {fig7_path}")

    # Figure 8: Holdout Forecast vs Actual with Empirical Intervals
    fig, ax = plt.subplots(figsize=(11, 5.2))
    hist_time = (train_full["timestamp"] - train_full["timestamp"].iloc[0]) / 3600.0
    holdout_time = (test_holdout["timestamp"] - train_full["timestamp"].iloc[0]) / 3600.0

    ax.plot(hist_time.iloc[-24:], train_full[target_col].iloc[-24:], color="black", lw=2, label="Historical Actual (Train)")
    ax.plot(holdout_time, y_holdout_true, color="#2ca02c", lw=2, marker="o", markersize=4, label="True Holdout Demand")
    ax.plot(holdout_time, best_holdout_pred, color="#1f77b4", lw=2.2, ls="--", label=f"Forecast ({best_model_name})")

    ax.fill_between(holdout_time, holdout_lb_80, holdout_ub_80, color="#1f77b4", alpha=0.25, label="80% Prediction Interval")
    ax.fill_between(holdout_time, holdout_lb_95, holdout_ub_95, color="#1f77b4", alpha=0.12, label="95% Prediction Interval")

    # Headroom capacity buffer: predicted + 95% one-sided buffer
    headroom_cap = best_holdout_pred + one_sided_buffer
    ax.plot(holdout_time, headroom_cap, color="#d62728", ls=":", lw=2.0, label=f"Headroom Allocation (+{one_sided_buffer:.3f}% Buffer)")

    ax.axvline(holdout_time.iloc[0], color="gray", ls="--", lw=1.2, label="Forecast Origin (t=110)")
    ax.set_xlabel("Elapsed Time (Hours)")
    ax.set_ylabel("Aggregate CPU Utilization (%)")
    ax.set_title(f"Held-out 2-Hour Workload Forecast with Calibrated Headroom ({best_model_name})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", framealpha=0.9)

    fig.tight_layout()
    fig8_path = FIG_DIR / "fig8_holdout_forecast_with_intervals.png"
    fig.savefig(fig8_path, dpi=300)
    plt.close(fig)
    print(f"Saved {fig8_path}")

    print("\nExperiments completed successfully!")


if __name__ == "__main__":
    run_experiments()
