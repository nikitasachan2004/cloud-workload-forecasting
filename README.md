# Cloud Workload Forecasting Pipeline

Time-series forecasting pipeline and operational capacity planning analysis for the **Cloud Workload Forecasting Challenge**, using a curated 11.08-hour trace from the **Microsoft Azure Public Dataset V2** (`data-v1` release).

---

## Reproduction Guide

### 1. Environment Setup

Requires Python 3.11 or newer (tested on Python 3.12).

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies in editable mode
pip install --upgrade pip
pip install -e ".[dev]"
```

### 2. Dataset Verification

Download and verify the fixed `data-v1` release asset:

```bash
mkdir -p challenge_data
curl -fSL -o data-v1.tar.gz https://github.com/atomityhq/forecasting-challenge/releases/download/data-v1/data-v1.tar.gz
tar -xzf data-v1.tar.gz -C challenge_data
rm data-v1.tar.gz

# Checksum verification
cd challenge_data && shasum -a 256 -c SHA256SUMS && cd ..
```

### 3. Generate Submission Predictions

To reproduce `predictions.csv` from a clean checkout:

```bash
python -m cwfc.predict --data challenge_data --output predictions.csv
```

Outputs 24 predictions covering the 2-hour holdout ($t=33000\dots39900$):
```text
timestamp,active_vm_count,aggregate_cpu_utilization
```

### 4. Run Test Suite

```bash
pytest -v
```
Runs 12 unit and integration tests covering metric definitions, temporal splitting, feature causality invariants, submission CSV contract validation, variance decomposition consistency, and CLI smoke-test execution on mock data.

### 5. Run Full Pipeline

```bash
make run-all
```
Executes test suite, exploratory data analysis, cross-validation modeling experiments, and holdout prediction generation.

---

## Measured Benchmark Results

All candidate models were benchmarked across 5 expanding-window validation folds on the 110 training observations ($N_{\min}=62$, $H=24$, step=6), followed by evaluation on the 24-step holdout:

### Development Cross-Validation (5 Expanding Folds, $H=24$, 2 Hours)

| Model Architecture | Validation MAE (%) | Validation RMSE (%) | Validation sMAPE (%) | Validation MASE |
|---|---|---|---|---|
| **Direct Ridge ($\alpha=1.0$) [Selected]** | **0.1759 ± 0.0228** | **0.2045 ± 0.0232** | **1.8740 ± 0.2482** | **1.4074 ± 0.2043** |
| **Recursive Ridge ($\alpha=1.0$)** | 0.1798 ± 0.0893 | 0.2088 ± 0.0905 | 1.9058 ± 0.9245 | 1.4328 ± 0.7040 |
| **Persistence Baseline** | 0.1886 ± 0.0917 | 0.2178 ± 0.1020 | 1.9949 ± 0.9469 | 1.5061 ± 0.7285 |
| **Recursive LightGBM** | 0.1917 ± 0.1033 | 0.2222 ± 0.1076 | 2.0273 ± 1.0710 | 1.5294 ± 0.8160 |
| **Moving Average ($w=3$, 15m)** | 0.2189 ± 0.0978 | 0.2490 ± 0.1021 | 2.3131 ± 1.0111 | 1.7466 ± 0.7715 |
| **Moving Average ($w=6$, 30m)** | 0.2371 ± 0.1048 | 0.2651 ± 0.1118 | 2.5016 ± 1.0863 | 1.8933 ± 0.8311 |
| **Moving Average ($w=12$, 60m)** | 0.2501 ± 0.1054 | 0.2773 ± 0.1050 | 2.6411 ± 1.0950 | 1.9928 ± 0.8306 |

### Held-Out 2-Hour Test Evaluation ($t=110\dots133$)

| Model | Holdout MAE (%) | Holdout RMSE (%) | Holdout sMAPE (%) | Holdout MASE |
|---|---|---|---|---|
| **Direct Ridge [Selected]** | **0.1101%** | **0.1330%** | **1.1856%** | **0.9092** |
| **Recursive LightGBM** | 0.1221% | 0.1598% | 1.3162% | 1.0079 |
| **Moving Average ($w=12$)** | 0.1230% | 0.1613% | 1.3263% | 1.0156 |
| **Moving Average ($w=6$)** | 0.1336% | 0.1713% | 1.4410% | 1.1028 |
| **Moving Average ($w=3$)** | 0.1466% | 0.1846% | 1.5828% | 1.2102 |
| **Persistence Baseline** | 0.1468% | 0.1848% | 1.5851% | 1.2119 |
| **Recursive Ridge** | 0.2825% | 0.3370% | 3.0822% | 2.3324 |

---

## Leakage Prevention & Methodology Notes

1. **Chronological Splitting:** All model selection and residual calibrations use expanding-window folds within the first 110 timestamps.
2. **Causal Lag Features:** Lag features ($k \ge 1$) and rolling statistics are shifted by at least 1 observation so target $y_t$ never enters its own feature set.
3. **Preprocessing Isolation:** Scalers and estimators fit strictly on training folds.
4. **Prior Holdout Inspection Disclosure:** In an initial prototyping pass, holdout scores were computed for recursive baselines (revealing Recursive Ridge's extrapolation failure). Model selection between direct and recursive strategies was then anchored and frozen strictly on the 5 development cross-validation folds before final evaluation.

---

## Operational Capacity Planning Headroom

To balance SLA protection against excess idle infrastructure:
- **Headroom Policy:** Provision capacity at $\hat{y}_{t+h} + 0.2194\%$ CPU (calibrated 95% single-sided upper residual quantile from development out-of-fold predictions).
- **Holdout Result:** **0 under-provisioning breaches** across all 24 holdout intervals while keeping idle headroom overhead strictly under **2.5%** of cluster capacity.

For detailed analysis, failure modes, and figures, see [`reports/final_report.md`](reports/final_report.md).
