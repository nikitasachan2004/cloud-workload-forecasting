# Cloud Workload Demand Forecasting and Capacity Planning Report

**Dataset:** Microsoft Azure Public Dataset V2 (Curated `data-v1` Release)  
**Evaluation Protocol:** 134 five-minute observations (~11.1 hours); 110 training observations ($t=0\dots32,700$ s), 24-step holdout horizon ($t=33,000\dots39,900$ s, 2 hours)  

---

## 1. Executive Summary

This report evaluates near-term aggregate workload demand forecasting and operational capacity planning using the Azure Public Dataset V2 (`data-v1` release). The curated trace provides 1,340,000 CPU readings across 10,000 virtual machines sampled every 5 minutes over an 11.08-hour window.

### Key Empirical Findings

1. **Workload Heterogeneity and Law of Total Variance:**
   VM-level CPU utilization is heavily right-skewed: the median CPU utilization across all VMs is **5.11%**, the mean is **9.39%**, and the 99th percentile reaches **87.35%**. Over **75.3%** of virtual machines operate at an average CPU utilization below 10%.
   A mathematically consistent variance decomposition ($ddof=0$) confirms the Law of Total Variance:
   $$\text{Total Variance} = 203.5188 \quad (100.0000\%)$$
   $$\text{Between-VM Variance} = 169.4691 \quad (83.2695\%)$$
   $$\text{Within-VM Variance} = 34.0497 \quad (16.7305\%)$$
   $$\text{Sum} = 203.5188 \quad (100.0000\%)$$
   Persistent sizing and workload differences between VMs account for over four-fifths of the overall variance, while temporal fluctuations within individual VMs account for under one-fifth.
2. **Aggregate Trace Dynamics & Absence of High-Frequency Seasonality:**
   Aggregating across 10,000 VMs cancels individual burstiness, yielding a smooth aggregate CPU trajectory between **8.98% and 9.91%** (mean **9.39%**, std **0.19%**). Active VM count is constant at **10,000 VMs** (zero VM lifecycle churn during the window). Autocorrelation analysis on the 110 training observations reveals strong short-term persistence ($\rho_1 = 0.662$ at 5 min, $\rho_3 = 0.664$ at 15 min, $\rho_6 = 0.450$ at 30 min). Spectral periodogram analysis confirms that power is concentrated in low-frequency drift (~4.5 to 9 hours). There is no physical or statistical basis for high-frequency seasonality (e.g. 15-minute cycles); short-lag correlation reflects autoregressive continuity.
3. **Model Selection on Development Folds (Direct vs. Recursive Forecasting):**
   Models were evaluated using a strictly chronological 5-fold expanding-window validation on the first 110 timestamps ($N_{\min}=62$, $H=24$, step=6). A **Direct Ridge Regression** architecture (fitting independent models for each lookahead step $h \in \{1\dots24\}$) achieved the lowest validation MAE (**0.1759% ± 0.0228%**), outperforming Recursive Ridge (0.1798% ± 0.0893%), Persistence (0.1886% ± 0.0917%), and Recursive LightGBM (0.1917% ± 0.1033%). Direct multi-step forecasting eliminates recursive error compounding and avoids extrapolating local trends into long horizons.
4. **Generalization to the 2-Hour Test Holdout:**
   Evaluating Direct Ridge on the 24-step holdout ($t=110\dots133$) produced:
   $$\text{Holdout MAE} = \mathbf{0.1101\%}, \quad \text{RMSE} = \mathbf{0.1330\%}, \quad \text{sMAPE} = \mathbf{1.1856\%}, \quad \text{MASE} = \mathbf{0.9092}$$
   Because MASE < 1.0, the model outperforms the naive training baseline out-of-sample. In contrast, Recursive Ridge extrapolated an upward training slope and suffered a holdout MAE of 0.2825%.
5. **Operational Capacity Planning Policy:**
   Using signed residuals ($e = y_{\text{actual}} - \hat{y}_{\text{pred}}$) from the 120 out-of-fold development predictions, we calibrated an empirical single-sided 95% capacity buffer:
   $$\text{Headroom Buffer} = +0.2194\% \text{ CPU}$$
   Provisioning capacity at $\hat{y} + 0.2194\%$ resulted in **0 under-provisioning breaches out of 24 holdout intervals (100.0% empirical coverage)** while maintaining idle capacity overhead strictly under 2.5% of cluster capacity.

---

## 2. Data Understanding and Preprocessing

### 2.1 Schema, Cadence, and Panel Balance

The release comprises two gzip-compressed files:
- `vm_cpu_readings.csv.gz`: 1,340,000 records across columns `timestamp` (int64), `vm_id` (str), `min_cpu` (float64), `max_cpu` (float64), `avg_cpu` (float64).
- `vm_metadata.csv.gz`: 10,000 records containing VM configuration (`vm_category`, `virtual_core_bucket`, `memory_gb_bucket`, etc.).

Integrity checks:
- **Panel Balance:** Exactly 10,000 unique VM IDs appear in both tables. Every VM contains exactly **134 consecutive observations**.
- **Cadence:** Timestamps span $t=0$ to $t=39,900$ seconds (11.08 hours). First-order timestamp differences $\Delta t = t_k - t_{k-1}$ are identically **300 seconds (5 minutes)**.
- **Physical Invariants:** All CPU measurements fall within $[0, 100]$. Zero records violate the physical ordering invariant: $\text{min\_cpu} \le \text{avg\_cpu} \le \text{max\_cpu}$.
- **Zero Duplicates:** Zero duplicate `(vm_id, timestamp)` records exist.

### 2.2 Missingness and Zero Treatment

In cloud telemetry, imputing missing values as zero biases aggregate capacity estimates downwards. In this dataset, there are zero missing records. However, our ingestion module (`cwfc/targets.py`) computes averages using `.dropna(subset=["avg_cpu"])`, ensuring that if a VM drops telemetry in production, it is excluded from the contributing denominator rather than falsely counted as an idle machine.

---

## 3. Exploratory Workload Analysis

### 3.1 VM-Level Skewness and Heavy Tails

VM-level CPU utilization is heavily skewed toward idle/low activity:

| Statistic / Quantile | CPU Utilization (%) | Operational Significance |
|---|---:|---|
| Minimum | 0.0072% | Negligible background activity |
| P10 | 1.2336% | Idle standby machines |
| P25 | 2.4771% | Low background services |
| Median (P50) | 5.1051% | Typical baseline daemon utilization |
| Mean ± Std | 9.3881% ± 14.2660% | Right-skewed distribution |
| P75 | 9.6167% | Modest steady load |
| P90 | 19.6015% | Active serving instances |
| P95 | 37.1329% | High-demand cluster nodes |
| P99 | 87.3511% | Saturated batch computation |
| Maximum | 99.4880% | Fully saturated VM |

```
VM Population Segmentation:
- Idle (mean avg_cpu < 1%): 582 VMs (5.8%)
- Low-utilization (1% <= mean < 10%): 6,952 VMs (69.5%)
- Medium-utilization (10% <= mean < 50%): 2,245 VMs (22.4%)
- High-utilization (mean >= 50%): 221 VMs (2.2%)
- Volatile (within-VM std > 15%): 367 VMs (3.7%)
```

### 3.2 Variance Decomposition

Using population degrees of freedom ($ddof=0$) across all components:
$$\text{Total Variance} = \frac{1}{NT} \sum_{i=1}^N \sum_{t=1}^T (X_{i,t} - \mu)^2 = 203.5188$$
$$\text{Between-VM Variance} = \frac{1}{N} \sum_{i=1}^N (\bar{X}_i - \mu)^2 = 169.4691 \quad (83.2695\%)$$
$$\text{Within-VM Variance} = \frac{1}{N} \sum_{i=1}^N \left[ \frac{1}{T} \sum_{t=1}^T (X_{i,t} - \bar{X}_i)^2 \right] = 34.0497 \quad (16.7305\%)$$
$$\text{Sum} = 169.4691 + 34.0497 = 203.5188 \quad (100.0000\%)$$

This confirms that individual VMs have stable baseline utilization characteristics over the 11-hour trace; variance between different machines dominates temporal volatility.

### 3.3 Workload Categories and Seasonality Limits

- **Delay-insensitive (6,180 VMs, 61.8%):** Mean CPU 8.24% ± 10.76% (Median 5.62%).
- **Interactive (3,091 VMs, 30.9%):** Mean CPU 11.49% ± 15.68% (Median 5.56%).
- **Unknown (729 VMs, 7.3%):** Mean CPU 10.23% ± 16.46% (Median 3.89%).

**Seasonality Assessment:**
Autocorrelation decays from $\rho_1 = 0.662$ to $\rho_6 = 0.450$ (30 min) and $\rho_{12} = 0.380$ (1 hour). An Augmented Dickey-Fuller test on the training series yields $t = -1.7348$ ($p = 0.413$), indicating a non-stationary local trend. Spectral analysis reveals power strictly at frequencies corresponding to multi-hour drift. Because an 11-hour window cannot capture diurnal (24h) cycles, seasonal models are excluded from primary consideration.

---

## 4. Forecasting Problem Formulation

Per the challenge contract in `DATASET.md`:
1. `active_vm_count(t)`: Count of unique VM IDs with valid readings at $t$. Because the population is constant at 10,000 VMs, the optimal forecast is persistence ($\hat{y}_{t+h} = 10,000$).
2. `aggregate_cpu_utilization(t)`: Arithmetic mean of VM-level `avg_cpu` readings at $t$, expressed as a percentage in $[0, 100]$.

Target horizons:
- $h=1$ (5 minutes): Immediate autoscaling and queue throttling.
- $h=6$ (30 minutes): Container pre-warming and pod scheduling.
- $h=24$ (2 hours): Cluster capacity reservations and host provisioning.

---

## 5. Leakage-Safe Validation Methodology

### 5.1 Chronological Splitting and Holdout Disclosure

- **Development Training Window:** First 110 observations ($t = 0 \dots 32,700$ s, ~9.17 hours).
- **Final Test Holdout:** Final 24 observations ($t = 33,000 \dots 39,900$ s, 2 hours).

*Holdout Integrity Disclosure:* Because the challenge distributes the public dataset as a single unmasked file without a hidden-label test server, initial exploratory prototyping computed holdout metrics for baseline recursive models (revealing Recursive Ridge's multi-step drift extrapolation). To ensure that model selection remained methodologically sound rather than overfitted to the holdout, the subsequent evaluation of direct multi-step forecasting was conducted and justified strictly using the 5-fold expanding-window cross-validation on the 110 training observations ($0.1759 \pm 0.0228$ vs $0.1798 \pm 0.0893$). We disclose this prior inspection candidly: holdout metrics reported in this document are retrospective confirmatory results, not blinded out-of-sample guarantees.

### 5.2 Expanding-Window Cross-Validation Design

Within the 110 training timestamps, we implemented 5 expanding-window folds ($H=24$, $N_{\min}=62$, step=6):
- Fold 0: Train [0..61] (62 steps) $\to$ Val [62..85] (24 steps)
- Fold 1: Train [0..67] (68 steps) $\to$ Val [68..91] (24 steps)
- Fold 2: Train [0..73] (74 steps) $\to$ Val [74..97] (24 steps)
- Fold 3: Train [0..79] (80 steps) $\to$ Val [80..103] (24 steps)
- Fold 4: Train [0..85] (86 steps) $\to$ Val [86..109] (24 steps)

---

## 6. Model Development & Validation Results

We evaluated five model families across all 5 expanding folds:
1. **Persistence:** $\hat{y}_{t+h} = y_t$.
2. **Moving Average:** Trailing window means over 3, 6, and 12 steps (15, 30, and 60 minutes).
3. **Recursive Ridge ($\alpha=1.0$):** 1-step autoregressive model rolled forward recursively.
4. **Direct Ridge ($\alpha=1.0$):** Independent regularized linear models for each step $h \in \{1\dots24\}$.
5. **Recursive LightGBM ($d=3, n=40$):** Gradient boosted tree rolled forward recursively.

### 6.1 Development Validation Benchmark (Mean ± Std over 5 Folds)

| Model Architecture | Validation MAE (%) | Validation RMSE (%) | Validation sMAPE (%) | Validation MASE |
|---|---|---|---|---|
| **Direct Ridge ($\alpha=1.0$)** | **0.1759 ± 0.0228** | **0.2045 ± 0.0232** | **1.8740 ± 0.2482** | **1.4074 ± 0.2043** |
| **Recursive Ridge ($\alpha=1.0$)** | 0.1798 ± 0.0893 | 0.2088 ± 0.0905 | 1.9058 ± 0.9245 | 1.4328 ± 0.7040 |
| **Persistence Baseline** | 0.1886 ± 0.0917 | 0.2178 ± 0.1020 | 1.9949 ± 0.9469 | 1.5061 ± 0.7285 |
| **Recursive LightGBM** | 0.1917 ± 0.1033 | 0.2222 ± 0.1076 | 2.0273 ± 1.0710 | 1.5294 ± 0.8160 |
| **Moving Average ($w=3$, 15m)** | 0.2189 ± 0.0978 | 0.2490 ± 0.1021 | 2.3131 ± 1.0111 | 1.7466 ± 0.7715 |
| **Moving Average ($w=6$, 30m)** | 0.2371 ± 0.1048 | 0.2651 ± 0.1118 | 2.5016 ± 1.0863 | 1.8933 ± 0.8311 |
| **Moving Average ($w=12$, 60m)** | 0.2501 ± 0.1054 | 0.2773 ± 0.1050 | 2.6411 ± 1.0950 | 1.9928 ± 0.8306 |

Direct Ridge achieved the lowest validation MAE and a four-fold reduction in variance across folds ($\pm 0.0228$ vs $\pm 0.0893$). Based strictly on development data, **Direct Ridge was selected as the final production model**.

---

## 7. Error Analysis & Holdout Evaluation

### 7.1 Performance on the 2-Hour Test Holdout

Evaluating all models on the final held-out 2-hour window ($t=110\dots133$, true mean 9.27%). As disclosed in §5.1, these figures represent retrospective performance on the public holdout segment:

| Model | Holdout MAE (%) | Holdout RMSE (%) | Holdout sMAPE (%) | Holdout MASE |
|---|---|---|---|---|
| **Direct Ridge (Selected)** | **0.1101%** | **0.1330%** | **1.1856%** | **0.9092** |
| **Recursive LightGBM** | 0.1221% | 0.1598% | 1.3162% | 1.0079 |
| **Moving Average ($w=12$)** | 0.1230% | 0.1613% | 1.3263% | 1.0156 |
| **Moving Average ($w=6$)** | 0.1336% | 0.1713% | 1.4410% | 1.1028 |
| **Moving Average ($w=3$)** | 0.1466% | 0.1846% | 1.5828% | 1.2102 |
| **Persistence Baseline** | 0.1468% | 0.1848% | 1.5851% | 1.2119 |
| **Recursive Ridge** | 0.2825% | 0.3370% | 3.0822% | 2.3324 |

### 7.2 Why Direct Ridge Outperformed Recursive Ridge

During the training period, aggregate demand had drifted upwards from 9.1% to 9.9% before settling around 9.4%.
- **Recursive Failure Mode:** Recursive Ridge learned a positive trend coefficient and rolled it forward 24 steps into the holdout. However, the true holdout drifted downwards to 9.27%. Compounding recursive error caused Ridge to over-forecast (MAE 0.2825%).
- **Direct Multi-Step Success:** Direct Ridge fits a separate model for each lookahead $h$. For distant steps ($h=20\dots24$), L2 regularization shrinks coefficients toward the unconditional historical mean ($9.41\%$) rather than extrapolating a local slope, achieving Holdout MAE **0.1101%** and MASE **0.9092**.

---

## 8. Capacity-Planning & Operational Recommendations

### 8.1 Calibrated One-Sided Headroom Buffer

In cloud infrastructure, under-provisioning causes SLA violations, request throttling, and latency degradation, whereas modest over-provisioning incurs minor idle cost.

Defining signed residual:
$$e = y_{\text{actual}} - \hat{y}_{\text{pred}}$$
A positive residual ($e > 0$) indicates under-forecasting.

Using $N=120$ out-of-fold development residuals from Direct Ridge:
$$\text{Residual Mean} = -0.1111\%, \quad \text{Std} = 0.1732\%$$
$$\text{Calibrated 95% Upper Buffer} = \text{Quantile}_{0.95}(e) = \mathbf{+0.2194\% \text{ CPU}}$$

**Operational Policy:**
$$\text{Allocated Capacity}(t+h) = \hat{y}_{t+h} + 0.2194\%$$

### 8.2 Holdout Diagnostic Verification

Applying this frozen buffer to the holdout forecast:
- **Under-provisioning Breaches:** **0 out of 24 timestamps (100.0% coverage)**.
- **Two-Sided 80% Interval ($[-0.3171\%, +0.1210\%]$):** 95.8% holdout coverage (23/24).
- **Two-Sided 95% Interval ($[-0.3662\%, +0.3249\%]$):** 100.0% holdout coverage (24/24).
- **Efficiency:** The $+0.2194\%$ buffer maintains cluster safety while keeping idle capacity overhead strictly under 2.5% of cluster capacity.

---

## 9. Limitations & Future Work

1. **Short Observation Duration:** An 11.1-hour trace cannot resolve diurnal (24h) cycles or weekly patterns. Models cannot be assumed to generalize across full multi-day operational cycles.
2. **Fixed Cohort:** The dataset selected VMs with 134 continuous observations. Production environments must model VM lifecycle churn and spot instance preemption.
3. **Next Steps:** Ingest 30-day traces from upstream Azure V2, evaluate multi-horizon quantile regression (pinball loss), and integrate core-weighted capacity metrics.
