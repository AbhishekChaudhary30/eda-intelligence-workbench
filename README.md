# EDA Intelligence Workbench

> **Tagline:** Understand Your Data. Discover What Matters.  
> **Category:** Interactive Exploratory Data Analysis, Visualization Engine & Automated Quality Platform  
> **Version:** 3.0.0 (Release-Ready Master Release)  
> **Security:** 100% Local &bull; Air-Gapped &bull; Zero External AI APIs &bull; Zero Telemetry &bull; CWE-1236 Hardened  

---

## 1. Project Overview

**EDA Intelligence Workbench** is a professional-grade exploratory data analysis (EDA), statistical discovery, and automated data quality auditing platform. Engineered for data scientists, analysts, ML engineers, students, and interview candidates, it eliminates repetitive exploratory scripting by synthesizing deep structural audits, interactive statistical graphics, rule-based findings, and non-destructive data transformations into an integrated, offline-first desktop web workbench.

Phase 2 elevates the platform into a capable analytical engine featuring a centralized **Chart Registry** (26 chart types across 7 families), an **Advanced Statistical Engine** (parametric and non-parametric hypothesis tests, confidence intervals, normality diagnostics), an **Advanced Correlation Explorer** (Pearson, Spearman, Kendall with sample-size matrices and FDR multiple-testing control), **Dedicated Missingness and Outlier Panels** (IQR, Z-score, MAD), a deterministic **Automated Insight Engine** (9 categories), an **Interactive Data Transformation Workbench** (with preview diffs, undo, and reset), and a self-contained **Consolidated Executive HTML Audit Report**.

---

## 2. Core Capabilities & Feature Overview

### 📊 1. Centralized Visualization Studio & Chart Registry
The visualization engine uses a centralized chart registry (`core/chart_registry.py`) that decouples chart definitions, data requirements, and mathematical compatibility predicates from rendering logic.

- **Seven Comprehensive Chart Families (26 Chart Types)**:
  1. **Distribution**: Histogram (with KDE/Box margins), Box plot, Violin plot, Empirical CDF (ECDF), KDE Density, Rug plot, Frequency polygon, Pareto chart (80/20 rule), Cumulative frequency, Q-Q distribution diagnostic plot.
  2. **Relationship**: Bivariate scatter (with OLS linear regression), Bubble chart, 2D density contour, Pair plot matrix, Categorical-versus-numeric comparisons.
  3. **Categorical**: Vertical bar, Horizontal bar, Grouped bar, 100% Stacked bar, Treemap, Sunburst hierarchical breakdown, Donut chart, Categorical frequency.
  4. **Time-Series**: Temporal line, Area chart, Resampled record counts (Daily, Weekly, Monthly, Quarterly, Yearly), Rolling mean and rolling standard deviation, Seasonal grouping (day-of-week, month), Autocorrelation (ACF), Lag plots.
  5. **Comparison**: Grouped box plots, Category violin comparisons, Mean vs Median comparison bar charts, Category count summaries.
  6. **Multivariate**: Parallel coordinates, Pairwise correlation heatmap, PCA 2D projections.
  7. **3D Visualizations**: 3D Continuous Scatter, 3D PCA standardized projection.
- **Strict Compatibility Predicates**: Charts are only offered when column types, cardinality, and sample sizes meet mathematical prerequisites. Incompatible charts display actionable feedback explaining required inputs.

### 📐 2. Advanced Statistical Analysis Engine
- **Extended Moments Table**: Count, mean, median, mode, standard deviation, sample variance, coefficient of variation ($CV$), min, max, range, quartiles ($Q_1, Q_3$), percentiles ($P_5, P_{95}$), IQR, skewness, and excess kurtosis.
- **Distribution Diagnostics & Normality Testing**:
  - **Shapiro-Wilk Test** ($3 \le N \le 5000$).
  - **D'Agostino-Pearson Omnibus Test** ($N \ge 20$) testing skewness and kurtosis moments.
  - Interactive Q-Q diagnostic plot mapping empirical quantiles against standard Gaussian quantiles.
- **Confidence Intervals**:
  - Parametric **Student's t** confidence intervals for population means.
  - Non-parametric **Percentile Bootstrap** confidence intervals for medians ($B=500$ resamples, deterministic seed).
- **Inferential Group Comparisons & Effect Sizes**:
  - **Welch's Two-Sample t-Test**: Robust against unequal group variances; calculates Cohen's $d$ effect size.
  - **Mann-Whitney U Test**: Non-parametric rank-sum test for stochastic equality across 2 groups.
  - **One-Way ANOVA F-Test**: Parametric variance comparison across $k \ge 3$ groups.
  - **Kruskal-Wallis H-Test**: Non-parametric rank comparison across $k \ge 3$ groups.
  - **Chi-Square Test of Independence**: Tests association across categorical variables with Cochran's rule warning for expected cell counts $< 5$.

### 🔗 3. Advanced Correlation Explorer
- Supports **Pearson** (linear), **Spearman** (monotonic rank), and **Kendall** ($\tau$ concordance) correlation coefficients.
- **Exact Pairwise Sample Sizes**: Displays an $N_{ij}$ matrix accounting for pairwise missing data patterns.
- **Inference & FDR Adjustment**: Computes raw p-values and applies the **Benjamini-Hochberg (BH)** False Discovery Rate procedure to prevent family-wise error rate inflation.
- **Degenerate Input Safeguards**: Constant features and zero-variance columns are automatically isolated and reported in an exclusion list.
- **Ranked Correlation Table**: Filterable by minimum correlation threshold with strength and direction badges.

### ❓ 4. Missing-Data Investigation Panel
- **Structural Missingness Metrics**: Total missing cells, overall missingness percentage, and complete rows count.
- **Nullity Correlation Matrix**: Correlation of missingness flags ($\phi$ coefficient) identifying co-occurring missing data patterns.
- **Row-Level Missingness Distribution**: Frequency distribution of rows by number of missing attributes.
- **Pseudo-Missing Token Audit**: Scans for disguised string placeholders (e.g. `"N/A"`, `"null"`, `"?"`, `"-999"`, `"missing"`).
- **Group-Wise Missingness**: Missingness rates broken down across categorical segments.

### 🎯 5. Statistical Outlier Investigation Panel
- **Three Multi-Method Detectors**:
  1. **Tukey's IQR Fences**: $[Q_1 - k \cdot IQR, Q_3 + k \cdot IQR]$ with configurable multiplier ($k=1.5$ standard, $k=3.0$ extreme).
  2. **Standard Z-Score**: Points exceeding $|z| > \text{threshold}$ (default 3.0) with normal distribution assumptions.
  3. **Median Absolute Deviation (MAD)**: Modified Z-score ($M_i = \frac{0.6745 \cdot |x_i - \tilde{x}|}{MAD}$) with zero-MAD safeguards.
- **Interactive Threshold Visualizations**: Box plots and distribution histograms with highlighted outlier cutoffs.
- **Flagged Record Inspector & CSV Export**: Preview and export exact records triggering outlier boundaries.

### 💡 6. Deterministic Automated Insight Engine
- Rule-based insight synthesizer deriving findings across **9 analytical categories**: Data Quality, Missingness, Distribution, Outliers, Relationships, Categorical Patterns, Time-Series Patterns, Potential Data Problems, Analytical Opportunities.
- **Transparent Priority Scoring**: Deterministic score ($0 - 100$) combining severity weight, affected record percentage, and statistical magnitude.
- **Zero Hallucination**: No generative AI APIs, no cloud telemetry. Every insight links directly to verifiable quantitative evidence, recommended next steps, and statistical boundaries.

### 🛠️ 7. Interactive Transformation Workbench
- **Strict Immutability**: The original ingested dataset (`original_df`) is never mutated. Operations apply solely to a working copy (`working_df`).
- **Interactive Operations (Zero `eval()` / `exec()`)**:
  - **Column Operations**: Select/drop columns, rename columns, cast data types (numeric, string, datetime, category).
  - **Row Operations**: Safe condition-based filtering (`equals`, `contains`, `greater_than`, `less_than`, `is_null`, etc.), deduplication, sampling.
  - **Missing Value Handling**: Constant replacement, numeric imputation (mean, median), categorical imputation (mode), missingness threshold dropping.
  - **Numeric Transformations**: Standard scaling ($z$-score), Min-Max scaling, natural log ($\ln(1+x)$ with domain checks), robust scaling.
  - **Categorical Encodings**: Top-N category grouping with `"Other"`, one-hot encoding with prefix control.
- **Preview & Reversibility**: Generates side-by-side shape diffs before application. Full audit trail with step-by-step **Undo** and **Reset to Original** actions.

### ⚙️ 8. Reproducible Analysis Settings
- Dataclass-backed configuration (`AnalysisSettings`) managing correlation thresholds, FDR toggles, outlier parameters, random seeds, and sample sizes.
- **JSON Import & Export**: Save and load exact analytical parameters across user sessions without storing sensitive raw data.

### 📑 9. Executive Reporting & Export Center
- **Consolidated Executive HTML Report**: Self-contained, responsive, dark-themed HTML report containing executive KPI tiles, ranked insights, descriptive statistics, outlier audits, correlation matrices, and transformation lineage. Strictly sanitized with `html.escape()` against XSS.
- **Standardized CSV Exports**: Clean UTF-8 CSV downloads for dataset, profile summary, quality issues, descriptive statistics, correlation matrices, ranked pairs, outlier summaries, and transformation history.

---

## 3. Technology Stack & Dependencies

- **Language:** Python 3.11+ (tested on Python 3.13)
- **Application Framework:** Streamlit (>= 1.42.0)
- **Data Manipulation:** Pandas (>= 2.2.0), NumPy (>= 2.0.0)
- **Scientific & Statistical Computation:** SciPy (>= 1.14.0), Statsmodels (>= 0.14.0)
- **Machine Learning & Preprocessing:** scikit-learn (>= 1.5.0)
- **Interactive Visualizations:** Plotly (>= 5.24.0)
- **Columnar Engine & Excel:** PyArrow (>= 18.0.0), openpyxl (>= 3.1.0)
- **Testing & Quality Assurance:** pytest (>= 8.0.0), Ruff (>= 0.8.0)

---

## 4. Local Installation & Quickstart

### Step 1: Clone or Navigate to Project Directory
```bash
git clone https://github.com/<your-username>/eda-intelligence-workbench.git
cd eda-intelligence-workbench
```

### Step 2: Create & Activate Virtual Environment
* **Windows (PowerShell):**
  ```powershell
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  ```
* **macOS / Linux (Bash/Zsh):**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```
*(For running tests and linters: `pip install -r requirements-dev.txt`)*

### Step 4: Launch Application
```bash
streamlit run app.py
```
The workbench opens automatically in your default browser at `http://localhost:8501`.

---

## 5. Free Cloud Deployment (Streamlit Community Cloud)

This workbench is engineered to deploy seamlessly to **Streamlit Community Cloud** with 100% free hosting:

1. **Push to GitHub**: Push your local repository to GitHub.
2. **Access Streamlit Cloud**: Sign in to [share.streamlit.io](https://share.streamlit.io) using your GitHub account.
3. **Deploy App**:
   - Repository: `<your-username>/eda-intelligence-workbench`
   - Branch: `main`
   - Main file path: `app.py`
   - Python Version: Automatically read from `.python-version` (`3.11`)
4. **Instant Access**: Streamlit Cloud builds and serves the application at a public `https://<your-subdomain>.streamlit.app` URL with automated HTTPS and instant redeployment on git push.

---

## 6. Verification & Automated Test Suite

The test suite covers data ingestion, type validation, profiling, data quality, extended statistics, correlation matrices, missingness audits, outlier detectors, deterministic insights, transformations, chart registry predicates, report XSS sanitization, analytical correctness edge cases, performance benchmarks, and end-to-end user workflows.

### Run Full Test Suite:
```bash
pytest -v
```

### Run Code Linter:
```bash
ruff check .
```

### Measured Performance Benchmarks:
| Dataset Tier | Dimensions | Profiling | Quality Audit | Missingness Audit | Outlier Audit | Peak Memory |
|:---|:---|:---|:---|:---|:---|:---|
| **Small** | 500 rows &times; 8 cols | 0.141s | 0.016s | 0.027s | 0.013s | 0.88 MB |
| **Medium** | 10,000 rows &times; 12 cols | 1.643s | 0.087s | 0.101s | 0.038s | 3.60 MB |
| **Large** | 50,000 rows &times; 16 cols | 3.838s | 0.116s | 1.003s | 0.341s | 21.15 MB |

---

## 7. Security, Integrity & Privacy Guarantees

1. **Air-Gapped Operation**: Completely offline architecture. No external cloud calls, third-party LLMs, or telemetry trackers.
2. **Spreadsheet Formula Injection Defense (CWE-1236)**: All exported CSV cells starting with formula triggers (`=`, `+`, `-`, `@`, `\t`, `\r`) are automatically sanitized with leading single quotes (`'`) while preserving legitimate negative/positive numeric strings.
3. **Safe Row Filtering**: Zero use of `eval()` or `exec()`. Conditions are evaluated via parameterized pandas boolean masks.
4. **Cross-Site Scripting (XSS) Prevention**: All user-controlled text, column headers, and data values are escaped using `html.escape()`.
5. **Data Immutability**: Pristine separation between `original_df` and `working_df` guarantees that raw datasets cannot be corrupted during exploratory operations.
6. **Path Traversal Protection**: Uploaded file names are strictly sanitized to prevent directory traversal attacks.

---

## 8. Documentation Index

- [Architecture & System Design](docs/ARCHITECTURE.md)
- [Statistical Methods & Methodology Guide](docs/STATISTICAL_METHODS.md)
