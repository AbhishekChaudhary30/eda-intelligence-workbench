# Statistical Methods & Analytical Methodology Guide
## EDA Intelligence Workbench — Phase 2

### Role & Purpose
This document provides complete mathematical formulations, assumptions, limitations, and interpretation rules for every statistical procedure implemented in the EDA Intelligence Workbench.

---

## 1. Descriptive Statistics & Distributional Moments

### 1.1 Central Tendency
- **Arithmetic Mean ($\bar{x}$)**: $\bar{x} = \frac{1}{N} \sum_{i=1}^N x_i$. Sensitive to extreme outliers and asymmetric skewness.
- **Median ($\tilde{x}$)**: 50th percentile. Non-parametric; highly resistant to extreme outliers.
- **Mode**: Most frequently occurring discrete value. Undefined or multi-modal when frequencies are identical.

### 1.2 Dispersion & Scale
- **Sample Standard Deviation ($s$)**: $s = \sqrt{\frac{1}{N-1} \sum_{i=1}^N (x_i - \bar{x})^2}$ (Bessel's correction $ddof=1$).
- **Sample Variance ($s^2$)**: $s^2 = \frac{1}{N-1} \sum_{i=1}^N (x_i - \bar{x})^2$.
- **Interquartile Range ($IQR$)**: $IQR = Q_3 - Q_1$, spanning the central 50% of the sample.
- **Coefficient of Variation ($CV$)**: $CV = \frac{s}{|\bar{x}|}$ for $|\bar{x}| > 10^{-9}$. Dimensionless measure of relative dispersion; only meaningful for ratio-scale data with a natural zero.

### 1.3 Higher Distributional Moments
- **Fisher-Pearson Skewness ($g_1$)**: Measures distributional asymmetry. Positive values indicate right/positive skew (long right tail); negative values indicate left/negative skew.
- **Excess Kurtosis ($g_2$)**: Measures tail extremity relative to a normal distribution ($g_2 = 0$). Positive values (leptokurtic) indicate heavier tails and higher outlier propensity; negative values (platykurtic) indicate lighter tails.

---

## 2. Distribution Diagnostics & Normality Testing

Normality tests assess whether sample observations are consistent with a normal distribution. **Failure to reject $H_0$ ($p \ge \alpha$) does NOT prove that the underlying population is exactly normal.**

### 2.1 Shapiro-Wilk Test
- **Null Hypothesis ($H_0$)**: The population is normally distributed.
- **Test Statistic ($W$)**: $W = \frac{(\sum_{i=1}^n a_i x_{(i)})^2}{\sum_{i=1}^n (x_i - \bar{x})^2}$, where $x_{(i)}$ are ordered sample values.
- **Assumptions**: Observations are independent and identically distributed (i.i.d.).
- **Sample Size Constraints**: Bounded to $3 \le N \le 5000$.
- **Limitations**: In very large sample sizes ($N > 500$), even negligible departures from normality cause statistical significance ($p < \alpha$), risking over-rejection of practically normal data.

### 2.2 D'Agostino-Pearson Omnibus Test
- **Null Hypothesis ($H_0$)**: The population skewness and kurtosis conform to Gaussian expectations.
- **Test Statistic ($K^2$)**: $K^2 = Z_1(g_1)^2 + Z_2(g_2)^2$, combining normalized transformations of sample skewness and kurtosis asymptotically following $\chi^2(2)$.
- **Assumptions**: Independent observations.
- **Sample Size Constraints**: Requires $N \ge 20$ for reliable moment approximations.

---

## 3. Confidence Intervals

### 3.1 Student's t Parametric Mean Confidence Interval
- **Formula**: $\bar{x} \pm t_{1-\alpha/2, \nu} \cdot \frac{s}{\sqrt{N}}$, where $\nu = N - 1$.
- **Interpretation**: In repeated sampling under identical conditions, $100(1-\alpha)\%$ of calculated intervals will encompass the true population mean $\mu$.
- **Assumptions**: Observations are independent and approximately normally distributed (or $N$ is sufficiently large under the Central Limit Theorem).

### 3.2 Non-Parametric Percentile Bootstrap Interval (Median)
- **Algorithm**: Resample $N$ observations with replacement $B$ times (default $B=500$, deterministic seed). Compute median $\tilde{x}^{*(b)}$ for each replicate.
- **Interval Bounds**: $[\hat{q}_{\alpha/2}, \hat{q}_{1 - \alpha/2}]$ of the empirical bootstrap distribution.
- **Assumptions**: Exchangeability of observations; sample is representative of the population.

---

## 4. Correlation Analysis & Multiple Testing Adjustments

### 4.1 Correlation Metrics
1. **Pearson Product-Moment ($r$)**: Measures linear co-variation: $r = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum (x_i - \bar{x})^2 \sum (y_i - \bar{y})^2}}$. Assumes bivariate normality and linearity.
2. **Spearman Rank Correlation ($\rho$)**: Pearson correlation computed on ranked variables. Measures monotonic relationships; resistant to monotonic non-linearities and outliers.
3. **Kendall Rank Correlation ($\tau$)**: Concordance metric: $\tau = \frac{C - D}{\frac{1}{2} n(n-1)}$, where $C$ is concordant pairs and $D$ is discordant pairs.

### 4.2 Degenerate Inputs & Pairwise Sample Sizes
- Constant columns ($s = 0$) are explicitly excluded to prevent ZeroDivisionError.
- Pairwise deletion computes $r$ and exact sample size $N_{ij}$ over mutually non-null observations.

### 4.3 False Discovery Rate (FDR) Adjustment
Evaluating $m$ pairs simultaneously inflates the family-wise type I error rate. We apply the **Benjamini-Hochberg (BH)** procedure:
1. Sort raw p-values in ascending order: $p_{(1)} \le p_{(2)} \le \dots \le p_{(m)}$.
2. Compute adjusted p-values: $p^{adj}_{(i)} = \min \left(1, \min_{k \ge i} \left( \frac{m}{k} p_{(k)} \right) \right)$.

---

## 5. Inferential Group Comparisons

### 5.1 Welch's Two-Sample t-Test
- **Null Hypothesis ($H_0$)**: $\mu_1 = \mu_2$.
- **Statistic**: $t = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}}$.
- **Degrees of Freedom**: Welch-Satterthwaite equation (robust against unequal variances $\sigma_1^2 \ne \sigma_2^2$).
- **Effect Size (Cohen's d)**: $d = \frac{\bar{x}_1 - \bar{x}_2}{s_{pooled}}$.

### 5.2 Mann-Whitney U Test (Wilcoxon Rank-Sum)
- **Null Hypothesis ($H_0$)**: Distributions of both groups are stochastically equal ($P(X > Y) = P(Y > X)$).
- **Assumptions**: Independent observations, ordinal or continuous scale. Does not assume normality.

### 5.3 One-Way ANOVA F-Test
- **Null Hypothesis ($H_0$)**: $\mu_1 = \mu_2 = \dots = \mu_k$ across $k \ge 3$ groups.
- **Assumptions**: Independent observations, normality within groups, homogeneity of variance (homoscedasticity).

### 5.4 Kruskal-Wallis H-Test
- **Null Hypothesis ($H_0$)**: Median ranks are equal across all $k \ge 3$ groups.
- **Assumptions**: Independent observations, ordinal or continuous scale. Non-parametric alternative to one-way ANOVA.

### 5.5 Chi-Square Test of Independence ($\chi^2$)
- **Null Hypothesis ($H_0$)**: Categorical feature $A$ is independent of categorical feature $B$.
- **Statistic**: $\chi^2 = \sum \frac{(O_{ij} - E_{ij})^2}{E_{ij}}$, where $E_{ij} = \frac{R_i C_j}{N}$.
- **Cochran's Rule Guardrail**: Displays a explicit warning if $> 20\%$ of contingency cells have expected counts $E_{ij} < 5$.

---

## 6. Statistical Outlier Detection Methods

**Important Governance Policy**: Statistically unusual observations may represent genuine extremes, natural variability, or data entry errors. The workbench **never** automatically deletes outliers.

### 6.1 Tukey's Fences (IQR Method)
- **Bounds**: $[Q_1 - k \cdot IQR, Q_3 + k \cdot IQR]$, default $k = 1.5$.
- **Strengths**: Distribution-agnostic, robust against heavy skewness.

### 6.2 Standard Z-Score
- **Bounds**: $[\bar{x} - c \cdot s, \bar{x} + c \cdot s]$, default $c = 3.0$.
- **Assumptions**: Assumes approximate normality. Extreme outliers artificially inflate $s$, leading to masking effects.

### 6.3 Median Absolute Deviation (MAD) / Modified Z-Score
- **Modified Z-Score**: $M_i = \frac{0.6745 \cdot |x_i - \tilde{x}|}{MAD}$, where $MAD = \text{median}(|x_i - \tilde{x}|)$.
- **Threshold**: Flagged when $M_i > 3.5$ (Iglewicz & Hoaglin).
- **Zero MAD Safeguard**: When $> 50\%$ of values are identical, $MAD = 0$; the system safely reports zero MAD rather than dividing by zero.

---

## 7. Dimensionality Reduction (PCA)

- **Standardization**: Continuous numeric features are preprocessed via z-score standardization ($z = \frac{x - \mu}{\sigma}$) to prevent variables with large absolute scales from dominating variance.
- **Missing Data Handling**: Rows with missing numeric values in the selected subset are dropped before eigen-decomposition.
- **Explained Variance**: Cumulative explained variance ratio ($R^2_{PCA}$) is documented explicitly to communicate information loss in 2D and 3D projections.
