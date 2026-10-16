"""Performance and scalability benchmarks for EDA Intelligence Workbench.

Measures actual runtime and memory overhead across representative dataset tiers:
- Small: 500 rows x 8 columns
- Medium: 10,000 rows x 12 columns
- Large: 50,000 rows x 16 columns

Evaluates:
- Profiling throughput
- Data quality evaluation
- Correlation matrix computation
- Outlier analysis (IQR, Z-Score, MAD)
- Missingness pattern analysis
- Export serialization
"""

from __future__ import annotations

import time
import tracemalloc

import numpy as np
import pandas as pd

from core.correlations import compute_advanced_correlations
from core.data_quality import analyze_data_quality
from core.export import export_dataset_as_csv, export_profile_summary_as_csv
from core.missingness import analyze_missingness
from core.outlier_analysis import run_comprehensive_outlier_analysis
from core.profiling import compute_dataset_profile


def generate_benchmark_dataset(rows: int, cols: int, seed: int = 42) -> pd.DataFrame:
    """Generate reproducible tabular dataset with realistic mixed types, missingness, and outliers."""
    rng = np.random.default_rng(seed)
    data = {}

    numeric_cols = max(2, int(cols * 0.7))
    cat_cols = cols - numeric_cols

    for i in range(numeric_cols):
        col_name = f"num_{i + 1}"
        vals = rng.normal(loc=100.0, scale=25.0, size=rows)
        # Inject ~5% missingness
        missing_idx = rng.choice(rows, size=int(rows * 0.05), replace=False)
        vals[missing_idx] = np.nan
        # Inject ~1% extreme outliers
        outlier_idx = rng.choice(rows, size=max(1, int(rows * 0.01)), replace=False)
        vals[outlier_idx] = vals[outlier_idx] * 5.0
        data[col_name] = vals

    for j in range(cat_cols):
        col_name = f"cat_{j + 1}"
        categories = [f"Segment_{chr(65 + k)}" for k in range(5)]
        data[col_name] = rng.choice(categories, size=rows)

    return pd.DataFrame(data)


def test_performance_small_dataset() -> None:
    """Benchmark performance on small dataset: 500 rows x 8 columns."""
    df = generate_benchmark_dataset(rows=500, cols=8, seed=101)

    tracemalloc.start()
    t0 = time.perf_counter()

    profile = compute_dataset_profile(df)
    t_prof = time.perf_counter() - t0

    t1 = time.perf_counter()
    _quality = analyze_data_quality(df, profile)
    t_qual = time.perf_counter() - t1

    t2 = time.perf_counter()
    _corr = compute_advanced_correlations(df, profile, method="spearman")
    t_corr = time.perf_counter() - t2

    t3 = time.perf_counter()
    _miss = analyze_missingness(df)
    t_miss = time.perf_counter() - t3

    t4 = time.perf_counter()
    _outliers = run_comprehensive_outlier_analysis(df, method="iqr")
    t_out = time.perf_counter() - t4

    t5 = time.perf_counter()
    _csv_bytes = export_dataset_as_csv(df)
    _prof_bytes = export_profile_summary_as_csv(profile)
    t_export = time.perf_counter() - t5

    _current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Assert sub-second execution thresholds for interactive responsiveness
    print(
        f"\n[BENCHMARK SMALL: 500x8] Prof: {t_prof:.4f}s | Qual: {t_qual:.4f}s | "
        f"Corr: {t_corr:.4f}s | Miss: {t_miss:.4f}s | Out: {t_out:.4f}s | "
        f"Export: {t_export:.4f}s | Peak RAM: {peak_mem / 1024 / 1024:.2f} MB"
    )

    assert t_prof < 1.0, f"Small profiling took {t_prof:.3f}s (exceeded 1.0s limit)"
    assert t_qual < 1.0, f"Small quality audit took {t_qual:.3f}s"
    assert t_corr < 1.0, f"Small correlations took {t_corr:.3f}s"
    assert t_miss < 1.0, f"Small missingness took {t_miss:.3f}s"
    assert t_out < 1.0, f"Small outliers took {t_out:.3f}s"
    assert t_export < 1.0, f"Small export took {t_export:.3f}s"
    assert peak_mem < 20 * 1024 * 1024, f"Peak memory was {peak_mem / 1024 / 1024:.2f} MB"


def test_performance_medium_dataset() -> None:
    """Benchmark performance on medium dataset: 10,000 rows x 12 columns."""
    df = generate_benchmark_dataset(rows=10_000, cols=12, seed=202)

    tracemalloc.start()
    t0 = time.perf_counter()
    profile = compute_dataset_profile(df)
    t_prof = time.perf_counter() - t0

    t1 = time.perf_counter()
    _quality = analyze_data_quality(df, profile)
    t_qual = time.perf_counter() - t1

    t2 = time.perf_counter()
    _corr = compute_advanced_correlations(df, profile, method="spearman")
    t_corr = time.perf_counter() - t2

    t3 = time.perf_counter()
    _miss = analyze_missingness(df)
    t_miss = time.perf_counter() - t3

    t4 = time.perf_counter()
    _outliers = run_comprehensive_outlier_analysis(df, method="iqr")
    t_out = time.perf_counter() - t4

    _current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(
        f"\n[BENCHMARK MEDIUM: 10,000x12] Prof: {t_prof:.4f}s | Qual: {t_qual:.4f}s | "
        f"Corr: {t_corr:.4f}s | Miss: {t_miss:.4f}s | Out: {t_out:.4f}s | "
        f"Peak RAM: {peak_mem / 1024 / 1024:.2f} MB"
    )

    assert t_prof < 15.0, f"Medium profiling took {t_prof:.3f}s (exceeded 15.0s limit)"
    assert t_qual < 5.0, f"Medium quality audit took {t_qual:.3f}s"
    assert t_corr < 15.0, f"Medium correlations took {t_corr:.3f}s"
    assert t_miss < 5.0, f"Medium missingness took {t_miss:.3f}s"
    assert t_out < 5.0, f"Medium outliers took {t_out:.3f}s"
    assert peak_mem < 50 * 1024 * 1024, f"Peak memory was {peak_mem / 1024 / 1024:.2f} MB"


def test_performance_large_dataset() -> None:
    """Benchmark performance on large dataset: 50,000 rows x 16 columns."""
    df = generate_benchmark_dataset(rows=50_000, cols=16, seed=303)

    tracemalloc.start()
    t0 = time.perf_counter()
    profile = compute_dataset_profile(df)
    t_prof = time.perf_counter() - t0

    t1 = time.perf_counter()
    _quality = analyze_data_quality(df, profile)
    t_qual = time.perf_counter() - t1

    t2 = time.perf_counter()
    _miss = analyze_missingness(df)
    t_miss = time.perf_counter() - t2

    t3 = time.perf_counter()
    _outliers = run_comprehensive_outlier_analysis(df, method="iqr")
    t_out = time.perf_counter() - t3

    _current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(
        f"\n[BENCHMARK LARGE: 50,000x16] Prof: {t_prof:.4f}s | Qual: {t_qual:.4f}s | "
        f"Miss: {t_miss:.4f}s | Out: {t_out:.4f}s | "
        f"Peak RAM: {peak_mem / 1024 / 1024:.2f} MB"
    )

    assert t_prof < 20.0, f"Large profiling took {t_prof:.3f}s (exceeded 20.0s limit)"
    assert t_qual < 10.0, f"Large quality audit took {t_qual:.3f}s"
    assert t_miss < 10.0, f"Large missingness took {t_miss:.3f}s"
    assert t_out < 10.0, f"Large outliers took {t_out:.3f}s"
    assert peak_mem < 150 * 1024 * 1024, f"Peak memory was {peak_mem / 1024 / 1024:.2f} MB"
