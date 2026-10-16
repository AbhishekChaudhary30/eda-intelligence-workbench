"""Shared pytest fixtures for the EDA Intelligence Workbench test suite."""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def clean_numeric_df() -> pd.DataFrame:
    """Fixture providing clean numeric data with known statistical moments."""
    np.random.seed(42)
    return pd.DataFrame({
        "var_a": np.linspace(10.0, 100.0, 50),
        "var_b": np.linspace(100.0, 10.0, 50),  # Perfectly inverse correlation r = -1.0
        "var_c": np.random.normal(50.0, 5.0, 50),
        "int_feat": np.arange(1, 51),
    })


@pytest.fixture
def mixed_dataset_df() -> pd.DataFrame:
    """Fixture providing diverse data types: numeric, categorical, boolean, datetime."""
    return pd.DataFrame({
        "id": [f"ID_{i}" for i in range(1, 21)],
        "age": [25, 30, 35, 40, 22, 28, 45, 50, 60, 29, 31, 33, 42, 55, 62, 26, 38, 48, 52, 24],
        "department": [
            "Engineering", "Product", "Sales", "Engineering", "Marketing",
            "Sales", "Engineering", "Product", "Engineering", "Sales",
            "Engineering", "Product", "Sales", "Marketing", "Human Resources",
            "Engineering", "Sales", "Engineering", "Product", "Sales",
        ],
        "is_active": [True, True, False, True, True, True, False, True, True, True,
                      True, False, True, True, True, True, True, False, True, True],
        "salary": [75000.0, 85000.0, 62000.0, 95000.0, 58000.0,
                   64000.0, 110000.0, 88000.0, 92000.0, 71000.0,
                   81000.0, 89000.0, 67000.0, 61000.0, 55000.0,
                   78000.0, 73000.0, 105000.0, 91000.0, 69000.0],
        "join_date": pd.date_range("2021-01-01", periods=20, freq="15D").strftime("%Y-%m-%d"),
    })


@pytest.fixture
def edge_case_df() -> pd.DataFrame:
    """Fixture containing degenerate conditions: constant, all-null, duplicate rows, whitespace."""
    df = pd.DataFrame({
        "constant_col": [42.0] * 20,
        "all_null_col": [np.nan] * 20,
        "mixed_nulls": [1.0, 2.0, np.nan, 4.0, 5.0, np.nan, 7.0, 8.0, np.nan, 10.0,
                        11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0],
        "text_with_blanks": ["Alpha", "Beta", "", "   ", "Gamma",
                             "Alpha", "Beta", "Delta", "", "   ",
                             "Alpha", "Beta", "Epsilon", "Zeta", "Eta",
                             "Alpha", "Beta", "Theta", "Iota", "Kappa"],
        "outlier_col": [10.0, 11.0, 12.0, 10.5, 11.2, 10.8, 12.1, 11.0, 10.9, 11.5,
                        10.7, 11.3, 10.6, 11.8, 11.1, 10.4, 11.9, 10.2, 500.0, -250.0],  # 2 massive outliers
        "dup_a": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
        "dup_b": ["A", "B", "C", "D", "E", "A", "B", "C", "D", "E",
                  "A", "B", "C", "D", "E", "A", "B", "C", "D", "E"],
    })
    return pd.concat([df, df.iloc[[0, 1]]], ignore_index=True)


@pytest.fixture
def sample_csv_bytes(mixed_dataset_df: pd.DataFrame) -> bytes:
    """Generate in-memory CSV bytes."""
    return mixed_dataset_df.to_csv(index=False).encode("utf-8")


@pytest.fixture
def sample_tsv_bytes(mixed_dataset_df: pd.DataFrame) -> bytes:
    """Generate in-memory TSV bytes."""
    return mixed_dataset_df.to_csv(index=False, sep="\t").encode("utf-8")


@pytest.fixture
def sample_xlsx_bytes(mixed_dataset_df: pd.DataFrame) -> bytes:
    """Generate in-memory XLSX bytes."""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        mixed_dataset_df.to_excel(writer, index=False, sheet_name="Employees")
    buffer.seek(0)
    return buffer.getvalue()


@pytest.fixture
def sample_parquet_bytes(mixed_dataset_df: pd.DataFrame) -> bytes:
    """Generate in-memory Parquet bytes."""
    buffer = io.BytesIO()
    mixed_dataset_df.to_parquet(buffer, engine="pyarrow", index=False)
    buffer.seek(0)
    return buffer.getvalue()
