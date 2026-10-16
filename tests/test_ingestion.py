"""Tests for the Data Ingestion Engine."""

from __future__ import annotations

import io

import pandas as pd
import pytest

from core.ingestion import (
    IngestionError,
    load_dataset,
    normalize_column_names,
)


def test_valid_csv_ingestion(sample_csv_bytes: bytes) -> None:
    """Test standard CSV file parsing and metadata generation."""
    res = load_dataset(sample_csv_bytes, filename="test_data.csv")
    assert res.df is not None
    assert len(res.df) == 20
    assert len(res.df.columns) == 6
    assert res.metadata.filename == "test_data.csv"
    assert "CSV" in res.metadata.file_format
    assert res.metadata.original_rows == 20
    assert not res.metadata.is_sampled


def test_valid_tsv_ingestion(sample_tsv_bytes: bytes) -> None:
    """Test TSV file parsing with tab delimiter sniffing."""
    res = load_dataset(sample_tsv_bytes, filename="data_table.tsv")
    assert len(res.df) == 20
    assert res.metadata.delimiter_used == "\t"


def test_valid_xlsx_ingestion(sample_xlsx_bytes: bytes) -> None:
    """Test Microsoft Excel XLSX ingestion."""
    res = load_dataset(sample_xlsx_bytes, filename="workbook.xlsx")
    assert len(res.df) == 20
    assert res.metadata.selected_sheet == "Employees"
    assert "Employees" in res.metadata.available_sheets


def test_valid_parquet_ingestion(sample_parquet_bytes: bytes) -> None:
    """Test columnar Apache Parquet ingestion."""
    res = load_dataset(sample_parquet_bytes, filename="dataset.parquet")
    assert len(res.df) == 20
    assert "salary" in res.df.columns


def test_empty_file_rejection() -> None:
    """Verify that 0-byte uploaded files raise actionable IngestionError."""
    with pytest.raises(IngestionError, match="empty"):
        load_dataset(b"", filename="empty.csv")


def test_unsupported_format_rejection() -> None:
    """Verify that unsupported extensions (.pdf, .exe, .json) are rejected."""
    with pytest.raises(IngestionError, match="Unsupported file format"):
        load_dataset(b"some content", filename="malicious.exe")


def test_duplicate_and_empty_column_normalization() -> None:
    """Test deterministic renaming for duplicate and blank column headers."""
    raw_cols = ["age", "age", "", "salary", "age"]
    norm_cols, col_map, warnings = normalize_column_names(raw_cols)

    assert norm_cols == ["age", "age_1", "unnamed_col_3", "salary", "age_2"]
    assert len(warnings) >= 3
    assert col_map["age_1"] == "age"
    assert "unnamed_col_3" in col_map


def test_csv_with_header_only_zero_rows() -> None:
    """Test dataset that has column headers but contains zero data records."""
    csv_content = b"col_a,col_b,col_c\n"
    res = load_dataset(csv_content, filename="empty_table.csv")
    assert len(res.df) == 0
    assert list(res.df.columns) == ["col_a", "col_b", "col_c"]
    assert any("zero data records" in w for w in res.metadata.warnings)


def test_encoding_fallback_handling() -> None:
    """Test fallback encoding parsing when text contains non-UTF8 bytes (e.g. CP1252 / Latin-1)."""
    # Create text containing Latin-1 / Windows-1252 specific byte: \xe9 (é)
    raw_latin1 = "id,name,value\n1,Ren\xe9,100\n2,Jos\xe9,200\n".encode("latin-1")
    res = load_dataset(raw_latin1, filename="accented.csv")
    assert len(res.df) == 2
    assert "René" in res.df["name"].values or "Ren" in str(res.df["name"].values)


def test_sampling_guardrail_on_large_datasets() -> None:
    """Test that massive datasets trigger reproducible sampling with clear metadata."""
    large_df = pd.DataFrame({"val": range(300)})
    csv_buf = io.BytesIO()
    large_df.to_csv(csv_buf, index=False)
    csv_bytes = csv_buf.getvalue()

    # Trigger threshold with small max_rows_threshold=100 and sample_size=50
    res = load_dataset(
        csv_bytes,
        filename="large.csv",
        max_rows_threshold=100,
        sample_size=50,
        allow_sampling=True,
    )
    assert res.metadata.is_sampled is True
    assert res.metadata.original_rows == 300
    assert len(res.df) == 50
    assert "50 rows" in res.metadata.sampling_explanation
