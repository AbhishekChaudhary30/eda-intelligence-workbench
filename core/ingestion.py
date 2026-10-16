"""Robust data ingestion engine for EDA Intelligence Workbench.

Handles multi-format ingestion (CSV, TSV, XLSX, Parquet), encoding detection,
deterministic column deduplication, delimiter handling, worksheet selection,
and safe memory safeguards.
"""

from __future__ import annotations

import csv
import io
import os
from dataclasses import dataclass, field
from typing import Any, BinaryIO

import pandas as pd

from utils.security import (
    SecurityValidationError,
    sanitize_filename,
    validate_file_size,
    verify_file_content_signature,
)

# Supported extensions
SUPPORTED_FORMATS: dict[str, str] = {
    ".csv": "Comma-Separated Values (CSV)",
    ".tsv": "Tab-Separated Values (TSV)",
    ".xlsx": "Microsoft Excel Workbook (XLSX)",
    ".parquet": "Apache Parquet (Columnar)",
}

DEFAULT_MAX_ROWS_THRESHOLD: int = 250_000
DEFAULT_SAMPLE_SIZE: int = 50_000


class IngestionError(Exception):
    """User-facing actionable error during ingestion."""


@dataclass
class DatasetMetadata:
    """Metadata describing the ingested dataset and ingestion conditions."""

    filename: str
    file_format: str
    file_size_bytes: int
    original_rows: int
    original_cols: int
    loaded_rows: int
    loaded_cols: int
    is_sampled: bool = False
    sampling_explanation: str = ""
    selected_sheet: str | None = None
    available_sheets: list[str] = field(default_factory=list)
    encoding_used: str | None = None
    delimiter_used: str | None = None
    warnings: list[str] = field(default_factory=list)
    column_mapping: dict[str, str] = field(default_factory=dict)  # normalized -> original


@dataclass
class IngestionResult:
    """Result returned by the ingestion engine."""

    df: pd.DataFrame
    metadata: DatasetMetadata


def normalize_column_names(columns: list[Any]) -> tuple[list[str], dict[str, str], list[str]]:
    """Normalize and deduplicate column names deterministically.

    Handles:
    - Empty or whitespace-only column headers -> 'unnamed_col_1', etc.
    - Duplicate column headers -> 'name', 'name_1', 'name_2'
    - Stripping trailing/leading whitespaces

    Returns:
        tuple of (normalized_columns_list, mapping_dict, warnings_list)
    """
    normalized: list[str] = []
    mapping: dict[str, str] = {}
    warnings: list[str] = []
    seen_counts: dict[str, int] = {}

    for idx, col in enumerate(columns):
        raw_name = str(col).strip() if col is not None else ""
        if not raw_name:
            col_name = f"unnamed_col_{idx + 1}"
            warnings.append(f"Column index {idx} had an empty header; renamed to '{col_name}'.")
        else:
            col_name = raw_name

        if col_name in seen_counts:
            seen_counts[col_name] += 1
            unique_name = f"{col_name}_{seen_counts[col_name]}"
            warnings.append(f"Duplicate column name '{col_name}' normalized to '{unique_name}'.")
            normalized.append(unique_name)
            mapping[unique_name] = raw_name or f"(empty_header_{idx})"
        else:
            seen_counts[col_name] = 0
            normalized.append(col_name)
            mapping[col_name] = raw_name or f"(empty_header_{idx})"

    return normalized, mapping, warnings


def inspect_excel_sheets(file_bytes: bytes) -> list[str]:
    """Inspect and return available worksheet names in an Excel workbook."""
    try:
        import openpyxl

        wb = openpyxl.load_workbook(io.BytesIO(file_bytes), read_only=True)
        sheet_names = wb.sheetnames
        wb.close()
        return sheet_names
    except Exception as exc:
        raise IngestionError(
            f"Failed to inspect Excel workbook sheets. Ensure it is a valid .xlsx file. Error: {exc}"
        ) from exc


def sniff_delimiter_and_encoding(
    raw_bytes: bytes,
    default_delimiter: str = ",",
) -> tuple[str, str, list[str]]:
    """Sniff text encoding and delimiter with safe fallbacks and warnings."""
    warnings: list[str] = []
    encoding_used = "utf-8"
    decoded_text = ""

    # Attempt UTF-8 with BOM first, then UTF-8, then Latin-1/cp1252
    encodings_to_try = ["utf-8-sig", "utf-8", "cp1252", "latin-1"]
    for enc in encodings_to_try:
        try:
            decoded_text = raw_bytes[:65536].decode(enc)
            encoding_used = enc
            if enc not in ("utf-8", "utf-8-sig"):
                warnings.append(
                    f"UTF-8 decoding failed. Ingested using fallback encoding '{enc}'. "
                    "Some special characters may require verification."
                )
            break
        except UnicodeDecodeError:
            continue

    if not decoded_text:
        raise IngestionError(
            "The file could not be decoded with standard UTF-8 or Latin-1 encodings. "
            "Please verify the file encoding or convert it to UTF-8."
        )

    # Delimiter detection
    delimiter = default_delimiter
    try:
        sample_snippet = "\n".join(
            [line for line in decoded_text.splitlines()[:20] if line.strip()]
        )
        if sample_snippet:
            sniffer = csv.Sniffer()
            detected_dialect = sniffer.sniff(sample_snippet, delimiters=[",", "\t", ";", "|"])
            delimiter = detected_dialect.delimiter
    except (csv.Error, TypeError, ValueError):
        # Sniffing failed, retain default
        delimiter = default_delimiter

    return delimiter, encoding_used, warnings


def load_dataset(
    file_source: BinaryIO | bytes | str,
    filename: str,
    user_delimiter: str | None = None,
    user_encoding: str | None = None,
    sheet_name: str | None = None,
    max_rows_threshold: int = DEFAULT_MAX_ROWS_THRESHOLD,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    allow_sampling: bool = True,
) -> IngestionResult:
    """Ingest a tabular dataset from a file-like object, raw bytes, or path.

    Args:
        file_source: Uploaded file, binary stream, or file path.
        filename: Original filename.
        user_delimiter: Explicit delimiter override (for CSV/TSV).
        user_encoding: Explicit encoding override (for CSV/TSV).
        sheet_name: Worksheet name for Excel files.
        max_rows_threshold: Row count beyond which safe sampling applies.
        sample_size: Number of rows to sample if file exceeds threshold.
        allow_sampling: Whether to allow sampling on massive files.

    Returns:
        IngestionResult containing clean DataFrame and detailed metadata.

    Raises:
        IngestionError: If file is invalid, empty, or parsing fails.
    """
    clean_filename = sanitize_filename(filename)
    _, ext = os.path.splitext(clean_filename.lower())

    if ext not in SUPPORTED_FORMATS:
        supported_list = ", ".join(SUPPORTED_FORMATS.keys())
        raise IngestionError(
            f"Unsupported file format '{ext}'. Supported formats are: {supported_list}."
        )

    # Read binary content
    if isinstance(file_source, (str, os.PathLike)):
        with open(file_source, "rb") as f:
            file_bytes = f.read()
    elif isinstance(file_source, bytes):
        file_bytes = file_source
    else:
        # File-like object (e.g. Streamlit UploadedFile)
        file_source.seek(0)
        file_bytes = file_source.read()

    file_size_bytes = len(file_bytes)
    try:
        validate_file_size(file_bytes)
        verify_file_content_signature(file_bytes[:16], ext)
    except SecurityValidationError as err:
        raise IngestionError(str(err)) from err

    warnings: list[str] = []
    available_sheets: list[str] = []
    selected_sheet_name: str | None = None
    delimiter_used: str | None = None
    encoding_used: str | None = None

    try:
        if ext in {".csv", ".tsv"}:
            default_delim = "\t" if ext == ".tsv" else ","
            sniffed_delim, auto_enc, enc_warnings = sniff_delimiter_and_encoding(
                file_bytes, default_delimiter=default_delim
            )
            warnings.extend(enc_warnings)

            delimiter_used = user_delimiter if user_delimiter else sniffed_delim
            encoding_used = user_encoding if user_encoding else auto_enc

            # Read CSV with pandas
            # Use on_bad_lines='skip' with alert if lines are skipped
            csv_buffer = io.BytesIO(file_bytes)
            try:
                df = pd.read_csv(
                    csv_buffer,
                    delimiter=delimiter_used,
                    encoding=encoding_used,
                    low_memory=False,
                )
            except UnicodeDecodeError:
                csv_buffer.seek(0)
                df = pd.read_csv(
                    csv_buffer,
                    delimiter=delimiter_used,
                    encoding="latin-1",
                    low_memory=False,
                )
                warnings.append(
                    "Primary encoding failed during full parse; loaded using 'latin-1' fallback."
                )

        elif ext == ".xlsx":
            available_sheets = inspect_excel_sheets(file_bytes)
            if not available_sheets:
                raise IngestionError("The Excel workbook contains no visible worksheets.")

            selected_sheet_name = (
                sheet_name if sheet_name and sheet_name in available_sheets else available_sheets[0]
            )

            df = pd.read_excel(
                io.BytesIO(file_bytes),
                sheet_name=selected_sheet_name,
                engine="openpyxl",
            )

        elif ext in {".parquet", ".parquet.gzip"}:
            df = pd.read_parquet(io.BytesIO(file_bytes), engine="pyarrow")

        else:
            raise IngestionError(f"Unsupported format handler for {ext}")

    except IngestionError:
        raise
    except Exception as exc:
        raise IngestionError(
            f"Failed to parse {ext.upper()} file. Please verify file integrity and structure. Details: {exc}"
        ) from exc

    # Check for empty dataframe
    if df.empty and len(df.columns) == 0:
        raise IngestionError("The uploaded file contains no tabular rows or columns.")

    if len(df) == 0:
        warnings.append("The dataset contains column headers but contains zero data records.")

    # Deduplicate and normalize column headers
    orig_cols = list(df.columns)
    norm_cols, col_map, norm_warnings = normalize_column_names(orig_cols)
    df.columns = norm_cols
    warnings.extend(norm_warnings)

    original_rows = len(df)
    original_cols = len(df.columns)
    is_sampled = False
    sampling_explanation = ""

    # Check row bounds safeguard
    if allow_sampling and original_rows > max_rows_threshold:
        is_sampled = True
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        sampling_explanation = (
            f"Dataset contains {original_rows:,} rows, exceeding safe interactive threshold "
            f"({max_rows_threshold:,} rows). A representative random sample of {sample_size:,} rows "
            "(reproducible seed=42) was loaded to preserve system responsiveness."
        )
        warnings.append(sampling_explanation)

    metadata = DatasetMetadata(
        filename=clean_filename,
        file_format=SUPPORTED_FORMATS.get(ext, ext.upper()),
        file_size_bytes=file_size_bytes,
        original_rows=original_rows,
        original_cols=original_cols,
        loaded_rows=len(df),
        loaded_cols=len(df.columns),
        is_sampled=is_sampled,
        sampling_explanation=sampling_explanation,
        selected_sheet=selected_sheet_name,
        available_sheets=available_sheets,
        encoding_used=encoding_used,
        delimiter_used=delimiter_used,
        warnings=warnings,
        column_mapping=col_map,
    )

    return IngestionResult(df=df.copy(), metadata=metadata)
