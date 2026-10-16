"""Security and sanitization utilities for the EDA Intelligence Workbench.

Provides filename sanitization, file signature verification, size checks,
and safe HTML escaping for dynamic output.
"""

from __future__ import annotations

import html
import os
import re
from typing import Any, BinaryIO

import pandas as pd

# Maximum permitted upload size in bytes (default 100 MB)
DEFAULT_MAX_UPLOAD_BYTES: int = 100 * 1024 * 1024

# Allowed file extensions
ALLOWED_EXTENSIONS: set[str] = {".csv", ".tsv", ".xlsx", ".parquet", ".parquet.gzip"}

# Signatures for known binary formats
SIGNATURE_PARQUET: bytes = b"PAR1"
SIGNATURE_ZIP_XLSX: bytes = b"PK\x03\x04"


class SecurityValidationError(ValueError):
    """Raised when an uploaded file violates security or integrity rules."""


def sanitize_filename(filename: str, max_length: int = 120) -> str:
    """Sanitize user-provided filename to prevent directory traversal and injection.

    Args:
        filename: Raw filename string from client.
        max_length: Maximum allowed length for the clean filename.

    Returns:
        Safe, stripped filename containing only alphanumeric characters,
        dots, underscores, and hyphens.
    """
    if not filename or not isinstance(filename, str):
        return "unnamed_dataset.csv"

    # Strip directory path separators and path traversal
    base_name = os.path.basename(filename.strip().replace("\\", "/"))

    # Remove null bytes or control characters
    clean_name = re.sub(r"[\x00-\x1f\x7f]", "", base_name)

    # Restrict to safe characters: alphanumerics, dots, hyphens, underscores
    clean_name = re.sub(r"[^\w\.\-]", "_", clean_name)

    # Prevent hidden files or double dots
    clean_name = clean_name.lstrip(".")
    clean_name = re.sub(r"\.{2,}", ".", clean_name)

    if not clean_name:
        clean_name = "unnamed_dataset"

    # Truncate if excessively long while preserving extension
    if len(clean_name) > max_length:
        name_part, ext = os.path.splitext(clean_name)
        clean_name = name_part[: max_length - len(ext)] + ext

    return clean_name


def validate_file_size(
    file_obj: BinaryIO | bytes,
    max_bytes: int = DEFAULT_MAX_UPLOAD_BYTES,
) -> int:
    """Verify file does not exceed maximum upload size.

    Args:
        file_obj: File-like binary stream or raw bytes.
        max_bytes: Maximum allowed bytes.

    Returns:
        The total byte size of the file.

    Raises:
        SecurityValidationError: If the file is empty or exceeds max_bytes.
    """
    if isinstance(file_obj, (bytes, bytearray)):
        size = len(file_obj)
    elif hasattr(file_obj, "size") and isinstance(file_obj.size, int):
        size = file_obj.size
    elif hasattr(file_obj, "seek") and hasattr(file_obj, "tell"):
        current_pos = file_obj.tell()
        file_obj.seek(0, os.SEEK_END)
        size = file_obj.tell()
        file_obj.seek(current_pos)
    else:
        raise SecurityValidationError("Cannot determine file size from input.")

    if size == 0:
        raise SecurityValidationError("The uploaded file is empty (0 bytes).")

    if size > max_bytes:
        max_mb = max_bytes / (1024 * 1024)
        size_mb = size / (1024 * 1024)
        raise SecurityValidationError(
            f"File size ({size_mb:.1f} MB) exceeds maximum permitted limit ({max_mb:.0f} MB)."
        )

    return size


def verify_file_content_signature(header_bytes: bytes, declared_ext: str) -> None:
    """Validate declared file extension against actual initial binary signature.

    Args:
        header_bytes: Initial bytes read from the file (at least 8 bytes recommended).
        declared_ext: The file extension (e.g., '.parquet', '.xlsx', '.csv').

    Raises:
        SecurityValidationError: If file content does not match expected signature.
    """
    declared_ext = declared_ext.lower()

    if declared_ext in {".parquet", ".parquet.gzip"}:
        if not header_bytes.startswith(SIGNATURE_PARQUET):
            raise SecurityValidationError(
                "File has .parquet extension but does not begin with the valid Parquet magic signature (PAR1)."
            )

    elif declared_ext == ".xlsx":
        if not header_bytes.startswith(SIGNATURE_ZIP_XLSX):
            raise SecurityValidationError(
                "File has .xlsx extension but does not contain a valid OpenXML/ZIP archive signature."
            )

    elif declared_ext in {".csv", ".tsv"}:
        # For CSV/TSV, ensure it does not start with executable or archive magic bytes
        if header_bytes.startswith(b"MZ"):  # Windows PE binary
            raise SecurityValidationError("File appears to be an executable binary, not a CSV/TSV file.")
        if header_bytes.startswith(b"\x7fELF"):  # Linux ELF binary
            raise SecurityValidationError("File appears to be an ELF executable binary, not a CSV/TSV file.")


def escape_html(text: object) -> str:
    """Escape text safely for injection into HTML elements to prevent XSS.

    Args:
        text: Any python object or string.

    Returns:
        HTML-escaped string.
    """
    if text is None:
        return ""
    return html.escape(str(text), quote=True)


def sanitize_spreadsheet_cell(val: Any) -> Any:
    """Neutralize spreadsheet formula injection (CWE-1236) for text cells.

    If a text value starts with formula triggers (=, +, -, @, \\t, \\r) and is
    not a valid numeric value, prefix it with a single quote (') to prevent
    formula execution in Microsoft Excel, LibreOffice Calc, or Google Sheets.
    """
    if not isinstance(val, str) or not val:
        return val

    if val[0] in ("=", "+", "-", "@", "\t", "\r"):
        # Preserve legitimate numbers stored as strings (e.g. "-42", "+100", "-3.14")
        try:
            float(val)
            return val
        except ValueError:
            return f"'{val}"
    return val


def sanitize_dataframe_for_spreadsheet(df: pd.DataFrame) -> pd.DataFrame:
    """Sanitize all text/object columns in a DataFrame against spreadsheet formula injection.

    Returns a clean copy of the DataFrame with formula injection triggers neutralized.
    Numeric, boolean, and datetime columns remain unchanged.
    """
    sanitized = df.copy()
    for col in sanitized.columns:
        if sanitized[col].dtype == "object" or pd.api.types.is_string_dtype(sanitized[col]):
            sanitized[col] = sanitized[col].apply(sanitize_spreadsheet_cell)
    return sanitized
