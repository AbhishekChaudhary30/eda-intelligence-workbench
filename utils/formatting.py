"""Numeric and string formatting utilities for clean analytics display.

Provides consistent formatting across metric cards, data tables,
and statistical summaries.
"""

from __future__ import annotations

import math
from typing import Any


def format_number(
    val: float | None,
    decimals: int = 2,
    compact: bool = False,
) -> str:
    """Format a floating point or numeric value with clean thousands separators or compact suffix.

    Args:
        val: Numeric value or None.
        decimals: Number of decimal places.
        compact: If True, uses 'K', 'M', 'B' abbreviations for large numbers.

    Returns:
        Formatted string representation, or 'N/A' if NaN/None.
    """
    if val is None:
        return "N/A"
    try:
        val_float = float(val)
    except (ValueError, TypeError):
        return str(val)

    if math.isnan(val_float):
        return "NaN"
    if math.isinf(val_float):
        return "∞" if val_float > 0 else "-∞"

    if compact:
        abs_v = abs(val_float)
        if abs_v >= 1_000_000_000:
            return f"{val_float / 1_000_000_000:.{decimals}f}B"
        if abs_v >= 1_000_000:
            return f"{val_float / 1_000_000:.{decimals}f}M"
        if abs_v >= 1_000:
            return f"{val_float / 1_000:.{decimals}f}K"

    # Avoid negative zero
    if abs(val_float) < 1e-12:
        val_float = 0.0

    return f"{val_float:,.{decimals}f}"


def format_integer(val: float | None) -> str:
    """Format an integer with thousands commas.

    Args:
        val: Integer value or float representing integer.

    Returns:
        Formatted string or 'N/A'.
    """
    if val is None:
        return "N/A"
    try:
        val_int = round(float(val))
        return f"{val_int:,}"
    except (ValueError, TypeError, OverflowError):
        return "N/A"


def format_percent(val: float | None, decimals: int = 1) -> str:
    """Format a percentage value (0.0 - 100.0 or 0.0 - 1.0 depending on scale).

    Args:
        val: Number representing percentage or fraction.
        decimals: Decimal precision.

    Returns:
        Formatted percentage string with '%' suffix.
    """
    if val is None:
        return "N/A"
    try:
        val_float = float(val)
    except (ValueError, TypeError):
        return "N/A"

    if math.isnan(val_float):
        return "N/A"

    # If value is between 0 and 1 (fraction), scale to 100
    if 0.0 <= val_float <= 1.0 and val_float != 0.0 and val_float != 1.0:
        val_float *= 100.0

    return f"{val_float:.{decimals}f}%"


def format_bytes(size_bytes: float | None) -> str:
    """Format byte counts into human readable units (B, KB, MB, GB).

    Args:
        size_bytes: Byte count.

    Returns:
        Readable string (e.g. '2.45 MB').
    """
    if size_bytes is None:
        return "0 B"
    try:
        b = float(size_bytes)
    except (ValueError, TypeError):
        return "0 B"

    if b <= 0:
        return "0 B"

    units = ["B", "KB", "MB", "GB", "TB"]
    unit_idx = 0
    while b >= 1024.0 and unit_idx < len(units) - 1:
        b /= 1024.0
        unit_idx += 1

    decimals = 0 if unit_idx == 0 else 2
    return f"{b:.{decimals}f} {units[unit_idx]}"


def format_short_text(text: Any, max_len: int = 35) -> str:
    """Safely truncate and format text string for table and card previews."""
    if text is None:
        return "—"
    s = str(text)
    if len(s) > max_len:
        return s[: max_len - 3] + "..."
    return s
