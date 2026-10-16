"""Caching and identity helper utilities for deterministic dataset processing.

Generates stable cache keys based on dataset structure, dimensions, and contents
to prevent redundant computations while keeping memory usage bounded.
"""

from __future__ import annotations

import hashlib
from typing import Any

import pandas as pd


def compute_dataset_hash(df: pd.DataFrame, extra_params: dict[str, Any] | None = None) -> str:
    """Compute a deterministic hash fingerprint for a pandas DataFrame and its options.

    Combines:
    - Shape (rows, cols)
    - Column names and order
    - Column data types
    - Fingerprint sample (first 10, middle 10, last 10 rows if large)
    - Any extra configuration parameters

    Args:
        df: Pandas DataFrame.
        extra_params: Optional dictionary of analytical parameters.

    Returns:
        Hexadecimal SHA-256 fingerprint string.
    """
    hasher = hashlib.sha256()

    # Dimensions
    hasher.update(f"shape:{df.shape[0]}x{df.shape[1]};".encode())

    # Column names & types
    col_sig = ",".join(f"{c}:{t!s}" for c, t in zip(df.columns, df.dtypes))
    hasher.update(col_sig.encode("utf-8"))

    # Content sample signature
    row_count = len(df)
    if row_count <= 50:
        sample_str = df.to_string(index=False)
    else:
        head_str = df.head(10).to_string(index=False)
        tail_str = df.tail(10).to_string(index=False)
        sample_str = f"{head_str}---{tail_str}"

    hasher.update(sample_str.encode("utf-8", errors="replace"))

    # Extra parameters
    if extra_params:
        for k in sorted(extra_params.keys()):
            hasher.update(f";{k}={extra_params[k]}".encode())

    return hasher.hexdigest()[:16]
