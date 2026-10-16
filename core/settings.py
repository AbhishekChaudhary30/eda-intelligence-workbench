"""Centralized, validated analysis configuration and reproducibility settings.

Maintains analytical parameters (correlation methods, outlier thresholds,
confidence levels, significance level alpha, random seeds) with serialization,
backward-compatibility aliases, and mathematical boundary validation.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, NamedTuple


class ValidationResult(NamedTuple):
    """Result of settings validation tuple for direct unpacking (is_valid, error_message)."""

    is_valid: bool
    error_message: str


@dataclass
class AnalysisSettings:
    """Configurable analytical parameters with validated guardrails."""

    # Correlation settings
    default_correlation_method: str = "spearman"  # "pearson", "spearman", "kendall"
    correlation_threshold: float = 0.30
    correlation_fdr_correction: bool = True

    # Outlier settings
    default_outlier_method: str = "iqr"  # "iqr", "zscore", "mad"
    iqr_multiplier: float = 1.5
    zscore_threshold: float = 3.0
    mad_threshold: float = 3.5

    # Statistical test & significance settings
    alpha: float = 0.05
    confidence_level: float = 0.95
    bootstrap_iterations: int = 500

    # Visualization settings
    histogram_bins: int = 30
    max_categorical_levels: int = 15
    default_time_series_freq: str = "Raw"  # "Raw", "Daily", "Weekly", "Monthly"
    rolling_window_size: int = 7

    # Data hygiene & missing tokens
    custom_missing_tokens: list[str] = field(
        default_factory=lambda: ["NA", "N/A", "null", "None", "none", "?", "-999", "missing"]
    )

    # Sampling & reproducibility
    random_seed: int = 42
    max_scatter_points: int = 5000
    max_pairwise_columns: int = 6

    # -------------------------------------------------------------------------
    # Backward Compatibility Aliases & Properties
    # -------------------------------------------------------------------------

    @property
    def correlation_method(self) -> str:
        """Alias for default_correlation_method."""
        return self.default_correlation_method

    @correlation_method.setter
    def correlation_method(self, val: str) -> None:
        self.default_correlation_method = val

    @property
    def min_correlation_threshold(self) -> float:
        """Alias for correlation_threshold."""
        return self.correlation_threshold

    @min_correlation_threshold.setter
    def min_correlation_threshold(self, val: float) -> None:
        self.correlation_threshold = val

    @property
    def outlier_method(self) -> str:
        """Alias for default_outlier_method."""
        return self.default_outlier_method

    @outlier_method.setter
    def outlier_method(self, val: str) -> None:
        self.default_outlier_method = val

    @property
    def normality_test_alpha(self) -> float:
        """Alias for alpha."""
        return self.alpha

    @normality_test_alpha.setter
    def normality_test_alpha(self, val: float) -> None:
        self.alpha = val

    @property
    def max_categories_display(self) -> int:
        """Alias for max_categorical_levels."""
        return self.max_categorical_levels

    @max_categories_display.setter
    def max_categories_display(self, val: int) -> None:
        self.max_categorical_levels = val

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    def validate(self) -> ValidationResult:
        """Validate settings against strict mathematical boundaries.

        Returns:
            ValidationResult (NamedTuple of is_valid: bool, error_message: str).
        """
        # Alpha validation (0 < alpha < 1)
        if not (0.0 < self.alpha < 1.0):
            return ValidationResult(
                False,
                f"Significance level alpha must be strictly between 0 and 1 (received {self.alpha}).",
            )

        # Correlation method validation
        valid_corr = ("pearson", "spearman", "kendall")
        if self.default_correlation_method.lower() not in valid_corr:
            return ValidationResult(
                False,
                f"Correlation method '{self.default_correlation_method}' is unsupported. "
                f"Choose from {', '.join(valid_corr)}.",
            )

        # Correlation threshold validation
        if not (0.0 <= self.correlation_threshold <= 1.0):
            return ValidationResult(
                False,
                f"Correlation threshold must be between 0.0 and 1.0 (received {self.correlation_threshold}).",
            )

        # Outlier method validation
        valid_out = ("iqr", "zscore", "mad")
        if self.default_outlier_method.lower() not in valid_out:
            return ValidationResult(
                False,
                f"Outlier method '{self.default_outlier_method}' is unsupported. "
                f"Choose from {', '.join(valid_out)}.",
            )

        # IQR multiplier validation
        if not (0.5 <= self.iqr_multiplier <= 5.0):
            return ValidationResult(
                False,
                f"IQR multiplier must be between 0.5 and 5.0 (received {self.iqr_multiplier}).",
            )

        # Z-score threshold validation
        if not (1.0 <= self.zscore_threshold <= 6.0):
            return ValidationResult(
                False,
                f"Z-score threshold must be between 1.0 and 6.0 (received {self.zscore_threshold}).",
            )

        # MAD threshold validation
        if not (1.0 <= self.mad_threshold <= 10.0):
            return ValidationResult(
                False,
                f"MAD threshold must be between 1.0 and 10.0 (received {self.mad_threshold}).",
            )

        # Confidence level validation
        if not (0.50 <= self.confidence_level < 1.0):
            return ValidationResult(
                False,
                f"Confidence level must be between 0.50 and 0.999 (received {self.confidence_level}).",
            )

        # Bootstrap iterations validation
        if not (100 <= self.bootstrap_iterations <= 5000):
            return ValidationResult(
                False,
                f"Bootstrap iterations must be between 100 and 5000 (received {self.bootstrap_iterations}).",
            )

        # Histogram bins validation
        if not (5 <= self.histogram_bins <= 200):
            return ValidationResult(
                False,
                f"Histogram bins must be between 5 and 200 (received {self.histogram_bins}).",
            )

        # Rolling window validation
        if not (2 <= self.rolling_window_size <= 100):
            return ValidationResult(
                False,
                f"Rolling window size must be between 2 and 100 (received {self.rolling_window_size}).",
            )

        # Time series frequency validation
        valid_freqs = ("Raw", "Daily", "Weekly", "Monthly")
        if self.default_time_series_freq not in valid_freqs:
            return ValidationResult(
                False,
                f"Time-series frequency '{self.default_time_series_freq}' is invalid. "
                f"Choose from {', '.join(valid_freqs)}.",
            )

        return ValidationResult(True, "")

    # -------------------------------------------------------------------------
    # JSON Serialization & Backward Compatibility Deserialization
    # -------------------------------------------------------------------------

    def to_json(self) -> str:
        """Export settings to formatted JSON string."""
        data = asdict(self)
        # Include legacy keys in export for cross-version compatibility
        data["correlation_method"] = self.default_correlation_method
        data["min_correlation_threshold"] = self.correlation_threshold
        data["outlier_method"] = self.default_outlier_method
        data["normality_test_alpha"] = self.alpha
        data["max_categories_display"] = self.max_categorical_levels
        return json.dumps(data, indent=2)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AnalysisSettings:
        """Instantiate AnalysisSettings from a dictionary with legacy field mapping."""
        clean_data = dict(data)

        # Map legacy keys to current keys if present
        if "correlation_method" in clean_data and "default_correlation_method" not in clean_data:
            clean_data["default_correlation_method"] = clean_data.pop("correlation_method")
        if "min_correlation_threshold" in clean_data and "correlation_threshold" not in clean_data:
            clean_data["correlation_threshold"] = clean_data.pop("min_correlation_threshold")
        if "outlier_method" in clean_data and "default_outlier_method" not in clean_data:
            clean_data["default_outlier_method"] = clean_data.pop("outlier_method")
        if "normality_test_alpha" in clean_data and "alpha" not in clean_data:
            clean_data["alpha"] = clean_data.pop("normality_test_alpha")
        if "max_categories_display" in clean_data and "max_categorical_levels" not in clean_data:
            clean_data["max_categorical_levels"] = clean_data.pop("max_categories_display")

        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in clean_data.items() if k in valid_fields}

        inst = cls(**filtered)
        val_res = inst.validate()
        if not val_res.is_valid:
            # Fall back invalid fields to defaults
            default_inst = cls()
            return default_inst
        return inst

    @classmethod
    def from_json(cls, json_str: str) -> AnalysisSettings:
        """Create AnalysisSettings from JSON string with fallback to defaults on error."""
        try:
            data = json.loads(json_str)
            if not isinstance(data, dict):
                return cls()
            return cls.from_dict(data)
        except (ValueError, KeyError, TypeError, json.JSONDecodeError):
            return cls()
