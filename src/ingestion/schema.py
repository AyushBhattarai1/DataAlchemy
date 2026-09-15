"""Semantic Schema Detection Module for DataAlchemy.

Analyzes ingested tabular DataFrames to classify columns into explainable
semantic categories: numerical, categorical, datetime, boolean, and identifier.
"""

from dataclasses import dataclass
from enum import Enum
import re
from typing import Any, Dict, List, Optional, Set, Union
import warnings

import pandas as pd


# -----------------------------------------------------------------------------
# Semantic Types
# -----------------------------------------------------------------------------

class SemanticType(str, Enum):
    """Enumeration of supported semantic column categories."""
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    BOOLEAN = "boolean"
    IDENTIFIER = "identifier"

    def __str__(self) -> str:
        return self.value


# -----------------------------------------------------------------------------
# Dataset Schema Representation
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class DatasetSchema:
    """Represents the semantic schema of an ingested dataset.

    Attributes:
        column_types: Mapping of column name to detected SemanticType.
    """
    column_types: Dict[str, SemanticType]

    @property
    def columns(self) -> List[str]:
        """Return all column names present in the schema."""
        return list(self.column_types.keys())

    def get_columns_by_type(self, semantic_type: Union[SemanticType, str]) -> List[str]:
        """Retrieve column names that match a specific semantic type.

        Args:
            semantic_type: SemanticType enum or string representation.

        Returns:
            List of column names matching the requested type.
        """
        target_value = semantic_type.value if isinstance(semantic_type, SemanticType) else str(semantic_type)
        return [col for col, st in self.column_types.items() if st.value == target_value]

    def to_dict(self) -> Dict[str, str]:
        """Convert schema to a plain dictionary mapping column name to type string."""
        return {col: st.value for col, st in self.column_types.items()}

    def summary(self) -> Dict[str, int]:
        """Return a count of columns for each detected semantic type."""
        counts: Dict[str, int] = {st.value: 0 for st in SemanticType}
        for st in self.column_types.values():
            counts[st.value] += 1
        return counts

    def __getitem__(self, col: str) -> SemanticType:
        return self.column_types[col]

    def __contains__(self, col: str) -> bool:
        return col in self.column_types

    def __repr__(self) -> str:
        items = ", ".join(f"{col}: {st.value}" for col, st in self.column_types.items())
        return f"DatasetSchema({items})"


# -----------------------------------------------------------------------------
# Heuristic Schema Detector
# -----------------------------------------------------------------------------

class SchemaDetector:
    """Heuristic-based detector for inferring semantic column types."""

    # Regex pattern for column names that strongly suggest identifiers
    ID_NAME_PATTERN = re.compile(
        r"(^|_)id$|^id(_|$)|uuid|guid|identifier|^pk$|_pk$",
        re.IGNORECASE,
    )

    # Regex pattern for UUID string values
    UUID_PATTERN = re.compile(
        r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
    )

    # Prefix/suffix indicators for boolean flag column names
    BOOLEAN_NAME_PREFIXES = ("is_", "has_", "can_", "should_", "was_", "flag_")
    BOOLEAN_NAME_SUFFIXES = ("_flag", "_bool")

    # Common date separators
    DATE_SEPARATORS = ("-", "/", ":", "T")

    def __init__(
        self,
        id_uniqueness_threshold: float = 0.70,
        datetime_sample_size: int = 50,
        datetime_match_threshold: float = 0.90,
    ) -> None:
        """Initialize detector with configurable heuristic thresholds.

        Args:
            id_uniqueness_threshold: Minimum uniqueness ratio (distinct / non-null)
                                     required when column name suggests an ID.
            datetime_sample_size: Max non-null samples to inspect for date parsing.
            datetime_match_threshold: Minimum parse success ratio for datetime detection.
        """
        self.id_uniqueness_threshold = id_uniqueness_threshold
        self.datetime_sample_size = datetime_sample_size
        self.datetime_match_threshold = datetime_match_threshold

    def is_boolean(self, series: pd.Series, col_name: str) -> bool:
        """Determine if a column is semantically boolean."""
        # 1. Direct pandas boolean dtype
        if pd.api.types.is_bool_dtype(series.dtype):
            return True

        non_null = series.dropna()
        if non_null.empty:
            return False

        unique_vals = list(non_null.unique())

        # 2. Python bool instances (type(v) is bool differentiates from int 0/1)
        if all(type(v) is bool for v in unique_vals):
            return True

        # 3. String boolean representations ("true", "false", "yes", "no")
        if all(isinstance(v, str) for v in unique_vals):
            str_norm = {v.strip().lower() for v in unique_vals if v.strip()}
            if str_norm and str_norm.issubset({"true", "false", "t", "f", "yes", "no", "y", "n"}):
                return True

        # 4. Conservative 0/1 handling:
        # Only treat 0/1 (or 0.0/1.0) as boolean if column name has a clear boolean naming cue
        col_lower = col_name.lower()
        has_bool_name = (
            col_lower.startswith(self.BOOLEAN_NAME_PREFIXES)
            or col_lower.endswith(self.BOOLEAN_NAME_SUFFIXES)
        )
        if has_bool_name:
            val_set = set(unique_vals)
            if val_set.issubset({0, 1, 0.0, 1.0, "0", "1"}):
                return True

        return False

    def is_identifier(self, series: pd.Series, col_name: str) -> bool:
        """Determine if a column is semantically an identifier."""
        non_null = series.dropna()
        if non_null.empty:
            return False

        total_non_null = len(non_null)
        num_unique = non_null.nunique()
        uniqueness_ratio = num_unique / total_non_null

        # Check for UUID formatted strings
        if all(isinstance(v, str) for v in non_null.head(20)):
            sample = non_null.head(self.datetime_sample_size)
            uuid_matches = sum(bool(self.UUID_PATTERN.match(str(v).strip())) for v in sample)
            if uuid_matches / len(sample) >= 0.8:
                return True

        # Check name cues
        col_lower = col_name.lower()
        has_id_name = bool(self.ID_NAME_PATTERN.search(col_lower))

        if has_id_name:
            # Identifier name signal + high uniqueness
            if uniqueness_ratio >= self.id_uniqueness_threshold:
                return True
            # Even in foreign key or repeated transaction contexts, if it's integer/string
            # and matches _id with reasonable cardinality, treat as identifier
            if num_unique > 1 and not pd.api.types.is_float_dtype(series.dtype):
                return True

        # Conservative: Continuous floats or generic columns without ID cues are NOT identifiers
        return False

    def is_datetime(self, series: pd.Series, col_name: str) -> bool:
        """Determine if a column is semantically datetime."""
        # 1. Native datetime dtype
        if pd.api.types.is_datetime64_any_dtype(series.dtype):
            return True

        # Don't aggressively parse purely numeric columns as datetimes
        if pd.api.types.is_numeric_dtype(series.dtype):
            return False

        non_null = series.dropna()
        if non_null.empty:
            return False

        # Must be string-like
        sample = non_null.astype(str).head(self.datetime_sample_size)
        
        # Check string characteristics: must contain date/time separator
        has_date_sep = all(
            any(sep in val for sep in self.DATE_SEPARATORS)
            for val in sample
        )
        if not has_date_sep:
            return False

        # Attempt parsing with pandas
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
                valid_count = parsed.notna().sum()
                success_ratio = valid_count / len(sample)
                if success_ratio >= self.datetime_match_threshold:
                    return True
            except Exception:
                return False

        return False

    def is_numerical(self, series: pd.Series, col_name: str) -> bool:
        """Determine if a column is numerical."""
        return pd.api.types.is_numeric_dtype(series.dtype) and not pd.api.types.is_bool_dtype(series.dtype)

    def classify_column(self, series: pd.Series, col_name: str) -> SemanticType:
        """Classify a single column into a SemanticType using modular heuristics."""
        # 1. Boolean check (handles explicit booleans and boolean-named binary flags)
        if self.is_boolean(series, col_name):
            return SemanticType.BOOLEAN

        # 2. Identifier check (handles ID naming, UUIDs, high-uniqueness entity keys)
        if self.is_identifier(series, col_name):
            return SemanticType.IDENTIFIER

        # 3. Datetime check (handles datetime64 and formatted date strings)
        if self.is_datetime(series, col_name):
            return SemanticType.DATETIME

        # 4. Numerical check (handles integers, floats)
        if self.is_numerical(series, col_name):
            return SemanticType.NUMERICAL

        # 5. Categorical fallback (strings, categoricals, discrete text)
        return SemanticType.CATEGORICAL

    def detect(self, df: pd.DataFrame) -> DatasetSchema:
        """Analyze a DataFrame and detect semantic types for all columns.

        Args:
            df: Input pandas DataFrame.

        Returns:
            DatasetSchema containing column-to-SemanticType mappings.
        """
        column_types: Dict[str, SemanticType] = {}
        for col in df.columns:
            column_types[str(col)] = self.classify_column(df[col], str(col))
        return DatasetSchema(column_types=column_types)


def detect_schema(df: pd.DataFrame) -> DatasetSchema:
    """Convenience function to detect semantic schema of a DataFrame.

    Args:
        df: Input pandas DataFrame.

    Returns:
        DatasetSchema with detected semantic types.
    """
    return SchemaDetector().detect(df)
