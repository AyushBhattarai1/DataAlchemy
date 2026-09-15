"""Automatic Dataset Profiling Engine for DataAlchemy.

Orchestrates dataset-level and column-level profiling from standardized Dataset
objects using existing schema detection and statistical calculations.
"""

from dataclasses import asdict, dataclass
import json
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from src.ingestion.loader import Dataset
from src.ingestion.schema import SemanticType, detect_schema
from src.profiling.statistics import (
    BooleanStats,
    CategoricalStats,
    CommonStats,
    DatetimeStats,
    IdentifierStats,
    NumericalStats,
    calculate_boolean_stats,
    calculate_categorical_stats,
    calculate_common_stats,
    calculate_datetime_stats,
    calculate_identifier_stats,
    calculate_numerical_stats,
)


# -----------------------------------------------------------------------------
# Column Profile Representation
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class ColumnProfile:
    """Comprehensive profile of an individual dataset column."""
    name: str
    semantic_type: str
    pandas_dtype: str
    row_count: int
    missing_count: int
    missing_percentage: float
    unique_count: int
    unique_percentage: float
    stats: Union[
        NumericalStats,
        CategoricalStats,
        DatetimeStats,
        BooleanStats,
        IdentifierStats,
        Dict[str, Any],
    ]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize column profile to a JSON-compatible dictionary."""
        stats_dict = self.stats.to_dict() if hasattr(self.stats, "to_dict") else self.stats
        return {
            "name": self.name,
            "semantic_type": self.semantic_type,
            "pandas_dtype": self.pandas_dtype,
            "row_count": self.row_count,
            "missing_count": self.missing_count,
            "missing_percentage": self.missing_percentage,
            "unique_count": self.unique_count,
            "unique_percentage": self.unique_percentage,
            "stats": stats_dict,
        }


# -----------------------------------------------------------------------------
# Dataset Profile Representation
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class DatasetProfile:
    """Standardized profile representing the entire dataset and all its columns."""
    dataset_name: str
    row_count: int
    column_count: int
    total_cells: int
    memory_usage_bytes: int
    missing_cells: int
    missing_percentage: float
    duplicate_rows: int
    duplicate_percentage: float
    semantic_type_counts: Dict[str, int]
    columns: Dict[str, ColumnProfile]

    def get_column(self, col_name: str) -> Optional[ColumnProfile]:
        """Retrieve a specific column profile by name."""
        return self.columns.get(col_name)

    def get_columns_by_type(self, semantic_type: Union[SemanticType, str]) -> List[ColumnProfile]:
        """Retrieve all column profiles matching a given semantic type."""
        target_val = semantic_type.value if isinstance(semantic_type, SemanticType) else str(semantic_type)
        return [cp for cp in self.columns.values() if cp.semantic_type == target_val]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize entire dataset profile into a JSON-compatible dictionary."""
        return {
            "dataset_name": self.dataset_name,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "total_cells": self.total_cells,
            "memory_usage_bytes": self.memory_usage_bytes,
            "missing_cells": self.missing_cells,
            "missing_percentage": self.missing_percentage,
            "duplicate_rows": self.duplicate_rows,
            "duplicate_percentage": self.duplicate_percentage,
            "semantic_type_counts": self.semantic_type_counts,
            "columns": {col_name: cp.to_dict() for col_name, cp in self.columns.items()},
        }

    def to_json(self, indent: int = 2) -> str:
        """Export dataset profile directly to a formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def summary(self) -> str:
        """Generate a human-readable text summary of the dataset profile."""
        mem_mb = round(self.memory_usage_bytes / (1024 * 1024), 2)
        lines = [
            "DATASET PROFILE",
            "=" * 30,
            f"Dataset:            {self.dataset_name}",
            f"Rows:               {self.row_count:,}",
            f"Columns:            {self.column_count:,}",
            f"Total cells:        {self.total_cells:,}",
            f"Memory:             {mem_mb} MB ({self.memory_usage_bytes:,} bytes)",
            "",
            "Semantic Type Breakdown:",
        ]
        for st_name, count in sorted(self.semantic_type_counts.items()):
            lines.append(f"  - {st_name.capitalize():<16}: {count}")

        lines.extend([
            "",
            f"Missing cells:      {self.missing_cells:,} ({self.missing_percentage}%)",
            f"Duplicate rows:     {self.duplicate_rows:,} ({self.duplicate_percentage}%)",
            "=" * 30,
        ])
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"DatasetProfile(dataset='{self.dataset_name}', "
            f"rows={self.row_count}, "
            f"cols={self.column_count}, "
            f"missing={self.missing_percentage}%, "
            f"duplicates={self.duplicate_percentage}%)"
        )


# -----------------------------------------------------------------------------
# Profiler Orchestrator
# -----------------------------------------------------------------------------

class DatasetProfiler:
    """Orchestrates schema detection and statistical computation to profile Datasets."""

    def __init__(self, top_n_categories: int = 10) -> None:
        """Initialize profiler with configurable categorical threshold.

        Args:
            top_n_categories: Maximum number of most frequent items to record
                              per categorical column.
        """
        self.top_n_categories = top_n_categories

    def profile(self, dataset: Dataset) -> DatasetProfile:
        """Generate a full DatasetProfile for a given Dataset without mutating raw data.

        Args:
            dataset: The standardized Dataset object from Step 2 ingestion.

        Returns:
            DatasetProfile containing dataset-level and column-level insights.
        """
        df = dataset.dataframe
        row_count, col_count = df.shape
        total_cells = row_count * col_count

        # Dataset-level missing values and duplicates
        missing_cells = int(df.isna().sum().sum())
        missing_pct = round((missing_cells / total_cells) * 100, 2) if total_cells > 0 else 0.0

        dup_rows = int(df.duplicated().sum()) if row_count > 0 else 0
        dup_pct = round((dup_rows / row_count) * 100, 2) if row_count > 0 else 0.0

        # Memory usage
        try:
            mem_bytes = int(df.memory_usage(deep=True).sum())
        except Exception:
            mem_bytes = dataset.metadata.get("memory_usage_bytes", 0)

        # Detect semantic schema using Step 2 schema module
        schema = detect_schema(df)
        semantic_counts = schema.summary()

        # Build column profiles
        columns: Dict[str, ColumnProfile] = {}

        for col in df.columns:
            col_str = str(col)
            series = df[col]
            sem_type = schema[col_str]
            sem_type_str = sem_type.value if isinstance(sem_type, SemanticType) else str(sem_type)

            common_stats = calculate_common_stats(series, semantic_type=sem_type_str)

            # Route to type-specific statistical calculation
            if sem_type == SemanticType.NUMERICAL:
                type_stats = calculate_numerical_stats(series)
            elif sem_type == SemanticType.CATEGORICAL:
                type_stats = calculate_categorical_stats(series, top_n=self.top_n_categories)
            elif sem_type == SemanticType.DATETIME:
                type_stats = calculate_datetime_stats(series)
            elif sem_type == SemanticType.BOOLEAN:
                type_stats = calculate_boolean_stats(series)
            elif sem_type == SemanticType.IDENTIFIER:
                type_stats = calculate_identifier_stats(series)
            else:
                type_stats = calculate_categorical_stats(series, top_n=self.top_n_categories)

            columns[col_str] = ColumnProfile(
                name=col_str,
                semantic_type=sem_type_str,
                pandas_dtype=common_stats.pandas_dtype,
                row_count=common_stats.total_count,
                missing_count=common_stats.missing_count,
                missing_percentage=common_stats.missing_percentage,
                unique_count=common_stats.unique_count,
                unique_percentage=common_stats.unique_percentage,
                stats=type_stats,
            )

        return DatasetProfile(
            dataset_name=dataset.filename,
            row_count=row_count,
            column_count=col_count,
            total_cells=total_cells,
            memory_usage_bytes=mem_bytes,
            missing_cells=missing_cells,
            missing_percentage=missing_pct,
            duplicate_rows=dup_rows,
            duplicate_percentage=dup_pct,
            semantic_type_counts=semantic_counts,
            columns=columns,
        )


def profile_dataset(dataset: Dataset, top_n_categories: int = 10) -> DatasetProfile:
    """Convenience function to profile a Dataset.

    Args:
        dataset: The standardized Dataset object from Step 2 ingestion.
        top_n_categories: Maximum number of most frequent items for categorical columns.

    Returns:
        DatasetProfile: Complete statistical and structural profile.
    """
    return DatasetProfiler(top_n_categories=top_n_categories).profile(dataset)
