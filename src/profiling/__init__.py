"""DataAlchemy Automatic Profiling Package.

Provides dataset-level and column-level statistical profiling for standardized Datasets.
"""

from src.profiling.profiler import (
    ColumnProfile,
    DatasetProfile,
    DatasetProfiler,
    profile_dataset,
)
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

__all__ = [
    # Profiler & Profile containers
    "DatasetProfile",
    "ColumnProfile",
    "DatasetProfiler",
    "profile_dataset",
    # Statistical Data Containers
    "CommonStats",
    "NumericalStats",
    "CategoricalStats",
    "DatetimeStats",
    "BooleanStats",
    "IdentifierStats",
    # Calculation Functions
    "calculate_common_stats",
    "calculate_numerical_stats",
    "calculate_categorical_stats",
    "calculate_datetime_stats",
    "calculate_boolean_stats",
    "calculate_identifier_stats",
]
