"""Statistical calculation engine for DataAlchemy data profiling.

Provides reusable, modular functions and typed containers for computing dataset
and column-level statistics across all semantic types without mutating data.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd


def _clean_float(val: Any) -> Optional[float]:
    """Convert numpy or pandas numerical scalar to standard Python float or None."""
    if val is None or pd.isna(val):
        return None
    try:
        f = float(val)
        if np.isneginf(f) or np.isposinf(f) or np.isnan(f):
            return None
        return f
    except (ValueError, TypeError):
        return None


# -----------------------------------------------------------------------------
# Data Containers
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class CommonStats:
    """Core statistics calculated across all columns regardless of semantic type."""
    total_count: int
    missing_count: int
    missing_percentage: float
    unique_count: int
    unique_percentage: float
    pandas_dtype: str
    semantic_type: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NumericalStats:
    """Detailed statistical distribution metrics for numerical columns."""
    count: int
    mean: Optional[float]
    median: Optional[float]
    std: Optional[float]
    variance: Optional[float]
    min: Optional[float]
    max: Optional[float]
    q25: Optional[float]
    q50: Optional[float]
    q75: Optional[float]
    skewness: Optional[float]
    kurtosis: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CategoricalStats:
    """Frequency and cardinality metrics for categorical columns."""
    unique_count: int
    missing_count: int
    missing_percentage: float
    most_frequent: Optional[Any]
    most_frequent_frequency: Optional[int]
    top_values: Dict[str, int]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DatetimeStats:
    """Temporal range and frequency metrics for datetime columns."""
    min: Optional[str]
    max: Optional[str]
    unique_count: int
    missing_count: int
    missing_percentage: float
    time_span: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BooleanStats:
    """Binary distribution metrics for boolean columns."""
    true_count: int
    false_count: int
    missing_count: int
    true_percentage: float
    false_percentage: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class IdentifierStats:
    """Integrity and uniqueness metrics for identifier/key columns."""
    unique_count: int
    unique_percentage: float
    missing_count: int
    missing_percentage: float
    is_unique: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# -----------------------------------------------------------------------------
# Calculation Functions
# -----------------------------------------------------------------------------

def calculate_common_stats(series: pd.Series, semantic_type: str) -> CommonStats:
    """Calculate common metrics shared by all column types."""
    total_count = len(series)
    missing_count = int(series.isna().sum())
    missing_percentage = round((missing_count / total_count) * 100, 2) if total_count > 0 else 0.0

    non_null_series = series.dropna()
    unique_count = int(non_null_series.nunique())
    unique_percentage = round((unique_count / total_count) * 100, 2) if total_count > 0 else 0.0

    return CommonStats(
        total_count=total_count,
        missing_count=missing_count,
        missing_percentage=missing_percentage,
        unique_count=unique_count,
        unique_percentage=unique_percentage,
        pandas_dtype=str(series.dtype),
        semantic_type=semantic_type,
    )


def calculate_numerical_stats(series: pd.Series) -> NumericalStats:
    """Calculate descriptive statistics for numerical columns."""
    non_null = series.dropna()
    count = len(non_null)

    if count == 0:
        return NumericalStats(
            count=0,
            mean=None,
            median=None,
            std=None,
            variance=None,
            min=None,
            max=None,
            q25=None,
            q50=None,
            q75=None,
            skewness=None,
            kurtosis=None,
        )

    # Convert to numeric safely
    numeric_series = pd.to_numeric(non_null, errors="coerce").dropna()
    count = len(numeric_series)
    if count == 0:
        return NumericalStats(
            count=0,
            mean=None,
            median=None,
            std=None,
            variance=None,
            min=None,
            max=None,
            q25=None,
            q50=None,
            q75=None,
            skewness=None,
            kurtosis=None,
        )

    mean = _clean_float(numeric_series.mean())
    median = _clean_float(numeric_series.median())
    min_val = _clean_float(numeric_series.min())
    max_val = _clean_float(numeric_series.max())

    q25 = _clean_float(numeric_series.quantile(0.25))
    q50 = _clean_float(numeric_series.quantile(0.50))
    q75 = _clean_float(numeric_series.quantile(0.75))

    # Single-row or constant series edge cases
    if count < 2:
        std = None
        var = None
        skew = None
        kurt = None
    else:
        var = _clean_float(numeric_series.var())
        std = _clean_float(numeric_series.std())
        skew = _clean_float(numeric_series.skew()) if count >= 3 else None
        kurt = _clean_float(numeric_series.kurt()) if count >= 4 else None

    return NumericalStats(
        count=count,
        mean=mean,
        median=median,
        std=std,
        variance=var,
        min=min_val,
        max=max_val,
        q25=q25,
        q50=q50,
        q75=q75,
        skewness=skew,
        kurtosis=kurt,
    )


def calculate_categorical_stats(series: pd.Series, top_n: int = 10) -> CategoricalStats:
    """Calculate frequency metrics and top values for categorical columns."""
    total_count = len(series)
    missing_count = int(series.isna().sum())
    missing_percentage = round((missing_count / total_count) * 100, 2) if total_count > 0 else 0.0

    non_null = series.dropna()
    unique_count = int(non_null.nunique())

    if non_null.empty:
        return CategoricalStats(
            unique_count=0,
            missing_count=missing_count,
            missing_percentage=missing_percentage,
            most_frequent=None,
            most_frequent_frequency=None,
            top_values={},
        )

    # Use value_counts with limit to preserve memory on high cardinality
    val_counts = non_null.astype(str).value_counts()
    top_counts = val_counts.head(top_n)

    top_values: Dict[str, int] = {str(k): int(v) for k, v in top_counts.items()}
    most_frequent = val_counts.index[0]
    most_frequent_freq = int(val_counts.iloc[0])

    return CategoricalStats(
        unique_count=unique_count,
        missing_count=missing_count,
        missing_percentage=missing_percentage,
        most_frequent=most_frequent,
        most_frequent_frequency=most_frequent_freq,
        top_values=top_values,
    )


def calculate_datetime_stats(series: pd.Series) -> DatetimeStats:
    """Calculate temporal boundaries and duration metrics for datetime columns."""
    total_count = len(series)
    missing_count = int(series.isna().sum())
    missing_percentage = round((missing_count / total_count) * 100, 2) if total_count > 0 else 0.0

    non_null = series.dropna()
    if non_null.empty:
        return DatetimeStats(
            min=None,
            max=None,
            unique_count=0,
            missing_count=missing_count,
            missing_percentage=missing_percentage,
            time_span=None,
        )

    # If not native datetime64, parse temporarily without mutating the input
    if not pd.api.types.is_datetime64_any_dtype(non_null.dtype):
        parsed = pd.to_datetime(non_null, errors="coerce")
        valid_dates = parsed.dropna()
    else:
        valid_dates = non_null

    if valid_dates.empty:
        return DatetimeStats(
            min=None,
            max=None,
            unique_count=0,
            missing_count=missing_count,
            missing_percentage=missing_percentage,
            time_span=None,
        )

    min_dt = valid_dates.min()
    max_dt = valid_dates.max()
    unique_count = int(valid_dates.nunique())

    min_str = min_dt.isoformat() if hasattr(min_dt, "isoformat") else str(min_dt)
    max_str = max_dt.isoformat() if hasattr(max_dt, "isoformat") else str(max_dt)

    try:
        delta = max_dt - min_dt
        if hasattr(delta, "days"):
            if delta.seconds == 0 and not getattr(delta, "microseconds", 0):
                time_span = f"{delta.days} days"
            else:
                time_span = str(delta)
        else:
            time_span = str(delta)
    except Exception:
        time_span = None

    return DatetimeStats(
        min=min_str,
        max=max_str,
        unique_count=unique_count,
        missing_count=missing_count,
        missing_percentage=missing_percentage,
        time_span=time_span,
    )


def calculate_boolean_stats(series: pd.Series) -> BooleanStats:
    """Calculate distribution metrics for boolean columns."""
    total_count = len(series)
    missing_count = int(series.isna().sum())
    non_null = series.dropna()
    non_null_count = len(non_null)

    if non_null_count == 0:
        return BooleanStats(
            true_count=0,
            false_count=0,
            missing_count=missing_count,
            true_percentage=0.0,
            false_percentage=0.0,
        )

    true_signals = {True, 1, 1.0, "true", "t", "yes", "y", "1"}
    false_signals = {False, 0, 0.0, "false", "f", "no", "n", "0"}

    true_count = 0
    false_count = 0

    for val in non_null:
        if isinstance(val, (bool, np.bool_)):
            if val:
                true_count += 1
            else:
                false_count += 1
        elif isinstance(val, str):
            norm = val.strip().lower()
            if norm in true_signals:
                true_count += 1
            elif norm in false_signals:
                false_count += 1
            else:
                false_count += 1
        elif isinstance(val, (int, float)):
            if val == 1:
                true_count += 1
            elif val == 0:
                false_count += 1
            else:
                if bool(val):
                    true_count += 1
                else:
                    false_count += 1
        else:
            if bool(val):
                true_count += 1
            else:
                false_count += 1

    true_pct = round((true_count / non_null_count) * 100, 2)
    false_pct = round((false_count / non_null_count) * 100, 2)

    return BooleanStats(
        true_count=true_count,
        false_count=false_count,
        missing_count=missing_count,
        true_percentage=true_pct,
        false_percentage=false_pct,
    )


def calculate_identifier_stats(series: pd.Series) -> IdentifierStats:
    """Calculate uniqueness and integrity metrics for identifier columns."""
    total_count = len(series)
    missing_count = int(series.isna().sum())
    missing_percentage = round((missing_count / total_count) * 100, 2) if total_count > 0 else 0.0

    non_null = series.dropna()
    unique_count = int(non_null.nunique())
    unique_percentage = round((unique_count / total_count) * 100, 2) if total_count > 0 else 0.0
    is_unique = (unique_count == total_count) and (missing_count == 0)

    return IdentifierStats(
        unique_count=unique_count,
        unique_percentage=unique_percentage,
        missing_count=missing_count,
        missing_percentage=missing_percentage,
        is_unique=is_unique,
    )
