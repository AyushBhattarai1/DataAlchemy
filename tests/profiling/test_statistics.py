"""Unit tests for statistical calculation functions in src/profiling/statistics.py."""

import numpy as np
import pandas as pd
import pytest

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


class TestCommonStats:
    """Test calculate_common_stats across typical and edge cases."""

    def test_common_stats_standard(self) -> None:
        series = pd.Series(["apple", "banana", "apple", None, "cherry"])
        stats = calculate_common_stats(series, semantic_type="categorical")

        assert isinstance(stats, CommonStats)
        assert stats.total_count == 5
        assert stats.missing_count == 1
        assert stats.missing_percentage == 20.0
        assert stats.unique_count == 3
        assert stats.unique_percentage == 60.0
        assert stats.semantic_type == "categorical"

    def test_common_stats_all_null(self) -> None:
        series = pd.Series([None, None, np.nan])
        stats = calculate_common_stats(series, semantic_type="numerical")

        assert stats.total_count == 3
        assert stats.missing_count == 3
        assert stats.missing_percentage == 100.0
        assert stats.unique_count == 0
        assert stats.unique_percentage == 0.0

    def test_common_stats_empty(self) -> None:
        series = pd.Series([], dtype="object")
        stats = calculate_common_stats(series, semantic_type="categorical")

        assert stats.total_count == 0
        assert stats.missing_count == 0
        assert stats.missing_percentage == 0.0
        assert stats.unique_count == 0
        assert stats.unique_percentage == 0.0


class TestNumericalStats:
    """Test calculate_numerical_stats with deterministic values and edge cases."""

    def test_numerical_stats_deterministic(self) -> None:
        # Values: [10, 20, 30, 40, 50]
        series = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
        stats = calculate_numerical_stats(series)

        assert isinstance(stats, NumericalStats)
        assert stats.count == 5
        assert stats.mean == pytest.approx(30.0)
        assert stats.median == pytest.approx(30.0)
        assert stats.min == pytest.approx(10.0)
        assert stats.max == pytest.approx(50.0)
        assert stats.q25 == pytest.approx(20.0)
        assert stats.q50 == pytest.approx(30.0)
        assert stats.q75 == pytest.approx(40.0)
        assert stats.std == pytest.approx(15.811388, rel=1e-4)
        assert stats.variance == pytest.approx(250.0)
        assert stats.skewness == pytest.approx(0.0)
        assert stats.kurtosis is not None

    def test_numerical_stats_with_missing(self) -> None:
        series = pd.Series([10.0, np.nan, 20.0, None, 30.0])
        stats = calculate_numerical_stats(series)

        assert stats.count == 3
        assert stats.mean == pytest.approx(20.0)
        assert stats.median == pytest.approx(20.0)
        assert stats.min == pytest.approx(10.0)
        assert stats.max == pytest.approx(30.0)

    def test_numerical_stats_all_null(self) -> None:
        series = pd.Series([np.nan, None, float("nan")])
        stats = calculate_numerical_stats(series)

        assert stats.count == 0
        assert stats.mean is None
        assert stats.median is None
        assert stats.std is None
        assert stats.variance is None
        assert stats.min is None
        assert stats.max is None
        assert stats.q25 is None
        assert stats.skewness is None
        assert stats.kurtosis is None

    def test_numerical_stats_constant_series(self) -> None:
        series = pd.Series([7.0, 7.0, 7.0, 7.0])
        stats = calculate_numerical_stats(series)

        assert stats.count == 4
        assert stats.mean == pytest.approx(7.0)
        assert stats.median == pytest.approx(7.0)
        assert stats.min == pytest.approx(7.0)
        assert stats.max == pytest.approx(7.0)
        assert stats.std == pytest.approx(0.0)
        assert stats.variance == pytest.approx(0.0)

    def test_numerical_stats_single_value(self) -> None:
        series = pd.Series([42.0])
        stats = calculate_numerical_stats(series)

        assert stats.count == 1
        assert stats.mean == pytest.approx(42.0)
        assert stats.median == pytest.approx(42.0)
        assert stats.min == pytest.approx(42.0)
        assert stats.max == pytest.approx(42.0)
        assert stats.std is None
        assert stats.variance is None
        assert stats.skewness is None
        assert stats.kurtosis is None


class TestCategoricalStats:
    """Test calculate_categorical_stats including cardinality limits."""

    def test_categorical_stats_standard(self) -> None:
        series = pd.Series(["NY", "CA", "TX", "NY", "CA", "NY", None])
        stats = calculate_categorical_stats(series, top_n=2)

        assert isinstance(stats, CategoricalStats)
        assert stats.unique_count == 3
        assert stats.missing_count == 1
        assert stats.missing_percentage == pytest.approx(14.29, abs=0.01)
        assert stats.most_frequent == "NY"
        assert stats.most_frequent_frequency == 3
        assert len(stats.top_values) == 2
        assert stats.top_values["NY"] == 3
        assert stats.top_values["CA"] == 2

    def test_categorical_stats_high_cardinality(self) -> None:
        # 100 distinct categories with different frequencies
        items = [f"item_{i % 25}" for i in range(100)]
        series = pd.Series(items)
        stats = calculate_categorical_stats(series, top_n=5)

        assert stats.unique_count == 25
        assert len(stats.top_values) == 5

    def test_categorical_stats_all_null(self) -> None:
        series = pd.Series([None, np.nan, None])
        stats = calculate_categorical_stats(series)

        assert stats.unique_count == 0
        assert stats.missing_count == 3
        assert stats.missing_percentage == 100.0
        assert stats.most_frequent is None
        assert stats.most_frequent_frequency is None
        assert stats.top_values == {}


class TestDatetimeStats:
    """Test calculate_datetime_stats for both native datetime and string formats."""

    def test_datetime_stats_native(self) -> None:
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        series = pd.Series(dates)
        stats = calculate_datetime_stats(series)

        assert isinstance(stats, DatetimeStats)
        assert stats.unique_count == 10
        assert stats.missing_count == 0
        assert "2024-01-01" in stats.min
        assert "2024-01-10" in stats.max
        assert "9 days" in stats.time_span

    def test_datetime_stats_strings(self) -> None:
        series = pd.Series(["2024-03-01 10:00:00", "2024-03-05 12:00:00", None])
        stats = calculate_datetime_stats(series)

        assert stats.unique_count == 2
        assert stats.missing_count == 1
        assert stats.missing_percentage == pytest.approx(33.33, abs=0.01)
        assert "2024-03-01" in stats.min
        assert "2024-03-05" in stats.max
        assert stats.time_span is not None

    def test_datetime_stats_all_null(self) -> None:
        series = pd.Series([None, np.nan], dtype="object")
        stats = calculate_datetime_stats(series)

        assert stats.min is None
        assert stats.max is None
        assert stats.unique_count == 0
        assert stats.missing_count == 2
        assert stats.time_span is None


class TestBooleanStats:
    """Test calculate_boolean_stats across native, string, and flag representations."""

    def test_boolean_stats_native(self) -> None:
        series = pd.Series([True, False, True, True, None])
        stats = calculate_boolean_stats(series)

        assert isinstance(stats, BooleanStats)
        assert stats.true_count == 3
        assert stats.false_count == 1
        assert stats.missing_count == 1
        assert stats.true_percentage == pytest.approx(75.0)
        assert stats.false_percentage == pytest.approx(25.0)

    def test_boolean_stats_strings(self) -> None:
        series = pd.Series(["yes", "no", "yes", "no", "no"])
        stats = calculate_boolean_stats(series)

        assert stats.true_count == 2
        assert stats.false_count == 3
        assert stats.true_percentage == pytest.approx(40.0)
        assert stats.false_percentage == pytest.approx(60.0)

    def test_boolean_stats_binary_flags(self) -> None:
        series = pd.Series([1, 0, 1, 1])
        stats = calculate_boolean_stats(series)

        assert stats.true_count == 3
        assert stats.false_count == 1
        assert stats.missing_count == 0
        assert stats.true_percentage == pytest.approx(75.0)

    def test_boolean_stats_all_null(self) -> None:
        series = pd.Series([None, None])
        stats = calculate_boolean_stats(series)

        assert stats.true_count == 0
        assert stats.false_count == 0
        assert stats.missing_count == 2
        assert stats.true_percentage == 0.0
        assert stats.false_percentage == 0.0


class TestIdentifierStats:
    """Test calculate_identifier_stats uniqueness and integrity checks."""

    def test_identifier_stats_unique(self) -> None:
        series = pd.Series(["ID_1", "ID_2", "ID_3"])
        stats = calculate_identifier_stats(series)

        assert isinstance(stats, IdentifierStats)
        assert stats.unique_count == 3
        assert stats.missing_count == 0
        assert stats.unique_percentage == pytest.approx(100.0)
        assert stats.is_unique is True

    def test_identifier_stats_with_duplicates(self) -> None:
        series = pd.Series(["ID_1", "ID_2", "ID_1"])
        stats = calculate_identifier_stats(series)

        assert stats.unique_count == 2
        assert stats.missing_count == 0
        assert stats.unique_percentage == pytest.approx(66.67, abs=0.01)
        assert stats.is_unique is False
