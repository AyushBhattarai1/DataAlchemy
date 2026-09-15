"""Unit tests for the DatasetProfiler in src/profiling/profiler.py."""

import json
from pathlib import Path

import pandas as pd
import pytest

from src.ingestion.loader import Dataset, load_dataset
from src.profiling.profiler import (
    ColumnProfile,
    DatasetProfile,
    DatasetProfiler,
    profile_dataset,
)
from src.profiling.statistics import (
    BooleanStats,
    CategoricalStats,
    DatetimeStats,
    IdentifierStats,
    NumericalStats,
)


@pytest.fixture
def sample_data_dir() -> Path:
    """Path to sample dataset directory."""
    return Path(__file__).resolve().parent.parent.parent / "data" / "samples"


class TestDatasetProfilerSamples:
    """Test profiler against real sample datasets."""

    def test_profile_customers_csv(self, sample_data_dir: Path) -> None:
        """Test full profile of customers.csv dataset."""
        ds = load_dataset(sample_data_dir / "customers.csv")
        profile = profile_dataset(ds)

        assert isinstance(profile, DatasetProfile)
        assert profile.dataset_name == "customers.csv"
        assert profile.row_count == 8
        assert profile.column_count == 7
        assert profile.total_cells == 56
        assert profile.memory_usage_bytes > 0

        # Missing cells: 1 in age, 1 in annual_income -> 2 cells total
        assert profile.missing_cells == 2
        assert profile.missing_percentage == pytest.approx((2 / 56) * 100, abs=0.01)

        # Duplicate rows: 1 duplicate (Bob Smith, CUST-1002)
        assert profile.duplicate_rows == 1
        assert profile.duplicate_percentage == pytest.approx((1 / 8) * 100, abs=0.01)

        # Semantic type counts
        assert profile.semantic_type_counts["identifier"] == 1
        assert profile.semantic_type_counts["numerical"] == 2
        assert profile.semantic_type_counts["categorical"] == 2
        assert profile.semantic_type_counts["datetime"] == 1
        assert profile.semantic_type_counts["boolean"] == 1

        # Check column-level stats types
        assert isinstance(profile.get_column("customer_id").stats, IdentifierStats)
        assert isinstance(profile.get_column("name").stats, CategoricalStats)
        assert isinstance(profile.get_column("age").stats, NumericalStats)
        assert isinstance(profile.get_column("annual_income").stats, NumericalStats)
        assert isinstance(profile.get_column("city").stats, CategoricalStats)
        assert isinstance(profile.get_column("signup_date").stats, DatetimeStats)
        assert isinstance(profile.get_column("is_active").stats, BooleanStats)

        # Explicit numerical assertions for age
        age_stats = profile.get_column("age").stats
        assert age_stats.count == 7
        assert age_stats.min == pytest.approx(24.0)
        assert age_stats.max == pytest.approx(61.0)

    def test_profile_sales_csv(self, sample_data_dir: Path) -> None:
        """Test full profile of sales.csv dataset."""
        ds = load_dataset(sample_data_dir / "sales.csv")
        profile = profile_dataset(ds)

        assert profile.row_count == 7
        assert profile.column_count == 8
        assert profile.total_cells == 56
        assert "sale_id" in profile.columns
        assert "is_discounted" in profile.columns

        disc_stats = profile.get_column("is_discounted").stats
        assert isinstance(disc_stats, BooleanStats)
        assert disc_stats.true_count + disc_stats.false_count == 7

    def test_profile_transactions_json(self, sample_data_dir: Path) -> None:
        """Test full profile of transactions.json dataset."""
        ds = load_dataset(sample_data_dir / "transactions.json")
        profile = profile_dataset(ds)

        assert profile.row_count == 5
        assert profile.column_count == 7
        assert profile.total_cells == 35
        assert profile.duplicate_rows == 0
        assert profile.missing_cells == 0

        tx_stats = profile.get_column("transaction_id").stats
        assert isinstance(tx_stats, IdentifierStats)
        assert tx_stats.is_unique is True


class TestDataPreservationDuringProfiling:
    """Verify that profiling NEVER mutates the input Dataset or DataFrame."""

    def test_dataframe_remains_unmodified(self, sample_data_dir: Path) -> None:
        """Original DataFrame must be identical before and after profiling."""
        ds = load_dataset(sample_data_dir / "customers.csv")
        # Snapshot copy of original dataframe
        df_snapshot = ds.dataframe.copy(deep=True)

        # Execute profiling
        profile = profile_dataset(ds)

        # Verify DataFrame is completely intact
        pd.testing.assert_frame_equal(ds.dataframe, df_snapshot)
        assert ds.dataframe["age"].isna().sum() == 1
        assert ds.dataframe.duplicated().sum() == 1


class TestProfilerEdgeCases:
    """Test profiler robustness on unusual and boundary datasets."""

    def test_single_row_dataset(self) -> None:
        """Single row dataset should profile without division by zero or errors."""
        df = pd.DataFrame({
            "id": ["ID_001"],
            "val": [100.0],
            "category": ["A"],
            "date": ["2026-01-01"],
            "active": [True],
        })
        ds = Dataset(
            dataframe=df,
            filename="single_row.csv",
            file_type="csv",
            row_count=1,
            column_count=5,
        )
        profile = profile_dataset(ds)

        assert profile.row_count == 1
        assert profile.total_cells == 5
        assert profile.duplicate_rows == 0
        assert profile.duplicate_percentage == 0.0
        assert profile.get_column("val").stats.mean == pytest.approx(100.0)

    def test_single_column_dataset(self) -> None:
        """Single column dataset should profile accurately."""
        df = pd.DataFrame({"count": [10, 20, 30, 40]})
        ds = Dataset(
            dataframe=df,
            filename="single_col.csv",
            file_type="csv",
            row_count=4,
            column_count=1,
        )
        profile = profile_dataset(ds)

        assert profile.column_count == 1
        assert "count" in profile.columns
        assert profile.get_column("count").stats.mean == pytest.approx(25.0)

    def test_all_null_column_dataset(self) -> None:
        """Dataset with an all-null column profiles cleanly without crashing."""
        df = pd.DataFrame({
            "id": ["1", "2", "3"],
            "empty_num": pd.Series([None, None, None], dtype="float64"),
            "empty_str": [None, None, None],
        })
        ds = Dataset(
            dataframe=df,
            filename="all_null.csv",
            file_type="csv",
            row_count=3,
            column_count=3,
        )
        profile = profile_dataset(ds)

        assert profile.missing_cells == 6
        assert profile.get_column("empty_num").missing_count == 3
        assert profile.get_column("empty_num").missing_percentage == 100.0
        assert profile.get_column("empty_num").stats.mean is None

    def test_constant_numerical_column(self) -> None:
        """Dataset with a constant numerical column profiles variance/std as 0."""
        df = pd.DataFrame({
            "id": ["1", "2", "3", "4"],
            "constant_val": [42.0, 42.0, 42.0, 42.0],
        })
        ds = Dataset(
            dataframe=df,
            filename="constant.csv",
            file_type="csv",
            row_count=4,
            column_count=2,
        )
        profile = profile_dataset(ds)

        col_stats = profile.get_column("constant_val").stats
        assert col_stats.std == pytest.approx(0.0)
        assert col_stats.variance == pytest.approx(0.0)

    def test_high_cardinality_categorical_threshold(self) -> None:
        """Profiler respects top_n_categories threshold for high cardinality."""
        df = pd.DataFrame({
            "category": [f"cat_{i}" for i in range(100)]
        })
        ds = Dataset(
            dataframe=df,
            filename="high_card.csv",
            file_type="csv",
            row_count=100,
            column_count=1,
        )
        profiler = DatasetProfiler(top_n_categories=5)
        profile = profiler.profile(ds)

        cat_stats = profile.get_column("category").stats
        assert cat_stats.unique_count == 100
        assert len(cat_stats.top_values) == 5


class TestSerializationAndHelpers:
    """Test serialization and querying helper methods."""

    def test_profile_to_dict_and_json(self, sample_data_dir: Path) -> None:
        """Ensure to_dict and to_json serialize without errors and round-trip."""
        ds = load_dataset(sample_data_dir / "customers.csv")
        profile = profile_dataset(ds)

        d = profile.to_dict()
        assert isinstance(d, dict)
        assert d["dataset_name"] == "customers.csv"
        assert d["row_count"] == 8
        assert "customer_id" in d["columns"]

        json_str = profile.to_json()
        deserialized = json.loads(json_str)
        assert deserialized["total_cells"] == 56

    def test_profile_summary_and_getters(self, sample_data_dir: Path) -> None:
        """Ensure summary string and helper query methods work as expected."""
        ds = load_dataset(sample_data_dir / "customers.csv")
        profile = profile_dataset(ds)

        summary_text = profile.summary()
        assert "DATASET PROFILE" in summary_text
        assert "customers.csv" in summary_text
        assert "Missing cells:" in summary_text

        num_cols = profile.get_columns_by_type("numerical")
        assert len(num_cols) == 2
        assert {c.name for c in num_cols} == {"age", "annual_income"}
