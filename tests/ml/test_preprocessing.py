"""Unit tests for MLPreprocessor in src/ml/preprocessing.py."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.ingestion.loader import load_dataset
from src.ml.preprocessing import MLPreprocessor
from src.profiling.profiler import profile_dataset


@pytest.fixture
def sample_data_dir() -> Path:
    return Path(__file__).resolve().parent.parent.parent / "data" / "samples"


class TestMLPreprocessor:
    """Test suite for tabular feature preprocessing."""

    def test_numerical_imputation_strategies(self) -> None:
        df = pd.DataFrame({
            "val": [10.0, 20.0, np.nan, 40.0, 100.0],
        })

        # Median strategy: median of [10, 20, 40, 100] is 30
        prep_med = MLPreprocessor(numeric_imputation="median", scaling=None)
        X_med = prep_med.fit_transform(df)
        assert np.isclose(X_med[2, 0], 30.0)

        # Mean strategy: mean of [10, 20, 40, 100] is 42.5
        prep_mean = MLPreprocessor(numeric_imputation="mean", scaling=None)
        X_mean = prep_mean.fit_transform(df)
        assert np.isclose(X_mean[2, 0], 42.5)

        # Zero strategy:
        prep_zero = MLPreprocessor(numeric_imputation="zero", scaling=None)
        X_zero = prep_zero.fit_transform(df)
        assert np.isclose(X_zero[2, 0], 0.0)

    def test_categorical_one_hot_and_unknowns(self) -> None:
        train_df = pd.DataFrame({
            "category": ["red", "blue", "red", "green"],
            "amount": [1.0, 2.0, 3.0, 4.0],
        })
        test_df = pd.DataFrame({
            "category": ["blue", "yellow"],  # 'yellow' is unseen
            "amount": [5.0, 6.0],
        })

        prep = MLPreprocessor(scaling=None)
        X_train = prep.fit_transform(train_df)
        assert X_train.shape[1] == 1 + 3  # 1 numeric + 3 categories (blue, green, red)

        # Unseen category 'yellow' must not crash and should be ignored (all 0s)
        X_test = prep.transform(test_df)
        assert X_test.shape == (2, 4)

    def test_datetime_feature_extraction(self) -> None:
        df = pd.DataFrame({
            "date": ["2023-01-15", "2023-06-20", "2024-12-31"],
            "value": [10, 20, 30],
        })
        prep = MLPreprocessor(scaling=None)
        X = prep.fit_transform(df)

        # 1 numeric + 4 datetime components (year, month, day, dayofweek)
        assert X.shape == (3, 5)
        names = prep.feature_names_
        assert "date_year" in names
        assert "date_month" in names
        assert "date_day" in names
        assert "date_dayofweek" in names

    def test_boolean_feature_conversion(self) -> None:
        df = pd.DataFrame({
            "is_active": [True, False, True],
            "flag_str": ["yes", "no", "yes"],
            "score": [1.0, 2.0, 3.0],
        })
        prep = MLPreprocessor(scaling=None)
        X = prep.fit_transform(df)

        assert X.shape == (3, 3)
        # Verify boolean columns converted to 0.0 / 1.0
        assert np.all(np.isin(X[:, 1], [0.0, 1.0]))

    def test_identifier_and_uniqueness_exclusion(self) -> None:
        df = pd.DataFrame({
            "customer_id": [f"CUST_{i}" for i in range(20)],
            "email": [f"user_{i}@domain.com" for i in range(20)],
            "metric": list(range(20)),
        })
        prep = MLPreprocessor(scaling=None)
        prep.fit(df)

        assert "customer_id" in prep.excluded_columns_
        assert "email" in prep.excluded_columns_
        assert "metric" in prep.numeric_columns_
        assert "identifier" in prep.exclusion_reasons_["customer_id"]

    def test_constant_and_all_null_feature_exclusion(self) -> None:
        df = pd.DataFrame({
            "const_num": [5.0, 5.0, 5.0, 5.0],
            "all_null": [np.nan, np.nan, np.nan, np.nan],
            "valid_col": [1.0, 2.0, 3.0, 4.0],
        })
        prep = MLPreprocessor(scaling=None)
        prep.fit(df)

        assert "const_num" in prep.excluded_columns_
        assert prep.exclusion_reasons_["const_num"] == "constant_feature"

        assert "all_null" in prep.excluded_columns_
        assert prep.exclusion_reasons_["all_null"] == "all_null"

        assert "valid_col" in prep.numeric_columns_

    def test_high_cardinality_categorical_protection(self) -> None:
        # Create column with 55 unique categories across 80 rows (not high uniqueness, but > 50 categories)
        cats = [f"cat_{i % 55}" for i in range(80)]
        df = pd.DataFrame({
            "high_card": cats,
            "value": [float(i) for i in range(80)],
        })
        prep = MLPreprocessor(max_categories=50)
        prep.fit(df)

        assert "high_card" in prep.excluded_columns_
        assert "high_cardinality" in prep.exclusion_reasons_["high_card"]

    def test_scaling_and_leak_free_transform(self) -> None:
        train_df = pd.DataFrame({"x": [0.0, 10.0, 20.0]})
        test_df = pd.DataFrame({"x": [30.0]})

        # StandardScaler on [0, 10, 20]: mean = 10, std = 8.1649658
        prep = MLPreprocessor(scaling="standard")
        X_train = prep.fit_transform(train_df)
        assert np.isclose(np.mean(X_train), 0.0)

        # Transform test_df should use fitted mean=10, not recomputed mean
        X_test = prep.transform(test_df)
        expected_scaled = (30.0 - 10.0) / np.std([0.0, 10.0, 20.0])
        assert np.isclose(X_test[0, 0], expected_scaled)

    def test_original_dataframe_immutability(self, sample_data_dir: Path) -> None:
        ds = load_dataset(sample_data_dir / "customers.csv")
        original_copy = ds.dataframe.copy(deep=True)

        prep = MLPreprocessor()
        _ = prep.fit_transform(ds)

        # Confirm original dataframe remains strictly identical
        pd.testing.assert_frame_equal(ds.dataframe, original_copy)

    def test_preprocessing_on_all_sample_datasets(self, sample_data_dir: Path) -> None:
        for fname in ["customers.csv", "sales.csv", "transactions.json"]:
            ds = load_dataset(sample_data_dir / fname)
            profile = profile_dataset(ds)

            prep = MLPreprocessor()
            X = prep.fit_transform(ds, profile=profile)
            meta = prep.get_preprocessing_result(ds)

            assert isinstance(X, np.ndarray)
            assert X.shape[0] == ds.row_count
            assert X.shape[1] == len(meta.feature_names)
            assert not np.isnan(X).any(), f"NaNs found in preprocessed matrix for {fname}"
