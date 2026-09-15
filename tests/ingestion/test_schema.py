"""Unit tests for the Semantic Schema Detection module."""

from pathlib import Path
import pandas as pd
import pytest

from src.ingestion.loader import load_dataset
from src.ingestion.schema import (
    DatasetSchema,
    SchemaDetector,
    SemanticType,
    detect_schema,
)


@pytest.fixture
def sample_data_dir() -> Path:
    """Path to sample dataset directory."""
    return Path(__file__).resolve().parent.parent.parent / "data" / "samples"


class TestSemanticTypeDetection:
    """Test individual semantic type detection heuristics."""

    def test_numerical_detection(self) -> None:
        """Verify integers, floats, negative numbers are detected as numerical."""
        df = pd.DataFrame({
            "age": [25, 30, 45, 50],
            "price": [19.99, 45.50, 100.0, 3.25],
            "temperature_delta": [-2.5, 0.0, 3.1, -1.2],
            "count": [0, 5, 12, 100],
        })
        schema = detect_schema(df)

        for col in df.columns:
            assert schema[col] == SemanticType.NUMERICAL
            assert schema[col].value == "numerical"

    def test_categorical_detection(self) -> None:
        """Verify text and discrete categories are classified as categorical."""
        df = pd.DataFrame({
            "country": ["USA", "Canada", "Germany", "France"],
            "tier": ["Gold", "Silver", "Platinum", "Gold"],
            "status": ["pending", "approved", "rejected", "pending"],
        })
        schema = detect_schema(df)

        for col in df.columns:
            assert schema[col] == SemanticType.CATEGORICAL
            assert schema[col].value == "categorical"

    def test_datetime_detection_native(self) -> None:
        """Verify native datetime64 columns are detected as datetime."""
        df = pd.DataFrame({
            "created_at": pd.date_range("2024-01-01", periods=4, freq="D"),
            "event_time": pd.to_datetime(["2024-02-01 10:00:00", "2024-02-02 11:30:00", "2024-02-03 14:15:00", "2024-02-04 18:00:00"]),
        })
        schema = detect_schema(df)

        assert schema["created_at"] == SemanticType.DATETIME
        assert schema["event_time"] == SemanticType.DATETIME

    def test_datetime_detection_strings(self) -> None:
        """Verify string-formatted dates are detected as datetime."""
        df = pd.DataFrame({
            "iso_date": ["2026-01-15", "2026-02-20", "2026-03-10", "2026-04-05"],
            "slash_date": ["15/01/2026", "20/02/2026", "10/03/2026", "05/04/2026"],
            "timestamp_str": ["2024-03-01 14:30:00", "2024-03-02 09:15:22", "2024-03-03 11:20:05", "2024-03-04 16:05:40"],
        })
        schema = detect_schema(df)

        assert schema["iso_date"] == SemanticType.DATETIME
        assert schema["slash_date"] == SemanticType.DATETIME
        assert schema["timestamp_str"] == SemanticType.DATETIME

    def test_boolean_detection_native(self) -> None:
        """Verify native boolean columns are detected as boolean."""
        df = pd.DataFrame({
            "is_active": [True, False, True, True],
            "verified": pd.Series([False, True, False, True], dtype="boolean"),
        })
        schema = detect_schema(df)

        assert schema["is_active"] == SemanticType.BOOLEAN
        assert schema["verified"] == SemanticType.BOOLEAN

    def test_boolean_detection_strings(self) -> None:
        """Verify string boolean representations are detected as boolean."""
        df = pd.DataFrame({
            "opted_in": ["True", "False", "True", "False"],
            "subscribed": ["yes", "no", "yes", "yes"],
        })
        schema = detect_schema(df)

        assert schema["opted_in"] == SemanticType.BOOLEAN
        assert schema["subscribed"] == SemanticType.BOOLEAN

    def test_boolean_detection_flag_name_with_0_1(self) -> None:
        """Verify 0/1 columns WITH boolean name indicators (is_*, has_*, flag_*) become boolean."""
        df = pd.DataFrame({
            "is_completed": [1, 0, 1, 1],
            "has_discount": [0, 1, 0, 0],
            "flag_fraud": [0, 0, 1, 0],
        })
        schema = detect_schema(df)

        assert schema["is_completed"] == SemanticType.BOOLEAN
        assert schema["has_discount"] == SemanticType.BOOLEAN
        assert schema["flag_fraud"] == SemanticType.BOOLEAN

    def test_identifier_detection_by_name(self) -> None:
        """Verify columns with ID naming patterns and high uniqueness are detected as identifier."""
        df = pd.DataFrame({
            "customer_id": ["C101", "C102", "C103", "C104"],
            "transaction_id": [10001, 10002, 10003, 10004],
            "user_uuid": ["u-1", "u-2", "u-3", "u-4"],
            "identifier": ["ID_A", "ID_B", "ID_C", "ID_D"],
        })
        schema = detect_schema(df)

        for col in df.columns:
            assert schema[col] == SemanticType.IDENTIFIER

    def test_identifier_detection_by_uuid_pattern(self) -> None:
        """Verify columns containing UUID format strings are detected as identifier even without ID name."""
        df = pd.DataFrame({
            "token": [
                "8fa21e56-54a8-4e50-93cb-986c7e3f0001",
                "8fa21e56-54a8-4e50-93cb-986c7e3f0002",
                "8fa21e56-54a8-4e50-93cb-986c7e3f0003",
                "8fa21e56-54a8-4e50-93cb-986c7e3f0004",
            ]
        })
        schema = detect_schema(df)
        assert schema["token"] == SemanticType.IDENTIFIER


class TestAmbiguousAndEdgeCases:
    """Test critical edge cases where heuristic classification must be conservative."""

    def test_binary_column_without_boolean_name_remains_numerical(self) -> None:
        """Binary 0/1 without boolean naming cue (e.g., target, cluster, code) must be numerical."""
        df = pd.DataFrame({
            "target": [0, 1, 1, 0, 1],
            "label_code": [0, 1, 0, 0, 1],
        })
        schema = detect_schema(df)

        assert schema["target"] == SemanticType.NUMERICAL
        assert schema["label_code"] == SemanticType.NUMERICAL

    def test_unique_continuous_floats_remain_numerical(self) -> None:
        """Continuous unique float values must not be misclassified as identifiers."""
        df = pd.DataFrame({
            "sensor_measurement": [12.456, 34.891, 78.102, 99.432],
            "latitude": [40.7128, 34.0522, 51.5074, 48.8566],
        })
        schema = detect_schema(df)

        assert schema["sensor_measurement"] == SemanticType.NUMERICAL
        assert schema["latitude"] == SemanticType.NUMERICAL

    def test_unique_text_names_remain_categorical(self) -> None:
        """Unique names/descriptions without ID naming cues must remain categorical."""
        df = pd.DataFrame({
            "full_name": ["Alice Johnson", "Bob Smith", "Charlie Brown", "Diana Prince"],
            "bio": ["Engineer at TechCorp", "Marketing Lead", "Product Manager", "Data Scientist"],
        })
        schema = detect_schema(df)

        assert schema["full_name"] == SemanticType.CATEGORICAL
        assert schema["bio"] == SemanticType.CATEGORICAL

    def test_general_text_with_embedded_numbers_remains_categorical(self) -> None:
        """Strings containing numbers that are not dates must not become datetime."""
        df = pd.DataFrame({
            "order_notes": [
                "Order delivered in 3 days",
                "Urgent delivery requested for room 204",
                "Customer called on 5 occasions",
            ],
            "serial_code": ["CODE-A1", "CODE-B2", "CODE-C3"],
        })
        schema = detect_schema(df)

        assert schema["order_notes"] == SemanticType.CATEGORICAL
        assert schema["serial_code"] == SemanticType.CATEGORICAL


class TestMixedDatasets:
    """Test schema detection on full realistic datasets."""

    def test_sample_customers_classification(self, sample_data_dir: Path) -> None:
        """Verify full schema classification of customers.csv."""
        ds = load_dataset(sample_data_dir / "customers.csv")
        schema = detect_schema(ds.dataframe)

        expected = {
            "customer_id": "identifier",
            "name": "categorical",
            "age": "numerical",
            "annual_income": "numerical",
            "city": "categorical",
            "signup_date": "datetime",
            "is_active": "boolean",
        }
        assert schema.to_dict() == expected

        # Check helper methods
        assert schema.get_columns_by_type(SemanticType.IDENTIFIER) == ["customer_id"]
        assert schema.get_columns_by_type("numerical") == ["age", "annual_income"]
        assert schema.get_columns_by_type(SemanticType.BOOLEAN) == ["is_active"]
        assert schema.get_columns_by_type(SemanticType.DATETIME) == ["signup_date"]
        assert schema.get_columns_by_type(SemanticType.CATEGORICAL) == ["name", "city"]

        summary = schema.summary()
        assert summary["identifier"] == 1
        assert summary["numerical"] == 2
        assert summary["categorical"] == 2
        assert summary["datetime"] == 1
        assert summary["boolean"] == 1

    def test_sample_sales_classification(self, sample_data_dir: Path) -> None:
        """Verify full schema classification of sales.csv."""
        ds = load_dataset(sample_data_dir / "sales.csv")
        schema = detect_schema(ds.dataframe)

        expected = {
            "sale_id": "identifier",
            "customer_id": "identifier",
            "product_category": "categorical",
            "quantity": "numerical",
            "unit_price": "numerical",
            "total_amount": "numerical",
            "order_date": "datetime",
            "is_discounted": "boolean",
        }
        assert schema.to_dict() == expected

    def test_sample_transactions_classification(self, sample_data_dir: Path) -> None:
        """Verify full schema classification of transactions.json."""
        ds = load_dataset(sample_data_dir / "transactions.json")
        schema = detect_schema(ds.dataframe)

        expected = {
            "transaction_id": "identifier",
            "account_id": "identifier",
            "amount": "numerical",
            "currency": "categorical",
            "merchant": "categorical",
            "timestamp": "datetime",
            "is_fraud": "boolean",
        }
        assert schema.to_dict() == expected
