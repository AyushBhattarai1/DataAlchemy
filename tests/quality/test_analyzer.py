"""Unit tests for QualityAnalyzer in src/quality/analyzer.py."""

from pathlib import Path
import pandas as pd
import pytest

from src.ingestion.loader import Dataset, load_dataset
from src.profiling.profiler import profile_dataset
from src.quality.analyzer import QualityAnalyzer, analyze_quality
from src.quality.report import DataQualityReport, Severity


@pytest.fixture
def sample_data_dir() -> Path:
    """Path to sample dataset directory."""
    return Path(__file__).resolve().parent.parent.parent / "data" / "samples"


class TestQualityAnalyzerSamples:
    """Test analyzer on sample datasets."""

    def test_analyze_customers_csv(self, sample_data_dir: Path) -> None:
        ds = load_dataset(sample_data_dir / "customers.csv")
        profile = profile_dataset(ds)
        report = analyze_quality(profile)

        assert isinstance(report, DataQualityReport)
        assert report.dataset_name == "customers.csv"
        assert 0.0 <= report.overall_score <= 100.0
        assert report.grade in ("Excellent", "Good", "Fair", "Poor", "Critical")

        # customers.csv has mild missingness and 1 duplicate, should score high (~94)
        assert report.overall_score >= 90.0
        assert report.grade == "Excellent"

        # Check issues detected
        issue_ids = [iss.rule_id for iss in report.issues]
        assert "MODERATE_MISSINGNESS" in issue_ids
        assert "DUPLICATE_ROWS" in issue_ids
        assert "IDENTIFIER_FEATURE" in issue_ids

        # Verify customer_id column score has no penalty
        assert report.column_scores["customer_id"] == 100.0

    def test_analyze_sales_csv(self, sample_data_dir: Path) -> None:
        ds = load_dataset(sample_data_dir / "sales.csv")
        profile = profile_dataset(ds)
        report = analyze_quality(profile)

        assert report.overall_score >= 95.0
        assert report.grade == "Excellent"
        assert report.column_scores["sale_id"] == 100.0

    def test_analyze_transactions_json(self, sample_data_dir: Path) -> None:
        ds = load_dataset(sample_data_dir / "transactions.json")
        profile = profile_dataset(ds)
        report = analyze_quality(profile)

        # Clean dataset with 0 missing and 0 duplicates
        assert report.overall_score == 100.0
        assert report.grade == "Excellent"
        assert report.critical_issues == 0
        assert report.error_issues == 0
        assert report.warning_issues == 0
        assert report.info_issues == 2  # 2 identifier columns noted


class TestQualityScoringSystem:
    """Verify deterministic scoring rules and score bounds."""

    def test_perfect_dataset_gets_100(self) -> None:
        df = pd.DataFrame({
            "feature_a": [10, 20, 30, 40],
            "feature_b": ["apple", "banana", "cherry", "date"],
        })
        profile = profile_dataset(Dataset(df, "perfect.csv", "csv", 4, 2))
        report = analyze_quality(profile)

        assert report.overall_score == 100.0
        assert report.grade == "Excellent"
        assert report.total_issues == 0

    def test_critical_issues_reduce_score_more_than_info(self) -> None:
        # Dataset 1: Pure info issues (identifiers)
        df_info = pd.DataFrame({
            "id_1": [f"ID_{i}" for i in range(20)],
            "id_2": [f"UUID_{i}" for i in range(20)],
        })
        rep_info = analyze_quality(profile_dataset(Dataset(df_info, "info.csv", "csv", 20, 2)))

        # Dataset 2: Critical issues (all-null column and heavy duplicates)
        df_crit = pd.DataFrame({
            "empty": [None] * 20,
            "repeated": [1] * 20,
        })
        rep_crit = analyze_quality(profile_dataset(Dataset(df_crit, "crit.csv", "csv", 20, 2)))

        assert rep_info.overall_score > rep_crit.overall_score
        assert rep_info.overall_score == 100.0
        assert rep_crit.overall_score < 70.0

    def test_score_bounded_between_0_and_100(self) -> None:
        # Heavily degraded dataset
        df_bad = pd.DataFrame({
            "all_null_1": [None] * 50,
            "all_null_2": [None] * 50,
            "const_col": [1] * 50,
        })
        profile = profile_dataset(Dataset(df_bad, "bad.csv", "csv", 50, 3))
        report = analyze_quality(profile)

        assert 0.0 <= report.overall_score <= 100.0

    def test_deterministic_scoring(self) -> None:
        df = pd.DataFrame({
            "a": [1, 2, None, 4],
            "b": ["x", "x", "x", "y"],
        })
        profile = profile_dataset(Dataset(df, "data.csv", "csv", 4, 2))
        report1 = analyze_quality(profile)
        report2 = analyze_quality(profile)

        assert report1.overall_score == report2.overall_score
        assert len(report1.issues) == len(report2.issues)
        assert report1.grade == report2.grade


class TestDataPreservationDuringQualityAnalysis:
    """Verify quality analysis never mutates the original data."""

    def test_dataframe_remains_strictly_unmodified(self, sample_data_dir: Path) -> None:
        ds = load_dataset(sample_data_dir / "customers.csv")
        df_snapshot = ds.dataframe.copy(deep=True)
        profile = profile_dataset(ds)

        # Run quality analyzer
        _ = analyze_quality(profile)

        # Verify DataFrame is completely intact
        pd.testing.assert_frame_equal(ds.dataframe, df_snapshot)


class TestAnalyzerEdgeCases:
    """Verify analyzer behaves gracefully on unusual data inputs."""

    def test_single_row_dataset(self) -> None:
        df = pd.DataFrame({"id": ["A1"], "metric": [42.0]})
        profile = profile_dataset(Dataset(df, "single_row.csv", "csv", 1, 2))
        report = analyze_quality(profile)

        assert 0.0 <= report.overall_score <= 100.0

    def test_single_column_dataset(self) -> None:
        df = pd.DataFrame({"single": [1, 2, 3, 4, 5]})
        profile = profile_dataset(Dataset(df, "single_col.csv", "csv", 5, 1))
        report = analyze_quality(profile)

        assert report.overall_score == 100.0

    def test_dataset_with_multiple_simultaneous_issues(self) -> None:
        df = pd.DataFrame({
            "all_null": [None, None, None, None, None],
            "constant": [1, 1, 1, 1, 1],
            "valid": [10, 20, 30, 40, 50],
        })
        profile = profile_dataset(Dataset(df, "multi.csv", "csv", 5, 3))
        report = analyze_quality(profile)

        rule_ids = {iss.rule_id for iss in report.issues}
        assert "ALL_NULL" in rule_ids
        assert "CONSTANT_COLUMN" in rule_ids
        assert report.column_scores["all_null"] == 0.0
