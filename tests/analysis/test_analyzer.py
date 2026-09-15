"""Integration unit tests for StatisticalAnalyzer and AnalysisReport in src/analysis/analyzer.py."""

import json
from pathlib import Path

import pandas as pd
import pytest

from src.analysis.analyzer import AnalysisReport, StatisticalAnalyzer, analyze_dataset
from src.analysis.findings import FindingImportance, FindingType
from src.ingestion.loader import Dataset, load_dataset
from src.profiling.profiler import profile_dataset
from src.quality.analyzer import analyze_quality


@pytest.fixture
def sample_data_dir() -> Path:
    """Path to sample dataset directory."""
    return Path(__file__).resolve().parent.parent.parent / "data" / "samples"


class TestStatisticalAnalyzerSamples:
    """Test full statistical analysis pipeline on real sample datasets."""

    def test_analyze_sales_sample(self, sample_data_dir: Path) -> None:
        ds = load_dataset(sample_data_dir / "sales.csv")
        prof = profile_dataset(ds)
        qual = analyze_quality(prof)
        report = analyze_dataset(ds, prof, qual)

        assert isinstance(report, AnalysisReport)
        assert report.dataset_name == "sales.csv"
        assert report.row_count == 7
        assert report.column_count == 8
        assert len(report.findings) > 0

        # Verify correlation between total_amount and unit_price
        corr_findings = report.get_findings_by_type(FindingType.CORRELATION)
        assert len(corr_findings) > 0
        top_corr = corr_findings[0]
        assert "total_amount" in top_corr.affected_columns
        assert "unit_price" in top_corr.affected_columns
        assert top_corr.measured_values["pearson_r"] > 0.7

    def test_analyze_customers_sample(self, sample_data_dir: Path) -> None:
        ds = load_dataset(sample_data_dir / "customers.csv")
        report = analyze_dataset(ds)

        assert report.dataset_name == "customers.csv"
        assert len(report.findings) > 0

        # Verify category findings
        cat_findings = report.get_findings_by_type(FindingType.CATEGORY)
        assert len(cat_findings) > 0

    def test_analyze_transactions_sample(self, sample_data_dir: Path) -> None:
        ds = load_dataset(sample_data_dir / "transactions.json")
        report = analyze_dataset(ds)

        assert report.dataset_name == "transactions.json"
        assert report.row_count == 5
        assert isinstance(report.to_dict(), dict)


class TestAnalysisReportBehaviors:
    """Test report helper methods, filtering, and serialization."""

    def test_report_filtering_helpers(self, sample_data_dir: Path) -> None:
        ds = load_dataset(sample_data_dir / "sales.csv")
        report = analyze_dataset(ds)

        # By importance
        high_findings = report.get_findings_by_importance(FindingImportance.HIGH)
        for f in high_findings:
            assert f.importance == FindingImportance.HIGH

        # By column
        price_findings = report.get_findings_by_column("unit_price")
        assert len(price_findings) > 0
        for f in price_findings:
            assert "unit_price" in f.affected_columns

    def test_report_serialization_and_summary(self, sample_data_dir: Path) -> None:
        ds = load_dataset(sample_data_dir / "sales.csv")
        report = analyze_dataset(ds)

        # to_dict
        d = report.to_dict()
        assert isinstance(d, dict)
        assert d["dataset_name"] == "sales.csv"
        assert "findings" in d

        # to_json roundtrip
        json_str = report.to_json()
        deserialized = json.loads(json_str)
        assert deserialized["total_findings"] == len(report.findings)

        # summary text
        summary = report.summary()
        assert "STATISTICAL ANALYSIS REPORT" in summary
        assert "sales.csv" in summary


class TestDeterminismAndDataPreservation:
    """Verify determinism and zero data mutation."""

    def test_dataframe_strictly_unmodified(self, sample_data_dir: Path) -> None:
        ds = load_dataset(sample_data_dir / "sales.csv")
        df_snapshot = ds.dataframe.copy(deep=True)

        _ = analyze_dataset(ds)

        pd.testing.assert_frame_equal(ds.dataframe, df_snapshot)

    def test_deterministic_output(self, sample_data_dir: Path) -> None:
        ds = load_dataset(sample_data_dir / "sales.csv")

        report1 = analyze_dataset(ds)
        report2 = analyze_dataset(ds)

        assert len(report1.findings) == len(report2.findings)
        assert [f.finding_id for f in report1.findings] == [f.finding_id for f in report2.findings]
        assert [f.importance for f in report1.findings] == [f.importance for f in report2.findings]


class TestAnalyzerEdgeCases:
    """Test analyzer robustness on boundary datasets."""

    def test_single_row_dataset(self) -> None:
        df = pd.DataFrame({"id": ["A1"], "val": [10.0]})
        report = analyze_dataset(Dataset(df, "single.csv", "csv", 1, 2))

        assert report.row_count == 1
        assert len(report.findings) == 0

    def test_all_null_dataset(self) -> None:
        df = pd.DataFrame({
            "col1": pd.Series([None, None, None], dtype="float64"),
            "col2": [None, None, None],
        })
        report = analyze_dataset(Dataset(df, "null.csv", "csv", 3, 2))
        assert isinstance(report, AnalysisReport)
        assert len(report.correlations) == 0

    def test_constant_dataset(self) -> None:
        df = pd.DataFrame({
            "const1": [5.0, 5.0, 5.0, 5.0],
            "const2": [10.0, 10.0, 10.0, 10.0],
        })
        report = analyze_dataset(Dataset(df, "const.csv", "csv", 4, 2))
        assert len(report.correlations) == 0

    def test_wide_dataset_caps_correlations(self) -> None:
        # 15 numerical columns -> 105 pairs, analyzer should cap at max_pairs
        cols = {f"c_{i}": [float(j * (i + 1)) for j in range(10)] for i in range(15)}
        df = pd.DataFrame(cols)
        analyzer = StatisticalAnalyzer(max_correlation_pairs=5)
        report = analyzer.analyze(Dataset(df, "wide.csv", "csv", 10, 15))

        assert len(report.correlations) <= 5
