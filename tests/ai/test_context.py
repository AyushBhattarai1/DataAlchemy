"""Unit and integration tests for AIContextBuilder in src/ai/context.py."""

from pathlib import Path
import pytest

from src.ai.context import AIContextBuilder
from src.analysis.analyzer import analyze_dataset
from src.analysis.findings import Finding, FindingImportance, FindingType
from src.ingestion.loader import load_dataset
from src.profiling.profiler import profile_dataset
from src.quality.analyzer import analyze_quality
from src.quality.report import DataQualityReport, QualityIssue, Severity


@pytest.fixture
def sample_data_dir() -> Path:
    """Path to sample dataset directory."""
    return Path(__file__).resolve().parent.parent.parent / "data" / "samples"


class TestAIContextBuilder:
    """Test suite for statistical context extraction and prompt text formatting."""

    def test_build_context_profile_only(self, sample_data_dir: Path) -> None:
        ds = load_dataset(sample_data_dir / "customers.csv")
        profile = profile_dataset(ds)

        builder = AIContextBuilder()
        context = builder.build_context(profile)

        assert "dataset" in context
        assert "quality" in context
        assert "findings" in context

        ds_info = context["dataset"]
        assert ds_info["name"] == "customers.csv"
        assert ds_info["row_count"] == ds.row_count
        assert ds_info["column_count"] == ds.column_count
        assert "columns" in ds_info
        assert len(ds_info["columns"]) <= ds.column_count

        # Quality and findings should be empty when not provided
        assert context["quality"] == {}
        assert context["findings"] == []

        prompt_text = builder.to_prompt_text(context)
        assert "DATASET SUMMARY" in prompt_text
        assert "customers.csv" in prompt_text
        assert "Quality Report: Not evaluated" in prompt_text
        assert "No significant statistical findings" in prompt_text

    def test_build_context_full_pipeline(self, sample_data_dir: Path) -> None:
        ds = load_dataset(sample_data_dir / "sales.csv")
        profile = profile_dataset(ds)
        quality = analyze_quality(profile)
        analysis = analyze_dataset(ds, profile, quality)

        builder = AIContextBuilder(max_findings=5, max_quality_issues=3, max_columns=4)
        context = builder.build_context(profile, quality, analysis)

        # Columns truncated to max_columns
        assert len(context["dataset"]["columns"]) <= 4

        # Quality issues truncated to max_quality_issues
        assert len(context["quality"]["top_issues"]) <= 3
        assert context["quality"]["overall_score"] == quality.overall_score
        assert context["quality"]["grade"] == quality.grade

        # Findings truncated to max_findings
        assert len(context["findings"]) <= 5

        # Verify prompt text generation
        prompt_text = builder.to_prompt_text(context)
        assert "### DATASET SUMMARY" in prompt_text
        assert "### DATA QUALITY HEALTH" in prompt_text
        assert "### VERIFIED STATISTICAL FINDINGS" in prompt_text
        assert "sales.csv" in prompt_text

    def test_quality_severity_prioritization(self, sample_data_dir: Path) -> None:
        ds = load_dataset(sample_data_dir / "customers.csv")
        profile = profile_dataset(ds)

        # Create mock quality report with mixed severities
        issues = [
            QualityIssue(
                rule_id="INFO_RULE",
                severity=Severity.INFO,
                title="Info Rule",
                description="Just info",
                column=None,
            ),
            QualityIssue(
                rule_id="CRITICAL_RULE",
                severity=Severity.CRITICAL,
                title="Critical Rule",
                description="Critical problem",
                column="id",
            ),
            QualityIssue(
                rule_id="WARNING_RULE",
                severity=Severity.WARNING,
                title="Warning Rule",
                description="Warning issue",
                column="email",
            ),
        ]
        quality = DataQualityReport(
            dataset_name="test",
            overall_score=85.0,
            grade="B",
            total_issues=3,
            critical_issues=1,
            error_issues=0,
            warning_issues=1,
            info_issues=1,
            issues=issues,
            column_scores={"id": 50.0, "email": 80.0},
        )

        builder = AIContextBuilder(max_quality_issues=2)
        context = builder.build_context(profile, quality_report=quality)

        top_issues = context["quality"]["top_issues"]
        assert len(top_issues) == 2
        # Critical should be first, then warning
        assert top_issues[0]["severity"] == "CRITICAL"
        assert top_issues[1]["severity"] == "WARNING"

    def test_findings_importance_prioritization(self, sample_data_dir: Path) -> None:
        ds = load_dataset(sample_data_dir / "customers.csv")
        profile = profile_dataset(ds)

        # Create mock findings with mixed importances
        findings = [
            Finding(
                finding_id="F_LOW",
                finding_type=FindingType.DISTRIBUTION,
                title="Low finding",
                description="Low description",
                importance=FindingImportance.LOW,
                affected_columns=["col1"],
                measured_values={"skewness": 0.2},
                statistical_measure="skewness",
            ),
            Finding(
                finding_id="F_HIGH",
                finding_type=FindingType.CORRELATION,
                title="High finding",
                description="High description",
                importance=FindingImportance.HIGH,
                affected_columns=["col1", "col2"],
                measured_values={"r": 0.85},
                statistical_measure="pearson_r",
            ),
            Finding(
                finding_id="F_MEDIUM",
                finding_type=FindingType.CATEGORY,
                title="Medium finding",
                description="Medium description",
                importance=FindingImportance.MEDIUM,
                affected_columns=["cat_col"],
                measured_values={"entropy": 1.2},
                statistical_measure="entropy",
            ),
        ]
        from src.analysis.analyzer import AnalysisReport

        report = AnalysisReport(
            dataset_name="test",
            row_count=10,
            column_count=2,
            findings=findings,
            correlations=[],
            distributions=[],
            relationships=[],
            trends=[],
        )

        builder = AIContextBuilder(max_findings=2)
        context = builder.build_context(profile, analysis_report=report)

        top_findings = context["findings"]
        assert len(top_findings) == 2
        # HIGH must come first, followed by MEDIUM
        assert top_findings[0]["importance"] == "HIGH"
        assert top_findings[1]["importance"] == "MEDIUM"

    def test_privacy_guarantee_no_raw_rows(self, sample_data_dir: Path) -> None:
        """Verify that context does not leak raw dataframe rows or unaggregated row dumps."""
        ds = load_dataset(sample_data_dir / "customers.csv")
        profile = profile_dataset(ds)
        quality = analyze_quality(profile)
        analysis = analyze_dataset(ds, profile, quality)

        builder = AIContextBuilder()
        context = builder.build_context(profile, quality, analysis)
        prompt_text = builder.to_prompt_text(context)

        # Verify no raw CSV lines or row tuples are dumped in prompt text
        for _, row in ds.dataframe.iterrows():
            row_line = f"{row['customer_id']},{row['name']}"
            assert row_line not in prompt_text
            # Individual customer IDs should not appear as standalone record dumps
            assert f"customer_id: {row['customer_id']}" not in prompt_text
