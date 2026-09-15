"""Unit tests for DataQualityReport and QualityIssue in src/quality/report.py."""

import json
import pytest

from src.quality.report import DataQualityReport, QualityIssue, Severity


@pytest.fixture
def sample_report() -> DataQualityReport:
    """Fixture providing a mock DataQualityReport."""
    issues = [
        QualityIssue(
            rule_id="HIGH_MISSINGNESS",
            severity=Severity.WARNING,
            title="High Missingness",
            description="Column 'income' has 23.4% missing values.",
            column="income",
            measured_value="23.4%",
            threshold=">20.0%",
            impact="Reduces model sample size.",
            metadata={"missing_count": 234},
        ),
        QualityIssue(
            rule_id="ALL_NULL",
            severity=Severity.CRITICAL,
            title="All-Null Column",
            description="Column 'notes' is 100% null.",
            column="notes",
            measured_value="100.0%",
            threshold="100.0%",
            impact="Column is completely useless.",
        ),
        QualityIssue(
            rule_id="IDENTIFIER_FEATURE",
            severity=Severity.INFO,
            title="Identifier Column Detected",
            description="Column 'cust_id' is an identifier.",
            column="cust_id",
            measured_value="identifier",
            threshold="semantic_type == 'identifier'",
        ),
    ]

    return DataQualityReport(
        dataset_name="test_data.csv",
        overall_score=72.5,
        grade="Fair",
        total_issues=3,
        critical_issues=1,
        error_issues=0,
        warning_issues=1,
        info_issues=1,
        issues=issues,
        column_scores={"income": 76.6, "notes": 0.0, "cust_id": 100.0},
        score_breakdown={"base_score": 100.0, "final_score": 72.5},
    )


class TestDataQualityReport:
    """Test DataQualityReport behaviors and serialization."""

    def test_report_attributes(self, sample_report: DataQualityReport) -> None:
        assert sample_report.dataset_name == "test_data.csv"
        assert sample_report.overall_score == 72.5
        assert sample_report.grade == "Fair"
        assert sample_report.total_issues == 3
        assert sample_report.critical_issues == 1
        assert sample_report.warning_issues == 1
        assert sample_report.info_issues == 1

    def test_filter_by_severity(self, sample_report: DataQualityReport) -> None:
        crit = sample_report.get_issues_by_severity(Severity.CRITICAL)
        assert len(crit) == 1
        assert crit[0].rule_id == "ALL_NULL"

        warn = sample_report.get_issues_by_severity("WARNING")
        assert len(warn) == 1
        assert warn[0].column == "income"

    def test_filter_by_column(self, sample_report: DataQualityReport) -> None:
        income_issues = sample_report.get_issues_by_column("income")
        assert len(income_issues) == 1
        assert income_issues[0].rule_id == "HIGH_MISSINGNESS"

        empty = sample_report.get_issues_by_column("non_existent")
        assert len(empty) == 0

    def test_to_dict_and_json_serialization(self, sample_report: DataQualityReport) -> None:
        d = sample_report.to_dict()
        assert isinstance(d, dict)
        assert d["dataset_name"] == "test_data.csv"
        assert len(d["issues"]) == 3
        assert d["column_scores"]["notes"] == 0.0

        json_str = sample_report.to_json()
        deserialized = json.loads(json_str)
        assert deserialized["overall_score"] == 72.5
        assert deserialized["grade"] == "Fair"

    def test_summary_formatting(self, sample_report: DataQualityReport) -> None:
        summary_text = sample_report.summary()
        assert "DATA QUALITY REPORT" in summary_text
        assert "72.5/100" in summary_text
        assert "Fair" in summary_text
        assert "[income] High Missingness" in summary_text
