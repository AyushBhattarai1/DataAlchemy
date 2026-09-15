"""Unit tests for AIOutputParser and AI models in src/ai/parser.py and src/ai/models.py."""

import json
import pytest

from src.ai.models import AIInsight, AIInsightReport, AIRecommendation
from src.ai.parser import AIOutputParser, AIOutputParsingError


class TestAIOutputParser:
    """Test suite for parsing and validating LLM-generated JSON."""

    @pytest.fixture
    def parser(self) -> AIOutputParser:
        return AIOutputParser()

    def test_parse_valid_json_report(self, parser: AIOutputParser) -> None:
        raw_json = json.dumps({
            "executive_summary": "Overall data health is good with strong correlation between x and y.",
            "insights": [
                {
                    "insight_id": "INSIGHT_01",
                    "title": "Strong correlation between age and income",
                    "summary": "Age and income move closely together.",
                    "explanation": "Correlation coefficient r=0.78 indicates high statistical association.",
                    "importance": "HIGH",
                    "category": "correlation",
                    "evidence": {"finding_id": "corr_age_income", "r": 0.78},
                    "affected_columns": ["age", "income"],
                    "caveats": "Does not prove causality.",
                }
            ],
            "recommendations": [
                {
                    "recommendation_id": "REC_01",
                    "title": "Investigate outliers",
                    "description": "Examine extreme income observations.",
                    "action_type": "investigation",
                    "supporting_evidence": ["corr_age_income"],
                    "priority": "medium",
                }
            ],
            "warnings": ["Sample size is small"],
        })

        report = parser.parse_insight_report(raw_json, dataset_name="test_data.csv")

        assert isinstance(report, AIInsightReport)
        assert report.dataset_name == "test_data.csv"
        assert "Overall data health" in report.executive_summary
        assert len(report.insights) == 1
        assert len(report.recommendations) == 1
        assert len(report.warnings) == 1

        ins = report.insights[0]
        assert isinstance(ins, AIInsight)
        assert ins.insight_id == "INSIGHT_01"
        assert ins.importance == "HIGH"
        assert ins.category == "correlation"
        assert ins.affected_columns == ["age", "income"]
        assert ins.evidence["r"] == 0.78
        assert ins.caveats == "Does not prove causality."

        rec = report.recommendations[0]
        assert isinstance(rec, AIRecommendation)
        assert rec.recommendation_id == "REC_01"
        assert rec.priority == "medium"
        assert rec.action_type == "investigation"

    def test_parse_markdown_code_fences(self, parser: AIOutputParser) -> None:
        raw_wrapped = """Here is your insight report:
```json
{
  "executive_summary": "Clean executive summary.",
  "insights": [],
  "recommendations": []
}
```
Hope this is helpful!"""

        report = parser.parse_insight_report(raw_wrapped, dataset_name="wrapped.csv")
        assert report.dataset_name == "wrapped.csv"
        assert report.executive_summary == "Clean executive summary."
        assert len(report.insights) == 0

    def test_parse_bare_fences(self, parser: AIOutputParser) -> None:
        raw_wrapped = """```
{
  "executive_summary": "Bare fences executive summary.",
  "insights": [],
  "recommendations": []
}
```"""
        report = parser.parse_insight_report(raw_wrapped, dataset_name="bare.csv")
        assert report.executive_summary == "Bare fences executive summary."

    def test_parse_empty_string_raises_error(self, parser: AIOutputParser) -> None:
        with pytest.raises(AIOutputParsingError) as exc_info:
            parser.parse_insight_report("", dataset_name="empty")
        assert "empty response" in str(exc_info.value).lower()

    def test_parse_invalid_json_raises_error(self, parser: AIOutputParser) -> None:
        with pytest.raises(AIOutputParsingError) as exc_info:
            parser.parse_insight_report("Not a JSON object at all", dataset_name="broken")
        assert "failed to parse" in str(exc_info.value).lower()

    def test_parse_non_dict_json_raises_error(self, parser: AIOutputParser) -> None:
        with pytest.raises(AIOutputParsingError) as exc_info:
            parser.parse_insight_report('["array", "of", "items"]', dataset_name="array")
        assert "expected json object" in str(exc_info.value).lower()

    def test_parse_missing_executive_summary_raises_error(self, parser: AIOutputParser) -> None:
        raw = '{"insights": [], "recommendations": []}'
        with pytest.raises(AIOutputParsingError) as exc_info:
            parser.parse_insight_report(raw, dataset_name="missing_field")
        assert "missing required field 'executive_summary'" in str(exc_info.value).lower()

    def test_parse_empty_executive_summary_raises_error(self, parser: AIOutputParser) -> None:
        raw = '{"executive_summary": "   ", "insights": [], "recommendations": []}'
        with pytest.raises(AIOutputParsingError) as exc_info:
            parser.parse_insight_report(raw, dataset_name="empty_field")
        assert "'executive_summary' cannot be empty" in str(exc_info.value).lower()

    def test_parse_non_list_insights_raises_error(self, parser: AIOutputParser) -> None:
        raw = '{"executive_summary": "Summary", "insights": "not a list", "recommendations": []}'
        with pytest.raises(AIOutputParsingError) as exc_info:
            parser.parse_insight_report(raw, dataset_name="bad_insights")
        assert "'insights' must be a list" in str(exc_info.value).lower()


class TestAIModelsMethods:
    """Test suite for AIInsightReport methods, filtering, and serialization."""

    def test_report_filters_and_export(self) -> None:
        ins1 = AIInsight(
            insight_id="I1",
            title="Correlation insight",
            summary="Corr summary",
            explanation="Corr explanation",
            importance="HIGH",
            category="correlation",
            evidence={"r": 0.9},
            affected_columns=["x", "y"],
        )
        ins2 = AIInsight(
            insight_id="I2",
            title="Quality insight",
            summary="Qual summary",
            explanation="Qual explanation",
            importance="LOW",
            category="quality",
            evidence={},
            affected_columns=["z"],
        )
        rec = AIRecommendation(
            recommendation_id="R1",
            title="Fix missing values",
            description="Impute z",
            action_type="quality_fix",
            supporting_evidence=["I2"],
            priority="high",
        )
        report = AIInsightReport(
            dataset_name="metrics.csv",
            executive_summary="Executive overview text",
            insights=[ins1, ins2],
            recommendations=[rec],
            warnings=["Watch out"],
        )

        # Filtering tests
        high_insights = report.get_insights_by_importance("HIGH")
        assert len(high_insights) == 1
        assert high_insights[0].insight_id == "I1"

        corr_insights = report.get_insights_by_category("correlation")
        assert len(corr_insights) == 1
        assert corr_insights[0].insight_id == "I1"

        # Serialization tests
        as_dict = report.to_dict()
        assert as_dict["dataset_name"] == "metrics.csv"
        assert as_dict["total_insights"] == 2
        assert as_dict["total_recommendations"] == 1

        json_str = report.to_json()
        assert "metrics.csv" in json_str
        assert "Executive overview text" in json_str

        # Summary string representation test
        summary_text = report.summary()
        assert "AI INSIGHT REPORT" in summary_text
        assert "metrics.csv" in summary_text
        assert "Correlation insight" in summary_text
        assert "Fix missing values" in summary_text
