"""Unit tests for Finding model and sorting logic in src/analysis/findings.py."""

import json
import pytest

from src.analysis.findings import Finding, FindingImportance, FindingType


class TestFindingModel:
    """Test Finding dataclass creation, serialization, and deterministic sorting."""

    def test_finding_creation_and_attributes(self) -> None:
        finding = Finding(
            finding_id="corr_rev_spend",
            finding_type=FindingType.CORRELATION,
            title="Strong association between revenue and spend",
            description="Revenue and spend correlate strongly.",
            importance=FindingImportance.HIGH,
            affected_columns=["revenue", "spend"],
            measured_values={"pearson_r": 0.81, "sample_size": 100},
            statistical_measure="pearson_correlation",
            confidence_or_pvalue=0.001,
            metadata={"source": "unit_test"},
        )

        assert finding.finding_id == "corr_rev_spend"
        assert finding.finding_type == FindingType.CORRELATION
        assert finding.importance == FindingImportance.HIGH
        assert finding.affected_columns == ["revenue", "spend"]
        assert finding.measured_values["pearson_r"] == 0.81
        assert finding.confidence_or_pvalue == 0.001

    def test_finding_to_dict_and_json(self) -> None:
        finding = Finding(
            finding_id="dist_revenue",
            finding_type=FindingType.DISTRIBUTION,
            title="Right skewed revenue",
            description="Revenue is right-skewed.",
            importance=FindingImportance.MEDIUM,
            affected_columns=["revenue"],
            measured_values={"skewness": 2.45},
            statistical_measure="skewness",
        )

        d = finding.to_dict()
        assert isinstance(d, dict)
        assert d["finding_id"] == "dist_revenue"
        assert d["finding_type"] == "DISTRIBUTION"
        assert d["importance"] == "MEDIUM"

        json_str = json.dumps(d)
        deserialized = json.loads(json_str)
        assert deserialized["measured_values"]["skewness"] == 2.45

    def test_deterministic_sorting(self) -> None:
        f_low = Finding("f_low", FindingType.DISTRIBUTION, "T1", "D1", FindingImportance.LOW, ["c"], {}, "m")
        f_high_b = Finding("f_high_b", FindingType.CORRELATION, "T2", "D2", FindingImportance.HIGH, ["a"], {}, "m")
        f_high_a = Finding("f_high_a", FindingType.CORRELATION, "T3", "D3", FindingImportance.HIGH, ["b"], {}, "m")
        f_info = Finding("f_info", FindingType.CATEGORY, "T4", "D4", FindingImportance.INFO, ["d"], {}, "m")
        f_med = Finding("f_med", FindingType.GROUP_RELATIONSHIP, "T5", "D5", FindingImportance.MEDIUM, ["e"], {}, "m")

        unsorted_findings = [f_low, f_high_b, f_info, f_high_a, f_med]
        sorted_findings = sorted(unsorted_findings)

        # High priority first, sorted by finding_id within priority
        assert [f.finding_id for f in sorted_findings] == [
            "f_high_a",
            "f_high_b",
            "f_med",
            "f_low",
            "f_info",
        ]
