"""Unit tests for RelationshipAnalyzer in src/analysis/relationships.py."""

import pandas as pd
import pytest

from src.analysis.findings import FindingImportance, FindingType
from src.analysis.relationships import RelationshipAnalyzer
from src.ingestion.loader import Dataset
from src.profiling.profiler import profile_dataset


class TestRelationshipAnalyzer:
    """Test category concentration and group breakdown analysis."""

    def test_high_category_concentration(self) -> None:
        # 16 out of 20 are "Alpha" = 80% dominance
        items = ["Alpha"] * 16 + ["Beta"] * 2 + ["Gamma"] * 2
        df = pd.DataFrame({"segment": items})
        profile = profile_dataset(Dataset(df, "cat.csv", "csv", 20, 1))

        findings = RelationshipAnalyzer(concentration_threshold_pct=50.0).analyze_categories(profile)

        assert len(findings) == 1
        assert findings[0].finding_type == FindingType.CATEGORY
        assert findings[0].importance == FindingImportance.HIGH
        assert findings[0].measured_values["dominant_category"] == "Alpha"
        assert findings[0].measured_values["dominant_share_pct"] == 80.0

    def test_balanced_category_distribution(self) -> None:
        # Balanced: 4 categories, 25% each
        items = ["A", "B", "C", "D"] * 10
        df = pd.DataFrame({"region": items})
        profile = profile_dataset(Dataset(df, "cat.csv", "csv", 40, 1))

        findings = RelationshipAnalyzer().analyze_categories(profile)

        assert len(findings) == 1
        assert findings[0].finding_type == FindingType.CATEGORY
        assert findings[0].importance == FindingImportance.INFO
        assert "Balanced" in findings[0].title

    def test_numerical_grouped_by_categorical_relationship(self) -> None:
        # Segment 1 has high sales, Segment 2 has low sales
        df = pd.DataFrame({
            "dept": ["Engineering", "Engineering", "Engineering", "HR", "HR", "HR"],
            "salary": [120000.0, 130000.0, 125000.0, 60000.0, 62000.0, 58000.0],
        })
        profile = profile_dataset(Dataset(df, "rel.csv", "csv", 6, 2))

        results, findings = RelationshipAnalyzer().analyze_relationships(df, profile)

        assert len(results) == 1
        assert results[0].categorical_column == "dept"
        assert results[0].numerical_column == "salary"
        assert results[0].variation_ratio > 1.5

        assert len(findings) == 1
        assert findings[0].finding_type == FindingType.GROUP_RELATIONSHIP
        assert findings[0].importance in (FindingImportance.HIGH, FindingImportance.MEDIUM)
        assert findings[0].affected_columns == ["dept", "salary"]
        assert "Engineering" in findings[0].measured_values["group_means"]

    def test_uniform_group_means_generates_no_disparity_finding(self) -> None:
        # Identical means across groups
        df = pd.DataFrame({
            "dept": ["A", "A", "A", "B", "B", "B"],
            "score": [50.0, 50.0, 50.0, 50.0, 50.0, 50.0],
        })
        profile = profile_dataset(Dataset(df, "uniform.csv", "csv", 6, 2))

        results, findings = RelationshipAnalyzer().analyze_relationships(df, profile)
        # Constant score means variance ratio is 1.0 -> no disparity finding
        assert len(findings) == 0
