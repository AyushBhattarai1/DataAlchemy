"""Unit tests for DistributionAnalyzer in src/analysis/distributions.py."""

import pandas as pd
import pytest

from src.analysis.distributions import DistributionAnalyzer, DistributionResult
from src.analysis.findings import FindingImportance, FindingType
from src.ingestion.loader import Dataset
from src.profiling.profiler import profile_dataset


class TestDistributionAnalyzer:
    """Test distribution shape analysis and skewness categorization."""

    def test_symmetric_distribution(self) -> None:
        # Perfectly symmetric distribution
        df = pd.DataFrame({"sym": [10.0, 20.0, 30.0, 40.0, 50.0]})
        profile = profile_dataset(Dataset(df, "sym.csv", "csv", 5, 1))

        results, findings = DistributionAnalyzer().analyze(profile)
        assert len(results) == 1
        assert results[0].skewness_classification == "approximately symmetric"
        assert results[0].skewness == pytest.approx(0.0)

        assert len(findings) == 1
        assert findings[0].importance == FindingImportance.INFO

    def test_strongly_positive_skew(self) -> None:
        # Long right tail: [1, 1, 1, 2, 2, 3, 50, 100]
        df = pd.DataFrame({"skewed": [1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 50.0, 100.0]})
        profile = profile_dataset(Dataset(df, "skew.csv", "csv", 8, 1))

        results, findings = DistributionAnalyzer().analyze(profile)
        assert len(results) == 1
        assert "right-skewed" in results[0].skewness_classification
        assert results[0].skewness > 1.0

        assert len(findings) == 1
        assert findings[0].importance in (FindingImportance.MEDIUM, FindingImportance.HIGH)
        assert findings[0].finding_type == FindingType.DISTRIBUTION

    def test_strongly_negative_skew(self) -> None:
        # Long left tail: [1, 50, 98, 98, 99, 99, 100, 100]
        df = pd.DataFrame({"skewed": [1.0, 50.0, 98.0, 98.0, 99.0, 99.0, 100.0, 100.0]})
        profile = profile_dataset(Dataset(df, "skew.csv", "csv", 8, 1))

        results, findings = DistributionAnalyzer().analyze(profile)
        assert len(results) == 1
        assert "left-skewed" in results[0].skewness_classification
        assert results[0].skewness < -1.0

    def test_constant_and_null_columns_handled_safely(self) -> None:
        df = pd.DataFrame({
            "const": [5.0, 5.0, 5.0],
            "empty": pd.Series([None, None, None], dtype="float64"),
        })
        profile = profile_dataset(Dataset(df, "edge.csv", "csv", 3, 2))

        results, findings = DistributionAnalyzer().analyze(profile)
        # Constant or null columns have no valid skewness to evaluate -> 0 findings
        assert len(results) == 0
        assert len(findings) == 0
