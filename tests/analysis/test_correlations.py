"""Unit tests for CorrelationAnalyzer in src/analysis/correlations.py."""

import numpy as np
import pandas as pd
import pytest

from src.analysis.correlations import CorrelationAnalyzer, CorrelationResult
from src.analysis.findings import FindingImportance, FindingType
from src.ingestion.loader import Dataset
from src.profiling.profiler import profile_dataset


class TestCorrelationAnalyzer:
    """Test correlation computation, filtering, and edge cases."""

    def test_perfect_positive_correlation(self) -> None:
        df = pd.DataFrame({
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "y": [2.0, 4.0, 6.0, 8.0, 10.0],
        })
        ds = Dataset(df, "test.csv", "csv", 5, 2)
        profile = profile_dataset(ds)

        results, findings = CorrelationAnalyzer().analyze(df, profile)

        assert len(results) == 1
        assert results[0].coefficient == pytest.approx(1.0)
        assert results[0].direction == "positive"
        assert results[0].strength == "very strong"

        assert len(findings) == 1
        assert findings[0].finding_type == FindingType.CORRELATION
        assert findings[0].importance == FindingImportance.HIGH
        assert findings[0].affected_columns == ["x", "y"]

    def test_perfect_negative_correlation(self) -> None:
        df = pd.DataFrame({
            "a": [10.0, 20.0, 30.0, 40.0],
            "b": [100.0, 80.0, 60.0, 40.0],
        })
        ds = Dataset(df, "test.csv", "csv", 4, 2)
        profile = profile_dataset(ds)

        results, findings = CorrelationAnalyzer().analyze(df, profile)

        assert len(results) == 1
        assert results[0].coefficient == pytest.approx(-1.0)
        assert results[0].direction == "negative"

    def test_weak_correlation_filtered_by_threshold(self) -> None:
        # Non-correlated data
        df = pd.DataFrame({
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "y": [5.0, 2.0, 6.0, 1.0, 4.0, 3.0],
        })
        ds = Dataset(df, "test.csv", "csv", 6, 2)
        profile = profile_dataset(ds)

        # High threshold filters out weak correlation
        results, findings = CorrelationAnalyzer(min_correlation=0.7).analyze(df, profile)
        assert len(results) == 0
        assert len(findings) == 0

    def test_missing_values_handled_pairwise(self) -> None:
        df = pd.DataFrame({
            "x": [1.0, 2.0, None, 4.0, 5.0],
            "y": [2.0, 4.0, 6.0, None, 10.0],
        })
        ds = Dataset(df, "test.csv", "csv", 5, 2)
        profile = profile_dataset(ds)

        results, findings = CorrelationAnalyzer().analyze(df, profile)
        assert len(results) == 1
        # Rows with valid pairs are 0, 1, 4 -> n = 3
        assert results[0].sample_size == 3
        assert results[0].coefficient == pytest.approx(1.0)

    def test_constant_column_skipped_safely(self) -> None:
        df = pd.DataFrame({
            "const": [5.0, 5.0, 5.0, 5.0],
            "num": [10.0, 20.0, 30.0, 40.0],
        })
        ds = Dataset(df, "test.csv", "csv", 4, 2)
        profile = profile_dataset(ds)

        results, findings = CorrelationAnalyzer().analyze(df, profile)
        assert len(results) == 0

    def test_identifier_columns_excluded(self) -> None:
        df = pd.DataFrame({
            "customer_id": [101, 102, 103, 104],  # Semantic identifier
            "income": [50000.0, 60000.0, 70000.0, 80000.0],
        })
        ds = Dataset(df, "test.csv", "csv", 4, 2)
        profile = profile_dataset(ds)

        results, findings = CorrelationAnalyzer().analyze(df, profile)
        # Only 1 numerical column (income) remains; customer_id is identifier -> 0 pairs
        assert len(results) == 0

    def test_no_duplicate_pairs_or_self_correlation(self) -> None:
        df = pd.DataFrame({
            "c1": [1.0, 2.0, 3.0, 4.0],
            "c2": [10.0, 20.0, 30.0, 40.0],
            "c3": [100.0, 200.0, 300.0, 400.0],
        })
        ds = Dataset(df, "test.csv", "csv", 4, 3)
        profile = profile_dataset(ds)

        results, findings = CorrelationAnalyzer().analyze(df, profile)
        # 3 columns = 3 distinct pairs: (c1, c2), (c1, c3), (c2, c3)
        assert len(results) == 3
        pairs = {(r.column_x, r.column_y) for r in results}
        assert ("c1", "c1") not in pairs
        assert ("c2", "c1") not in pairs

    def test_max_pairs_limit_enforced(self) -> None:
        df = pd.DataFrame({
            "c1": [1, 2, 3, 4],
            "c2": [2, 4, 6, 8],
            "c3": [3, 6, 9, 12],
            "c4": [4, 8, 12, 16],
        })
        ds = Dataset(df, "test.csv", "csv", 4, 4)
        profile = profile_dataset(ds)

        # 6 possible pairs, limit to 2
        results, findings = CorrelationAnalyzer(max_pairs=2).analyze(df, profile)
        assert len(results) == 2
        assert len(findings) == 2
