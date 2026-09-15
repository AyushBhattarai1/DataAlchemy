"""Unit tests for TrendAnalyzer in src/analysis/trends.py."""

import pandas as pd
import pytest

from src.analysis.findings import FindingImportance, FindingType
from src.analysis.trends import TrendAnalyzer
from src.ingestion.loader import Dataset
from src.profiling.profiler import profile_dataset


class TestTrendAnalyzer:
    """Test temporal trend trajectory detection and period aggregation."""

    def test_increasing_temporal_trend(self) -> None:
        # Clear upward progression over monthly dates
        dates = pd.date_range("2024-01-01", periods=6, freq="MS").strftime("%Y-%m-%d").tolist()
        df = pd.DataFrame({
            "record_date": dates,
            "revenue": [1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0],
        })
        profile = profile_dataset(Dataset(df, "trend_up.csv", "csv", 6, 2))

        results, findings = TrendAnalyzer().analyze(df, profile)

        assert len(results) == 1
        assert results[0].direction == "increasing"
        assert results[0].slope > 0
        assert results[0].r_squared == pytest.approx(1.0)

        assert len(findings) == 1
        assert findings[0].finding_type == FindingType.TEMPORAL_TREND
        assert findings[0].importance == FindingImportance.HIGH
        assert findings[0].measured_values["direction"] == "increasing"

    def test_decreasing_temporal_trend(self) -> None:
        # Downward trajectory
        dates = pd.date_range("2024-01-01", periods=5, freq="MS").strftime("%Y-%m-%d").tolist()
        df = pd.DataFrame({
            "sale_date": dates,
            "churn": [50.0, 40.0, 30.0, 20.0, 10.0],
        })
        profile = profile_dataset(Dataset(df, "trend_down.csv", "csv", 5, 2))

        results, findings = TrendAnalyzer().analyze(df, profile)

        assert len(results) == 1
        assert results[0].direction == "decreasing"
        assert results[0].slope < 0
        assert len(findings) == 1
        assert findings[0].measured_values["direction"] == "decreasing"

    def test_stable_fluctuation_generates_no_directional_finding(self) -> None:
        # Flat / random fluctuation with low R^2
        dates = pd.date_range("2024-01-01", periods=6, freq="MS").strftime("%Y-%m-%d").tolist()
        df = pd.DataFrame({
            "log_date": dates,
            "metric": [10.0, 12.0, 9.0, 11.0, 10.0, 10.5],
        })
        profile = profile_dataset(Dataset(df, "flat.csv", "csv", 6, 2))

        results, findings = TrendAnalyzer(r2_threshold=0.5).analyze(df, profile)

        assert len(results) == 1
        assert results[0].direction == "stable"
        # Stable trend does not generate directional finding
        assert len(findings) == 0

    def test_insufficient_periods_handled_safely(self) -> None:
        df = pd.DataFrame({
            "date": ["2024-01-01", "2024-01-02"],
            "val": [10.0, 20.0],
        })
        profile = profile_dataset(Dataset(df, "short.csv", "csv", 2, 2))

        results, findings = TrendAnalyzer(min_periods=3).analyze(df, profile)
        assert len(results) == 0
        assert len(findings) == 0
