"""Unit tests for individual data quality rules in src/quality/rules.py."""

import pandas as pd
import pytest

from src.ingestion.loader import Dataset
from src.profiling.profiler import profile_dataset
from src.quality.report import Severity
from src.quality.rules import (
    ConstantColumnRule,
    DuplicateRowRule,
    HighCardinalityRule,
    IdentifierFeatureRule,
    LowVarianceRule,
    MissingnessRule,
    NumericalSanityRule,
)


class TestMissingnessRule:
    """Test MissingnessRule column and dataset tiers."""

    def test_clean_dataset_no_missing_issues(self) -> None:
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        profile = profile_dataset(Dataset(df, "clean.csv", "csv", 3, 2))
        rule = MissingnessRule()
        issues = rule.evaluate(profile)
        assert len(issues) == 0

    def test_moderate_missingness_tier(self) -> None:
        # 1 out of 10 = 10% (between 5% and 20%) -> MODERATE_MISSINGNESS (INFO)
        df = pd.DataFrame({"col": [1, 2, 3, 4, 5, 6, 7, 8, 9, None]})
        profile = profile_dataset(Dataset(df, "data.csv", "csv", 10, 1))
        issues = MissingnessRule().evaluate(profile)

        assert len(issues) == 1
        assert issues[0].rule_id == "MODERATE_MISSINGNESS"
        assert issues[0].severity == Severity.INFO
        assert issues[0].column == "col"

    def test_high_missingness_tier(self) -> None:
        # 3 out of 10 = 30% (between 20% and 50%) -> HIGH_MISSINGNESS (WARNING)
        df = pd.DataFrame({"col": [1, 2, 3, 4, 5, 6, 7, None, None, None]})
        profile = profile_dataset(Dataset(df, "data.csv", "csv", 10, 1))
        issues = MissingnessRule().evaluate(profile)

        col_issues = [i for i in issues if i.column == "col"]
        assert len(col_issues) == 1
        assert col_issues[0].rule_id == "HIGH_MISSINGNESS"
        assert col_issues[0].severity == Severity.WARNING

    def test_extreme_missingness_tier(self) -> None:
        # 7 out of 10 = 70% (between 50% and 99%) -> EXTREME_MISSINGNESS (ERROR)
        df = pd.DataFrame({"col": [1, 2, 3, None, None, None, None, None, None, None]})
        profile = profile_dataset(Dataset(df, "data.csv", "csv", 10, 1))
        issues = MissingnessRule().evaluate(profile)

        # Triggers column EXTREME_MISSINGNESS and dataset-wide high missingness (>20%)
        col_issues = [i for i in issues if i.column == "col"]
        assert len(col_issues) == 1
        assert col_issues[0].rule_id == "EXTREME_MISSINGNESS"
        assert col_issues[0].severity == Severity.ERROR

    def test_all_null_tier(self) -> None:
        # 100% missing -> ALL_NULL (CRITICAL)
        df = pd.DataFrame({"col": [None, None, None]})
        profile = profile_dataset(Dataset(df, "data.csv", "csv", 3, 1))
        issues = MissingnessRule().evaluate(profile)

        all_null = [i for i in issues if i.rule_id == "ALL_NULL"]
        assert len(all_null) == 1
        assert all_null[0].severity == Severity.CRITICAL


class TestDuplicateRowRule:
    """Test DuplicateRowRule thresholds."""

    def test_no_duplicates(self) -> None:
        df = pd.DataFrame({"id": [1, 2, 3], "val": ["a", "b", "c"]})
        profile = profile_dataset(Dataset(df, "data.csv", "csv", 3, 2))
        issues = DuplicateRowRule().evaluate(profile)
        assert len(issues) == 0

    def test_small_duplicates_info(self) -> None:
        # 1 duplicate in 25 rows = 4% (<= 5%) -> INFO
        rows = [{"id": i, "val": "A"} for i in range(24)]
        rows.append({"id": 0, "val": "A"})  # 1 duplicate
        df = pd.DataFrame(rows)
        profile = profile_dataset(Dataset(df, "data.csv", "csv", 25, 2))
        issues = DuplicateRowRule().evaluate(profile)

        assert len(issues) == 1
        assert issues[0].severity == Severity.INFO

    def test_moderate_duplicates_warning(self) -> None:
        # 2 duplicates in 20 rows = 10% (between 5% and 20%) -> WARNING
        rows = [{"id": i, "val": "A"} for i in range(18)]
        rows.extend([{"id": 0, "val": "A"}, {"id": 1, "val": "A"}])
        df = pd.DataFrame(rows)
        profile = profile_dataset(Dataset(df, "data.csv", "csv", 20, 2))
        issues = DuplicateRowRule().evaluate(profile)

        assert len(issues) == 1
        assert issues[0].severity == Severity.WARNING

    def test_critical_duplicates(self) -> None:
        # 6 duplicates in 10 rows = 60% (> 50%) -> CRITICAL
        df = pd.DataFrame({"id": [1, 2, 3, 4, 1, 1, 1, 1, 1, 1]})
        profile = profile_dataset(Dataset(df, "data.csv", "csv", 10, 1))
        issues = DuplicateRowRule().evaluate(profile)

        assert len(issues) == 1
        assert issues[0].severity == Severity.CRITICAL


class TestConstantColumnRule:
    """Test ConstantColumnRule detection."""

    def test_normal_column_no_issue(self) -> None:
        df = pd.DataFrame({"col": [1, 2, 3, 4]})
        profile = profile_dataset(Dataset(df, "data.csv", "csv", 4, 1))
        issues = ConstantColumnRule().evaluate(profile)
        assert len(issues) == 0

    def test_constant_column_detected(self) -> None:
        df = pd.DataFrame({"const_col": ["active", "active", "active", "active"]})
        profile = profile_dataset(Dataset(df, "data.csv", "csv", 4, 1))
        issues = ConstantColumnRule().evaluate(profile)

        assert len(issues) == 1
        assert issues[0].rule_id == "CONSTANT_COLUMN"
        assert issues[0].severity == Severity.WARNING
        assert issues[0].column == "const_col"


class TestLowVarianceRule:
    """Test LowVarianceRule for high single-category dominance."""

    def test_balanced_distribution_no_issue(self) -> None:
        df = pd.DataFrame({"category": ["A", "B", "A", "B", "C"]})
        profile = profile_dataset(Dataset(df, "data.csv", "csv", 5, 1))
        issues = LowVarianceRule(dominance_threshold_pct=95.0).evaluate(profile)
        assert len(issues) == 0

    def test_dominated_distribution_triggers_warning(self) -> None:
        # 98 out of 100 are 'A' (98%) -> triggers >= 95% threshold
        items = ["A"] * 98 + ["B", "C"]
        df = pd.DataFrame({"category": items})
        profile = profile_dataset(Dataset(df, "data.csv", "csv", 100, 1))
        issues = LowVarianceRule(dominance_threshold_pct=95.0).evaluate(profile)

        assert len(issues) == 1
        assert issues[0].rule_id == "LOW_VARIANCE"
        assert issues[0].severity == Severity.WARNING
        assert issues[0].column == "category"


class TestHighCardinalityRule:
    """Test HighCardinalityRule for non-identifier categorical columns."""

    def test_low_cardinality_no_issue(self) -> None:
        df = pd.DataFrame({"status": ["pending", "approved", "rejected"] * 20})
        profile = profile_dataset(Dataset(df, "data.csv", "csv", 60, 1))
        issues = HighCardinalityRule(min_unique_count=50).evaluate(profile)
        assert len(issues) == 0

    def test_high_cardinality_triggers_warning(self) -> None:
        # 60 unique descriptions in 60 rows = 100% uniqueness
        df = pd.DataFrame({"description": [f"Item description note #{i}" for i in range(60)]})
        profile = profile_dataset(Dataset(df, "data.csv", "csv", 60, 1))
        issues = HighCardinalityRule(min_unique_count=50, uniqueness_threshold_pct=80.0).evaluate(profile)

        assert len(issues) == 1
        assert issues[0].rule_id == "HIGH_CARDINALITY"
        assert issues[0].column == "description"


class TestIdentifierFeatureRule:
    """Test IdentifierFeatureRule detection."""

    def test_identifier_detected_as_info(self) -> None:
        df = pd.DataFrame({"customer_id": [f"C_{i}" for i in range(10)]})
        profile = profile_dataset(Dataset(df, "data.csv", "csv", 10, 1))
        issues = IdentifierFeatureRule().evaluate(profile)

        assert len(issues) == 1
        assert issues[0].rule_id == "IDENTIFIER_FEATURE"
        assert issues[0].severity == Severity.INFO
        assert issues[0].column == "customer_id"


class TestNumericalSanityRule:
    """Test NumericalSanityRule for non-finite values."""

    def test_normal_numerical_no_issue(self) -> None:
        df = pd.DataFrame({"val": [1.0, 2.5, -3.0, 0.0]})
        profile = profile_dataset(Dataset(df, "data.csv", "csv", 4, 1))
        issues = NumericalSanityRule().evaluate(profile)
        assert len(issues) == 0
