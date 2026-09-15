"""Quality Rules Engine for DataAlchemy.

Defines independent, modular, and configurable rules for evaluating data quality
defects from a DatasetProfile without mutating data.
"""

from abc import ABC, abstractmethod
import math
from typing import Any, Dict, List, Optional

from src.profiling.profiler import DatasetProfile
from src.profiling.statistics import (
    BooleanStats,
    CategoricalStats,
    NumericalStats,
)
from src.quality.report import QualityIssue, Severity


# -----------------------------------------------------------------------------
# Base Quality Rule
# -----------------------------------------------------------------------------

class QualityRule(ABC):
    """Abstract base class for all data quality rules."""
    rule_id: str
    name: str
    description: str

    @abstractmethod
    def evaluate(self, profile: DatasetProfile) -> List[QualityIssue]:
        """Evaluate the rule against a DatasetProfile.

        Args:
            profile: Standardized dataset profile from Step 3.

        Returns:
            List of QualityIssue instances found.
        """
        pass


# -----------------------------------------------------------------------------
# Missingness Rule
# -----------------------------------------------------------------------------

class MissingnessRule(QualityRule):
    """Detects missing value anomalies at column and dataset levels."""
    rule_id = "MISSINGNESS"
    name = "Missing Values Rule"
    description = "Evaluates columns and overall dataset for excessive missingness."

    def __init__(
        self,
        moderate_pct: float = 5.0,
        high_pct: float = 20.0,
        extreme_pct: float = 50.0,
        dataset_high_pct: float = 20.0,
    ) -> None:
        self.moderate_pct = moderate_pct
        self.high_pct = high_pct
        self.extreme_pct = extreme_pct
        self.dataset_high_pct = dataset_high_pct

    def evaluate(self, profile: DatasetProfile) -> List[QualityIssue]:
        issues: List[QualityIssue] = []

        # 1. Column-level missingness (highest tier applies per column to avoid double-counting)
        for col_name, cp in profile.columns.items():
            pct = cp.missing_percentage
            count = cp.missing_count

            if pct >= 100.0:
                issues.append(QualityIssue(
                    rule_id="ALL_NULL",
                    severity=Severity.CRITICAL,
                    title="All-Null Column",
                    description=f"Column '{col_name}' contains 100% missing values.",
                    column=col_name,
                    measured_value=f"{pct:.1f}%",
                    threshold="100.0%",
                    impact="Column provides zero information and is completely unusable in its current state.",
                    metadata={"missing_count": count, "missing_percentage": pct},
                ))
            elif pct > self.extreme_pct:
                issues.append(QualityIssue(
                    rule_id="EXTREME_MISSINGNESS",
                    severity=Severity.ERROR,
                    title="Extreme Missingness",
                    description=f"Column '{col_name}' has {pct:.1f}% missing values (exceeds {self.extreme_pct}%).",
                    column=col_name,
                    measured_value=f"{pct:.1f}%",
                    threshold=f">{self.extreme_pct}%",
                    impact="More than half the column is missing, likely requiring severe imputation or drop.",
                    metadata={"missing_count": count, "missing_percentage": pct},
                ))
            elif pct > self.high_pct:
                issues.append(QualityIssue(
                    rule_id="HIGH_MISSINGNESS",
                    severity=Severity.WARNING,
                    title="High Missingness",
                    description=f"Column '{col_name}' has {pct:.1f}% missing values (exceeds {self.high_pct}%).",
                    column=col_name,
                    measured_value=f"{pct:.1f}%",
                    threshold=f">{self.high_pct}%",
                    impact="High proportion of missing values may bias analysis or models.",
                    metadata={"missing_count": count, "missing_percentage": pct},
                ))
            elif pct > self.moderate_pct:
                issues.append(QualityIssue(
                    rule_id="MODERATE_MISSINGNESS",
                    severity=Severity.INFO,
                    title="Moderate Missingness",
                    description=f"Column '{col_name}' has {pct:.1f}% missing values.",
                    column=col_name,
                    measured_value=f"{pct:.1f}%",
                    threshold=f">{self.moderate_pct}%",
                    impact="Mild missingness requiring standard handling or imputation.",
                    metadata={"missing_count": count, "missing_percentage": pct},
                ))

        # 2. Dataset-wide missingness
        if profile.missing_percentage > self.dataset_high_pct:
            issues.append(QualityIssue(
                rule_id="DATASET_HIGH_MISSINGNESS",
                severity=Severity.WARNING,
                title="High Dataset-Wide Missingness",
                description=(
                    f"Dataset overall missingness is {profile.missing_percentage:.1f}% "
                    f"({profile.missing_cells:,} missing cells)."
                ),
                column=None,
                measured_value=f"{profile.missing_percentage:.1f}%",
                threshold=f">{self.dataset_high_pct}%",
                impact="Overall dataset sparsity may compromise multi-variable analysis.",
                metadata={
                    "total_missing_cells": profile.missing_cells,
                    "missing_percentage": profile.missing_percentage,
                },
            ))

        return issues


# -----------------------------------------------------------------------------
# Duplicate Rows Rule
# -----------------------------------------------------------------------------

class DuplicateRowRule(QualityRule):
    """Detects duplicate rows across the dataset."""
    rule_id = "DUPLICATE_ROWS"
    name = "Duplicate Rows Rule"
    description = "Checks dataset for repeated identical rows."

    def __init__(
        self,
        warning_pct: float = 5.0,
        error_pct: float = 20.0,
        critical_pct: float = 50.0,
    ) -> None:
        self.warning_pct = warning_pct
        self.error_pct = error_pct
        self.critical_pct = critical_pct

    def evaluate(self, profile: DatasetProfile) -> List[QualityIssue]:
        issues: List[QualityIssue] = []
        pct = profile.duplicate_percentage
        count = profile.duplicate_rows

        if count == 0:
            return issues

        if pct > self.critical_pct:
            severity = Severity.CRITICAL
        elif pct > self.error_pct:
            severity = Severity.ERROR
        elif pct > self.warning_pct:
            severity = Severity.WARNING
        else:
            severity = Severity.INFO

        issues.append(QualityIssue(
            rule_id="DUPLICATE_ROWS",
            severity=severity,
            title="Duplicate Rows Detected",
            description=f"Dataset contains {count:,} duplicate rows ({pct:.1f}% of total).",
            column=None,
            measured_value=f"{pct:.1f}%",
            threshold=f">0%",
            impact="Duplicate rows can falsely inflate metric counts and cause statistical bias.",
            metadata={"duplicate_rows": count, "duplicate_percentage": pct},
        ))
        return issues


# -----------------------------------------------------------------------------
# Constant Column Rule
# -----------------------------------------------------------------------------

class ConstantColumnRule(QualityRule):
    """Detects columns where all non-null values are identical."""
    rule_id = "CONSTANT_COLUMN"
    name = "Constant Column Rule"
    description = "Identifies zero-variance columns containing only a single distinct value."

    def evaluate(self, profile: DatasetProfile) -> List[QualityIssue]:
        issues: List[QualityIssue] = []

        if profile.row_count <= 1:
            return issues

        for col_name, cp in profile.columns.items():
            # If all null, handled by ALL_NULL in MissingnessRule
            if cp.missing_percentage >= 100.0:
                continue

            if cp.unique_count == 1:
                # Find dominant value
                dom_val = "constant"
                if isinstance(cp.stats, CategoricalStats) and cp.stats.most_frequent is not None:
                    dom_val = str(cp.stats.most_frequent)
                elif isinstance(cp.stats, NumericalStats) and cp.stats.min is not None:
                    dom_val = str(cp.stats.min)
                elif isinstance(cp.stats, BooleanStats):
                    dom_val = "True" if cp.stats.true_count > 0 else "False"

                issues.append(QualityIssue(
                    rule_id="CONSTANT_COLUMN",
                    severity=Severity.WARNING,
                    title="Constant Column",
                    description=f"Column '{col_name}' has zero variance with constant value: '{dom_val}'.",
                    column=col_name,
                    measured_value=1,
                    threshold=">1 unique values",
                    impact="Zero variance feature provides no discriminative or predictive information.",
                    metadata={"dominant_value": dom_val, "unique_count": 1},
                ))

        return issues


# -----------------------------------------------------------------------------
# Low Variance / High Dominance Rule
# -----------------------------------------------------------------------------

class LowVarianceRule(QualityRule):
    """Detects columns where a single value overwhelmingly dominates."""
    rule_id = "LOW_VARIANCE"
    name = "Low Variance Rule"
    description = "Identifies columns overwhelmingly dominated by a single value (e.g. >= 95%)."

    def __init__(self, dominance_threshold_pct: float = 95.0) -> None:
        self.dominance_threshold_pct = dominance_threshold_pct

    def evaluate(self, profile: DatasetProfile) -> List[QualityIssue]:
        issues: List[QualityIssue] = []

        for col_name, cp in profile.columns.items():
            # Skip constant or all-null columns
            if cp.unique_count <= 1 or cp.missing_percentage >= 100.0:
                continue

            non_null_rows = cp.row_count - cp.missing_count
            if non_null_rows <= 1:
                continue

            dominant_val = None
            dominant_freq = 0

            if isinstance(cp.stats, CategoricalStats) and cp.stats.most_frequent_frequency:
                dominant_val = str(cp.stats.most_frequent)
                dominant_freq = cp.stats.most_frequent_frequency
            elif isinstance(cp.stats, BooleanStats):
                if cp.stats.true_count >= cp.stats.false_count:
                    dominant_val = "True"
                    dominant_freq = cp.stats.true_count
                else:
                    dominant_val = "False"
                    dominant_freq = cp.stats.false_count

            if dominant_freq > 0:
                dom_pct = round((dominant_freq / non_null_rows) * 100, 2)
                if dom_pct >= self.dominance_threshold_pct:
                    issues.append(QualityIssue(
                        rule_id="LOW_VARIANCE",
                        severity=Severity.WARNING,
                        title="Low Variance / High Dominance",
                        description=(
                            f"Column '{col_name}' is {dom_pct:.1f}% dominated by value '{dominant_val}'."
                        ),
                        column=col_name,
                        measured_value=f"{dom_pct:.1f}%",
                        threshold=f">={self.dominance_threshold_pct}%",
                        impact="Near-constant columns provide minimal information for analytical tasks.",
                        metadata={
                            "dominant_value": dominant_val,
                            "dominant_frequency": dominant_freq,
                            "dominance_percentage": dom_pct,
                            "unique_count": cp.unique_count,
                        },
                    ))

        return issues


# -----------------------------------------------------------------------------
# High Cardinality Rule
# -----------------------------------------------------------------------------

class HighCardinalityRule(QualityRule):
    """Detects categorical columns with excessively high distinct value counts."""
    rule_id = "HIGH_CARDINALITY"
    name = "High Cardinality Rule"
    description = "Detects categorical columns with extreme cardinality unsuitable for basic categorical encoding."

    def __init__(
        self,
        uniqueness_threshold_pct: float = 80.0,
        min_unique_count: int = 50,
    ) -> None:
        self.uniqueness_threshold_pct = uniqueness_threshold_pct
        self.min_unique_count = min_unique_count

    def evaluate(self, profile: DatasetProfile) -> List[QualityIssue]:
        issues: List[QualityIssue] = []

        for col_name, cp in profile.columns.items():
            # Only evaluate categorical columns (identifiers have their own separate rule)
            if cp.semantic_type != "categorical":
                continue

            if cp.unique_count >= self.min_unique_count and cp.unique_percentage >= self.uniqueness_threshold_pct:
                issues.append(QualityIssue(
                    rule_id="HIGH_CARDINALITY",
                    severity=Severity.WARNING,
                    title="High Cardinality Categorical",
                    description=(
                        f"Categorical column '{col_name}' has {cp.unique_count:,} unique values "
                        f"({cp.unique_percentage:.1f}% uniqueness)."
                    ),
                    column=col_name,
                    measured_value=f"{cp.unique_percentage:.1f}%",
                    threshold=f">={self.uniqueness_threshold_pct}% and >={self.min_unique_count} distinct",
                    impact="High cardinality may degrade model performance or blow up one-hot encoding dimensions.",
                    metadata={"unique_count": cp.unique_count, "unique_percentage": cp.unique_percentage},
                ))

        return issues


# -----------------------------------------------------------------------------
# Identifier Feature Rule
# -----------------------------------------------------------------------------

class IdentifierFeatureRule(QualityRule):
    """Identifies columns functioning as entity keys rather than analytical features."""
    rule_id = "IDENTIFIER_FEATURE"
    name = "Identifier Feature Rule"
    description = "Notes columns identified as primary/foreign keys or entity identifiers."

    def evaluate(self, profile: DatasetProfile) -> List[QualityIssue]:
        issues: List[QualityIssue] = []

        for col_name, cp in profile.columns.items():
            if cp.semantic_type == "identifier":
                issues.append(QualityIssue(
                    rule_id="IDENTIFIER_FEATURE",
                    severity=Severity.INFO,
                    title="Identifier Column Detected",
                    description=(
                        f"Column '{col_name}' appears to be an entity identifier / key "
                        f"and may not provide predictive signal."
                    ),
                    column=col_name,
                    measured_value="identifier",
                    threshold="semantic_type == 'identifier'",
                    impact="Informational finding: identifiers should typically be excluded from ML feature sets.",
                    metadata={"semantic_type": cp.semantic_type, "unique_count": cp.unique_count},
                ))

        return issues


# -----------------------------------------------------------------------------
# Numerical Sanity Rule
# -----------------------------------------------------------------------------

class NumericalSanityRule(QualityRule):
    """Detects non-finite or invalid numeric values such as +/- infinity."""
    rule_id = "NUMERICAL_SANITY"
    name = "Numerical Sanity Rule"
    description = "Checks numerical columns for infinite or invalid values."

    def evaluate(self, profile: DatasetProfile) -> List[QualityIssue]:
        issues: List[QualityIssue] = []

        for col_name, cp in profile.columns.items():
            if cp.semantic_type != "numerical" or not isinstance(cp.stats, NumericalStats):
                continue

            stats = cp.stats
            has_inf = False

            for bound in [stats.min, stats.max, stats.mean]:
                if bound is not None and (math.isinf(bound) or math.isnan(bound)):
                    has_inf = True
                    break

            if has_inf:
                issues.append(QualityIssue(
                    rule_id="INVALID_NUMERICAL_VALUES",
                    severity=Severity.ERROR,
                    title="Invalid Numerical Values",
                    description=f"Column '{col_name}' contains infinite or non-finite values.",
                    column=col_name,
                    measured_value="infinity/NaN",
                    threshold="finite real numbers",
                    impact="Infinite values cause numerical instability and will break downstream models.",
                    metadata={"min": stats.min, "max": stats.max},
                ))

        return issues
