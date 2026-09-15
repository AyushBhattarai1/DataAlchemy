"""Data Quality Analyzer Engine for DataAlchemy.

Executes deterministic, rule-based quality evaluation against a DatasetProfile
to compute explainable quality scores, grade levels, and structured DataQualityReports.
"""

from typing import Dict, List, Optional, Tuple

from src.profiling.profiler import DatasetProfile
from src.quality.report import DataQualityReport, QualityIssue, Severity
from src.quality.rules import (
    ConstantColumnRule,
    DuplicateRowRule,
    HighCardinalityRule,
    IdentifierFeatureRule,
    LowVarianceRule,
    MissingnessRule,
    NumericalSanityRule,
    QualityRule,
)


class QualityAnalyzer:
    """Orchestrates quality evaluation rules and computes deterministic health scores."""

    DEFAULT_RULES = (
        MissingnessRule,
        DuplicateRowRule,
        ConstantColumnRule,
        LowVarianceRule,
        HighCardinalityRule,
        IdentifierFeatureRule,
        NumericalSanityRule,
    )

    def __init__(self, rules: Optional[List[QualityRule]] = None) -> None:
        """Initialize analyzer with custom or default rules.

        Args:
            rules: Optional list of QualityRule instances. If None, default rules are used.
        """
        if rules is not None:
            self.rules = rules
        else:
            self.rules = [rule_cls() for rule_cls in self.DEFAULT_RULES]

    def _determine_grade(self, score: float) -> str:
        """Map numerical score to transparent qualitative grade."""
        if score >= 90.0:
            return "Excellent"
        elif score >= 75.0:
            return "Good"
        elif score >= 60.0:
            return "Fair"
        elif score >= 40.0:
            return "Poor"
        else:
            return "Critical"

    def _calculate_scores(
        self, profile: DatasetProfile, issues: List[QualityIssue]
    ) -> Tuple[float, Dict[str, float], Dict[str, float]]:
        """Calculate deterministic overall and column-level quality scores.

        Returns:
            Tuple of (overall_score, column_scores, score_breakdown).
        """
        # 1. Dataset-level penalty dimensions
        # Completeness penalty: based on overall missing cells % (max 35 pts)
        completeness_penalty = min(35.0, (profile.missing_percentage / 100.0) * 40.0)

        # Uniqueness penalty: based on duplicate rows % (max 25 pts)
        duplicate_penalty = min(25.0, (profile.duplicate_percentage / 100.0) * 35.0)

        # Usability penalties from column-level critical issues
        all_null_count = sum(1 for issue in issues if issue.rule_id == "ALL_NULL")
        constant_count = sum(1 for issue in issues if issue.rule_id == "CONSTANT_COLUMN")
        invalid_num_count = sum(1 for issue in issues if issue.rule_id == "INVALID_NUMERICAL_VALUES")
        low_var_count = sum(1 for issue in issues if issue.rule_id == "LOW_VARIANCE")

        usability_penalty = min(30.0, (all_null_count * 15.0) + (constant_count * 5.0))
        validity_penalty = min(10.0, invalid_num_count * 10.0)
        low_info_penalty = min(10.0, low_var_count * 2.5)

        total_penalty = (
            completeness_penalty
            + duplicate_penalty
            + usability_penalty
            + validity_penalty
            + low_info_penalty
        )

        overall_score = max(0.0, min(100.0, round(100.0 - total_penalty, 1)))

        score_breakdown = {
            "base_score": 100.0,
            "completeness_deduction": round(completeness_penalty, 1),
            "uniqueness_deduction": round(duplicate_penalty, 1),
            "usability_deduction": round(usability_penalty, 1),
            "validity_deduction": round(validity_penalty, 1),
            "low_info_deduction": round(low_info_penalty, 1),
            "final_score": overall_score,
        }

        # 2. Column-level quality scores
        column_scores: Dict[str, float] = {}
        for col_name, cp in profile.columns.items():
            if cp.missing_percentage >= 100.0:
                column_scores[col_name] = 0.0
                continue

            col_penalty = 0.0
            # Missingness penalty on column
            col_penalty += min(50.0, cp.missing_percentage * 0.6)

            # Check issues targeting this column
            col_issues = [iss for iss in issues if iss.column == col_name]
            for iss in col_issues:
                if iss.rule_id == "CONSTANT_COLUMN":
                    col_penalty += 25.0
                elif iss.rule_id == "LOW_VARIANCE":
                    col_penalty += 10.0
                elif iss.rule_id == "INVALID_NUMERICAL_VALUES":
                    col_penalty += 30.0
                elif iss.rule_id == "HIGH_CARDINALITY":
                    col_penalty += 5.0
                # Identifier columns receive NO penalty (informational only)

            col_score = max(0.0, min(100.0, round(100.0 - col_penalty, 1)))
            column_scores[col_name] = col_score

        return overall_score, column_scores, score_breakdown

    def analyze(self, profile: DatasetProfile) -> DataQualityReport:
        """Run all configured quality rules on a DatasetProfile.

        Args:
            profile: DatasetProfile instance from Step 3 profiling.

        Returns:
            DataQualityReport: Comprehensive data quality analysis and metrics.
        """
        issues: List[QualityIssue] = []

        # Execute rules
        for rule in self.rules:
            rule_issues = rule.evaluate(profile)
            issues.extend(rule_issues)

        # Count severity
        critical_count = sum(1 for iss in issues if iss.severity == Severity.CRITICAL)
        error_count = sum(1 for iss in issues if iss.severity == Severity.ERROR)
        warning_count = sum(1 for iss in issues if iss.severity == Severity.WARNING)
        info_count = sum(1 for iss in issues if iss.severity == Severity.INFO)

        # Calculate scores and grade
        overall_score, column_scores, breakdown = self._calculate_scores(profile, issues)
        grade = self._determine_grade(overall_score)

        return DataQualityReport(
            dataset_name=profile.dataset_name,
            overall_score=overall_score,
            grade=grade,
            total_issues=len(issues),
            critical_issues=critical_count,
            error_issues=error_count,
            warning_issues=warning_count,
            info_issues=info_count,
            issues=issues,
            column_scores=column_scores,
            score_breakdown=breakdown,
        )


def analyze_quality(profile: DatasetProfile) -> DataQualityReport:
    """Convenience function to analyze data quality of a DatasetProfile.

    Args:
        profile: The DatasetProfile object from Step 3 profiling.

    Returns:
        DataQualityReport: Quality evaluation with issues, scores, and grade.
    """
    return QualityAnalyzer().analyze(profile)
