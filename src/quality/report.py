"""Data Quality Report and Issue Representations for DataAlchemy.

Defines typed containers for individual quality issues, severity classifications,
and the overarching DataQualityReport with scoring and serialization support.
"""

from dataclasses import asdict, dataclass, field
from enum import Enum
import json
from typing import Any, Dict, List, Optional, Union


# -----------------------------------------------------------------------------
# Severity Levels
# -----------------------------------------------------------------------------

class Severity(str, Enum):
    """Severity levels for data quality issues."""
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

    def __str__(self) -> str:
        return self.value


# -----------------------------------------------------------------------------
# Quality Issue Representation
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class QualityIssue:
    """Represents a discrete data quality defect or observational finding."""
    rule_id: str
    severity: Severity
    title: str
    description: str
    column: Optional[str] = None
    measured_value: Optional[Any] = None
    threshold: Optional[Any] = None
    impact: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the issue to a JSON-compatible dictionary."""
        return {
            "rule_id": self.rule_id,
            "severity": self.severity.value,
            "title": self.title,
            "description": self.description,
            "column": self.column,
            "measured_value": self.measured_value,
            "threshold": self.threshold,
            "impact": self.impact,
            "metadata": self.metadata,
        }


# -----------------------------------------------------------------------------
# Data Quality Report Representation
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class DataQualityReport:
    """Comprehensive quality evaluation report for an ingested and profiled dataset."""
    dataset_name: str
    overall_score: float
    grade: str
    total_issues: int
    critical_issues: int
    error_issues: int
    warning_issues: int
    info_issues: int
    issues: List[QualityIssue]
    column_scores: Dict[str, float]
    score_breakdown: Dict[str, float] = field(default_factory=dict)

    def get_issues_by_severity(self, severity: Union[Severity, str]) -> List[QualityIssue]:
        """Filter issues by severity level."""
        target_val = severity.value if isinstance(severity, Severity) else str(severity).upper()
        return [issue for issue in self.issues if issue.severity.value == target_val]

    def get_issues_by_column(self, column_name: str) -> List[QualityIssue]:
        """Filter issues affecting a specific column."""
        return [issue for issue in self.issues if issue.column == column_name]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the complete report to a JSON-compatible dictionary."""
        return {
            "dataset_name": self.dataset_name,
            "overall_score": self.overall_score,
            "grade": self.grade,
            "total_issues": self.total_issues,
            "critical_issues": self.critical_issues,
            "error_issues": self.error_issues,
            "warning_issues": self.warning_issues,
            "info_issues": self.info_issues,
            "issues": [issue.to_dict() for issue in self.issues],
            "column_scores": self.column_scores,
            "score_breakdown": self.score_breakdown,
        }

    def to_json(self, indent: int = 2) -> str:
        """Export report directly to a formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def summary(self) -> str:
        """Generate a clean human-readable ASCII summary of the report."""
        lines = [
            "DATA QUALITY REPORT",
            "=" * 30,
            f"Dataset:            {self.dataset_name}",
            f"Overall Score:      {self.overall_score:.1f}/100",
            f"Grade:              {self.grade}",
            "",
            "Issues Summary:",
            f"  - Critical:       {self.critical_issues}",
            f"  - Errors:         {self.error_issues}",
            f"  - Warnings:       {self.warning_issues}",
            f"  - Info:           {self.info_issues}",
            f"  - Total:          {self.total_issues}",
            "",
            "Major Problems:",
        ]

        if not self.issues:
            lines.append("  None detected (Clean dataset).")
        else:
            for i, issue in enumerate(self.issues[:10], 1):
                col_prefix = f"[{issue.column}] " if issue.column else "[Dataset] "
                lines.append(f"  {i}. {col_prefix}{issue.title} ({issue.severity.value})")
                lines.append(f"     Description: {issue.description}")
                if issue.measured_value is not None:
                    lines.append(f"     Measured: {issue.measured_value} (Threshold: {issue.threshold})")

        lines.append("=" * 30)
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"DataQualityReport(dataset='{self.dataset_name}', "
            f"score={self.overall_score:.1f}, "
            f"grade='{self.grade}', "
            f"issues={self.total_issues})"
        )
