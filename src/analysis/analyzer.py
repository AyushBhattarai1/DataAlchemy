"""Statistical Analysis Orchestrator for DataAlchemy.

Coordinates correlation analysis, distribution assessments, categorical patterns,
group relationships, and temporal trends into an integrated, deterministic AnalysisReport.
"""

from dataclasses import asdict, dataclass, field
import json
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from src.analysis.correlations import CorrelationAnalyzer, CorrelationResult
from src.analysis.distributions import DistributionAnalyzer, DistributionResult
from src.analysis.findings import Finding, FindingImportance, FindingType
from src.analysis.relationships import RelationshipAnalyzer, RelationshipResult
from src.analysis.trends import TrendAnalyzer, TrendResult
from src.ingestion.loader import Dataset
from src.profiling.profiler import DatasetProfile, profile_dataset
from src.quality.report import DataQualityReport


# -----------------------------------------------------------------------------
# Analysis Report Representation
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class AnalysisReport:
    """Comprehensive statistical intelligence report containing structured findings."""
    dataset_name: str
    row_count: int
    column_count: int
    findings: List[Finding]
    correlations: List[CorrelationResult]
    distributions: List[DistributionResult]
    relationships: List[RelationshipResult]
    trends: List[TrendResult]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_findings_by_type(self, finding_type: Union[FindingType, str]) -> List[Finding]:
        """Filter findings by finding type."""
        target_val = finding_type.value if isinstance(finding_type, FindingType) else str(finding_type)
        return [f for f in self.findings if f.finding_type.value == target_val]

    def get_findings_by_importance(self, importance: Union[FindingImportance, str]) -> List[Finding]:
        """Filter findings by importance level."""
        target_val = importance.value if isinstance(importance, FindingImportance) else str(importance)
        return [f for f in self.findings if f.importance.value == target_val]

    def get_findings_by_column(self, column_name: str) -> List[Finding]:
        """Filter findings that reference a specific column."""
        return [f for f in self.findings if column_name in f.affected_columns]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize complete analysis report into a JSON-compatible dictionary."""
        return {
            "dataset_name": self.dataset_name,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "total_findings": len(self.findings),
            "findings": [f.to_dict() for f in self.findings],
            "correlations": [c.to_dict() for c in self.correlations],
            "distributions": [d.to_dict() for d in self.distributions],
            "relationships": [r.to_dict() for r in self.relationships],
            "trends": [t.to_dict() for t in self.trends],
            "metadata": self.metadata,
        }

    def to_json(self, indent: int = 2) -> str:
        """Export report directly to a formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def summary(self) -> str:
        """Produce a clean human-readable ASCII summary of key statistical findings."""
        lines = [
            "STATISTICAL ANALYSIS REPORT",
            "=" * 35,
            f"Dataset:            {self.dataset_name}",
            f"Dimensions:         {self.row_count:,} rows x {self.column_count:,} columns",
            f"Total Findings:     {len(self.findings)}",
            "",
            "Findings Breakdown by Type:",
        ]

        # Breakdown by type
        for ft in FindingType:
            count = len(self.get_findings_by_type(ft))
            if count > 0:
                lines.append(f"  - {ft.value:<22}: {count}")

        lines.extend([
            "",
            "Top Statistical Discoveries:",
        ])

        if not self.findings:
            lines.append("  No significant statistical patterns discovered.")
        else:
            for i, f in enumerate(self.findings[:10], 1):
                col_tag = f"[{', '.join(f.affected_columns)}] " if f.affected_columns else ""
                lines.append(f"  {i}. [{f.importance.value}] {col_tag}{f.title}")
                lines.append(f"     {f.description}")

        lines.append("=" * 35)
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"AnalysisReport(dataset='{self.dataset_name}', "
            f"findings={len(self.findings)}, "
            f"correlations={len(self.correlations)}, "
            f"trends={len(self.trends)})"
        )


# -----------------------------------------------------------------------------
# Statistical Analyzer Orchestrator
# -----------------------------------------------------------------------------

class StatisticalAnalyzer:
    """Coordinates deterministic statistical discovery across multiple dimensions."""

    def __init__(
        self,
        min_correlation: float = 0.3,
        max_correlation_pairs: int = 50,
        strong_skew_threshold: float = 1.0,
        concentration_threshold_pct: float = 50.0,
        trend_r2_threshold: float = 0.25,
    ) -> None:
        self.correlation_analyzer = CorrelationAnalyzer(
            min_correlation=min_correlation,
            max_pairs=max_correlation_pairs,
        )
        self.distribution_analyzer = DistributionAnalyzer(
            strong_skew_threshold=strong_skew_threshold,
        )
        self.relationship_analyzer = RelationshipAnalyzer(
            concentration_threshold_pct=concentration_threshold_pct,
        )
        self.trend_analyzer = TrendAnalyzer(
            r2_threshold=trend_r2_threshold,
        )

    def analyze(
        self,
        dataset: Dataset,
        profile: Optional[DatasetProfile] = None,
        quality_report: Optional[DataQualityReport] = None,
    ) -> AnalysisReport:
        """Run all exploratory intelligence modules on the dataset.

        Args:
            dataset: The standardized Dataset object from Step 2 ingestion.
            profile: Optional precomputed DatasetProfile from Step 3 profiling.
            quality_report: Optional precomputed DataQualityReport from Step 4.

        Returns:
            AnalysisReport containing structured statistical findings and evidence.
        """
        # Ensure profile is available
        if profile is None:
            profile = profile_dataset(dataset)

        df = dataset.dataframe
        all_findings: Dict[str, Finding] = {}

        # 1. Numerical Correlations
        corr_results, corr_findings = self.correlation_analyzer.analyze(df, profile)
        for f in corr_findings:
            all_findings[f.finding_id] = f

        # 2. Distributions & Skewness
        dist_results, dist_findings = self.distribution_analyzer.analyze(profile)
        for f in dist_findings:
            all_findings[f.finding_id] = f

        # 3. Categorical Concentration
        cat_findings = self.relationship_analyzer.analyze_categories(profile)
        for f in cat_findings:
            all_findings[f.finding_id] = f

        # 4. Group Relationships (Numerical by Categorical)
        rel_results, rel_findings = self.relationship_analyzer.analyze_relationships(df, profile)
        for f in rel_findings:
            all_findings[f.finding_id] = f

        # 5. Temporal Trends
        trend_results, trend_findings = self.trend_analyzer.analyze(df, profile)
        for f in trend_findings:
            all_findings[f.finding_id] = f

        # Deterministic sorting of findings (by importance rank, then finding_id)
        sorted_findings = sorted(all_findings.values())

        metadata: Dict[str, Any] = {
            "has_quality_report": quality_report is not None,
            "correlations_calculated": len(corr_results),
            "distributions_evaluated": len(dist_results),
            "relationships_evaluated": len(rel_results),
            "trends_evaluated": len(trend_results),
        }

        return AnalysisReport(
            dataset_name=dataset.filename,
            row_count=dataset.row_count,
            column_count=dataset.column_count,
            findings=sorted_findings,
            correlations=corr_results,
            distributions=dist_results,
            relationships=rel_results,
            trends=trend_results,
            metadata=metadata,
        )


def analyze_dataset(
    dataset: Dataset,
    profile: Optional[DatasetProfile] = None,
    quality_report: Optional[DataQualityReport] = None,
) -> AnalysisReport:
    """Convenience function to perform exploratory intelligence on a dataset.

    Args:
        dataset: The standardized Dataset object from Step 2 ingestion.
        profile: Optional precomputed DatasetProfile from Step 3.
        quality_report: Optional precomputed DataQualityReport from Step 4.

    Returns:
        AnalysisReport: Complete statistical intelligence report.
    """
    return StatisticalAnalyzer().analyze(
        dataset=dataset,
        profile=profile,
        quality_report=quality_report,
    )
