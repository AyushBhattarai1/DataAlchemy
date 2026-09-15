"""Numerical distribution and skewness analysis module for DataAlchemy.

Evaluates distribution characteristics (symmetry, skewness, kurtosis, mean-median spread)
reusing statistics calculated in DatasetProfile without redundant data scans.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

from src.analysis.findings import Finding, FindingImportance, FindingType
from src.profiling.profiler import DatasetProfile
from src.profiling.statistics import NumericalStats


@dataclass(frozen=True)
class DistributionResult:
    """Detailed distribution assessment for a numerical column."""
    column: str
    mean: Optional[float]
    median: Optional[float]
    std: Optional[float]
    skewness: Optional[float]
    kurtosis: Optional[float]
    skewness_classification: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DistributionAnalyzer:
    """Analyzes distribution symmetry, skewness, and tail characteristics."""

    def __init__(
        self,
        strong_skew_threshold: float = 1.0,
        moderate_skew_threshold: float = 0.5,
    ) -> None:
        self.strong_skew_threshold = strong_skew_threshold
        self.moderate_skew_threshold = moderate_skew_threshold

    def _classify_skewness(self, skew: float) -> Tuple[str, FindingImportance]:
        abs_skew = abs(skew)
        if abs_skew >= 2.0:
            direction = "right" if skew > 0 else "left"
            return f"extremely {direction}-skewed", FindingImportance.HIGH
        elif abs_skew >= self.strong_skew_threshold:
            direction = "right" if skew > 0 else "left"
            return f"strongly {direction}-skewed", FindingImportance.MEDIUM
        elif abs_skew >= self.moderate_skew_threshold:
            direction = "right" if skew > 0 else "left"
            return f"moderately {direction}-skewed", FindingImportance.LOW
        else:
            return "approximately symmetric", FindingImportance.INFO

    def analyze(self, profile: DatasetProfile) -> Tuple[List[DistributionResult], List[Finding]]:
        """Evaluate distributions for all numerical columns in profile.

        Args:
            profile: DatasetProfile containing precomputed column statistics.

        Returns:
            Tuple of (List[DistributionResult], List[Finding]).
        """
        results: List[DistributionResult] = []
        findings: List[Finding] = []

        for col_name in sorted(profile.columns.keys()):
            cp = profile.columns[col_name]
            if cp.semantic_type != "numerical" or not isinstance(cp.stats, NumericalStats):
                continue

            # Skip constant or zero-variance columns
            if cp.unique_count <= 1:
                continue

            stats = cp.stats
            if stats.count < 3 or stats.skewness is None or stats.std is None or stats.std == 0.0:
                continue

            skew = stats.skewness
            classification, importance = self._classify_skewness(skew)

            dist_res = DistributionResult(
                column=col_name,
                mean=stats.mean,
                median=stats.median,
                std=stats.std,
                skewness=round(skew, 4),
                kurtosis=round(stats.kurtosis, 4) if stats.kurtosis is not None else None,
                skewness_classification=classification,
            )
            results.append(dist_res)

            # Generate finding for non-trivial skewness or symmetry
            if abs(skew) >= self.moderate_skew_threshold:
                title = f"Column '{col_name}' is {classification}"
                desc = (
                    f"Distribution of '{col_name}' is {classification} "
                    f"(skewness = {skew:.2f}, mean = {stats.mean:.2f}, median = {stats.median:.2f})."
                )
            else:
                title = f"Column '{col_name}' has an approximately symmetric distribution"
                desc = (
                    f"Distribution of '{col_name}' appears approximately symmetric "
                    f"(skewness = {skew:.2f})."
                )

            finding = Finding(
                finding_id=f"dist_{col_name}",
                finding_type=FindingType.DISTRIBUTION,
                title=title,
                description=desc,
                importance=importance,
                affected_columns=[col_name],
                measured_values={
                    "skewness": round(skew, 4),
                    "kurtosis": round(stats.kurtosis, 4) if stats.kurtosis is not None else None,
                    "mean": stats.mean,
                    "median": stats.median,
                    "std": stats.std,
                    "min": stats.min,
                    "max": stats.max,
                },
                statistical_measure="skewness",
                confidence_or_pvalue=None,
                metadata={"classification": classification},
            )
            findings.append(finding)

        return results, findings
