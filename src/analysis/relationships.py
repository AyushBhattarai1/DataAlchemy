"""Categorical distribution and group relationship analysis for DataAlchemy.

Analyzes category concentrations and numerical distributions grouped by categorical
features to uncover segment differences without combinatorial explosion or causal claims.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.analysis.findings import Finding, FindingImportance, FindingType
from src.profiling.profiler import DatasetProfile
from src.profiling.statistics import CategoricalStats


@dataclass(frozen=True)
class GroupSummary:
    """Statistical summary for a numerical variable within a categorical group."""
    group_name: str
    count: int
    mean: float
    median: float
    std: Optional[float]
    min: float
    max: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RelationshipResult:
    """Structured result of a numerical vs categorical relationship analysis."""
    categorical_column: str
    numerical_column: str
    group_summaries: Dict[str, GroupSummary]
    variation_ratio: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "categorical_column": self.categorical_column,
            "numerical_column": self.numerical_column,
            "group_summaries": {k: v.to_dict() for k, v in self.group_summaries.items()},
            "variation_ratio": self.variation_ratio,
        }


class RelationshipAnalyzer:
    """Analyzes categorical patterns and numerical-by-categorical relationships."""

    def __init__(
        self,
        concentration_threshold_pct: float = 50.0,
        max_categorical_columns: int = 5,
        max_numerical_columns: int = 5,
        max_groups_per_category: int = 8,
    ) -> None:
        self.concentration_threshold_pct = concentration_threshold_pct
        self.max_categorical_columns = max_categorical_columns
        self.max_numerical_columns = max_numerical_columns
        self.max_groups_per_category = max_groups_per_category

    def analyze_categories(self, profile: DatasetProfile) -> List[Finding]:
        """Analyze categorical concentration and balance from profile metadata."""
        findings: List[Finding] = []

        for col_name in sorted(profile.columns.keys()):
            cp = profile.columns[col_name]
            if cp.semantic_type != "categorical" or not isinstance(cp.stats, CategoricalStats):
                continue

            stats = cp.stats
            non_null_rows = cp.row_count - cp.missing_count
            if non_null_rows < 5 or not stats.most_frequent_frequency:
                continue

            share = (stats.most_frequent_frequency / non_null_rows) * 100.0
            top_val = str(stats.most_frequent)

            if share >= self.concentration_threshold_pct:
                importance = FindingImportance.HIGH if share >= 80.0 else FindingImportance.MEDIUM
                findings.append(Finding(
                    finding_id=f"cat_concentration_{col_name}",
                    finding_type=FindingType.CATEGORY,
                    title=f"High category concentration in '{col_name}'",
                    description=(
                        f"In column '{col_name}', the dominant category '{top_val}' "
                        f"accounts for {share:.1f}% of observations ({stats.most_frequent_frequency:,}/{non_null_rows:,})."
                    ),
                    importance=importance,
                    affected_columns=[col_name],
                    measured_values={
                        "dominant_category": top_val,
                        "dominant_share_pct": round(share, 2),
                        "dominant_count": stats.most_frequent_frequency,
                        "total_non_null": non_null_rows,
                        "unique_categories": stats.unique_count,
                    },
                    statistical_measure="category_concentration",
                    metadata={"top_values": stats.top_values},
                ))
            elif stats.unique_count >= 3 and share <= 35.0:
                findings.append(Finding(
                    finding_id=f"cat_balanced_{col_name}",
                    finding_type=FindingType.CATEGORY,
                    title=f"Balanced category distribution in '{col_name}'",
                    description=(
                        f"Column '{col_name}' has {stats.unique_count} distinct categories "
                        f"with the top category '{top_val}' representing only {share:.1f}%."
                    ),
                    importance=FindingImportance.INFO,
                    affected_columns=[col_name],
                    measured_values={
                        "dominant_category": top_val,
                        "dominant_share_pct": round(share, 2),
                        "unique_categories": stats.unique_count,
                    },
                    statistical_measure="category_balance",
                    metadata={"top_values": stats.top_values},
                ))

        return findings

    def analyze_relationships(
        self, df: pd.DataFrame, profile: DatasetProfile
    ) -> Tuple[List[RelationshipResult], List[Finding]]:
        """Analyze numerical statistics segmented across categorical groups."""
        results: List[RelationshipResult] = []
        findings: List[Finding] = []

        if len(df) < 5:
            return results, findings

        # Select candidate columns (excluding identifiers, constants, high cardinality > 20)
        cat_cols = [
            c for c, cp in profile.columns.items()
            if cp.semantic_type == "categorical" and 2 <= cp.unique_count <= 20
        ][: self.max_categorical_columns]

        num_cols = [
            c for c, cp in profile.columns.items()
            if cp.semantic_type == "numerical" and cp.unique_count > 1 and cp.missing_percentage < 100.0
        ][: self.max_numerical_columns]

        for cat_col in sorted(cat_cols):
            for num_col in sorted(num_cols):
                sub_df = df[[cat_col, num_col]].dropna()
                if len(sub_df) < 5:
                    continue

                # Group by category
                grouped = sub_df.groupby(cat_col)[num_col]
                group_counts = grouped.count()
                # Keep groups with at least 2 observations
                valid_groups = group_counts[group_counts >= 1].index.tolist()
                if len(valid_groups) < 2:
                    continue

                # Limit to top N largest groups
                top_groups = sub_df[cat_col].value_counts().head(self.max_groups_per_category).index.tolist()
                group_summaries: Dict[str, GroupSummary] = {}
                means: Dict[str, float] = {}

                for grp in top_groups:
                    vals = sub_df[sub_df[cat_col] == grp][num_col]
                    numeric_vals = pd.to_numeric(vals, errors="coerce").dropna()
                    if numeric_vals.empty:
                        continue

                    m = float(numeric_vals.mean())
                    means[str(grp)] = m
                    group_summaries[str(grp)] = GroupSummary(
                        group_name=str(grp),
                        count=len(numeric_vals),
                        mean=round(m, 2),
                        median=round(float(numeric_vals.median()), 2),
                        std=round(float(numeric_vals.std()), 2) if len(numeric_vals) > 1 else None,
                        min=round(float(numeric_vals.min()), 2),
                        max=round(float(numeric_vals.max()), 2),
                    )

                if len(means) < 2:
                    continue

                # Calculate variation ratio
                min_m = min(means.values())
                max_m = max(means.values())
                var_ratio = None

                if min_m > 0:
                    var_ratio = round(max_m / min_m, 2)
                elif min_m < 0 and max_m > 0:
                    var_ratio = round((max_m - min_m) / max(abs(min_m), abs(max_m)), 2)

                rel_res = RelationshipResult(
                    categorical_column=cat_col,
                    numerical_column=num_col,
                    group_summaries=group_summaries,
                    variation_ratio=var_ratio,
                )
                results.append(rel_res)

                # Determine if difference between groups is noteworthy
                if var_ratio is not None and var_ratio >= 1.3:
                    top_grp = max(means, key=means.get)
                    bottom_grp = min(means, key=means.get)
                    importance = FindingImportance.HIGH if var_ratio >= 2.0 else FindingImportance.MEDIUM

                    findings.append(Finding(
                        finding_id=f"rel_{cat_col}_{num_col}",
                        finding_type=FindingType.GROUP_RELATIONSHIP,
                        title=f"'{num_col}' varies substantially by '{cat_col}'",
                        description=(
                            f"Average {num_col} varies across {cat_col} categories "
                            f"(highest in '{top_grp}' at {means[top_grp]:.2f}, "
                            f"lowest in '{bottom_grp}' at {means[bottom_grp]:.2f}, ratio = {var_ratio}x)."
                        ),
                        importance=importance,
                        affected_columns=[cat_col, num_col],
                        measured_values={
                            "variation_ratio": var_ratio,
                            "top_group": top_grp,
                            "top_mean": round(means[top_grp], 2),
                            "bottom_group": bottom_grp,
                            "bottom_mean": round(means[bottom_grp], 2),
                            "group_means": {k: round(v, 2) for k, v in means.items()},
                        },
                        statistical_measure="group_mean_variation",
                        metadata={"group_counts": {k: v.count for k, v in group_summaries.items()}},
                    ))

        return results, findings
