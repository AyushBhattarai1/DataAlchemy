"""Numerical correlation analysis module for DataAlchemy.

Computes pairwise Pearson correlation coefficients, p-values, and creates structured
findings without duplicate pairs, self-correlations, or identifier evaluations.
"""

from dataclasses import asdict, dataclass
import math
from typing import Any, Dict, List, Optional, Tuple
import warnings

import pandas as pd
import scipy.stats as stats

from src.analysis.findings import Finding, FindingImportance, FindingType
from src.profiling.profiler import DatasetProfile


@dataclass(frozen=True)
class CorrelationResult:
    """Detailed result of a pairwise correlation calculation."""
    column_x: str
    column_y: str
    coefficient: float
    p_value: Optional[float]
    sample_size: int
    strength: str
    direction: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CorrelationAnalyzer:
    """Analyzes numerical columns to compute correlations and significance."""

    def __init__(
        self,
        min_correlation: float = 0.3,
        max_pairs: int = 50,
        significance_threshold: float = 0.05,
    ) -> None:
        """Initialize correlation analyzer.

        Args:
            min_correlation: Minimum absolute correlation required to report.
            max_pairs: Maximum number of correlation pairs to report.
            significance_threshold: P-value threshold for statistical significance.
        """
        self.min_correlation = min_correlation
        self.max_pairs = max_pairs
        self.significance_threshold = significance_threshold

    def _classify_strength(self, abs_r: float) -> str:
        if abs_r >= 0.7:
            return "very strong"
        elif abs_r >= 0.5:
            return "strong"
        elif abs_r >= 0.3:
            return "moderate"
        else:
            return "weak"

    def _determine_importance(self, abs_r: float, p_value: Optional[float]) -> FindingImportance:
        if p_value is not None and p_value > self.significance_threshold:
            # Low significance reduces priority
            return FindingImportance.LOW
        if abs_r >= 0.7:
            return FindingImportance.HIGH
        elif abs_r >= 0.5:
            return FindingImportance.MEDIUM
        elif abs_r >= 0.3:
            return FindingImportance.LOW
        return FindingImportance.INFO

    def analyze(
        self, df: pd.DataFrame, profile: DatasetProfile
    ) -> Tuple[List[CorrelationResult], List[Finding]]:
        """Compute correlations across valid numerical columns.

        Args:
            df: Raw pandas DataFrame.
            profile: DatasetProfile providing column metadata and types.

        Returns:
            Tuple of (List[CorrelationResult], List[Finding]).
        """
        results: List[CorrelationResult] = []
        findings: List[Finding] = []

        # 1. Identify eligible numerical columns (excluding identifiers, constants, all-nulls)
        numerical_cols: List[str] = []
        for col_name, cp in profile.columns.items():
            if cp.semantic_type == "numerical":
                if cp.unique_count > 1 and cp.missing_percentage < 100.0:
                    if col_name in df.columns:
                        numerical_cols.append(col_name)

        # Sort columns to enforce deterministic pairing
        numerical_cols = sorted(numerical_cols)

        if len(numerical_cols) < 2 or len(df) < 3:
            return results, findings

        # 2. Pairwise evaluation (strictly col1 < col2 to avoid duplicates and self-pairs)
        pair_candidates: List[Tuple[float, CorrelationResult, Finding]] = []

        for i in range(len(numerical_cols)):
            col1 = numerical_cols[i]
            for j in range(i + 1, len(numerical_cols)):
                col2 = numerical_cols[j]

                # Drop rows with NaN in either column
                valid_subset = df[[col1, col2]].dropna()
                n = len(valid_subset)
                if n < 3:
                    continue

                s1 = pd.to_numeric(valid_subset[col1], errors="coerce")
                s2 = pd.to_numeric(valid_subset[col2], errors="coerce")

                valid_mask = s1.notna() & s2.notna()
                clean_s1 = s1[valid_mask]
                clean_s2 = s2[valid_mask]
                n_clean = len(clean_s1)

                if n_clean < 3 or clean_s1.nunique() <= 1 or clean_s2.nunique() <= 1:
                    continue

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        res = stats.pearsonr(clean_s1, clean_s2)
                        r = float(res.statistic)
                        p_val = float(res.pvalue)
                    except Exception:
                        continue

                if math.isnan(r) or math.isinf(r):
                    continue

                abs_r = abs(r)
                if abs_r < self.min_correlation:
                    continue

                strength = self._classify_strength(abs_r)
                direction = "positive" if r >= 0 else "negative"
                importance = self._determine_importance(abs_r, p_val)

                corr_res = CorrelationResult(
                    column_x=col1,
                    column_y=col2,
                    coefficient=round(r, 4),
                    p_value=round(p_val, 6) if not math.isnan(p_val) else None,
                    sample_size=n_clean,
                    strength=strength,
                    direction=direction,
                )

                p_str = f", p = {p_val:.4g}" if (p_val is not None and not math.isnan(p_val)) else ""
                desc = (
                    f"Columns '{col1}' and '{col2}' exhibit a {strength} {direction} "
                    f"statistical association (Pearson r = {r:.2f}{p_str}, n = {n_clean:,})."
                )

                finding = Finding(
                    finding_id=f"corr_{col1}_{col2}",
                    finding_type=FindingType.CORRELATION,
                    title=f"{strength.capitalize()} {direction} association between {col1} and {col2}",
                    description=desc,
                    importance=importance,
                    affected_columns=[col1, col2],
                    measured_values={
                        "pearson_r": round(r, 4),
                        "abs_pearson_r": round(abs_r, 4),
                        "p_value": round(p_val, 6) if not math.isnan(p_val) else None,
                        "sample_size": n_clean,
                        "strength": strength,
                        "direction": direction,
                    },
                    statistical_measure="pearson_correlation",
                    confidence_or_pvalue=round(p_val, 6) if not math.isnan(p_val) else None,
                    metadata={"column_x": col1, "column_y": col2},
                )

                pair_candidates.append((abs_r, corr_res, finding))

        # 3. Sort by absolute correlation descending and enforce max_pairs
        pair_candidates.sort(key=lambda x: x[0], reverse=True)
        top_candidates = pair_candidates[: self.max_pairs]

        for _, corr_res, finding in top_candidates:
            results.append(corr_res)
            findings.append(finding)

        return results, findings
