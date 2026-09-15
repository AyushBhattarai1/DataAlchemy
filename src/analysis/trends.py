"""Temporal trend detection and period aggregation module for DataAlchemy.

Evaluates historical patterns and trajectory slopes over time without forecasting
or causal assertions, using deterministic linear fits on temporal aggregations.
"""

from dataclasses import asdict, dataclass
import math
from typing import Any, Dict, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
import scipy.stats as stats

from src.analysis.findings import Finding, FindingImportance, FindingType
from src.profiling.profiler import DatasetProfile


@dataclass(frozen=True)
class TrendResult:
    """Quantitative evaluation of a metric's trajectory over time."""
    datetime_column: str
    numerical_column: str
    period_frequency: str
    slope: float
    r_squared: float
    p_value: Optional[float]
    direction: str
    start_date: str
    end_date: str
    periods_count: int
    aggregated_values: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TrendAnalyzer:
    """Identifies historical temporal trajectories in numerical columns across time."""

    def __init__(
        self,
        min_periods: int = 3,
        r2_threshold: float = 0.25,
        max_numerical_columns: int = 5,
    ) -> None:
        self.min_periods = min_periods
        self.r2_threshold = r2_threshold
        self.max_numerical_columns = max_numerical_columns

    def analyze(
        self, df: pd.DataFrame, profile: DatasetProfile
    ) -> Tuple[List[TrendResult], List[Finding]]:
        """Identify temporal trends across datetime and numerical column pairs.

        Args:
            df: Raw input DataFrame.
            profile: DatasetProfile containing column semantic types.

        Returns:
            Tuple of (List[TrendResult], List[Finding]).
        """
        results: List[TrendResult] = []
        findings: List[Finding] = []

        if len(df) < self.min_periods:
            return results, findings

        # Identify datetime and numerical columns
        date_cols = [
            c for c, cp in profile.columns.items()
            if cp.semantic_type == "datetime" and c in df.columns
        ]
        num_cols = [
            c for c, cp in profile.columns.items()
            if cp.semantic_type == "numerical" and cp.unique_count > 1 and cp.missing_percentage < 100.0 and c in df.columns
        ][: self.max_numerical_columns]

        if not date_cols or not num_cols:
            return results, findings

        for date_col in sorted(date_cols):
            # Parse dates temporarily without mutating original df
            parsed_dates = pd.to_datetime(df[date_col], errors="coerce")
            valid_mask = parsed_dates.notna()
            if valid_mask.sum() < self.min_periods:
                continue

            sub_dates = parsed_dates[valid_mask]
            min_dt = sub_dates.min()
            max_dt = sub_dates.max()
            span_days = (max_dt - min_dt).days if hasattr(max_dt - min_dt, "days") else 0

            # Determine aggregation frequency
            if span_days <= 60:
                freq_str = "D"
                freq_label = "daily"
            elif span_days <= 365:
                freq_str = "W"
                freq_label = "weekly"
            else:
                freq_str = "ME"
                freq_label = "monthly"

            for num_col in sorted(num_cols):
                # Form subframe
                sub_df = pd.DataFrame({
                    "dt": sub_dates,
                    "val": pd.to_numeric(df.loc[valid_mask, num_col], errors="coerce"),
                }).dropna()

                if len(sub_df) < self.min_periods:
                    continue

                sub_df = sub_df.sort_values("dt")

                # Resample / aggregate
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        resampled = sub_df.set_index("dt").resample(freq_str)["val"].mean().dropna()
                except Exception:
                    # Fallback to simple integer bucket aggregation if resample fails
                    continue

                n_periods = len(resampled)
                if n_periods < self.min_periods:
                    continue

                y = resampled.values
                x = np.arange(n_periods)

                if np.all(y == y[0]):
                    continue

                try:
                    res = stats.linregress(x, y)
                    slope = float(res.slope)
                    r_val = float(res.rvalue)
                    r2 = float(r_val ** 2)
                    p_val = float(res.pvalue)
                except Exception:
                    continue

                if math.isnan(slope) or math.isnan(r2):
                    continue

                # Classify direction
                if r2 >= self.r2_threshold and p_val < 0.1:
                    direction = "increasing" if slope > 0 else "decreasing"
                else:
                    direction = "stable"

                agg_dict = {
                    idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx): round(float(v), 2)
                    for idx, v in resampled.items()
                }

                trend_res = TrendResult(
                    datetime_column=date_col,
                    numerical_column=num_col,
                    period_frequency=freq_label,
                    slope=round(slope, 4),
                    r_squared=round(r2, 4),
                    p_value=round(p_val, 6) if not math.isnan(p_val) else None,
                    direction=direction,
                    start_date=min_dt.strftime("%Y-%m-%d") if hasattr(min_dt, "strftime") else str(min_dt),
                    end_date=max_dt.strftime("%Y-%m-%d") if hasattr(max_dt, "strftime") else str(max_dt),
                    periods_count=n_periods,
                    aggregated_values=agg_dict,
                )
                results.append(trend_res)

                # Generate finding if trend is non-trivial or strong
                if direction in ("increasing", "decreasing"):
                    importance = FindingImportance.HIGH if r2 >= 0.6 else FindingImportance.MEDIUM
                    findings.append(Finding(
                        finding_id=f"trend_{date_col}_{num_col}",
                        finding_type=FindingType.TEMPORAL_TREND,
                        title=f"{direction.capitalize()} temporal trend in '{num_col}' over time",
                        description=(
                            f"Column '{num_col}' exhibits an {direction} trend across {n_periods} "
                            f"{freq_label} periods from {trend_res.start_date} to {trend_res.end_date} "
                            f"(slope = {slope:.2f}, R² = {r2:.2f}, p = {p_val:.4g})."
                        ),
                        importance=importance,
                        affected_columns=[date_col, num_col],
                        measured_values={
                            "direction": direction,
                            "slope": round(slope, 4),
                            "r_squared": round(r2, 4),
                            "p_value": round(p_val, 6) if not math.isnan(p_val) else None,
                            "periods_count": n_periods,
                            "frequency": freq_label,
                        },
                        statistical_measure="linear_trend",
                        confidence_or_pvalue=round(p_val, 6) if not math.isnan(p_val) else None,
                        metadata={"start_date": trend_res.start_date, "end_date": trend_res.end_date},
                    ))

        return results, findings
