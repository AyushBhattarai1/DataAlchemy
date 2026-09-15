"""DataAlchemy Data Quality and Health Engine Package.

Provides deterministic, rule-based quality evaluation and scoring for profiled datasets.
"""

from src.quality.analyzer import QualityAnalyzer, analyze_quality
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

__all__ = [
    # Report & Issue Models
    "DataQualityReport",
    "QualityIssue",
    "Severity",
    # Analyzer & Main API
    "QualityAnalyzer",
    "analyze_quality",
    # Rule Base & Built-in Rules
    "QualityRule",
    "MissingnessRule",
    "DuplicateRowRule",
    "ConstantColumnRule",
    "LowVarianceRule",
    "HighCardinalityRule",
    "IdentifierFeatureRule",
    "NumericalSanityRule",
]
