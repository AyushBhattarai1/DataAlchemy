"""DataAlchemy Statistical / Exploratory Intelligence Package.

Provides deterministic statistical discovery across numerical correlations, distribution shapes,
categorical concentrations, group differences, and historical temporal trends.
"""

from src.analysis.analyzer import (
    AnalysisReport,
    StatisticalAnalyzer,
    analyze_dataset,
)
from src.analysis.correlations import (
    CorrelationAnalyzer,
    CorrelationResult,
)
from src.analysis.distributions import (
    DistributionAnalyzer,
    DistributionResult,
)
from src.analysis.findings import (
    Finding,
    FindingImportance,
    FindingType,
)
from src.analysis.relationships import (
    GroupSummary,
    RelationshipAnalyzer,
    RelationshipResult,
)
from src.analysis.trends import (
    TrendAnalyzer,
    TrendResult,
)

__all__ = [
    # Report & Orchestrator
    "AnalysisReport",
    "StatisticalAnalyzer",
    "analyze_dataset",
    # Findings
    "Finding",
    "FindingType",
    "FindingImportance",
    # Correlations
    "CorrelationAnalyzer",
    "CorrelationResult",
    # Distributions
    "DistributionAnalyzer",
    "DistributionResult",
    # Relationships
    "RelationshipAnalyzer",
    "RelationshipResult",
    "GroupSummary",
    # Trends
    "TrendAnalyzer",
    "TrendResult",
]
