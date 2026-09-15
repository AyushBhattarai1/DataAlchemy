"""Finding data models and importance structures for DataAlchemy statistical analysis.

Provides typed, deterministic containers for statistical discoveries that can be easily
serialized, prioritized, and consumed by downstream analytical or LLM layers.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class FindingType(str, Enum):
    """Categorization of statistical findings."""
    CORRELATION = "CORRELATION"
    DISTRIBUTION = "DISTRIBUTION"
    CATEGORY = "CATEGORY"
    GROUP_RELATIONSHIP = "GROUP_RELATIONSHIP"
    TEMPORAL_TREND = "TEMPORAL_TREND"

    def __str__(self) -> str:
        return self.value


class FindingImportance(str, Enum):
    """Priority and significance level of a finding."""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"

    def __str__(self) -> str:
        return self.value

    @property
    def rank(self) -> int:
        """Numerical rank for sorting (lower is higher priority)."""
        ranks = {
            FindingImportance.HIGH: 1,
            FindingImportance.MEDIUM: 2,
            FindingImportance.LOW: 3,
            FindingImportance.INFO: 4,
        }
        return ranks.get(self, 99)


@dataclass(frozen=True)
class Finding:
    """Represents a discrete statistical discovery with full numerical evidence."""
    finding_id: str
    finding_type: FindingType
    title: str
    description: str
    importance: FindingImportance
    affected_columns: List[str]
    measured_values: Dict[str, Any]
    statistical_measure: str
    confidence_or_pvalue: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert finding into a JSON-compatible dictionary."""
        return {
            "finding_id": self.finding_id,
            "finding_type": self.finding_type.value,
            "title": self.title,
            "description": self.description,
            "importance": self.importance.value,
            "affected_columns": self.affected_columns,
            "measured_values": self.measured_values,
            "statistical_measure": self.statistical_measure,
            "confidence_or_pvalue": self.confidence_or_pvalue,
            "metadata": self.metadata,
        }

    def __lt__(self, other: "Finding") -> bool:
        """Deterministic sorting by importance rank, then finding_id."""
        if self.importance.rank != other.importance.rank:
            return self.importance.rank < other.importance.rank
        return self.finding_id < other.finding_id

    def __repr__(self) -> str:
        return (
            f"Finding(id='{self.finding_id}', "
            f"type='{self.finding_type.value}', "
            f"importance='{self.importance.value}', "
            f"cols={self.affected_columns})"
        )
