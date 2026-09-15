"""Structured AI insight and recommendation models for DataAlchemy.

Defines typed representations for AI-synthesized explanations, evidence-linked insights,
and grounded recommendations that can be serialized and consumed by downstream systems.
"""

from dataclasses import asdict, dataclass, field
import json
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class AIInsight:
    """Represents a discrete AI-generated insight grounded in verified statistical evidence."""
    insight_id: str
    title: str
    summary: str
    explanation: str
    importance: str
    category: str
    evidence: Dict[str, Any]
    affected_columns: List[str]
    caveats: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize insight to a JSON-compatible dictionary."""
        return {
            "insight_id": self.insight_id,
            "title": self.title,
            "summary": self.summary,
            "explanation": self.explanation,
            "importance": self.importance,
            "category": self.category,
            "evidence": self.evidence,
            "affected_columns": self.affected_columns,
            "caveats": self.caveats,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class AIRecommendation:
    """Actionable recommendation directly supported by statistical or quality evidence."""
    recommendation_id: str
    title: str
    description: str
    action_type: str
    supporting_evidence: List[str]
    priority: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize recommendation to a JSON-compatible dictionary."""
        return {
            "recommendation_id": self.recommendation_id,
            "title": self.title,
            "description": self.description,
            "action_type": self.action_type,
            "supporting_evidence": self.supporting_evidence,
            "priority": self.priority,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class AIInsightReport:
    """Comprehensive AI-generated synthesis report linking facts to explanations."""
    dataset_name: str
    executive_summary: str
    insights: List[AIInsight]
    recommendations: List[AIRecommendation]
    warnings: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_insights_by_importance(self, importance: str) -> List[AIInsight]:
        """Filter insights by importance level (HIGH, MEDIUM, LOW, INFO)."""
        target = importance.upper()
        return [i for i in self.insights if i.importance.upper() == target]

    def get_insights_by_category(self, category: str) -> List[AIInsight]:
        """Filter insights by analytical category."""
        target = category.lower()
        return [i for i in self.insights if i.category.lower() == target]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize complete AI report into a JSON-compatible dictionary."""
        return {
            "dataset_name": self.dataset_name,
            "executive_summary": self.executive_summary,
            "total_insights": len(self.insights),
            "total_recommendations": len(self.recommendations),
            "insights": [i.to_dict() for i in self.insights],
            "recommendations": [r.to_dict() for r in self.recommendations],
            "warnings": self.warnings,
            "metadata": self.metadata,
        }

    def to_json(self, indent: int = 2) -> str:
        """Export report directly to a formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def summary(self) -> str:
        """Generate a human-readable ASCII summary of the AI insight report."""
        lines = [
            "AI INSIGHT REPORT",
            "=" * 35,
            f"Dataset:            {self.dataset_name}",
            f"Total Insights:     {len(self.insights)}",
            f"Recommendations:    {len(self.recommendations)}",
            "",
            "Executive Summary:",
            f"  {self.executive_summary}",
            "",
            "Key Insights:",
        ]

        if not self.insights:
            lines.append("  No insights generated.")
        else:
            for idx, ins in enumerate(self.insights[:5], 1):
                lines.append(f"  {idx}. [{ins.importance}] {ins.title}")
                lines.append(f"     Summary: {ins.summary}")
                if ins.caveats:
                    lines.append(f"     Caveat:  {ins.caveats}")

        if self.recommendations:
            lines.extend(["", "Evidence-Based Recommendations:"])
            for idx, rec in enumerate(self.recommendations[:3], 1):
                lines.append(f"  {idx}. [{rec.priority.upper()}] {rec.title}")
                lines.append(f"     Action: {rec.description}")

        lines.append("=" * 35)
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"AIInsightReport(dataset='{self.dataset_name}', "
            f"insights={len(self.insights)}, "
            f"recommendations={len(self.recommendations)})"
        )
