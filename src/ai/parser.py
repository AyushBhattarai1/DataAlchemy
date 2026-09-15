"""Output parsing and validation module for DataAlchemy AI synthesis.

Extracts, validates, and parses structured JSON responses from LLM generation into
typed AIInsightReport objects with robust error handling and markdown fence stripping.
"""

import json
import re
from typing import Any, Dict, List, Optional

from src.ai.models import AIInsight, AIInsightReport, AIRecommendation


class AIOutputParsingError(Exception):
    """Raised when LLM response is malformed, invalid JSON, or fails schema validation."""
    pass


class AIOutputParser:
    """Parses and validates LLM generation into structured AIInsightReport objects."""

    def _extract_json_text(self, raw_text: str) -> str:
        """Extract JSON substring, stripping markdown code fences if present."""
        text = raw_text.strip()

        # Handle ```json ... ``` or ``` ... ```
        fence_pattern = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)
        match = fence_pattern.search(text)
        if match:
            return match.group(1).strip()

        # If no fences, find outermost '{' and '}'
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return text[start : end + 1].strip()

        return text

    def parse_insight_report(self, raw_text: str, dataset_name: str) -> AIInsightReport:
        """Parse raw LLM output into an AIInsightReport.

        Args:
            raw_text: Raw string output from LLM generation.
            dataset_name: Name of the dataset for report identification.

        Returns:
            AIInsightReport: Structured, validated insight report.

        Raises:
            AIOutputParsingError: If text is empty, invalid JSON, or missing required schema fields.
        """
        if not raw_text or not raw_text.strip():
            raise AIOutputParsingError("Model returned an empty response.")

        clean_json_str = self._extract_json_text(raw_text)

        try:
            payload = json.loads(clean_json_str)
        except json.JSONDecodeError as err:
            raise AIOutputParsingError(
                f"Failed to parse LLM response as JSON: {err}. Raw excerpt: {raw_text[:200]}"
            ) from err

        if not isinstance(payload, dict):
            raise AIOutputParsingError(f"Expected JSON object, got {type(payload).__name__}")

        # Required fields validation
        if "executive_summary" not in payload:
            raise AIOutputParsingError("Missing required field 'executive_summary' in model response.")

        exec_summary = str(payload["executive_summary"]).strip()
        if not exec_summary:
            raise AIOutputParsingError("'executive_summary' cannot be empty.")

        # Parse insights
        raw_insights = payload.get("insights", [])
        if not isinstance(raw_insights, list):
            raise AIOutputParsingError(f"'insights' must be a list, got {type(raw_insights).__name__}")

        insights: List[AIInsight] = []
        for idx, item in enumerate(raw_insights, 1):
            if not isinstance(item, dict):
                continue

            insight_id = str(item.get("insight_id", f"INSIGHT_{idx:02d}"))
            title = str(item.get("title", "Untitled Insight"))
            summary = str(item.get("summary", title))
            explanation = str(item.get("explanation", summary))
            importance = str(item.get("importance", "INFO")).upper()
            category = str(item.get("category", "general")).lower()
            evidence = item.get("evidence", {})
            if not isinstance(evidence, dict):
                evidence = {"raw_evidence": str(evidence)}

            affected_cols = item.get("affected_columns", [])
            if not isinstance(affected_cols, list):
                affected_cols = [str(affected_cols)] if affected_cols else []
            affected_cols = [str(c) for c in affected_cols]

            caveats = item.get("caveats")
            caveats_str = str(caveats) if (caveats is not None and str(caveats).lower() != "null") else None

            insights.append(AIInsight(
                insight_id=insight_id,
                title=title,
                summary=summary,
                explanation=explanation,
                importance=importance,
                category=category,
                evidence=evidence,
                affected_columns=affected_cols,
                caveats=caveats_str,
            ))

        # Parse recommendations
        raw_recs = payload.get("recommendations", [])
        if not isinstance(raw_recs, list):
            raise AIOutputParsingError(f"'recommendations' must be a list, got {type(raw_recs).__name__}")

        recommendations: List[AIRecommendation] = []
        for idx, item in enumerate(raw_recs, 1):
            if not isinstance(item, dict):
                continue

            rec_id = str(item.get("recommendation_id", f"REC_{idx:02d}"))
            title = str(item.get("title", "Technical Recommendation"))
            desc = str(item.get("description", title))
            action_type = str(item.get("action_type", "investigation")).lower()

            evidence_refs = item.get("supporting_evidence", [])
            if not isinstance(evidence_refs, list):
                evidence_refs = [str(evidence_refs)] if evidence_refs else []
            evidence_refs = [str(e) for e in evidence_refs]

            priority = str(item.get("priority", "medium")).lower()

            recommendations.append(AIRecommendation(
                recommendation_id=rec_id,
                title=title,
                description=desc,
                action_type=action_type,
                supporting_evidence=evidence_refs,
                priority=priority,
            ))

        # Warnings
        raw_warnings = payload.get("warnings", [])
        warnings_list = [str(w) for w in raw_warnings] if isinstance(raw_warnings, list) else []

        return AIInsightReport(
            dataset_name=dataset_name,
            executive_summary=exec_summary,
            insights=insights,
            recommendations=recommendations,
            warnings=warnings_list,
        )
