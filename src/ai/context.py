"""Context building module for DataAlchemy AI synthesis.

Transforms DatasetProfile, DataQualityReport, and AnalysisReport into a compact,
token-controlled statistical context suitable for local LLM consumption without leaking raw data.
"""

from typing import Any, Dict, List, Optional

from src.analysis.analyzer import AnalysisReport
from src.analysis.findings import FindingImportance
from src.profiling.profiler import DatasetProfile
from src.quality.report import DataQualityReport, Severity


class AIContextBuilder:
    """Constructs structured statistical context dictionaries and prompt text."""

    def __init__(
        self,
        max_findings: int = 15,
        max_quality_issues: int = 10,
        max_columns: int = 20,
    ) -> None:
        self.max_findings = max_findings
        self.max_quality_issues = max_quality_issues
        self.max_columns = max_columns

    def build_context(
        self,
        profile: DatasetProfile,
        quality_report: Optional[DataQualityReport] = None,
        analysis_report: Optional[AnalysisReport] = None,
        ml_report: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Convert profiled facts, quality metrics, findings, and ML reports into structured AI context.

        Args:
            profile: DatasetProfile from Step 3.
            quality_report: Optional DataQualityReport from Step 4.
            analysis_report: Optional AnalysisReport from Step 5.
            ml_report: Optional MLReport from Step 7.

        Returns:
            Dictionary containing structured factual evidence.
        """
        # 1. Dataset Overview
        dataset_info: Dict[str, Any] = {
            "name": profile.dataset_name,
            "row_count": profile.row_count,
            "column_count": profile.column_count,
            "total_cells": profile.total_cells,
            "missing_cells": profile.missing_cells,
            "missing_percentage": profile.missing_percentage,
            "duplicate_rows": profile.duplicate_rows,
            "duplicate_percentage": profile.duplicate_percentage,
            "semantic_type_counts": profile.semantic_type_counts,
        }

        # Select top columns
        selected_columns: List[Dict[str, Any]] = []
        for col_name in list(profile.columns.keys())[: self.max_columns]:
            cp = profile.columns[col_name]
            selected_columns.append({
                "name": cp.name,
                "semantic_type": cp.semantic_type,
                "missing_percentage": cp.missing_percentage,
                "unique_count": cp.unique_count,
            })
        dataset_info["columns"] = selected_columns

        # 2. Quality Context
        quality_info: Dict[str, Any] = {}
        if quality_report is not None:
            # Prioritize higher severity issues
            prioritized_issues = sorted(
                quality_report.issues,
                key=lambda iss: (
                    0 if iss.severity == Severity.CRITICAL else
                    1 if iss.severity == Severity.ERROR else
                    2 if iss.severity == Severity.WARNING else 3
                ),
            )
            top_issues = [
                {
                    "rule_id": iss.rule_id,
                    "severity": iss.severity.value,
                    "title": iss.title,
                    "column": iss.column,
                    "description": iss.description,
                    "measured_value": iss.measured_value,
                }
                for iss in prioritized_issues[: self.max_quality_issues]
            ]
            quality_info = {
                "overall_score": quality_report.overall_score,
                "grade": quality_report.grade,
                "total_issues": quality_report.total_issues,
                "critical_issues": quality_report.critical_issues,
                "error_issues": quality_report.error_issues,
                "warning_issues": quality_report.warning_issues,
                "top_issues": top_issues,
            }

        # 3. Statistical Findings Context
        findings_info: List[Dict[str, Any]] = []
        if analysis_report is not None:
            # Prioritize findings by importance
            sorted_findings = sorted(
                analysis_report.findings,
                key=lambda f: (
                    0 if f.importance == FindingImportance.HIGH else
                    1 if f.importance == FindingImportance.MEDIUM else
                    2 if f.importance == FindingImportance.LOW else 3
                ),
            )
            for f in sorted_findings[: self.max_findings]:
                findings_info.append({
                    "finding_id": f.finding_id,
                    "type": f.finding_type.value,
                    "title": f.title,
                    "importance": f.importance.value,
                    "affected_columns": f.affected_columns,
                    "evidence": f.measured_values,
                    "description": f.description,
                })

        # 4. Machine Learning Context
        ml_info: Dict[str, Any] = {}
        if ml_report is not None:
            # Extract ML summary
            ml_info = {
                "anomaly": {
                    "algorithm": ml_report.anomaly.algorithm,
                    "n_anomalies": ml_report.anomaly.n_anomalies,
                    "anomaly_percentage": ml_report.anomaly.anomaly_percentage,
                    "status": ml_report.anomaly.status,
                },
                "clustering": {
                    "algorithm": ml_report.clustering.algorithm,
                    "n_clusters": ml_report.clustering.n_clusters,
                    "silhouette_score": ml_report.clustering.silhouette_score,
                    "status": ml_report.clustering.status,
                },
                "dimensionality": {
                    "n_components": ml_report.dimensionality.n_components,
                    "cumulative_variance": (
                        ml_report.dimensionality.cumulative_explained_variance[-1]
                        if ml_report.dimensionality.cumulative_explained_variance
                        else None
                    ),
                    "status": ml_report.dimensionality.status,
                },
            }

            # Ingest ML findings into findings list
            for mf in ml_report.findings:
                findings_info.append({
                    "finding_id": mf.finding_id,
                    "type": f"ml_{mf.finding_type}",
                    "title": mf.title,
                    "importance": mf.importance,
                    "affected_columns": mf.affected_columns,
                    "evidence": mf.measured_values,
                    "description": mf.description,
                })

        return {
            "dataset": dataset_info,
            "quality": quality_info,
            "findings": findings_info,
            "ml": ml_info,
        }

    def to_prompt_text(self, context: Dict[str, Any]) -> str:
        """Format the structured context into a clean text block for prompt injection."""
        ds = context.get("dataset", {})
        qual = context.get("quality", {})
        findings = context.get("findings", [])
        ml = context.get("ml", {})

        lines = [
            "### DATASET SUMMARY",
            f"- Name: {ds.get('name')}",
            f"- Dimensions: {ds.get('row_count'):,} rows x {ds.get('column_count'):,} columns",
            f"- Missing Cells: {ds.get('missing_cells'):,} ({ds.get('missing_percentage')}%)",
            f"- Duplicate Rows: {ds.get('duplicate_rows'):,} ({ds.get('duplicate_percentage')}%)",
            f"- Semantic Types: {ds.get('semantic_type_counts')}",
            "",
            "### DATA QUALITY HEALTH",
        ]

        if qual:
            lines.append(f"- Overall Score: {qual.get('overall_score')}/100 (Grade: {qual.get('grade')})")
            lines.append(f"- Issues: {qual.get('total_issues')} total ({qual.get('critical_issues')} critical, {qual.get('error_issues')} error, {qual.get('warning_issues')} warning)")
            for iss in qual.get("top_issues", [])[:5]:
                col_str = f"[{iss.get('column')}] " if iss.get('column') else ""
                lines.append(f"  * {col_str}{iss.get('title')} ({iss.get('severity')}): {iss.get('description')}")
        else:
            lines.append("- Quality Report: Not evaluated")

        if ml:
            lines.extend(["", "### MACHINE LEARNING PATTERNS & CLUSTERING"])
            anom = ml.get("anomaly", {})
            clust = ml.get("clustering", {})
            dim = ml.get("dimensionality", {})
            if anom.get("status") == "completed":
                lines.append(f"- Anomaly Detection ({anom.get('algorithm')}): {anom.get('n_anomalies')} outliers identified ({anom.get('anomaly_percentage'):.1f}%)")
            if clust.get("status") == "completed":
                sil_str = f", silhouette = {clust.get('silhouette_score'):.2f}" if clust.get('silhouette_score') is not None else ""
                lines.append(f"- Clustering Discovery ({clust.get('algorithm')}): {clust.get('n_clusters')} clusters discovered{sil_str}")
            if dim.get("status") == "completed" and dim.get("cumulative_variance") is not None:
                lines.append(f"- PCA Dimensionality: {dim.get('n_components')} components capture {dim.get('cumulative_variance') * 100:.1f}% variance")

        lines.extend(["", "### VERIFIED STATISTICAL FINDINGS"])
        if findings:
            for f in findings:
                cols = ", ".join(f.get("affected_columns", []))
                lines.append(f"- [{f.get('importance')}] Finding ID: {f.get('finding_id')} | Type: {f.get('type')} | Columns: [{cols}]")
                lines.append(f"  Title: {f.get('title')}")
                lines.append(f"  Evidence: {f.get('evidence')}")
        else:
            lines.append("- No significant statistical findings detected.")

        return "\n".join(lines)
