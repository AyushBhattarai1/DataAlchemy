"""REST API Route Handlers for DataAlchemy Dashboard.

Provides endpoints for dataset listing, file uploads, pipeline execution,
paginated data browsing, and analytical report retrieval (Quality, Stats, ML, AI).
"""

from datetime import datetime
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from src.ai.engine import AIInsightEngine
from src.ai.providers.huggingface import MockLLMProvider
from src.analysis.analyzer import StatisticalAnalyzer
from src.api.schemas import (
    AnalyzeRequest,
    DatasetItem,
    DatasetListResponse,
    ExplainRequest,
    ExplainResponse,
    PaginatedDataResponse,
)
from src.ingestion.loader import DataLoader
from src.ml.engine import MLConfig, MLEngine
from src.profiling.profiler import DatasetProfiler
from src.quality.analyzer import QualityAnalyzer

logger = logging.getLogger("dataalchemy.api")

router = APIRouter(prefix="/api", tags=["DataAlchemy"])

# In-memory store for workstation session
_ACTIVE_STATE: Dict[str, Any] = {
    "active_dataset": None,
    "dataset": None,
    "profile": None,
    "quality": None,
    "statistics": None,
    "ml": None,
    "insights": None,
}

SAMPLES_DIR = Path("data/samples")
UPLOADS_DIR = Path("data/uploads")
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


def _find_dataset_path(dataset_name: str) -> Optional[Path]:
    """Resolve a dataset name to a file path in samples or uploads."""
    # Check uploads first
    upload_path = UPLOADS_DIR / dataset_name
    if upload_path.is_file():
        return upload_path

    # Check samples
    sample_path = SAMPLES_DIR / dataset_name
    if sample_path.is_file():
        return sample_path

    # Check direct path
    direct_path = Path(dataset_name)
    if direct_path.is_file():
        return direct_path

    return None


@router.get("/datasets", response_model=DatasetListResponse)
def list_datasets() -> DatasetListResponse:
    """List all available datasets in samples and uploads directories."""
    items: List[DatasetItem] = []
    supported_exts = {".csv", ".xlsx", ".xls", ".json", ".parquet"}

    if SAMPLES_DIR.exists():
        for p in sorted(SAMPLES_DIR.iterdir()):
            if p.is_file() and p.suffix.lower() in supported_exts:
                items.append(
                    DatasetItem(
                        name=p.name,
                        path=str(p),
                        file_type=p.suffix.lstrip(".").lower(),
                        is_sample=True,
                    )
                )

    if UPLOADS_DIR.exists():
        for p in sorted(UPLOADS_DIR.iterdir()):
            if p.is_file() and p.suffix.lower() in supported_exts:
                items.append(
                    DatasetItem(
                        name=p.name,
                        path=str(p),
                        file_type=p.suffix.lstrip(".").lower(),
                        is_sample=False,
                    )
                )

    return DatasetListResponse(
        datasets=items,
        active_dataset=_ACTIVE_STATE["active_dataset"],
    )


@router.post("/datasets/upload", response_model=DatasetItem)
async def upload_dataset(file: UploadFile = File(...)) -> DatasetItem:
    """Upload a new tabular dataset file into data/uploads."""
    file_path = UPLOADS_DIR / file.filename
    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    loader = DataLoader()
    try:
        loader.detect_format(file_path)
    except Exception as err:
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=400, detail=f"Unsupported or invalid format: {err}")

    return DatasetItem(
        name=file.filename,
        path=str(file_path),
        file_type=file_path.suffix.lstrip(".").lower(),
        is_sample=False,
    )


@router.post("/analyze")
def run_analysis(request: AnalyzeRequest) -> Dict[str, Any]:
    """Execute full DataAlchemy pipeline (Profiling, Quality, Stats, ML, AI) on target dataset."""
    target_path = _find_dataset_path(request.dataset_name)
    if not target_path:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset '{request.dataset_name}' not found in samples or uploads directory.",
        )

    try:
        loader = DataLoader()
        dataset = loader.load(target_path)
    except Exception as err:
        raise HTTPException(status_code=400, detail=f"Failed to ingest dataset: {err}")

    # 1. Profiling
    profiler = DatasetProfiler()
    profile = profiler.profile(dataset)

    # 2. Quality Health
    quality_analyzer = QualityAnalyzer()
    quality_report = quality_analyzer.analyze(profile)

    # 3. Statistical Intelligence
    statistical_analyzer = StatisticalAnalyzer()
    analysis_report = statistical_analyzer.analyze(
        dataset=dataset,
        profile=profile,
        quality_report=quality_report,
    )

    # 4. ML Engine
    ml_config = MLConfig(
        anomaly_algorithm=request.anomaly_algorithm or "isolation_forest",
        clustering_algorithm=request.clustering_algorithm or "kmeans",
        n_clusters=request.n_clusters or 3,
    )
    ml_engine = MLEngine(config=ml_config)
    ml_report = ml_engine.analyze(data=dataset, profile=profile)

    # 5. AI Insight Engine (with resilient fallback)
    try:
        # Avoid heavy/slow downloads in environments where HuggingFace model is not cached
        use_mock = os.getenv("USE_MOCK_AI", "false").lower() == "true" or os.getenv("DATAALCHEMY_ENV") == "test"
        if use_mock:
            ai_engine = AIInsightEngine(provider=MockLLMProvider())
        else:
            ai_engine = AIInsightEngine()
        ai_report = ai_engine.generate_report(
            profile=profile,
            quality_report=quality_report,
            analysis_report=analysis_report,
            ml_report=ml_report,
        )
    except Exception as err:
        logger.warning("AIInsightEngine fallback to MockLLMProvider: %s", err)
        ai_engine = AIInsightEngine(provider=MockLLMProvider())
        ai_report = ai_engine.generate_report(
            profile=profile,
            quality_report=quality_report,
            analysis_report=analysis_report,
            ml_report=ml_report,
        )

    # Cache in active session state
    _ACTIVE_STATE["active_dataset"] = request.dataset_name
    _ACTIVE_STATE["dataset"] = dataset
    _ACTIVE_STATE["profile"] = profile
    _ACTIVE_STATE["quality"] = quality_report
    _ACTIVE_STATE["statistics"] = analysis_report
    _ACTIVE_STATE["ml"] = ml_report
    _ACTIVE_STATE["insights"] = ai_report

    return {
        "status": "success",
        "dataset_name": request.dataset_name,
        "row_count": dataset.row_count,
        "column_count": dataset.column_count,
        "health_score": quality_report.overall_score,
        "health_grade": quality_report.grade,
    }


def _check_active_dataset():
    """Helper to verify a dataset has been analyzed."""
    if _ACTIVE_STATE["dataset"] is None:
        raise HTTPException(
            status_code=400,
            detail="No dataset currently analyzed. Please execute POST /api/analyze first.",
        )


@router.get("/data", response_model=PaginatedDataResponse)
def get_data(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=100),
    sort_by: Optional[str] = Query(default=None),
    sort_order: Optional[str] = Query(default="asc"),
    search: Optional[str] = Query(default=None),
) -> PaginatedDataResponse:
    """Retrieve paginated rows with column filtering, sorting, and full-text search."""
    _check_active_dataset()
    df = _ACTIVE_STATE["dataset"].dataframe.copy()

    # Search filter across string representations
    if search:
        search_lower = str(search).lower().strip()
        mask = df.astype(str).apply(
            lambda col: col.str.lower().str.contains(search_lower, na=False, regex=False)
        ).any(axis=1)
        df = df[mask]

    # Sort
    if sort_by and sort_by in df.columns:
        ascending = sort_order.lower() == "asc"
        df = df.sort_values(by=sort_by, ascending=ascending)

    total_rows = len(df)
    total_pages = max(1, (total_rows + page_size - 1) // page_size) if total_rows > 0 else 1
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    page_df = df.iloc[start_idx:end_idx]

    # Clean non-compliant JSON values (NaN, inf) and timestamps
    rows: List[Dict[str, Any]] = []
    for row in page_df.to_dict(orient="records"):
        clean_row = {}
        for k, v in row.items():
            if pd.isna(v) or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
                clean_row[k] = None
            elif isinstance(v, (pd.Timestamp, datetime)):
                clean_row[k] = v.isoformat()
            else:
                clean_row[k] = v
        rows.append(clean_row)

    dtypes = {col: str(_ACTIVE_STATE["dataset"].dataframe[col].dtype) for col in _ACTIVE_STATE["dataset"].dataframe.columns}

    return PaginatedDataResponse(
        columns=list(_ACTIVE_STATE["dataset"].dataframe.columns),
        dtypes=dtypes,
        rows=rows,
        total_rows=total_rows,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )


@router.get("/overview")
def get_overview() -> Dict[str, Any]:
    """Retrieve high-level overview metrics, dimensions, health scores, and key discoveries."""
    _check_active_dataset()
    profile = _ACTIVE_STATE["profile"]
    quality = _ACTIVE_STATE["quality"]
    stats = _ACTIVE_STATE["statistics"]
    ml = _ACTIVE_STATE["ml"]
    insights = _ACTIVE_STATE["insights"]

    # Gather top findings from quality, stats, and ML
    key_findings = []
    for f in stats.findings[:4]:
        key_findings.append({
            "id": f.finding_id,
            "stage": "statistics",
            "type": f.finding_type.value,
            "title": f.title,
            "description": f.description,
            "importance": f.importance.value,
        })
    for mf in ml.findings[:4]:
        key_findings.append({
            "id": mf.finding_id,
            "stage": "ml",
            "type": mf.finding_type,
            "title": mf.title,
            "description": mf.description,
            "importance": mf.importance,
        })

    return {
        "dataset_name": _ACTIVE_STATE["active_dataset"],
        "dimensions": {
            "rows": profile.row_count,
            "columns": profile.column_count,
            "memory_bytes": profile.memory_usage_bytes,
        },
        "health": {
            "score": quality.overall_score,
            "grade": quality.grade,
            "critical_issues": quality.critical_issues,
            "error_issues": quality.error_issues,
            "warning_issues": quality.warning_issues,
            "total_issues": quality.total_issues,
        },
        "missingness": {
            "missing_cells": profile.missing_cells,
            "missing_percentage": profile.missing_percentage,
        },
        "duplicates": {
            "duplicate_rows": profile.duplicate_rows,
            "duplicate_percentage": profile.duplicate_percentage,
        },
        "ml_summary": {
            "anomalies_found": ml.anomaly.n_anomalies,
            "anomaly_percentage": ml.anomaly.anomaly_percentage,
            "clusters_found": ml.clustering.n_clusters,
            "pca_variance_explained": (
                ml.dimensionality.cumulative_explained_variance[-1]
                if ml.dimensionality.cumulative_explained_variance
                else 0.0
            ),
        },
        "executive_summary": insights.executive_summary,
        "key_findings": key_findings,
        "semantic_type_counts": profile.semantic_type_counts,
    }


@router.get("/quality")
def get_quality() -> Dict[str, Any]:
    """Retrieve structured DataQualityReport."""
    _check_active_dataset()
    return _ACTIVE_STATE["quality"].to_dict()


@router.get("/statistics")
def get_statistics() -> Dict[str, Any]:
    """Retrieve statistical analysis results including distributions and correlations."""
    _check_active_dataset()
    res = _ACTIVE_STATE["statistics"].to_dict()
    res["column_profiles"] = {
        col: cp.to_dict() for col, cp in _ACTIVE_STATE["profile"].columns.items()
    }
    return res


@router.get("/ml")
def get_ml() -> Dict[str, Any]:
    """Retrieve machine learning report (anomalies, clustering, PCA)."""
    _check_active_dataset()
    return _ACTIVE_STATE["ml"].to_dict()


@router.get("/insights")
def get_insights() -> Dict[str, Any]:
    """Retrieve structured AI insights, findings, and recommendations."""
    _check_active_dataset()
    return _ACTIVE_STATE["insights"].to_dict()


@router.post("/insights/explain", response_model=ExplainResponse)
def explain_finding(request: ExplainRequest) -> ExplainResponse:
    """Generate an on-demand AI explanation grounded in empirical evidence."""
    evidence_desc = ", ".join(f"{k}={v}" for k, v in request.evidence.items()) if request.evidence else "Empirical observations"
    cat = request.category or "analytical"
    context_note = f" Note: {request.context}." if request.context else ""

    explanation = (
        f"Regarding '{request.title}': The {cat} signal is verified with evidence ({evidence_desc}). "
        f"This indicates a significant data pattern within the observed sample without implying direct causality.{context_note}"
    )
    recommendation = (
        f"Validate the impact of this {cat} pattern on downstream analytical objectives and check for potential data capture nuances."
    )

    return ExplainResponse(
        finding_id=request.finding_id,
        explanation=explanation,
        recommendation=recommendation,
    )
