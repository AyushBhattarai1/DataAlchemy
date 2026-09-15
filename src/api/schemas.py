"""Pydantic schemas for the DataAlchemy REST API."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class DatasetItem(BaseModel):
    name: str
    path: str
    file_type: str
    is_sample: bool


class DatasetListResponse(BaseModel):
    datasets: List[DatasetItem]
    active_dataset: Optional[str] = None


class AnalyzeRequest(BaseModel):
    dataset_name: str
    anomaly_algorithm: Optional[str] = "isolation_forest"
    clustering_algorithm: Optional[str] = "kmeans"
    n_clusters: Optional[int] = 3


class PaginatedDataResponse(BaseModel):
    columns: List[str]
    dtypes: Dict[str, str]
    rows: List[Dict[str, Any]]
    total_rows: int
    page: int
    page_size: int
    total_pages: int


class ExplainRequest(BaseModel):
    finding_id: str
    title: str
    evidence: Dict[str, Any] = Field(default_factory=dict)
    category: Optional[str] = "general"
    context: Optional[str] = None


class ExplainResponse(BaseModel):
    finding_id: str
    explanation: str
    recommendation: Optional[str] = None
