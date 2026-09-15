"""Tests for DataAlchemy API routes."""

import os
import pytest
from fastapi.testclient import TestClient

os.environ["DATAALCHEMY_ENV"] = "test"
from main import app

client = TestClient(app)


def test_health_check():
    """Verify health check endpoint returns 200."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_list_datasets():
    """Verify listing available sample datasets."""
    response = client.get("/api/datasets")
    assert response.status_code == 200
    data = response.json()
    assert "datasets" in data
    names = [d["name"] for d in data["datasets"]]
    assert "customers.csv" in names
    assert "sales.csv" in names


def test_unauthorized_access_before_analyze():
    """Verify that querying data endpoints before running analyze returns 400."""
    # Reset active state if any
    from src.api.routes import _ACTIVE_STATE
    _ACTIVE_STATE["dataset"] = None

    for endpoint in ["/api/data", "/api/overview", "/api/quality", "/api/statistics", "/api/ml", "/api/insights"]:
        res = client.get(endpoint)
        assert res.status_code == 400
        assert "No dataset currently analyzed" in res.json()["detail"]


def test_analyze_not_found():
    """Verify analyzing non-existent dataset returns 404."""
    response = client.post("/api/analyze", json={"dataset_name": "non_existent.csv"})
    assert response.status_code == 404


def test_analyze_and_query_endpoints():
    """Verify successful analysis of sample dataset and querying of all analytical endpoints."""
    # Run analysis on customers.csv
    analyze_res = client.post(
        "/api/analyze",
        json={
            "dataset_name": "customers.csv",
            "anomaly_algorithm": "isolation_forest",
            "clustering_algorithm": "kmeans",
            "n_clusters": 3,
        },
    )
    assert analyze_res.status_code == 200
    result = analyze_res.json()
    assert result["status"] == "success"
    assert result["dataset_name"] == "customers.csv"
    assert result["row_count"] > 0
    assert result["column_count"] > 0
    assert "health_score" in result

    # Query overview
    overview_res = client.get("/api/overview")
    assert overview_res.status_code == 200
    overview_data = overview_res.json()
    assert overview_data["dataset_name"] == "customers.csv"
    assert "dimensions" in overview_data
    assert "health" in overview_data
    assert "key_findings" in overview_data
    assert "executive_summary" in overview_data

    # Query data pagination
    data_res = client.get("/api/data?page=1&page_size=10")
    assert data_res.status_code == 200
    page_data = data_res.json()
    assert page_data["page"] == 1
    assert page_data["page_size"] == 10
    assert len(page_data["rows"]) <= 10
    assert len(page_data["columns"]) > 0

    # Test search in data
    search_res = client.get("/api/data?search=Smith")
    assert search_res.status_code == 200
    assert "rows" in search_res.json()

    # Query quality
    quality_res = client.get("/api/quality")
    assert quality_res.status_code == 200
    quality_data = quality_res.json()
    assert "overall_score" in quality_data
    assert "issues" in quality_data

    # Query statistics
    stats_res = client.get("/api/statistics")
    assert stats_res.status_code == 200
    stats_data = stats_res.json()
    assert "correlations" in stats_data
    assert "distributions" in stats_data
    assert "column_profiles" in stats_data

    # Query ML
    ml_res = client.get("/api/ml")
    assert ml_res.status_code == 200
    ml_data = ml_res.json()
    assert "anomaly" in ml_data
    assert "clustering" in ml_data
    assert "dimensionality" in ml_data

    # Query Insights
    insights_res = client.get("/api/insights")
    assert insights_res.status_code == 200
    insights_data = insights_res.json()
    assert "executive_summary" in insights_data
    assert "insights" in insights_data


def test_explain_finding_endpoint():
    """Verify on-demand AI explanation generation."""
    res = client.post(
        "/api/insights/explain",
        json={
            "finding_id": "FINDING_TEST_1",
            "title": "High missingness in phone numbers",
            "evidence": {"missing_pct": 45.2},
            "category": "quality",
            "context": "Phone number column",
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert data["finding_id"] == "FINDING_TEST_1"
    assert "High missingness in phone numbers" in data["explanation"]
    assert "recommendation" in data


def test_upload_dataset_endpoint(tmp_path):
    """Verify uploading a CSV dataset."""
    sample_content = b"colA,colB\n1,10\n2,20\n3,30\n"
    files = {"file": ("test_upload.csv", sample_content, "text/csv")}
    res = client.post("/api/datasets/upload", files=files)
    assert res.status_code == 200
    data = res.json()
    assert data["name"] == "test_upload.csv"
    assert data["file_type"] == "csv"
    assert data["is_sample"] is False
