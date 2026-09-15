"""Unit tests for ML models, serialization, and visualization contracts in src/ml/models.py."""

import json
import pytest

from src.ml.models import (
    AnomalyResult,
    ClusterResult,
    ClusterSummary,
    MLEvaluation,
    MLFinding,
    MLPreprocessingResult,
    MLReport,
    PCAResult,
    _sanitize_for_json,
)


class TestMLModelsSerialization:
    """Test suite verifying serialization and data contracts of ML structures."""

    def test_sanitize_for_json_numpy_types(self) -> None:
        import numpy as np

        data = {
            "int_val": np.int64(42),
            "float_val": np.float64(3.1415),
            "nan_val": np.float64(np.nan),
            "bool_val": np.bool_(True),
            "array_val": np.array([1.0, 2.0, 3.0]),
            "nested": {"inner": np.int32(10)},
        }
        sanitized = _sanitize_for_json(data)
        assert isinstance(sanitized["int_val"], int)
        assert isinstance(sanitized["float_val"], float)
        assert sanitized["nan_val"] is None
        assert isinstance(sanitized["bool_val"], bool)
        assert isinstance(sanitized["array_val"], list)
        assert isinstance(sanitized["nested"]["inner"], int)

        # Confirm strict JSON compatibility
        dumped = json.dumps(sanitized)
        assert "42" in dumped
        assert "3.1415" in dumped

    def test_ml_report_to_dict_and_to_json(self) -> None:
        prep = MLPreprocessingResult(
            feature_names=["f1", "f2"],
            original_columns=["f1", "f2", "id"],
            numeric_columns=["f1", "f2"],
            categorical_columns=[],
            datetime_columns=[],
            boolean_columns=[],
            excluded_columns=["id"],
            exclusion_reasons={"id": "identifier"},
            generated_features=[],
            imputation_metadata={"numeric_strategy": "median"},
            encoding_metadata={},
            scaling_metadata={"strategy": "standard"},
            n_rows=100,
            n_features=2,
        )
        anomaly = AnomalyResult(
            algorithm="isolation_forest",
            n_anomalies=5,
            anomaly_percentage=5.0,
            anomaly_indices=[0, 1, 2, 3, 4],
            raw_scores=[-0.2, -0.1, 0.1, 0.2, 0.3],
            normalized_scores=[0.9, 0.8, 0.7, 0.6, 0.5],
            score_summary={"min": 0.5, "max": 0.9, "mean": 0.7, "median": 0.7},
        )
        cluster_sum = ClusterSummary(
            cluster_id=0,
            size=50,
            proportion=0.5,
            numerical_profiles={"f1": {"mean": 10.0, "median": 10.0, "std": 1.0, "min": 5.0, "max": 15.0}},
            categorical_profiles={},
        )
        clustering = ClusterResult(
            algorithm="kmeans",
            n_clusters=2,
            cluster_labels=[0] * 50 + [1] * 50,
            cluster_sizes={0: 50, 1: 50},
            cluster_proportions={0: 0.5, 1: 0.5},
            cluster_summaries=[cluster_sum],
            distinguishing_features=[{"feature": "f1", "separation_score": 2.5}],
            silhouette_score=0.62,
        )
        pca = PCAResult(
            n_components=2,
            explained_variance_ratio=[0.7, 0.3],
            cumulative_explained_variance=[0.7, 1.0],
            loadings={"PC1": {"f1": 0.71, "f2": 0.71}},
            coordinates=[{"row_id": 0, "PC1": 1.0, "PC2": 0.0}],
            top_features_per_component={"PC1": [{"feature": "f1", "loading": 0.71}]},
        )
        evaluation = MLEvaluation(
            metrics={"silhouette": 0.62, "anomaly_percentage": 5.0},
            metric_reasons={},
        )
        finding = MLFinding(
            finding_id="ML_ANOMALY_01",
            finding_type="anomaly",
            title="5% Outliers Detected",
            description="5 anomalies were discovered.",
            importance="HIGH",
            measured_values={"n_anomalies": 5},
            affected_columns=["f1"],
        )
        report = MLReport(
            dataset_name="test.csv",
            row_count=100,
            feature_count=2,
            preprocessing=prep,
            anomaly=anomaly,
            clustering=clustering,
            dimensionality=pca,
            evaluation=evaluation,
            findings=[finding],
        )

        d = report.to_dict()
        assert d["dataset_name"] == "test.csv"
        assert d["row_count"] == 100
        assert d["total_findings"] == 1

        json_str = report.to_json()
        assert "test.csv" in json_str
        assert "isolation_forest" in json_str

        # Check ASCII summary
        summary_text = report.summary()
        assert "MACHINE LEARNING INTELLIGENCE REPORT" in summary_text
        assert "test.csv" in summary_text
        assert "isolation_forest" in summary_text
        assert "kmeans" in summary_text

    def test_visualization_contracts(self) -> None:
        prep = MLPreprocessingResult(
            feature_names=["f1"],
            original_columns=["f1"],
            numeric_columns=["f1"],
            categorical_columns=[],
            datetime_columns=[],
            boolean_columns=[],
            excluded_columns=[],
            exclusion_reasons={},
            generated_features=[],
            imputation_metadata={},
            encoding_metadata={},
            scaling_metadata={},
            n_rows=2,
            n_features=1,
        )
        anomaly = AnomalyResult(
            algorithm="isolation_forest",
            n_anomalies=1,
            anomaly_percentage=50.0,
            anomaly_indices=[1],
            raw_scores=[0.1, -0.2],
            normalized_scores=[0.2, 0.9],
            score_summary={"mean": 0.55},
        )
        clustering = ClusterResult(
            algorithm="kmeans",
            n_clusters=2,
            cluster_labels=[0, 1],
            cluster_sizes={0: 1, 1: 1},
            cluster_proportions={0: 0.5, 1: 0.5},
            cluster_summaries=[],
            distinguishing_features=[],
        )
        pca = PCAResult(
            n_components=2,
            explained_variance_ratio=[0.8, 0.2],
            cumulative_explained_variance=[0.8, 1.0],
            loadings={"PC1": {"f1": 1.0}},
            coordinates=[
                {"row_id": 0, "PC1": 1.5, "PC2": 0.5},
                {"row_id": 1, "PC1": -1.5, "PC2": -0.5},
            ],
            top_features_per_component={},
        )
        report = MLReport(
            dataset_name="viz_test.csv",
            row_count=2,
            feature_count=1,
            preprocessing=prep,
            anomaly=anomaly,
            clustering=clustering,
            dimensionality=pca,
            evaluation=MLEvaluation(metrics={}, metric_reasons={}),
            findings=[],
        )

        viz = report.get_visualization_data()
        assert "pca_scatter" in viz
        assert viz["pca_scatter"]["chart_type"] == "scatter"
        assert len(viz["pca_scatter"]["points"]) == 2
        assert viz["pca_scatter"]["points"][1]["is_anomaly"] is True

        assert "pca_scree" in viz
        assert viz["pca_scree"]["chart_type"] == "bar_and_line"

        assert "cluster_distribution" in viz
        assert viz["cluster_distribution"]["chart_type"] == "bar"

        assert "anomaly_distribution" in viz
        assert viz["anomaly_distribution"]["chart_type"] == "histogram"
