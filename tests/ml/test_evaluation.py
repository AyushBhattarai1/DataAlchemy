"""Unit tests for MLEvaluator in src/ml/evaluation.py."""

from src.ml.evaluation import MLEvaluator
from src.ml.models import (
    AnomalyResult,
    ClusterResult,
    MLEvaluation,
    PCAResult,
)


class TestMLEvaluator:
    """Test suite for ML evaluation metrics and explanation generation."""

    def test_evaluate_valid_metrics(self) -> None:
        evaluator = MLEvaluator()

        anomaly = AnomalyResult(
            algorithm="isolation_forest",
            n_anomalies=4,
            anomaly_percentage=4.0,
            anomaly_indices=[0, 1, 2, 3],
            raw_scores=[0.1] * 100,
            normalized_scores=[0.8] * 100,
            score_summary={"mean": 0.45},
            status="completed",
        )
        clustering = ClusterResult(
            algorithm="kmeans",
            n_clusters=3,
            cluster_labels=[0] * 33 + [1] * 33 + [2] * 34,
            cluster_sizes={0: 33, 1: 33, 2: 34},
            cluster_proportions={0: 0.33, 1: 0.33, 2: 0.34},
            cluster_summaries=[],
            distinguishing_features=[],
            silhouette_score=0.55,
            inertia=120.5,
            status="completed",
        )
        pca = PCAResult(
            n_components=2,
            explained_variance_ratio=[0.60, 0.25],
            cumulative_explained_variance=[0.60, 0.85],
            loadings={},
            coordinates=[],
            top_features_per_component={},
            status="completed",
        )

        res = evaluator.evaluate(anomaly, clustering, pca)
        assert isinstance(res, MLEvaluation)

        assert res.metrics["anomaly_count"] == 4.0
        assert res.metrics["anomaly_percentage"] == 4.0
        assert res.metrics["cluster_count"] == 3.0
        assert res.metrics["cluster_silhouette"] == 0.55
        assert res.metrics["cluster_inertia"] == 120.5
        assert res.metrics["pca_variance_top2"] == 0.85

    def test_evaluate_unavailable_and_none_reasons(self) -> None:
        evaluator = MLEvaluator()

        anomaly = AnomalyResult(
            algorithm="isolation_forest",
            n_anomalies=0,
            anomaly_percentage=0.0,
            anomaly_indices=[],
            raw_scores=[],
            normalized_scores=[],
            score_summary={},
            status="unavailable",
            error_message="Feature matrix is empty.",
        )
        # 1-cluster result where silhouette is mathematically invalid
        clustering = ClusterResult(
            algorithm="kmeans",
            n_clusters=1,
            cluster_labels=[0] * 10,
            cluster_sizes={0: 10},
            cluster_proportions={0: 1.0},
            cluster_summaries=[],
            distinguishing_features=[],
            silhouette_score=None,
            inertia=50.0,
            status="completed",
        )
        pca = PCAResult(
            n_components=0,
            explained_variance_ratio=[],
            cumulative_explained_variance=[],
            loadings={},
            coordinates=[],
            top_features_per_component={},
            status="unavailable",
            error_message="Insufficient features.",
        )

        res = evaluator.evaluate(anomaly, clustering, pca)

        # Anomaly metrics should be None with explanation
        assert res.metrics["anomaly_count"] is None
        assert "unavailable" in res.metric_reasons["anomaly_count"]

        # Silhouette should be None with clear reason
        assert res.metrics["cluster_silhouette"] is None
        assert "mathematically undefined" in res.metric_reasons["cluster_silhouette"].lower()

        # PCA metrics should be None with explanation
        assert res.metrics["pca_variance_top2"] is None
        assert "unavailable" in res.metric_reasons["pca_variance_top2"]
