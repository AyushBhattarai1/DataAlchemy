"""Consolidated mathematical evaluation subsystem for DataAlchemy machine learning.

Computes mathematically valid evaluation metrics for anomaly detection, clustering,
and dimensionality reduction with transparent reasoning when metrics are undefined.
"""

from typing import Dict, Optional
from src.ml.models import (
    AnomalyResult,
    ClusterResult,
    MLEvaluation,
    PCAResult,
)


class MLEvaluator:
    """Evaluates mathematical soundness and performance metrics of unsupervised ML models."""

    def evaluate(
        self,
        anomaly: AnomalyResult,
        clustering: ClusterResult,
        pca: PCAResult,
    ) -> MLEvaluation:
        """Consolidate evaluation metrics across all ML components.

        Args:
            anomaly: Results from anomaly detection.
            clustering: Results from clustering discovery.
            pca: Results from dimensionality reduction.

        Returns:
            MLEvaluation: Dictionary of metric values and explanation reasons.
        """
        metrics: Dict[str, Optional[float]] = {}
        reasons: Dict[str, str] = {}

        # 1. Anomaly Metrics
        if anomaly.status == "completed":
            metrics["anomaly_count"] = float(anomaly.n_anomalies)
            metrics["anomaly_percentage"] = float(anomaly.anomaly_percentage)
            metrics["anomaly_mean_score"] = float(anomaly.score_summary.get("mean", 0.0))
        else:
            metrics["anomaly_count"] = None
            reasons["anomaly_count"] = f"Anomaly detection {anomaly.status}: {anomaly.error_message}"
            metrics["anomaly_percentage"] = None
            reasons["anomaly_percentage"] = "Anomaly detection was not completed."
            metrics["anomaly_mean_score"] = None
            reasons["anomaly_mean_score"] = "Anomaly detection was not completed."

        # 2. Clustering Metrics
        if clustering.status == "completed":
            metrics["cluster_count"] = float(clustering.n_clusters)
            metrics["cluster_silhouette"] = clustering.silhouette_score
            if clustering.silhouette_score is None:
                reasons["cluster_silhouette"] = (
                    "Silhouette score is mathematically undefined when fewer than 2 clusters are present."
                )

            metrics["cluster_inertia"] = clustering.inertia
            if clustering.inertia is None:
                reasons["cluster_inertia"] = (
                    "Inertia is only defined for centroid-based clustering (e.g. K-Means)."
                )

            if clustering.algorithm == "dbscan":
                metrics["dbscan_noise_percentage"] = float(clustering.noise_percentage)
        else:
            metrics["cluster_count"] = None
            reasons["cluster_count"] = f"Clustering {clustering.status}: {clustering.error_message}"
            metrics["cluster_silhouette"] = None
            reasons["cluster_silhouette"] = "Clustering was not completed."
            metrics["cluster_inertia"] = None
            reasons["cluster_inertia"] = "Clustering was not completed."

        # 3. PCA Metrics
        if pca.status == "completed":
            if len(pca.explained_variance_ratio) >= 2:
                metrics["pca_variance_top2"] = float(sum(pca.explained_variance_ratio[:2]))
            elif len(pca.explained_variance_ratio) == 1:
                metrics["pca_variance_top2"] = float(pca.explained_variance_ratio[0])
            else:
                metrics["pca_variance_top2"] = None
                reasons["pca_variance_top2"] = "Fewer than 1 component extracted."

            if pca.cumulative_explained_variance:
                metrics["pca_cumulative_variance"] = float(pca.cumulative_explained_variance[-1])
            else:
                metrics["pca_cumulative_variance"] = None
                reasons["pca_cumulative_variance"] = "No cumulative variance available."
        else:
            metrics["pca_variance_top2"] = None
            reasons["pca_variance_top2"] = f"PCA {pca.status}: {pca.error_message}"
            metrics["pca_cumulative_variance"] = None
            reasons["pca_cumulative_variance"] = "PCA was not completed."

        return MLEvaluation(
            metrics=metrics,
            metric_reasons=reasons,
        )
