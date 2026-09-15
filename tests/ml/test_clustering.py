"""Unit tests for unsupervised clustering in src/ml/clustering.py."""

import numpy as np
import pytest

from src.ml.clustering import DBSCANClusterer, KMeansClusterer
from src.ml.models import ClusterResult


class TestClustering:
    """Test suite for KMeans and DBSCAN clustering algorithms."""

    @pytest.fixture
    def three_clusters_data(self) -> np.ndarray:
        """Create synthetic data with 3 clearly separated Gaussian clusters."""
        np.random.seed(42)
        c1 = np.random.normal(loc=[0.0, 0.0], scale=0.5, size=(30, 2))
        c2 = np.random.normal(loc=[10.0, 10.0], scale=0.5, size=(30, 2))
        c3 = np.random.normal(loc=[0.0, 10.0], scale=0.5, size=(30, 2))
        return np.vstack([c1, c2, c3])

    def test_kmeans_fixed_k(self, three_clusters_data: np.ndarray) -> None:
        clusterer = KMeansClusterer(n_clusters=3, random_state=42)
        res = clusterer.fit_predict(three_clusters_data, feature_names=["x1", "x2"])

        assert isinstance(res, ClusterResult)
        assert res.algorithm == "kmeans"
        assert res.n_clusters == 3
        assert len(res.cluster_labels) == 90
        assert res.silhouette_score is not None
        assert res.silhouette_score > 0.6  # High separation
        assert res.inertia is not None

        # Verify cluster sizes and profiles
        assert len(res.cluster_summaries) == 3
        assert sum(res.cluster_sizes.values()) == 90
        assert len(res.distinguishing_features) > 0

    def test_kmeans_automatic_k_selection(self, three_clusters_data: np.ndarray) -> None:
        clusterer = KMeansClusterer(auto_k=True, min_k=2, max_k=5, random_state=42)
        res = clusterer.fit_predict(three_clusters_data)

        assert res.status == "completed"
        # Automatic search should select K=3 as optimal
        assert res.n_clusters == 3
        assert res.auto_k_evaluation is not None
        assert res.auto_k_evaluation["selected_k"] == 3
        assert "silhouette_scores" in res.auto_k_evaluation
        assert len(res.auto_k_evaluation["candidate_k_values"]) >= 3

    def test_dbscan_clustering_and_noise(self) -> None:
        np.random.seed(42)
        c1 = np.random.normal(loc=[0.0, 0.0], scale=0.2, size=(20, 2))
        c2 = np.random.normal(loc=[5.0, 5.0], scale=0.2, size=(20, 2))
        noise = np.array([[100.0, 100.0], [-50.0, 50.0]])  # Clear noise
        X = np.vstack([c1, c2, noise])

        clusterer = DBSCANClusterer(eps=0.8, min_samples=5)
        res = clusterer.fit_predict(X)

        assert res.algorithm == "dbscan"
        assert res.status == "completed"
        assert res.n_clusters == 2
        assert res.noise_count == 2
        assert np.isclose(res.noise_percentage, (2 / 42) * 100.0)

        # Check noise label exists in cluster summaries
        noise_summary = [cs for cs in res.cluster_summaries if cs.is_noise]
        assert len(noise_summary) == 1
        assert noise_summary[0].size == 2

    def test_clustering_edge_cases(self) -> None:
        empty_X = np.zeros((0, 0))
        km = KMeansClusterer()
        res_empty = km.fit_predict(empty_X)
        assert res_empty.status == "unavailable"

        single_X = np.array([[1.0, 2.0]])
        res_single = km.fit_predict(single_X)
        assert res_single.status == "completed"
        assert res_single.n_clusters == 1
