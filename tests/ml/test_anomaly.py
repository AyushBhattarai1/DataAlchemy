"""Unit tests for unsupervised anomaly detection in src/ml/anomaly.py."""

import numpy as np
import pytest

from src.ml.anomaly import IsolationForestDetector, LocalOutlierFactorDetector
from src.ml.models import AnomalyResult


class TestAnomalyDetection:
    """Test suite for Isolation Forest and Local Outlier Factor anomaly detectors."""

    @pytest.fixture
    def synthetic_data(self) -> np.ndarray:
        """Create synthetic cluster with 50 inliers and 5 obvious outliers."""
        np.random.seed(42)
        inliers = np.random.normal(loc=0.0, scale=1.0, size=(50, 4))
        outliers = np.random.uniform(low=10.0, high=15.0, size=(5, 4))
        return np.vstack([inliers, outliers])

    def test_isolation_forest_detection_and_normalization(self, synthetic_data: np.ndarray) -> None:
        detector = IsolationForestDetector(contamination=0.1, random_state=42)
        res = detector.fit_predict(synthetic_data, feature_names=["f1", "f2", "f3", "f4"])

        assert isinstance(res, AnomalyResult)
        assert res.algorithm == "isolation_forest"
        assert res.status == "completed"
        assert res.n_anomalies > 0
        assert 5.0 <= res.anomaly_percentage <= 15.0

        # Verify scores are normalized into [0, 1]
        assert all(0.0 <= s <= 1.0 for s in res.normalized_scores)
        assert res.score_summary["min"] >= 0.0
        assert res.score_summary["max"] <= 1.0

        # Verify the known outliers (indices 50-54) have higher anomaly scores than inliers
        inlier_mean_score = np.mean(res.normalized_scores[:50])
        outlier_mean_score = np.mean(res.normalized_scores[50:])
        assert outlier_mean_score > inlier_mean_score

        # Feature contributions present
        assert res.feature_contributions is not None
        assert len(res.feature_contributions) == 4

    def test_isolation_forest_determinism(self, synthetic_data: np.ndarray) -> None:
        d1 = IsolationForestDetector(contamination=0.1, random_state=42)
        r1 = d1.fit_predict(synthetic_data)

        d2 = IsolationForestDetector(contamination=0.1, random_state=42)
        r2 = d2.fit_predict(synthetic_data)

        assert r1.anomaly_indices == r2.anomaly_indices
        assert np.allclose(r1.normalized_scores, r2.normalized_scores)

    def test_local_outlier_factor_detection(self, synthetic_data: np.ndarray) -> None:
        detector = LocalOutlierFactorDetector(n_neighbors=15, contamination=0.1)
        res = detector.fit_predict(synthetic_data)

        assert isinstance(res, AnomalyResult)
        assert res.algorithm == "local_outlier_factor"
        assert res.status == "completed"
        assert res.n_anomalies > 0
        assert all(0.0 <= s <= 1.0 for s in res.normalized_scores)

    def test_small_dataset_adaptation(self) -> None:
        # Small dataset with only 4 samples where default n_neighbors=20 exceeds sample size
        small_X = np.array([
            [1.0, 2.0],
            [1.1, 2.1],
            [0.9, 1.9],
            [10.0, 20.0],
        ])

        # LOF should adapt neighbors to 3 without raising an exception
        lof = LocalOutlierFactorDetector(n_neighbors=20, contamination=0.25)
        res_lof = lof.fit_predict(small_X)
        assert res_lof.status == "completed"
        assert len(res_lof.normalized_scores) == 4

        # Isolation Forest on small dataset
        ifo = IsolationForestDetector(contamination=0.25, random_state=42)
        res_ifo = ifo.fit_predict(small_X)
        assert res_ifo.status == "completed"
        assert len(res_ifo.normalized_scores) == 4

    def test_empty_and_single_sample_edge_cases(self) -> None:
        empty_X = np.zeros((0, 0))
        ifo = IsolationForestDetector()
        res_empty = ifo.fit_predict(empty_X)
        assert res_empty.status == "unavailable"

        single_X = np.array([[1.0, 2.0]])
        res_single = ifo.fit_predict(single_X)
        assert res_single.status == "completed"
        assert res_single.n_anomalies == 0
