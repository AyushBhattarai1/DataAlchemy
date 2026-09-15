"""Unsupervised anomaly detection subsystem for DataAlchemy.

Implements Isolation Forest and Local Outlier Factor algorithms with normalized anomaly scoring,
small-dataset resilience, and feature contribution attribution.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from src.ml.models import AnomalyRecord, AnomalyResult


class AnomalyDetector(ABC):
    """Abstract base class for unsupervised anomaly detectors."""

    @abstractmethod
    def fit_predict(
        self,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> AnomalyResult:
        """Fit detector on feature matrix and return structured AnomalyResult."""
        pass


class IsolationForestDetector(AnomalyDetector):
    """Isolation Forest anomaly detection with normalized scoring and feature contribution."""

    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 100,
        random_state: int = 42,
    ) -> None:
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state

    def fit_predict(
        self,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> AnomalyResult:
        """Run Isolation Forest detection on feature matrix X."""
        n_rows, n_cols = X.shape
        if n_rows == 0 or n_cols == 0:
            return AnomalyResult(
                algorithm="isolation_forest",
                n_anomalies=0,
                anomaly_percentage=0.0,
                anomaly_indices=[],
                raw_scores=[],
                normalized_scores=[],
                score_summary={"min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0},
                status="unavailable",
                error_message="Feature matrix is empty or has zero features.",
            )

        if n_rows < 2:
            return AnomalyResult(
                algorithm="isolation_forest",
                n_anomalies=0,
                anomaly_percentage=0.0,
                anomaly_indices=[],
                raw_scores=[0.0],
                normalized_scores=[0.0],
                score_summary={"min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0},
                status="completed",
                metadata={"note": "Dataset has fewer than 2 samples; no anomalies identified."},
            )

        try:
            model = IsolationForest(
                contamination=self.contamination,
                n_estimators=self.n_estimators,
                random_state=self.random_state,
            )
            preds = model.fit_predict(X)  # -1 = anomaly, 1 = normal
            raw_scores = model.decision_function(X)  # Lower is more abnormal

            is_anomaly_mask = preds == -1
            anomaly_indices = [int(i) for i in np.where(is_anomaly_mask)[0]]
            n_anomalies = len(anomaly_indices)
            anomaly_percentage = (n_anomalies / n_rows) * 100.0

            # Normalize scores into [0, 1] where 1.0 = most anomalous
            min_score = float(np.min(raw_scores))
            max_score = float(np.max(raw_scores))
            if max_score > min_score:
                normalized_scores = (max_score - raw_scores) / (max_score - min_score)
            else:
                normalized_scores = np.zeros_like(raw_scores)

            score_summary = {
                "min": float(np.min(normalized_scores)),
                "max": float(np.max(normalized_scores)),
                "mean": float(np.mean(normalized_scores)),
                "median": float(np.median(normalized_scores)),
                "raw_decision_min": min_score,
                "raw_decision_max": max_score,
            }

            # Approximate feature contribution for anomalies
            feature_contributions: Optional[Dict[str, float]] = None
            if feature_names and n_anomalies > 0 and n_anomalies < n_rows:
                inlier_mask = ~is_anomaly_mask
                inlier_means = np.mean(X[inlier_mask], axis=0)
                inlier_stds = np.std(X[inlier_mask], axis=0) + 1e-6

                outlier_X = X[is_anomaly_mask]
                deviations = np.mean(np.abs(outlier_X - inlier_means) / inlier_stds, axis=0)
                total_dev = np.sum(deviations)
                if total_dev > 0:
                    norm_devs = deviations / total_dev
                    feature_contributions = {
                        feature_names[i]: float(norm_devs[i])
                        for i in range(len(feature_names))
                    }
                    # Sort by contribution descending
                    feature_contributions = dict(
                        sorted(feature_contributions.items(), key=lambda x: x[1], reverse=True)
                    )

            return AnomalyResult(
                algorithm="isolation_forest",
                n_anomalies=n_anomalies,
                anomaly_percentage=float(anomaly_percentage),
                anomaly_indices=anomaly_indices,
                raw_scores=[float(s) for s in raw_scores],
                normalized_scores=[float(s) for s in normalized_scores],
                score_summary=score_summary,
                feature_contributions=feature_contributions,
                status="completed",
            )
        except Exception as err:
            return AnomalyResult(
                algorithm="isolation_forest",
                n_anomalies=0,
                anomaly_percentage=0.0,
                anomaly_indices=[],
                raw_scores=[],
                normalized_scores=[],
                score_summary={},
                status="failed",
                error_message=str(err),
            )


class LocalOutlierFactorDetector(AnomalyDetector):
    """Local Outlier Factor (LOF) anomaly detection with small-dataset adaptation."""

    def __init__(
        self,
        n_neighbors: int = 20,
        contamination: float = 0.05,
        metric: str = "euclidean",
    ) -> None:
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.metric = metric

    def fit_predict(
        self,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> AnomalyResult:
        """Run Local Outlier Factor detection on feature matrix X."""
        n_rows, n_cols = X.shape
        if n_rows == 0 or n_cols == 0:
            return AnomalyResult(
                algorithm="local_outlier_factor",
                n_anomalies=0,
                anomaly_percentage=0.0,
                anomaly_indices=[],
                raw_scores=[],
                normalized_scores=[],
                score_summary={"min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0},
                status="unavailable",
                error_message="Feature matrix is empty or has zero features.",
            )

        if n_rows < 3:
            return AnomalyResult(
                algorithm="local_outlier_factor",
                n_anomalies=0,
                anomaly_percentage=0.0,
                anomaly_indices=[],
                raw_scores=[0.0] * n_rows,
                normalized_scores=[0.0] * n_rows,
                score_summary={"min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0},
                status="completed",
                metadata={"note": "Insufficient samples for LOF density estimation."},
            )

        # Adapt neighbors for smaller datasets
        effective_k = min(self.n_neighbors, max(1, n_rows - 1))

        try:
            model = LocalOutlierFactor(
                n_neighbors=effective_k,
                contamination=self.contamination,
                metric=self.metric,
            )
            preds = model.fit_predict(X)  # -1 = anomaly, 1 = normal
            # negative_outlier_factor_: larger negative is more anomalous
            raw_nof = model.negative_outlier_factor_

            is_anomaly_mask = preds == -1
            anomaly_indices = [int(i) for i in np.where(is_anomaly_mask)[0]]
            n_anomalies = len(anomaly_indices)
            anomaly_percentage = (n_anomalies / n_rows) * 100.0

            # Normalize raw scores to [0, 1] where 1.0 = highest outlier factor
            min_score = float(np.min(raw_nof))
            max_score = float(np.max(raw_nof))
            if max_score > min_score:
                normalized_scores = (max_score - raw_nof) / (max_score - min_score)
            else:
                normalized_scores = np.zeros_like(raw_nof)

            score_summary = {
                "min": float(np.min(normalized_scores)),
                "max": float(np.max(normalized_scores)),
                "mean": float(np.mean(normalized_scores)),
                "median": float(np.median(normalized_scores)),
                "raw_nof_min": min_score,
                "raw_nof_max": max_score,
            }

            return AnomalyResult(
                algorithm="local_outlier_factor",
                n_anomalies=n_anomalies,
                anomaly_percentage=float(anomaly_percentage),
                anomaly_indices=anomaly_indices,
                raw_scores=[float(s) for s in raw_nof],
                normalized_scores=[float(s) for s in normalized_scores],
                score_summary=score_summary,
                feature_contributions=None,  # LOF does not support native feature attribution
                status="completed",
                metadata={"effective_neighbors": effective_k},
            )
        except Exception as err:
            return AnomalyResult(
                algorithm="local_outlier_factor",
                n_anomalies=0,
                anomaly_percentage=0.0,
                anomaly_indices=[],
                raw_scores=[],
                normalized_scores=[],
                score_summary={},
                status="failed",
                error_message=str(err),
            )
