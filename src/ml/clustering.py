"""Unsupervised clustering discovery subsystem for DataAlchemy.

Implements K-Means (with automatic K selection, inertia, and silhouette optimization)
and DBSCAN density clustering with noise handling and cluster profiling.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score

from src.ml.models import ClusterResult, ClusterSummary


class Clusterer(ABC):
    """Abstract base class for clustering algorithms."""

    @abstractmethod
    def fit_predict(
        self,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None,
        original_df: Optional[pd.DataFrame] = None,
    ) -> ClusterResult:
        """Run clustering on feature matrix X and return structured ClusterResult."""
        pass


def _characterize_clusters(
    X: np.ndarray,
    labels: np.ndarray,
    feature_names: Optional[List[str]] = None,
    original_df: Optional[pd.DataFrame] = None,
) -> Tuple[List[ClusterSummary], List[Dict[str, Any]]]:
    """Calculate descriptive statistics for each cluster and identify distinguishing features."""
    unique_labels = sorted(list(set(labels)))
    n_samples, n_features = X.shape
    names = feature_names or [f"feature_{i}" for i in range(n_features)]

    summaries: List[ClusterSummary] = []
    cluster_means: Dict[int, np.ndarray] = {}

    for c_id in unique_labels:
        mask = labels == c_id
        cluster_X = X[mask]
        c_size = int(np.sum(mask))
        c_prop = float(c_size / n_samples)
        is_noise = c_id == -1

        num_profiles: Dict[str, Dict[str, float]] = {}
        if c_size > 0:
            cluster_means[c_id] = np.mean(cluster_X, axis=0)
            for idx, feat_name in enumerate(names[:25]):  # Cap at top 25 features for summary
                vals = cluster_X[:, idx]
                num_profiles[feat_name] = {
                    "mean": float(np.mean(vals)),
                    "median": float(np.median(vals)),
                    "std": float(np.std(vals)) if c_size > 1 else 0.0,
                    "min": float(np.min(vals)),
                    "max": float(np.max(vals)),
                }

        cat_profiles: Dict[str, Dict[str, Any]] = {}
        if original_df is not None and c_size > 0:
            c_df = original_df.iloc[mask]
            cat_cols = [c for c in c_df.columns if not pd.api.types.is_numeric_dtype(c_df[c])][:5]
            for col in cat_cols:
                series = c_df[col].dropna()
                if len(series) > 0:
                    top_val = str(series.mode().iloc[0])
                    top_cnt = int((series == top_val).sum())
                    cat_profiles[col] = {
                        "dominant_category": top_val,
                        "count": top_cnt,
                        "frequency": float(top_cnt / len(series)),
                    }

        summaries.append(ClusterSummary(
            cluster_id=int(c_id),
            size=c_size,
            proportion=c_prop,
            numerical_profiles=num_profiles,
            categorical_profiles=cat_profiles,
            is_noise=is_noise,
        ))

    # Rank features by variance between non-noise cluster centroids
    distinguishing_features: List[Dict[str, Any]] = []
    valid_c_ids = [c for c in unique_labels if c != -1]
    if len(valid_c_ids) > 1 and n_features > 0:
        valid_means = np.array([cluster_means[c] for c in valid_c_ids])
        between_variance = np.var(valid_means, axis=0)
        overall_std = np.std(X, axis=0) + 1e-6
        separation_ratio = between_variance / (overall_std ** 2)

        sorted_indices = np.argsort(separation_ratio)[::-1]
        for idx in sorted_indices[:5]:
            distinguishing_features.append({
                "feature": names[idx],
                "separation_score": float(separation_ratio[idx]),
                "cluster_means": {
                    f"Cluster {c}": float(cluster_means[c][idx]) for c in valid_c_ids
                },
            })

    return summaries, distinguishing_features


class KMeansClusterer(Clusterer):
    """K-Means clustering with optional automatic K selection and cluster profiling."""

    def __init__(
        self,
        n_clusters: Optional[int] = 3,
        auto_k: bool = False,
        min_k: int = 2,
        max_k: int = 8,
        random_state: int = 42,
        n_init: int = 10,
        max_iter: int = 300,
    ) -> None:
        self.n_clusters = n_clusters
        self.auto_k = auto_k
        self.min_k = min_k
        self.max_k = max_k
        self.random_state = random_state
        self.n_init = n_init
        self.max_iter = max_iter

    def fit_predict(
        self,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None,
        original_df: Optional[pd.DataFrame] = None,
    ) -> ClusterResult:
        """Run K-Means clustering on feature matrix X."""
        n_rows, n_cols = X.shape
        if n_rows == 0 or n_cols == 0:
            return ClusterResult(
                algorithm="kmeans",
                n_clusters=0,
                cluster_labels=[],
                cluster_sizes={},
                cluster_proportions={},
                cluster_summaries=[],
                distinguishing_features=[],
                status="unavailable",
                error_message="Feature matrix is empty or has zero features.",
            )

        if n_rows < 2:
            return ClusterResult(
                algorithm="kmeans",
                n_clusters=1,
                cluster_labels=[0] * n_rows,
                cluster_sizes={0: n_rows},
                cluster_proportions={0: 1.0},
                cluster_summaries=[],
                distinguishing_features=[],
                status="completed",
                metadata={"note": "Single sample dataset; assigned to cluster 0."},
            )

        try:
            auto_k_eval: Optional[Dict[str, Any]] = None
            selected_k = self.n_clusters or 3

            # Run automatic K search if requested or if n_clusters is None
            if self.auto_k or self.n_clusters is None:
                max_candidate = min(self.max_k, n_rows - 1)
                min_candidate = min(self.min_k, max_candidate)

                if min_candidate < max_candidate:
                    candidate_ks = list(range(min_candidate, max_candidate + 1))
                    inertias: List[float] = []
                    silhouettes: List[float] = []
                    best_k = candidate_ks[0]
                    best_score = -1.0

                    for k in candidate_ks:
                        km = KMeans(
                            n_clusters=k,
                            random_state=self.random_state,
                            n_init=self.n_init,
                            max_iter=self.max_iter,
                        )
                        labels_k = km.fit_predict(X)
                        score = float(silhouette_score(X, labels_k))
                        inertias.append(float(km.inertia_))
                        silhouettes.append(score)
                        if score > best_score:
                            best_score = score
                            best_k = k

                    selected_k = best_k
                    auto_k_eval = {
                        "candidate_k_values": candidate_ks,
                        "inertias": inertias,
                        "silhouette_scores": silhouettes,
                        "selected_k": selected_k,
                        "selection_metric": "silhouette_score",
                    }
                else:
                    selected_k = min_candidate

            # Cap selected_k to valid range
            selected_k = max(1, min(selected_k, n_rows))

            model = KMeans(
                n_clusters=selected_k,
                random_state=self.random_state,
                n_init=self.n_init,
                max_iter=self.max_iter,
            )
            labels = model.fit_predict(X)
            inertia = float(model.inertia_)

            # Calculate silhouette score if valid
            sil_score: Optional[float] = None
            if 2 <= selected_k < n_rows:
                sil_score = float(silhouette_score(X, labels))

            # Cluster counts and proportions
            cluster_sizes: Dict[int, int] = {}
            cluster_props: Dict[int, float] = {}
            for c_id in range(selected_k):
                cnt = int(np.sum(labels == c_id))
                cluster_sizes[c_id] = cnt
                cluster_props[c_id] = float(cnt / n_rows)

            summaries, distinguishing = _characterize_clusters(
                X=X,
                labels=labels,
                feature_names=feature_names,
                original_df=original_df,
            )

            return ClusterResult(
                algorithm="kmeans",
                n_clusters=selected_k,
                cluster_labels=[int(lbl) for lbl in labels],
                cluster_sizes=cluster_sizes,
                cluster_proportions=cluster_props,
                cluster_summaries=summaries,
                distinguishing_features=distinguishing,
                inertia=inertia,
                silhouette_score=sil_score,
                auto_k_evaluation=auto_k_eval,
                status="completed",
            )
        except Exception as err:
            return ClusterResult(
                algorithm="kmeans",
                n_clusters=0,
                cluster_labels=[],
                cluster_sizes={},
                cluster_proportions={},
                cluster_summaries=[],
                distinguishing_features=[],
                status="failed",
                error_message=str(err),
            )


class DBSCANClusterer(Clusterer):
    """Density-Based Spatial Clustering of Applications with Noise (DBSCAN)."""

    def __init__(
        self,
        eps: float = 0.5,
        min_samples: int = 5,
        metric: str = "euclidean",
    ) -> None:
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric

    def fit_predict(
        self,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None,
        original_df: Optional[pd.DataFrame] = None,
    ) -> ClusterResult:
        """Run DBSCAN clustering on feature matrix X."""
        n_rows, n_cols = X.shape
        if n_rows == 0 or n_cols == 0:
            return ClusterResult(
                algorithm="dbscan",
                n_clusters=0,
                cluster_labels=[],
                cluster_sizes={},
                cluster_proportions={},
                cluster_summaries=[],
                distinguishing_features=[],
                status="unavailable",
                error_message="Feature matrix is empty or has zero features.",
            )

        if n_rows < self.min_samples:
            effective_min = max(2, n_rows)
        else:
            effective_min = self.min_samples

        try:
            model = DBSCAN(
                eps=self.eps,
                min_samples=effective_min,
                metric=self.metric,
            )
            labels = model.fit_predict(X)

            unique_labels = set(labels)
            core_clusters = [lbl for lbl in unique_labels if lbl != -1]
            n_clusters = len(core_clusters)

            noise_count = int(np.sum(labels == -1))
            noise_pct = float((noise_count / n_rows) * 100.0)

            cluster_sizes: Dict[int, int] = {}
            cluster_props: Dict[int, float] = {}
            for c_id in sorted(list(unique_labels)):
                cnt = int(np.sum(labels == c_id))
                cluster_sizes[int(c_id)] = cnt
                cluster_props[int(c_id)] = float(cnt / n_rows)

            sil_score: Optional[float] = None
            if 2 <= n_clusters < n_rows and noise_count < n_rows:
                # Compute silhouette on non-noise points if possible
                non_noise_mask = labels != -1
                if np.sum(non_noise_mask) > n_clusters:
                    try:
                        sil_score = float(silhouette_score(X[non_noise_mask], labels[non_noise_mask]))
                    except Exception:
                        pass

            summaries, distinguishing = _characterize_clusters(
                X=X,
                labels=labels,
                feature_names=feature_names,
                original_df=original_df,
            )

            return ClusterResult(
                algorithm="dbscan",
                n_clusters=n_clusters,
                cluster_labels=[int(lbl) for lbl in labels],
                cluster_sizes=cluster_sizes,
                cluster_proportions=cluster_props,
                cluster_summaries=summaries,
                distinguishing_features=distinguishing,
                silhouette_score=sil_score,
                noise_count=noise_count,
                noise_percentage=noise_pct,
                status="completed",
                metadata={"eps": self.eps, "min_samples": effective_min},
            )
        except Exception as err:
            return ClusterResult(
                algorithm="dbscan",
                n_clusters=0,
                cluster_labels=[],
                cluster_sizes={},
                cluster_proportions={},
                cluster_summaries=[],
                distinguishing_features=[],
                status="failed",
                error_message=str(err),
            )
