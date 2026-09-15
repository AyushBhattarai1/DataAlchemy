"""High-level ML Intelligence Engine orchestrator for DataAlchemy.

Coordinates preprocessing, anomaly detection, clustering discovery, dimensionality reduction,
and structured evidence extraction into a unified, serializable MLReport.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd

from src.ingestion.loader import Dataset
from src.ml.anomaly import AnomalyDetector, IsolationForestDetector, LocalOutlierFactorDetector
from src.ml.clustering import Clusterer, DBSCANClusterer, KMeansClusterer
from src.ml.dimensionality import PCAReducer
from src.ml.evaluation import MLEvaluator
from src.ml.models import (
    AnomalyResult,
    ClusterResult,
    MLFinding,
    MLPreprocessingResult,
    MLReport,
    PCAResult,
)
from src.ml.preprocessing import MLPreprocessor
from src.profiling.profiler import DatasetProfile


@dataclass
class MLConfig:
    """Unified configuration for DataAlchemy machine learning algorithms."""
    # Preprocessing
    numeric_imputation: str = "median"
    scaling: Optional[str] = "standard"
    max_categories: int = 50

    # Anomaly Detection
    anomaly_algorithm: str = "isolation_forest"  # "isolation_forest" or "local_outlier_factor"
    anomaly_contamination: float = 0.05

    # Clustering
    clustering_algorithm: str = "kmeans"  # "kmeans", "auto_kmeans", or "dbscan"
    n_clusters: Optional[int] = 3
    auto_k: bool = False
    min_k: int = 2
    max_k: int = 8
    dbscan_eps: float = 0.5
    dbscan_min_samples: int = 5

    # PCA
    pca_components: int = 2

    # Global
    random_state: int = 42
    run_anomaly: bool = True
    run_clustering: bool = True
    run_pca: bool = True


class MLEngine:
    """Orchestrates machine learning analysis pipelines across tabular datasets."""

    def __init__(self, config: Optional[MLConfig] = None) -> None:
        self.config = config or MLConfig()
        self.evaluator = MLEvaluator()

    def _extract_df(self, data: Union[Dataset, pd.DataFrame]) -> Tuple[pd.DataFrame, str]:
        """Extract DataFrame and dataset name safely."""
        if isinstance(data, Dataset):
            return data.dataframe, data.filename
        elif isinstance(data, pd.DataFrame):
            return data, "dataframe"
        raise TypeError(f"Expected Dataset or pd.DataFrame, got {type(data).__name__}")

    def _extract_ml_findings(
        self,
        anomaly: AnomalyResult,
        clustering: ClusterResult,
        pca: PCAResult,
    ) -> List[MLFinding]:
        """Convert ML discoveries into prioritized, evidence-linked MLFinding objects."""
        findings: List[MLFinding] = []

        # 1. Anomaly Findings
        if anomaly.status == "completed" and anomaly.n_anomalies > 0:
            importance = "HIGH" if anomaly.anomaly_percentage >= 5.0 else "MEDIUM"
            top_feats = (
                list(anomaly.feature_contributions.keys())[:3]
                if anomaly.feature_contributions
                else []
            )
            feat_desc = f" Top contributing attributes: {', '.join(top_feats)}." if top_feats else ""
            findings.append(MLFinding(
                finding_id=f"ML_ANOMALY_{anomaly.algorithm.upper()}",
                finding_type="anomaly",
                title=f"Statistical anomalies identified ({anomaly.anomaly_percentage:.1f}%)",
                description=(
                    f"{anomaly.n_anomalies:,} observations ({anomaly.anomaly_percentage:.1f}%) "
                    f"were identified as statistical outliers using {anomaly.algorithm}.{feat_desc}"
                ),
                importance=importance,
                measured_values={
                    "n_anomalies": anomaly.n_anomalies,
                    "anomaly_percentage": anomaly.anomaly_percentage,
                    "mean_score": anomaly.score_summary.get("mean", 0.0),
                    "algorithm": anomaly.algorithm,
                },
                affected_columns=top_feats,
                affected_rows=anomaly.anomaly_indices[:20],
            ))

        # 2. Clustering Findings
        if clustering.status == "completed" and clustering.n_clusters > 1:
            importance = (
                "HIGH"
                if (clustering.silhouette_score is not None and clustering.silhouette_score >= 0.45)
                else "MEDIUM"
            )
            top_diff_feat = (
                clustering.distinguishing_features[0]["feature"]
                if clustering.distinguishing_features
                else "key features"
            )
            sil_str = (
                f" with silhouette score of {clustering.silhouette_score:.2f}"
                if clustering.silhouette_score is not None
                else ""
            )
            findings.append(MLFinding(
                finding_id=f"ML_CLUSTER_{clustering.algorithm.upper()}",
                finding_type="cluster",
                title=f"{clustering.n_clusters} distinct behavioural clusters discovered",
                description=(
                    f"Unsupervised clustering grouped data into {clustering.n_clusters} natural clusters{sil_str}. "
                    f"Primary distinguishing feature across clusters is '{top_diff_feat}'."
                ),
                importance=importance,
                measured_values={
                    "n_clusters": clustering.n_clusters,
                    "silhouette_score": clustering.silhouette_score,
                    "cluster_sizes": clustering.cluster_sizes,
                    "noise_count": clustering.noise_count,
                },
                affected_columns=[
                    d["feature"] for d in clustering.distinguishing_features[:3]
                ],
            ))

        # 3. PCA Variance Findings
        if pca.status == "completed" and pca.explained_variance_ratio:
            top2_var = sum(pca.explained_variance_ratio[:2])
            importance = "HIGH" if top2_var >= 0.60 else "INFO"
            top_pc1_feats = [
                item["feature"]
                for item in pca.top_features_per_component.get("PC1", [])[:3]
            ]
            findings.append(MLFinding(
                finding_id="ML_PCA_VARIANCE",
                finding_type="pca",
                title=f"Top 2 principal components capture {top2_var * 100:.1f}% of variance",
                description=(
                    f"Dimensionality reduction reveals {top2_var * 100:.1f}% of data variance is compressed "
                    f"within the first 2 principal components. Primary PC1 drivers: {', '.join(top_pc1_feats)}."
                ),
                importance=importance,
                measured_values={
                    "top2_variance": float(top2_var),
                    "pc1_variance": float(pca.explained_variance_ratio[0]),
                    "components_count": pca.n_components,
                },
                affected_columns=top_pc1_feats,
            ))

        return findings

    def analyze(
        self,
        data: Union[Dataset, pd.DataFrame],
        profile: Optional[DatasetProfile] = None,
    ) -> MLReport:
        """Execute complete ML pipeline on dataset.

        Args:
            data: Dataset or DataFrame to analyze.
            profile: Optional pre-computed DatasetProfile.

        Returns:
            MLReport: Consolidated machine learning intelligence report.
        """
        df, dataset_name = self._extract_df(data)
        n_rows, n_cols = df.shape

        # 1. Preprocessing
        preprocessor = MLPreprocessor(
            numeric_imputation=self.config.numeric_imputation,
            scaling=self.config.scaling,
            max_categories=self.config.max_categories,
        )

        try:
            X = preprocessor.fit_transform(df, profile=profile)
            prep_result = preprocessor.get_preprocessing_result(df)
        except Exception as err:
            empty_prep = MLPreprocessingResult(
                feature_names=[],
                original_columns=list(df.columns),
                numeric_columns=[],
                categorical_columns=[],
                datetime_columns=[],
                boolean_columns=[],
                excluded_columns=list(df.columns),
                exclusion_reasons={c: f"preprocessing_failed: {err}" for c in df.columns},
                generated_features=[],
                imputation_metadata={},
                encoding_metadata={},
                scaling_metadata={},
                n_rows=n_rows,
                n_features=0,
            )
            return MLReport(
                dataset_name=dataset_name,
                row_count=n_rows,
                feature_count=0,
                preprocessing=empty_prep,
                anomaly=AnomalyResult(
                    algorithm="none",
                    n_anomalies=0,
                    anomaly_percentage=0.0,
                    anomaly_indices=[],
                    raw_scores=[],
                    normalized_scores=[],
                    score_summary={},
                    status="unavailable",
                    error_message=f"Preprocessing failed: {err}",
                ),
                clustering=ClusterResult(
                    algorithm="none",
                    n_clusters=0,
                    cluster_labels=[],
                    cluster_sizes={},
                    cluster_proportions={},
                    cluster_summaries=[],
                    distinguishing_features=[],
                    status="unavailable",
                    error_message=f"Preprocessing failed: {err}",
                ),
                dimensionality=PCAResult(
                    n_components=0,
                    explained_variance_ratio=[],
                    cumulative_explained_variance=[],
                    loadings={},
                    coordinates=[],
                    top_features_per_component={},
                    status="unavailable",
                    error_message=f"Preprocessing failed: {err}",
                ),
                evaluation=self.evaluator.evaluate(
                    AnomalyResult("none", 0, 0, [], [], [], {}, status="unavailable"),
                    ClusterResult("none", 0, [], {}, {}, [], [], status="unavailable"),
                    PCAResult(0, [], [], {}, [], {}, status="unavailable"),
                ),
                findings=[],
                metadata={"error": str(err)},
            )

        n_transformed_features = X.shape[1]

        # 2. Anomaly Detection
        if not self.config.run_anomaly or n_transformed_features == 0:
            anomaly_result = AnomalyResult(
                algorithm=self.config.anomaly_algorithm,
                n_anomalies=0,
                anomaly_percentage=0.0,
                anomaly_indices=[],
                raw_scores=[],
                normalized_scores=[],
                score_summary={},
                status="unavailable" if n_transformed_features == 0 else "skipped",
                error_message="No features available" if n_transformed_features == 0 else "Skipped by config",
            )
        else:
            if self.config.anomaly_algorithm == "local_outlier_factor":
                detector: AnomalyDetector = LocalOutlierFactorDetector(
                    contamination=self.config.anomaly_contamination,
                )
            else:
                detector = IsolationForestDetector(
                    contamination=self.config.anomaly_contamination,
                    random_state=self.config.random_state,
                )
            anomaly_result = detector.fit_predict(X, feature_names=prep_result.feature_names)

        # 3. Clustering
        if not self.config.run_clustering or n_transformed_features == 0:
            cluster_result = ClusterResult(
                algorithm=self.config.clustering_algorithm,
                n_clusters=0,
                cluster_labels=[],
                cluster_sizes={},
                cluster_proportions={},
                cluster_summaries=[],
                distinguishing_features=[],
                status="unavailable" if n_transformed_features == 0 else "skipped",
                error_message="No features available" if n_transformed_features == 0 else "Skipped by config",
            )
        else:
            if self.config.clustering_algorithm == "dbscan":
                clusterer: Clusterer = DBSCANClusterer(
                    eps=self.config.dbscan_eps,
                    min_samples=self.config.dbscan_min_samples,
                )
            else:
                is_auto = self.config.auto_k or self.config.clustering_algorithm == "auto_kmeans"
                clusterer = KMeansClusterer(
                    n_clusters=self.config.n_clusters,
                    auto_k=is_auto,
                    min_k=self.config.min_k,
                    max_k=self.config.max_k,
                    random_state=self.config.random_state,
                )
            cluster_result = clusterer.fit_predict(
                X=X,
                feature_names=prep_result.feature_names,
                original_df=df,
            )

        # 4. Dimensionality Reduction (PCA)
        if not self.config.run_pca or n_transformed_features == 0:
            pca_result = PCAResult(
                n_components=0,
                explained_variance_ratio=[],
                cumulative_explained_variance=[],
                loadings={},
                coordinates=[],
                top_features_per_component={},
                status="unavailable" if n_transformed_features == 0 else "skipped",
                error_message="No features available" if n_transformed_features == 0 else "Skipped by config",
            )
        else:
            reducer = PCAReducer(n_components=self.config.pca_components)
            pca_result = reducer.fit_transform(X, feature_names=prep_result.feature_names)

        # 5. Consolidated Evaluation
        evaluation = self.evaluator.evaluate(
            anomaly=anomaly_result,
            clustering=cluster_result,
            pca=pca_result,
        )

        # 6. Structured Evidence Findings
        findings = self._extract_ml_findings(
            anomaly=anomaly_result,
            clustering=cluster_result,
            pca=pca_result,
        )

        return MLReport(
            dataset_name=dataset_name,
            row_count=n_rows,
            feature_count=n_transformed_features,
            preprocessing=prep_result,
            anomaly=anomaly_result,
            clustering=cluster_result,
            dimensionality=pca_result,
            evaluation=evaluation,
            findings=findings,
            metadata={
                "random_state": self.config.random_state,
                "unscaled_shape": list(X.shape),
            },
        )


def run_ml_analysis(
    data: Union[Dataset, pd.DataFrame],
    config: Optional[MLConfig] = None,
    profile: Optional[DatasetProfile] = None,
) -> MLReport:
    """Functional convenience entrypoint to execute ML analysis on tabular data.

    Args:
        data: Ingested Dataset or pandas DataFrame.
        config: Optional MLConfig parameterization.
        profile: Optional pre-computed DatasetProfile.

    Returns:
        MLReport: Comprehensive ML intelligence report.
    """
    engine = MLEngine(config=config)
    return engine.analyze(data, profile=profile)
