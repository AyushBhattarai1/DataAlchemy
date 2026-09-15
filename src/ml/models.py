"""Data models and result containers for DataAlchemy ML Intelligence Engine.

Provides typed, serializable structures for preprocessing results, anomaly detection,
clustering, dimensionality reduction (PCA), evaluation metrics, and unified ML reports.
"""

from dataclasses import asdict, dataclass, field
import json
from typing import Any, Dict, List, Optional, Union
import numpy as np


def _sanitize_for_json(obj: Any) -> Any:
    """Recursively convert NumPy data types and non-serializable objects to native Python."""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        val = float(obj)
        if np.isnan(val) or np.isinf(val):
            return None
        return val
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return [_sanitize_for_json(x) for x in obj.tolist()]
    elif isinstance(obj, dict):
        return {str(k): _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(x) for x in obj]
    elif obj is None:
        return None
    return obj


@dataclass(frozen=True)
class MLPreprocessingResult:
    """Represents the results and metadata of the ML preprocessing pipeline."""
    feature_names: List[str]
    original_columns: List[str]
    numeric_columns: List[str]
    categorical_columns: List[str]
    datetime_columns: List[str]
    boolean_columns: List[str]
    excluded_columns: List[str]
    exclusion_reasons: Dict[str, str]
    generated_features: List[str]
    imputation_metadata: Dict[str, Any]
    encoding_metadata: Dict[str, Any]
    scaling_metadata: Dict[str, Any]
    n_rows: int
    n_features: int
    is_sparse: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert preprocessing metadata to a JSON-compatible dictionary."""
        return _sanitize_for_json(asdict(self))


@dataclass(frozen=True)
class AnomalyRecord:
    """Represents an anomaly evaluation for an individual row."""
    row_index: int
    is_anomaly: bool
    raw_score: float
    normalized_score: float
    row_identifier: Optional[str] = None
    feature_contributions: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return _sanitize_for_json(asdict(self))


@dataclass(frozen=True)
class AnomalyResult:
    """Results from unsupervised anomaly detection."""
    algorithm: str
    n_anomalies: int
    anomaly_percentage: float
    anomaly_indices: List[int]
    raw_scores: List[float]
    normalized_scores: List[float]
    score_summary: Dict[str, float]
    status: str = "completed"
    error_message: Optional[str] = None
    feature_contributions: Optional[Dict[str, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return _sanitize_for_json(asdict(self))


@dataclass(frozen=True)
class ClusterSummary:
    """Characterization of an individual discovered cluster."""
    cluster_id: int
    size: int
    proportion: float
    numerical_profiles: Dict[str, Dict[str, float]]  # feature -> {mean, median, std, min, max}
    categorical_profiles: Dict[str, Dict[str, Any]]  # feature -> {dominant_category, count, frequency}
    is_noise: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return _sanitize_for_json(asdict(self))


@dataclass(frozen=True)
class ClusterResult:
    """Results from unsupervised clustering analysis."""
    algorithm: str
    n_clusters: int
    cluster_labels: List[int]
    cluster_sizes: Dict[int, int]
    cluster_proportions: Dict[int, float]
    cluster_summaries: List[ClusterSummary]
    distinguishing_features: List[Dict[str, Any]]
    inertia: Optional[float] = None
    silhouette_score: Optional[float] = None
    auto_k_evaluation: Optional[Dict[str, Any]] = None
    noise_count: int = 0
    noise_percentage: float = 0.0
    status: str = "completed"
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return _sanitize_for_json(asdict(self))


@dataclass(frozen=True)
class PCAResult:
    """Results from principal component analysis (dimensionality reduction)."""
    n_components: int
    explained_variance_ratio: List[float]
    cumulative_explained_variance: List[float]
    loadings: Dict[str, Dict[str, float]]  # component -> {feature: loading_val}
    coordinates: List[Dict[str, Any]]  # [{row_id: 0, PC1: 1.2, PC2: -0.4, ...}]
    top_features_per_component: Dict[str, List[Dict[str, Any]]]
    status: str = "completed"
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return _sanitize_for_json(asdict(self))


@dataclass(frozen=True)
class MLEvaluation:
    """Consolidated mathematical evaluation metrics across ML components."""
    metrics: Dict[str, Optional[float]]
    metric_reasons: Dict[str, str]

    def to_dict(self) -> Dict[str, Any]:
        return _sanitize_for_json(asdict(self))


@dataclass(frozen=True)
class MLFinding:
    """Discrete structured machine learning discovery backed by numerical evidence."""
    finding_id: str
    finding_type: str  # "anomaly", "cluster", "pca"
    title: str
    description: str
    importance: str  # "HIGH", "MEDIUM", "LOW", "INFO"
    measured_values: Dict[str, Any]
    affected_columns: List[str]
    affected_rows: Optional[List[int]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return _sanitize_for_json(asdict(self))


@dataclass(frozen=True)
class MLReport:
    """Unified report containing complete machine learning analysis results and contracts."""
    dataset_name: str
    row_count: int
    feature_count: int
    preprocessing: MLPreprocessingResult
    anomaly: AnomalyResult
    clustering: ClusterResult
    dimensionality: PCAResult
    evaluation: MLEvaluation
    findings: List[MLFinding]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert ML report to a JSON-compatible dictionary."""
        return _sanitize_for_json({
            "dataset_name": self.dataset_name,
            "row_count": self.row_count,
            "feature_count": self.feature_count,
            "preprocessing": self.preprocessing.to_dict(),
            "anomaly": self.anomaly.to_dict(),
            "clustering": self.clustering.to_dict(),
            "dimensionality": self.dimensionality.to_dict(),
            "evaluation": self.evaluation.to_dict(),
            "total_findings": len(self.findings),
            "findings": [f.to_dict() for f in self.findings],
            "visualization_data": self.get_visualization_data(),
            "metadata": self.metadata,
        })

    def to_json(self, indent: int = 2) -> str:
        """Export report directly to a formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def get_visualization_data(self) -> Dict[str, Any]:
        """Generate clean, structured contracts specifically designed for Step 8 visualization."""
        viz: Dict[str, Any] = {}

        # 1. PCA 2D/3D Scatter Plot coordinates with anomaly & cluster labels attached
        if self.dimensionality.status == "completed" and self.dimensionality.coordinates:
            scatter_points: List[Dict[str, Any]] = []
            anomaly_labels = set(self.anomaly.anomaly_indices) if self.anomaly.status == "completed" else set()
            cluster_labels = self.clustering.cluster_labels if self.clustering.status == "completed" else []

            for i, coord in enumerate(self.dimensionality.coordinates):
                pt = {
                    "row_id": coord.get("row_id", i),
                    "pc1": coord.get("PC1"),
                    "pc2": coord.get("PC2"),
                    "pc3": coord.get("PC3"),
                    "is_anomaly": i in anomaly_labels,
                    "anomaly_score": (
                        self.anomaly.normalized_scores[i]
                        if self.anomaly.status == "completed" and i < len(self.anomaly.normalized_scores)
                        else None
                    ),
                    "cluster": cluster_labels[i] if i < len(cluster_labels) else None,
                }
                scatter_points.append(pt)

            viz["pca_scatter"] = {
                "chart_type": "scatter",
                "x_axis": "PC1",
                "y_axis": "PC2",
                "points": scatter_points,
            }

        # 2. PCA Scree Plot (Explained Variance)
        if self.dimensionality.status == "completed":
            components = [f"PC{i+1}" for i in range(len(self.dimensionality.explained_variance_ratio))]
            viz["pca_scree"] = {
                "chart_type": "bar_and_line",
                "components": components,
                "explained_variance_ratio": self.dimensionality.explained_variance_ratio,
                "cumulative_explained_variance": self.dimensionality.cumulative_explained_variance,
            }

        # 3. PCA Loadings Heatmap / Bar
        if self.dimensionality.status == "completed" and self.dimensionality.loadings:
            viz["pca_loadings"] = {
                "chart_type": "matrix",
                "loadings": self.dimensionality.loadings,
                "top_features": self.dimensionality.top_features_per_component,
            }

        # 4. Cluster Distribution Bar Chart
        if self.clustering.status == "completed":
            viz["cluster_distribution"] = {
                "chart_type": "bar",
                "clusters": [f"Cluster {c}" if c != -1 else "Noise" for c in self.clustering.cluster_sizes.keys()],
                "counts": list(self.clustering.cluster_sizes.values()),
                "proportions": list(self.clustering.cluster_proportions.values()),
            }

        # 5. Cluster Profiles (Radar / Comparison)
        if self.clustering.status == "completed" and self.clustering.cluster_summaries:
            profiles = []
            for cs in self.clustering.cluster_summaries:
                profiles.append({
                    "cluster_id": cs.cluster_id,
                    "label": f"Cluster {cs.cluster_id}" if not cs.is_noise else "Noise",
                    "size": cs.size,
                    "numerical_means": {
                        feat: vals.get("mean") for feat, vals in cs.numerical_profiles.items()
                    },
                })
            viz["cluster_profiles"] = {
                "chart_type": "grouped_bar",
                "profiles": profiles,
                "distinguishing_features": self.clustering.distinguishing_features,
            }

        # 6. Anomaly Score Distribution Histogram
        if self.anomaly.status == "completed" and self.anomaly.normalized_scores:
            viz["anomaly_distribution"] = {
                "chart_type": "histogram",
                "n_anomalies": self.anomaly.n_anomalies,
                "anomaly_percentage": self.anomaly.anomaly_percentage,
                "scores": self.anomaly.normalized_scores,
                "summary": self.anomaly.score_summary,
            }

        return _sanitize_for_json(viz)

    def summary(self) -> str:
        """Produce a clean human-readable ASCII summary of the ML analysis."""
        lines = [
            "MACHINE LEARNING INTELLIGENCE REPORT",
            "=" * 45,
            f"Dataset:              {self.dataset_name}",
            f"Rows Evaluated:       {self.row_count:,}",
            f"Transformed Features: {self.feature_count:,}",
            "",
            "1. ML PREPROCESSING:",
            f"   - Numeric Features:     {len(self.preprocessing.numeric_columns)}",
            f"   - Categorical Features: {len(self.preprocessing.categorical_columns)}",
            f"   - Datetime Features:    {len(self.preprocessing.datetime_columns)}",
            f"   - Excluded Features:    {len(self.preprocessing.excluded_columns)}",
        ]
        if self.preprocessing.excluded_columns:
            lines.append(f"     Excluded: {', '.join(self.preprocessing.excluded_columns)}")

        lines.extend(["", "2. ANOMALY DETECTION:"])
        if self.anomaly.status == "completed":
            lines.append(f"   - Algorithm:          {self.anomaly.algorithm}")
            lines.append(f"   - Anomalies Found:    {self.anomaly.n_anomalies:,} ({self.anomaly.anomaly_percentage:.2f}%)")
            lines.append(f"   - Mean Anomaly Score: {self.anomaly.score_summary.get('mean', 0.0):.4f}")
        else:
            lines.append(f"   - Status:             {self.anomaly.status} ({self.anomaly.error_message})")

        lines.extend(["", "3. CLUSTERING DISCOVERY:"])
        if self.clustering.status == "completed":
            lines.append(f"   - Algorithm:          {self.clustering.algorithm}")
            lines.append(f"   - Clusters Found:     {self.clustering.n_clusters}")
            if self.clustering.silhouette_score is not None:
                lines.append(f"   - Silhouette Score:   {self.clustering.silhouette_score:.4f}")
            if self.clustering.noise_count > 0:
                lines.append(f"   - Noise Observations: {self.clustering.noise_count:,} ({self.clustering.noise_percentage:.2f}%)")
        else:
            lines.append(f"   - Status:             {self.clustering.status} ({self.clustering.error_message})")

        lines.extend(["", "4. DIMENSIONALITY REDUCTION (PCA):"])
        if self.dimensionality.status == "completed":
            lines.append(f"   - Components:         {self.dimensionality.n_components}")
            if self.dimensionality.cumulative_explained_variance:
                cum_var = self.dimensionality.cumulative_explained_variance[-1]
                lines.append(f"   - Total Variance:     {cum_var * 100:.2f}%")
        else:
            lines.append(f"   - Status:             {self.dimensionality.status} ({self.dimensionality.error_message})")

        if self.findings:
            lines.extend(["", f"5. TOP ML EVIDENCE DISCOVERIES ({len(self.findings)}):"])
            for idx, f in enumerate(self.findings[:5], 1):
                lines.append(f"   {idx}. [{f.importance}] {f.title}")
                lines.append(f"      {f.description}")

        lines.append("=" * 45)
        return "\n".join(lines)
