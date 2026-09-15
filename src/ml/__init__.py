"""DataAlchemy Machine Learning Intelligence Engine Package.

Provides ML preprocessing, unsupervised anomaly detection, cluster discovery,
dimensionality reduction (PCA), and evidence extraction for tabular datasets.
"""

from src.ml.anomaly import (
    AnomalyDetector,
    IsolationForestDetector,
    LocalOutlierFactorDetector,
)
from src.ml.clustering import (
    Clusterer,
    DBSCANClusterer,
    KMeansClusterer,
)
from src.ml.dimensionality import PCAReducer
from src.ml.engine import MLConfig, MLEngine, run_ml_analysis
from src.ml.evaluation import MLEvaluator
from src.ml.models import (
    AnomalyRecord,
    AnomalyResult,
    ClusterResult,
    ClusterSummary,
    MLEvaluation,
    MLFinding,
    MLPreprocessingResult,
    MLReport,
    PCAResult,
)
from src.ml.preprocessing import MLPreprocessor

__all__ = [
    # Engine & Config
    "MLEngine",
    "MLConfig",
    "run_ml_analysis",
    # Preprocessor
    "MLPreprocessor",
    # Anomaly
    "AnomalyDetector",
    "IsolationForestDetector",
    "LocalOutlierFactorDetector",
    # Clustering
    "Clusterer",
    "KMeansClusterer",
    "DBSCANClusterer",
    # Dimensionality
    "PCAReducer",
    # Evaluation
    "MLEvaluator",
    # Models
    "MLPreprocessingResult",
    "AnomalyRecord",
    "AnomalyResult",
    "ClusterSummary",
    "ClusterResult",
    "PCAResult",
    "MLEvaluation",
    "MLFinding",
    "MLReport",
]
