"""Dimensionality reduction subsystem for DataAlchemy using Principal Component Analysis.

Extracts principal components, explained variance ratios, component feature loadings,
and visualization-ready 2D/3D projection coordinates.
"""

from typing import Any, Dict, List, Optional
import numpy as np
from sklearn.decomposition import PCA

from src.ml.models import PCAResult


class PCAReducer:
    """Principal Component Analysis reducer for feature projection and interpretation."""

    def __init__(self, n_components: int = 2) -> None:
        self.requested_components = n_components

    def fit_transform(
        self,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> PCAResult:
        """Fit PCA on feature matrix X and project into principal component coordinates."""
        n_rows, n_cols = X.shape
        if n_rows == 0 or n_cols == 0:
            return PCAResult(
                n_components=0,
                explained_variance_ratio=[],
                cumulative_explained_variance=[],
                loadings={},
                coordinates=[],
                top_features_per_component={},
                status="unavailable",
                error_message="Feature matrix is empty or has zero features.",
            )

        if n_rows < 2 or n_cols < 1:
            return PCAResult(
                n_components=0,
                explained_variance_ratio=[],
                cumulative_explained_variance=[],
                loadings={},
                coordinates=[],
                top_features_per_component={},
                status="unavailable",
                error_message="PCA requires at least 2 samples and 1 feature.",
            )

        # Ensure valid number of components
        effective_k = min(self.requested_components, n_cols, n_rows)

        try:
            model = PCA(n_components=effective_k)
            coords = model.fit_transform(X)

            explained_ratios = [float(r) for r in model.explained_variance_ratio_]
            cumulative_ratios = [float(c) for c in np.cumsum(explained_ratios)]

            names = feature_names or [f"feature_{i}" for i in range(n_cols)]

            # Calculate component loadings
            # Loadings indicate how strongly each original feature influences each PC
            loadings: Dict[str, Dict[str, float]] = {}
            top_features: Dict[str, List[Dict[str, Any]]] = {}

            for comp_idx in range(effective_k):
                comp_name = f"PC{comp_idx + 1}"
                comp_weights = model.components_[comp_idx]
                feat_dict = {
                    names[j]: float(comp_weights[j]) for j in range(len(names))
                }
                loadings[comp_name] = feat_dict

                # Sort features by absolute loading value
                sorted_feats = sorted(
                    feat_dict.items(),
                    key=lambda item: abs(item[1]),
                    reverse=True,
                )
                top_features[comp_name] = [
                    {"feature": feat, "loading": val, "abs_loading": abs(val)}
                    for feat, val in sorted_feats[:5]
                ]

            # Format projection coordinates for visualization
            coordinates: List[Dict[str, Any]] = []
            for row_idx in range(n_rows):
                pt: Dict[str, Any] = {"row_id": int(row_idx)}
                for comp_idx in range(effective_k):
                    pt[f"PC{comp_idx + 1}"] = float(coords[row_idx, comp_idx])
                coordinates.append(pt)

            return PCAResult(
                n_components=effective_k,
                explained_variance_ratio=explained_ratios,
                cumulative_explained_variance=cumulative_ratios,
                loadings=loadings,
                coordinates=coordinates,
                top_features_per_component=top_features,
                status="completed",
                metadata={"original_features_count": n_cols},
            )
        except Exception as err:
            return PCAResult(
                n_components=0,
                explained_variance_ratio=[],
                cumulative_explained_variance=[],
                loadings={},
                coordinates=[],
                top_features_per_component={},
                status="failed",
                error_message=str(err),
            )
