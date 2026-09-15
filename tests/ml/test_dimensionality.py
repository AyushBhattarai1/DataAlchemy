"""Unit tests for PCA dimensionality reduction in src/ml/dimensionality.py."""

import numpy as np
import pytest

from src.ml.dimensionality import PCAReducer
from src.ml.models import PCAResult


class TestPCAReducer:
    """Test suite for PCA dimensionality reduction and coordinate projection."""

    @pytest.fixture
    def multi_feature_data(self) -> np.ndarray:
        np.random.seed(42)
        # 50 rows, 5 features
        return np.random.normal(loc=0.0, scale=1.0, size=(50, 5))

    def test_pca_2_components(self, multi_feature_data: np.ndarray) -> None:
        reducer = PCAReducer(n_components=2)
        res = reducer.fit_transform(
            multi_feature_data,
            feature_names=["f1", "f2", "f3", "f4", "f5"],
        )

        assert isinstance(res, PCAResult)
        assert res.status == "completed"
        assert res.n_components == 2
        assert len(res.explained_variance_ratio) == 2
        assert len(res.cumulative_explained_variance) == 2
        assert res.cumulative_explained_variance[-1] <= 1.0

        # Verify coordinates
        assert len(res.coordinates) == 50
        coord_0 = res.coordinates[0]
        assert "row_id" in coord_0
        assert "PC1" in coord_0
        assert "PC2" in coord_0

        # Verify loadings
        assert "PC1" in res.loadings
        assert "PC2" in res.loadings
        assert len(res.loadings["PC1"]) == 5
        assert "PC1" in res.top_features_per_component

    def test_pca_component_capping(self) -> None:
        # 10 rows, 3 features, but 10 components requested
        X = np.random.normal(size=(10, 3))
        reducer = PCAReducer(n_components=10)
        res = reducer.fit_transform(X)

        assert res.status == "completed"
        # Must be capped at min(10, 3) = 3
        assert res.n_components == 3
        assert len(res.explained_variance_ratio) == 3

    def test_pca_edge_cases(self) -> None:
        empty_X = np.zeros((0, 0))
        reducer = PCAReducer()
        res_empty = reducer.fit_transform(empty_X)
        assert res_empty.status == "unavailable"

        single_row_X = np.array([[1.0, 2.0, 3.0]])
        res_single = reducer.fit_transform(single_row_X)
        assert res_single.status == "unavailable"
