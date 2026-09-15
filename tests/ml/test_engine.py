"""Integration tests for MLEngine and run_ml_analysis in src/ml/engine.py."""

from pathlib import Path
import pandas as pd
import pytest

from src.ai.context import AIContextBuilder
from src.ai.engine import generate_ai_insights
from src.ai.providers.huggingface import MockLLMProvider
from src.analysis.analyzer import analyze_dataset
from src.ingestion.loader import load_dataset
from src.ml.engine import MLConfig, MLEngine, run_ml_analysis
from src.ml.models import MLReport
from src.profiling.profiler import profile_dataset
from src.quality.analyzer import analyze_quality


@pytest.fixture
def sample_data_dir() -> Path:
    return Path(__file__).resolve().parent.parent.parent / "data" / "samples"


class TestMLEngineIntegration:
    """End-to-end integration tests on real sample datasets."""

    def test_analyze_sales_sample(self, sample_data_dir: Path) -> None:
        ds = load_dataset(sample_data_dir / "sales.csv")
        profile = profile_dataset(ds)

        report = run_ml_analysis(ds, profile=profile)

        assert isinstance(report, MLReport)
        assert report.dataset_name == "sales.csv"
        assert report.row_count == 7
        assert report.feature_count > 0

        # Anomaly detection completed
        assert report.anomaly.status == "completed"
        assert len(report.anomaly.raw_scores) == 7

        # Clustering completed
        assert report.clustering.status == "completed"
        assert len(report.clustering.cluster_labels) == 7

        # PCA completed
        assert report.dimensionality.status == "completed"
        assert report.dimensionality.n_components >= 2
        assert len(report.dimensionality.coordinates) == 7

        # Structured findings generated
        assert len(report.findings) > 0

    def test_analyze_customers_sample(self, sample_data_dir: Path) -> None:
        ds = load_dataset(sample_data_dir / "customers.csv")
        profile = profile_dataset(ds)

        engine = MLEngine()
        report = engine.analyze(ds, profile=profile)

        assert report.dataset_name == "customers.csv"
        assert report.row_count == 8
        assert report.anomaly.status == "completed"
        assert report.clustering.status == "completed"
        assert report.dimensionality.status == "completed"

    def test_analyze_transactions_sample(self, sample_data_dir: Path) -> None:
        ds = load_dataset(sample_data_dir / "transactions.json")
        report = run_ml_analysis(ds)

        assert report.dataset_name == "transactions.json"
        assert report.row_count > 0
        assert report.feature_count > 0

    def test_rerunnability_with_different_config(self, sample_data_dir: Path) -> None:
        ds = load_dataset(sample_data_dir / "sales.csv")

        # Run 1: Isolation Forest + KMeans
        config1 = MLConfig(
            anomaly_algorithm="isolation_forest",
            clustering_algorithm="kmeans",
            n_clusters=2,
            random_state=42,
        )
        report1 = run_ml_analysis(ds, config=config1)
        assert report1.anomaly.algorithm == "isolation_forest"
        assert report1.clustering.n_clusters == 2

        # Run 2: Local Outlier Factor + DBSCAN (without mutating dataset)
        config2 = MLConfig(
            anomaly_algorithm="local_outlier_factor",
            clustering_algorithm="dbscan",
            dbscan_eps=1.5,
            dbscan_min_samples=2,
        )
        report2 = run_ml_analysis(ds, config=config2)
        assert report2.anomaly.algorithm == "local_outlier_factor"
        assert report2.clustering.algorithm == "dbscan"

    def test_selective_component_execution(self, sample_data_dir: Path) -> None:
        ds = load_dataset(sample_data_dir / "sales.csv")

        # Run only clustering, skip anomaly and PCA
        config = MLConfig(
            run_anomaly=False,
            run_clustering=True,
            run_pca=False,
        )
        report = run_ml_analysis(ds, config=config)

        assert report.anomaly.status == "skipped"
        assert report.clustering.status == "completed"
        assert report.dimensionality.status == "skipped"

    def test_partial_failure_resilience(self) -> None:
        # 1-row dataset where PCA cannot extract multiple components
        single_row_df = pd.DataFrame({
            "num1": [10.0],
            "cat1": ["A"],
        })
        engine = MLEngine()
        report = engine.analyze(single_row_df)

        assert report.row_count == 1
        assert report.preprocessing.n_rows == 1
        # Pipeline must not crash: PCA marks status unavailable
        assert report.dimensionality.status == "unavailable"
        assert "PCA requires at least 2 samples" in report.dimensionality.error_message

    def test_integration_with_step6_ai_insight_engine(self, sample_data_dir: Path) -> None:
        ds = load_dataset(sample_data_dir / "sales.csv")
        profile = profile_dataset(ds)
        quality = analyze_quality(profile)
        analysis = analyze_dataset(ds, profile, quality)
        ml_report = run_ml_analysis(ds, profile=profile)

        # 1. Verify AI Context Builder handles ml_report
        builder = AIContextBuilder()
        context = builder.build_context(
            profile=profile,
            quality_report=quality,
            analysis_report=analysis,
            ml_report=ml_report,
        )
        assert "ml" in context
        assert "anomaly" in context["ml"]
        assert "clustering" in context["ml"]

        prompt_text = builder.to_prompt_text(context)
        assert "### MACHINE LEARNING PATTERNS & CLUSTERING" in prompt_text
        assert "Anomaly Detection" in prompt_text

        # 2. Verify generate_ai_insights accepts ml_report seamlessly
        mock_provider = MockLLMProvider()
        ai_report = generate_ai_insights(
            profile=profile,
            quality_report=quality,
            analysis_report=analysis,
            ml_report=ml_report,
            provider=mock_provider,
        )
        assert ai_report.dataset_name == "sales.csv"
        assert mock_provider.last_prompt is not None
        assert "MACHINE LEARNING PATTERNS" in mock_provider.last_prompt
