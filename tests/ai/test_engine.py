"""Integration tests for AIInsightEngine and generate_ai_insights in src/ai/engine.py."""

from pathlib import Path
import pytest

from src.ai.engine import AIInsightEngine, generate_ai_insights
from src.ai.models import AIInsightReport
from src.ai.parser import AIOutputParsingError
from src.ai.providers import LLMProviderError
from src.ai.providers.huggingface import MockLLMProvider
from src.analysis.analyzer import analyze_dataset
from src.ingestion.loader import load_dataset
from src.profiling.profiler import profile_dataset
from src.quality.analyzer import analyze_quality


@pytest.fixture
def sample_data_dir() -> Path:
    """Path to sample dataset directory."""
    return Path(__file__).resolve().parent.parent.parent / "data" / "samples"


class TestAIInsightEngine:
    """End-to-end test suite for AI insight generation."""

    def test_generate_ai_insights_sales(self, sample_data_dir: Path) -> None:
        ds = load_dataset(sample_data_dir / "sales.csv")
        profile = profile_dataset(ds)
        quality = analyze_quality(profile)
        analysis = analyze_dataset(ds, profile, quality)

        mock_provider = MockLLMProvider()
        engine = AIInsightEngine(provider=mock_provider)
        report = engine.generate_report(profile, quality, analysis)

        assert isinstance(report, AIInsightReport)
        assert report.dataset_name == "sales.csv"
        assert len(report.insights) > 0
        assert len(report.recommendations) > 0

        # Verify prompt passed to provider contained factual context
        assert mock_provider.last_prompt is not None
        assert "sales.csv" in mock_provider.last_prompt
        assert "DATASET SUMMARY" in mock_provider.last_prompt
        assert "VERIFIED STATISTICAL FINDINGS" in mock_provider.last_prompt

    def test_generate_ai_insights_customers_functional_api(self, sample_data_dir: Path) -> None:
        ds = load_dataset(sample_data_dir / "customers.csv")
        profile = profile_dataset(ds)
        quality = analyze_quality(profile)
        analysis = analyze_dataset(ds, profile, quality)

        mock_provider = MockLLMProvider()
        report = generate_ai_insights(
            profile=profile,
            quality_report=quality,
            analysis_report=analysis,
            provider=mock_provider,
        )

        assert report.dataset_name == "customers.csv"
        assert len(report.insights) >= 2
        assert mock_provider.call_count == 1

    def test_generate_ai_insights_transactions(self, sample_data_dir: Path) -> None:
        ds = load_dataset(sample_data_dir / "transactions.json")
        profile = profile_dataset(ds)
        quality = analyze_quality(profile)
        analysis = analyze_dataset(ds, profile, quality)

        mock_provider = MockLLMProvider()
        report = generate_ai_insights(profile, quality, analysis, provider=mock_provider)

        assert report.dataset_name == "transactions.json"
        assert report.executive_summary is not None
        assert len(report.executive_summary) > 0

    def test_generate_ai_insights_profile_only(self, sample_data_dir: Path) -> None:
        ds = load_dataset(sample_data_dir / "customers.csv")
        profile = profile_dataset(ds)

        mock_provider = MockLLMProvider()
        report = generate_ai_insights(profile=profile, provider=mock_provider)

        assert report.dataset_name == "customers.csv"
        assert len(report.insights) > 0

    def test_provider_error_propagation(self, sample_data_dir: Path) -> None:
        ds = load_dataset(sample_data_dir / "sales.csv")
        profile = profile_dataset(ds)

        class FailingProvider(MockLLMProvider):
            def generate(self, prompt: str, **kwargs: object) -> str:
                raise LLMProviderError("Connection timeout during generation")

        engine = AIInsightEngine(provider=FailingProvider())
        with pytest.raises(LLMProviderError) as exc_info:
            engine.generate_report(profile)

        assert "Connection timeout" in str(exc_info.value)

    def test_parser_error_propagation(self, sample_data_dir: Path) -> None:
        ds = load_dataset(sample_data_dir / "sales.csv")
        profile = profile_dataset(ds)

        mock_provider = MockLLMProvider(custom_response="Malformed non-JSON model answer")
        engine = AIInsightEngine(provider=mock_provider)

        with pytest.raises(AIOutputParsingError) as exc_info:
            engine.generate_report(profile)

        assert "failed to parse" in str(exc_info.value).lower()
