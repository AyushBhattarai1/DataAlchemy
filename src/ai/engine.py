"""AI Insight Engine orchestrator for DataAlchemy.

Coordinates statistical context construction, prompt generation, LLM provider invocation,
and output validation to produce structured, evidence-grounded AIInsightReports.
"""

from typing import Optional

from src.ai.context import AIContextBuilder
from src.ai.models import AIInsightReport
from src.ai.parser import AIOutputParser
from src.ai.prompts import build_insight_prompt, build_system_prompt
from src.ai.providers import LLMProvider
from src.ai.providers.huggingface import HuggingFaceProvider
from src.analysis.analyzer import AnalysisReport
from src.profiling.profiler import DatasetProfile
from src.quality.report import DataQualityReport


class AIInsightEngine:
    """Orchestrates structured statistical context into evidence-grounded AI insights."""

    def __init__(
        self,
        provider: Optional[LLMProvider] = None,
        context_builder: Optional[AIContextBuilder] = None,
        parser: Optional[AIOutputParser] = None,
    ) -> None:
        """Initialize the AI Insight Engine with configurable providers.

        Args:
            provider: LLM provider implementation (defaults to HuggingFaceProvider).
            context_builder: Component converting profiles to prompt contexts.
            parser: Component validating and converting raw model output to reports.
        """
        self.provider = provider if provider is not None else HuggingFaceProvider()
        self.context_builder = context_builder if context_builder is not None else AIContextBuilder()
        self.parser = parser if parser is not None else AIOutputParser()

    def generate_report(
        self,
        profile: DatasetProfile,
        quality_report: Optional[DataQualityReport] = None,
        analysis_report: Optional[AnalysisReport] = None,
        ml_report: Optional[Any] = None,
    ) -> AIInsightReport:
        """Generate structured AI insights grounded in factual statistical evidence and ML patterns.

        Args:
            profile: DatasetProfile from Step 3.
            quality_report: Optional DataQualityReport from Step 4.
            analysis_report: Optional AnalysisReport from Step 5.
            ml_report: Optional MLReport from Step 7.

        Returns:
            AIInsightReport: Validated report with executive summary, insights, and recommendations.
        """
        # 1. Build compact statistical context
        context_dict = self.context_builder.build_context(
            profile=profile,
            quality_report=quality_report,
            analysis_report=analysis_report,
            ml_report=ml_report,
        )
        context_text = self.context_builder.to_prompt_text(context_dict)

        # 2. Build system and user prompts
        system_prompt = build_system_prompt()
        prompt = build_insight_prompt(context_text)

        # 3. Invoke LLM provider
        raw_output = self.provider.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.0,
        )

        # 4. Parse and validate structured output
        return self.parser.parse_insight_report(
            raw_text=raw_output,
            dataset_name=profile.dataset_name,
        )


def generate_ai_insights(
    profile: DatasetProfile,
    quality_report: Optional[DataQualityReport] = None,
    analysis_report: Optional[AnalysisReport] = None,
    ml_report: Optional[Any] = None,
    provider: Optional[LLMProvider] = None,
) -> AIInsightReport:
    """Convenience functional API to generate AI insight reports.

    Args:
        profile: DatasetProfile from Step 3.
        quality_report: Optional DataQualityReport from Step 4.
        analysis_report: Optional AnalysisReport from Step 5.
        ml_report: Optional MLReport from Step 7.
        provider: Optional custom or mock LLM provider.

    Returns:
        AIInsightReport: Complete AI insight report.
    """
    engine = AIInsightEngine(provider=provider)
    return engine.generate_report(
        profile=profile,
        quality_report=quality_report,
        analysis_report=analysis_report,
        ml_report=ml_report,
    )
