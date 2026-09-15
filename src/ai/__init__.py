"""DataAlchemy AI Insight Engine Package.

Provides local-first generative AI synthesis, transforming verified statistical facts
into evidence-grounded, human-readable insights and actionable recommendations.
"""

from src.ai.context import AIContextBuilder
from src.ai.engine import AIInsightEngine, generate_ai_insights
from src.ai.models import AIInsight, AIInsightReport, AIRecommendation
from src.ai.parser import AIOutputParser, AIOutputParsingError
from src.ai.prompts import build_insight_prompt, build_system_prompt
from src.ai.providers import LLMProvider, LLMProviderError
from src.ai.providers.huggingface import HuggingFaceProvider, MockLLMProvider

__all__ = [
    # Engine & Main API
    "AIInsightEngine",
    "generate_ai_insights",
    # Models
    "AIInsight",
    "AIRecommendation",
    "AIInsightReport",
    # Context & Prompts
    "AIContextBuilder",
    "build_system_prompt",
    "build_insight_prompt",
    # Parser
    "AIOutputParser",
    "AIOutputParsingError",
    # Providers
    "LLMProvider",
    "LLMProviderError",
    "HuggingFaceProvider",
    "MockLLMProvider",
]
