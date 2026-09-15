"""LLM Provider abstractions and implementations for DataAlchemy.

Provides a clean interface for model inference, decoupling business logic from
specific local Hugging Face or mock test implementations.
"""

from abc import ABC, abstractmethod
from typing import Optional


class LLMProviderError(Exception):
    """Exception raised for provider configuration, loading, or generation failures."""
    pass


class LLMProvider(ABC):
    """Abstract interface for local or mock LLM providers."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        """Generate text response from the model given a prompt.

        Args:
            prompt: Main user or task prompt.
            system_prompt: Optional system role prompt.
            temperature: Sampling temperature (0.0 for deterministic output).
            max_tokens: Maximum tokens to generate.

        Returns:
            Raw response text from the model.
        """
        pass
