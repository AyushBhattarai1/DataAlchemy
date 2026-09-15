"""Unit tests for HuggingFaceProvider and MockLLMProvider in src/ai/providers/huggingface.py."""

import json
import os
from unittest.mock import MagicMock, patch
import pytest

from src.ai.providers import LLMProvider, LLMProviderError
from src.ai.providers.huggingface import HuggingFaceProvider, MockLLMProvider


class TestMockLLMProvider:
    """Test suite for MockLLMProvider testing utility."""

    def test_mock_provider_default_response(self) -> None:
        provider = MockLLMProvider()
        assert isinstance(provider, LLMProvider)
        assert provider.call_count == 0

        res = provider.generate("Test prompt", system_prompt="System instructions")
        assert provider.call_count == 1
        assert provider.last_prompt == "Test prompt"
        assert provider.last_system_prompt == "System instructions"

        # Verify response is valid JSON adhering to schema
        data = json.loads(res)
        assert "executive_summary" in data
        assert "insights" in data
        assert "recommendations" in data
        assert len(data["insights"]) > 0

    def test_mock_provider_custom_response(self) -> None:
        custom = '{"executive_summary": "Custom summary", "insights": [], "recommendations": []}'
        provider = MockLLMProvider(custom_response=custom)

        res = provider.generate("Prompt")
        assert res == custom
        assert provider.call_count == 1


class TestHuggingFaceProvider:
    """Test suite for HuggingFaceProvider configuration, device detection, and lazy loading."""

    def test_initialization_defaults(self) -> None:
        provider = HuggingFaceProvider()
        assert provider.model_name == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        assert provider.max_tokens == 1024
        assert provider._pipeline is None
        assert provider.device in ["cpu", "cuda", "mps"]

    def test_initialization_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DATAALCHEMY_HF_MODEL", "custom-hf/model-name")
        provider = HuggingFaceProvider()
        assert provider.model_name == "custom-hf/model-name"

    def test_detect_device(self) -> None:
        provider = HuggingFaceProvider()
        detected = provider._detect_device()
        assert detected in ["cpu", "cuda", "mps"]

    def test_lazy_loading_and_pipeline_mock(self) -> None:
        provider = HuggingFaceProvider(model_name="mock-model", device="cpu")
        assert provider._pipeline is None

        mock_pipe = MagicMock()
        mock_pipe.return_value = [{"generated_text": '{"executive_summary": "Generated from HF"}'}]

        with patch("transformers.pipelines.pipeline", return_value=mock_pipe) as mock_pipeline_factory:
            output = provider.generate(
                prompt="Analyze data",
                system_prompt="You are an analyst",
                temperature=0.0,
                max_tokens=256,
            )

            mock_pipeline_factory.assert_called_once_with(
                "text-generation",
                model="mock-model",
                device=-1,
            )
            assert output == '{"executive_summary": "Generated from HF"}'

    def test_generation_pipeline_error_raises_llm_provider_error(self) -> None:
        provider = HuggingFaceProvider(model_name="mock-model")

        with patch("transformers.pipelines.pipeline", side_effect=RuntimeError("CUDA out of memory")):
            with pytest.raises(LLMProviderError) as exc_info:
                provider.generate("Prompt")

            assert "Failed to load Hugging Face model" in str(exc_info.value)

    def test_generation_inference_error_raises_llm_provider_error(self) -> None:
        provider = HuggingFaceProvider(model_name="mock-model")
        mock_pipe = MagicMock(side_effect=ValueError("Token generation failed"))

        with patch("transformers.pipelines.pipeline", return_value=mock_pipe):
            with pytest.raises(LLMProviderError) as exc_info:
                provider.generate("Prompt")

            assert "Hugging Face inference error" in str(exc_info.value)
