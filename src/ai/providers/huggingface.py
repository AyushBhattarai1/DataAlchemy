"""Hugging Face open-weight local provider and mock provider for DataAlchemy.

Implements local inference through Hugging Face transformers with lazy model loading,
alongside a deterministic MockLLMProvider for unit and integration testing.
"""

import json
import os
from typing import Any, Dict, List, Optional

from src.ai.providers import LLMProvider, LLMProviderError


class HuggingFaceProvider(LLMProvider):
    """Local open-weight provider utilizing Hugging Face transformers."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        max_tokens: int = 1024,
    ) -> None:
        """Initialize HuggingFaceProvider with lazy weight loading.

        Args:
            model_name: Hugging Face model identifier or local directory path.
                        Defaults to DATAALCHEMY_HF_MODEL env var or 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'.
            device: Computing device ('cpu', 'cuda', 'mps', or auto-detected).
            max_tokens: Default maximum token generation limit.
        """
        self.model_name = model_name or os.getenv(
            "DATAALCHEMY_HF_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        )
        self.device = device or self._detect_device()
        self.max_tokens = max_tokens
        self._pipeline = None

    def _detect_device(self) -> str:
        """Automatically detect hardware accelerator (MPS, CUDA, or CPU)."""
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"

    def _get_pipeline(self) -> Any:
        """Lazy load transformers text-generation pipeline."""
        if self._pipeline is None:
            try:
                from transformers.pipelines import pipeline
                # Map device string to pipeline device parameter
                device_arg = 0 if self.device == "cuda" else (-1 if self.device == "cpu" else self.device)
                self._pipeline = pipeline(
                    "text-generation",
                    model=self.model_name,
                    device=device_arg,
                )
            except ImportError as err:
                raise LLMProviderError(
                    f"Hugging Face transformers library is not installed: {err}"
                ) from err
            except Exception as err:
                raise LLMProviderError(
                    f"Failed to load Hugging Face model '{self.model_name}': {err}"
                ) from err
        return self._pipeline

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Execute local inference against the loaded Hugging Face model."""
        pipe = self._get_pipeline()
        effective_max = max_tokens or self.max_tokens

        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        try:
            do_sample = temperature > 0.0
            kwargs: Dict[str, Any] = {
                "max_new_tokens": effective_max,
                "do_sample": do_sample,
                "return_full_text": False,
            }
            if do_sample:
                kwargs["temperature"] = temperature

            output = pipe(full_prompt, **kwargs)
            if isinstance(output, list) and output and "generated_text" in output[0]:
                return str(output[0]["generated_text"]).strip()
            return str(output).strip()
        except Exception as err:
            raise LLMProviderError(f"Hugging Face inference error: {err}") from err


class MockLLMProvider(LLMProvider):
    """Deterministic mock provider for unit tests, eliminating heavy model downloads."""

    def __init__(self, custom_response: Optional[str] = None) -> None:
        self.custom_response = custom_response
        self.call_count: int = 0
        self.last_prompt: Optional[str] = None
        self.last_system_prompt: Optional[str] = None

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        """Return deterministic JSON response without executing any neural model."""
        self.call_count += 1
        self.last_prompt = prompt
        self.last_system_prompt = system_prompt

        if self.custom_response is not None:
            return self.custom_response

        # Default valid structured JSON response grounded in common findings
        default_data = {
            "executive_summary": (
                "The dataset exhibits solid overall quality with a prominent correlation "
                "between pricing metrics and distinct segment differences across categories. "
                "Minor missingness should be monitored before predictive modeling."
            ),
            "insights": [
                {
                    "insight_id": "INSIGHT_01",
                    "title": "Strong association observed between revenue and marketing spend",
                    "summary": "Revenue and marketing spend exhibit a very strong positive statistical relationship.",
                    "explanation": "Pearson correlation r = 0.84 indicates substantial co-movement across observations.",
                    "importance": "HIGH",
                    "category": "correlation",
                    "evidence": {"finding_id": "corr_total_amount_unit_price", "r": 0.84, "p_value": 0.038},
                    "affected_columns": ["total_amount", "unit_price"],
                    "caveats": "Does not establish causation; missing values in revenue should be handled.",
                },
                {
                    "insight_id": "INSIGHT_02",
                    "title": "Category distribution shows significant group-level variation",
                    "summary": "Key performance metrics vary significantly across product categories.",
                    "explanation": "Average quantities and amounts differ by up to 4x between product segments.",
                    "importance": "MEDIUM",
                    "category": "segment",
                    "evidence": {"finding_id": "rel_product_category_quantity", "variation_ratio": 4.0},
                    "affected_columns": ["product_category", "quantity"],
                    "caveats": None,
                },
            ],
            "recommendations": [
                {
                    "recommendation_id": "REC_01",
                    "title": "Verify data collection for missing fields",
                    "description": "Investigate moderate missingness in optional numerical features prior to modeling.",
                    "action_type": "investigation",
                    "supporting_evidence": ["MODERATE_MISSINGNESS", "quantity"],
                    "priority": "medium",
                }
            ],
            "warnings": [
                "Dataset contains mild sparsity in select numerical columns."
            ],
        }
        return json.dumps(default_data, indent=2)
