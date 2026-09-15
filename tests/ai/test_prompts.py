"""Unit tests for prompt generation and anti-hallucination instructions in src/ai/prompts.py."""

from src.ai.prompts import (
    SYSTEM_PROMPT,
    build_insight_prompt,
    build_system_prompt,
)


class TestAIPrompts:
    """Test suite verifying prompt safety guidelines, templates, and formatting."""

    def test_system_prompt_anti_hallucination_rules(self) -> None:
        sys_prompt = build_system_prompt()
        assert sys_prompt == SYSTEM_PROMPT

        # Check explicit anti-hallucination constraints
        assert "STRICT FACTUAL GROUNDING" in sys_prompt
        assert "Never invent or hallucinate numbers" in sys_prompt

        # Check explicit non-causal constraints
        assert "NO CAUSAL CLAIMS" in sys_prompt
        assert "Correlation does not imply causation" in sys_prompt
        assert "statistical association" in sys_prompt

        # Check quality caveats instruction
        assert "ACKNOWLEDGE DATA QUALITY" in sys_prompt

        # Check evidence-based recommendations requirement
        assert "EVIDENCE-BASED RECOMMENDATIONS" in sys_prompt
        assert "Never invent unsupported business strategy" in sys_prompt

        # Check JSON output mandate
        assert "STRUCTURED JSON OUTPUT" in sys_prompt

    def test_build_insight_prompt_formatting(self) -> None:
        mock_context = "### DATASET SUMMARY\n- Name: demo_data.csv\n- Dimensions: 100 rows x 5 columns"
        prompt = build_insight_prompt(mock_context)

        # Ensure context is embedded
        assert mock_context in prompt

        # Ensure schema structure instructions are present
        assert "executive_summary" in prompt
        assert "insights" in prompt
        assert "insight_id" in prompt
        assert "importance" in prompt
        assert "affected_columns" in prompt
        assert "recommendations" in prompt
        assert "supporting_evidence" in prompt
        assert "warnings" in prompt
