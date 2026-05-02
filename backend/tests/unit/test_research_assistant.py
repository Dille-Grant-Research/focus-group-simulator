"""Tests for the Research Assistant.

Covers:
- analyze_turn: valid JSON, markdown-fenced JSON, malformed JSON, contribution counting
- generate_question_summary: valid JSON, malformed JSON defaults, empty transcript
- JSON parsing: clean JSON, fenced JSON, fenced without label, garbage, empty string
- Prompt verification: correct prompt content sent to provider
"""

import json

import pytest

from focus_group.core.research_assistant import ResearchAssistant
from focus_group.models.memory import (
    ParticipantStance,
    StanceMatrix,
    StancePosition,
)
from tests.mocks.mock_provider import MockLLMProvider

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def stance_matrix() -> StanceMatrix:
    """Provide a fresh StanceMatrix for each test."""
    return StanceMatrix()


@pytest.fixture
def research_provider() -> MockLLMProvider:
    """Provide a MockLLMProvider configured for research assistant tests.

    Clears default patterns so we can control responses exactly.
    """
    provider = MockLLMProvider(
        model_name="mock-research-model",
        max_context_tokens=8192,
        available=True,
    )
    # Clear default patterns so our explicit responses take priority
    return provider


@pytest.fixture
def assistant(research_provider: MockLLMProvider) -> ResearchAssistant:
    """Provide a ResearchAssistant with a mock provider."""
    return ResearchAssistant(research_provider)


@pytest.fixture
def sample_participants() -> list[str]:
    """Provide sample participant names."""
    return ["Sarah Chen", "Robert Wilson", "Jordan Martinez"]


# =============================================================================
# ANALYZE TURN
# =============================================================================


class TestAnalyzeTurn:
    """Tests for ResearchAssistant.analyze_turn()."""

    async def test_valid_json_parsed(
        self,
        research_provider: MockLLMProvider,
        assistant: ResearchAssistant,
        stance_matrix: StanceMatrix,
        sample_participants: list[str],
    ):
        """Valid JSON response is correctly parsed into a ParticipantStance."""
        valid_response = json.dumps(
            {
                "position": "supportive",
                "key_points": ["Remote work improves flexibility", "Better work-life balance"],
                "sentiment_score": 0.8,
            }
        )
        research_provider.add_response(r".*", valid_response)

        result = await assistant.analyze_turn(
            speaker_name="Sarah Chen",
            statement="I think remote work is great for flexibility.",
            question="What do you think about remote work?",
            question_index=0,
            stance_matrix=stance_matrix,
            participants=sample_participants,
        )

        assert isinstance(result, ParticipantStance)
        assert result.position == StancePosition.SUPPORTIVE
        assert len(result.key_points) == 2
        assert result.sentiment_score == 0.8
        assert result.contribution_count == 1

    async def test_markdown_fenced_json_parsed(
        self,
        research_provider: MockLLMProvider,
        assistant: ResearchAssistant,
        stance_matrix: StanceMatrix,
        sample_participants: list[str],
    ):
        """JSON wrapped in markdown code fences is correctly parsed."""
        fenced_response = (
            "```json\n"
            '{"position": "skeptical", "key_points": ["Cost concerns"], "sentiment_score": 0.3}\n'
            "```"
        )
        research_provider.add_response(r".*", fenced_response)

        result = await assistant.analyze_turn(
            speaker_name="Robert Wilson",
            statement="I'm not sure about that approach.",
            question="What do you think about remote work?",
            question_index=0,
            stance_matrix=stance_matrix,
            participants=sample_participants,
        )

        assert result.position == StancePosition.SKEPTICAL
        assert result.key_points == ["Cost concerns"]
        assert result.sentiment_score == 0.3

    async def test_malformed_json_returns_neutral_default(
        self,
        research_provider: MockLLMProvider,
        assistant: ResearchAssistant,
        stance_matrix: StanceMatrix,
        sample_participants: list[str],
    ):
        """Malformed JSON response gracefully falls back to NEUTRAL stance."""
        research_provider.add_response(r".*", "This is not valid JSON at all!")

        result = await assistant.analyze_turn(
            speaker_name="Jordan Martinez",
            statement="Interesting point.",
            question="What do you think about remote work?",
            question_index=0,
            stance_matrix=stance_matrix,
            participants=sample_participants,
        )

        assert result.position == StancePosition.NEUTRAL
        assert result.key_points == []
        assert result.sentiment_score == 0.5
        assert result.contribution_count == 1

    async def test_contribution_count_increments(
        self,
        research_provider: MockLLMProvider,
        assistant: ResearchAssistant,
        stance_matrix: StanceMatrix,
        sample_participants: list[str],
    ):
        """Each call to analyze_turn increments the contribution count."""
        valid_response = json.dumps(
            {"position": "supportive", "key_points": ["Point A"], "sentiment_score": 0.7}
        )
        research_provider.add_response(r".*", valid_response)

        await assistant.analyze_turn(
            speaker_name="Sarah Chen",
            statement="First statement.",
            question="Question?",
            question_index=0,
            stance_matrix=stance_matrix,
            participants=sample_participants,
        )
        result = await assistant.analyze_turn(
            speaker_name="Sarah Chen",
            statement="Second statement.",
            question="Question?",
            question_index=0,
            stance_matrix=stance_matrix,
            participants=sample_participants,
        )

        assert result.contribution_count == 2

    async def test_correct_prompt_sent_to_provider(
        self,
        research_provider: MockLLMProvider,
        assistant: ResearchAssistant,
        stance_matrix: StanceMatrix,
        sample_participants: list[str],
    ):
        """Verify the prompt sent to the LLM contains expected content."""
        valid_response = json.dumps(
            {"position": "neutral", "key_points": [], "sentiment_score": 0.5}
        )
        research_provider.add_response(r".*", valid_response)

        await assistant.analyze_turn(
            speaker_name="Sarah Chen",
            statement="Testing the prompt.",
            question="Is this a test?",
            question_index=0,
            stance_matrix=stance_matrix,
            participants=sample_participants,
        )

        last_call = research_provider.get_last_call()
        assert last_call is not None
        prompt = last_call["prompt"]
        assert "Sarah Chen" in prompt
        assert "Testing the prompt." in prompt
        assert "Is this a test?" in prompt
        assert "Sarah Chen" in prompt
        assert "Robert Wilson" in prompt
        assert "Jordan Martinez" in prompt

    async def test_invalid_position_defaults_to_neutral(
        self,
        research_provider: MockLLMProvider,
        assistant: ResearchAssistant,
        stance_matrix: StanceMatrix,
        sample_participants: list[str],
    ):
        """An unrecognized position string defaults to NEUTRAL."""
        response = json.dumps(
            {"position": "confused", "key_points": ["Not a real stance"], "sentiment_score": 0.5}
        )
        research_provider.add_response(r".*", response)

        result = await assistant.analyze_turn(
            speaker_name="Sarah Chen",
            statement="Something.",
            question="Question?",
            question_index=0,
            stance_matrix=stance_matrix,
            participants=sample_participants,
        )

        assert result.position == StancePosition.NEUTRAL


# =============================================================================
# GENERATE QUESTION SUMMARY
# =============================================================================


class TestGenerateQuestionSummary:
    """Tests for ResearchAssistant.generate_question_summary()."""

    async def test_valid_json_parsed(
        self,
        research_provider: MockLLMProvider,
        assistant: ResearchAssistant,
        stance_matrix: StanceMatrix,
    ):
        """Valid summary JSON is correctly parsed."""
        valid_response = json.dumps(
            {
                "summary": "The group discussed remote work with mixed opinions.",
                "key_themes": ["flexibility", "isolation"],
                "consensus_level": "moderate",
                "notable_interactions": ["Sarah and Robert disagreed on productivity"],
            }
        )
        research_provider.add_response(r".*", valid_response)

        result = await assistant.generate_question_summary(
            question="What do you think about remote work?",
            question_index=0,
            transcript=[
                {"speaker": "Sarah Chen", "content": "I love it."},
                {"speaker": "Robert Wilson", "content": "I'm not so sure."},
            ],
            stance_matrix=stance_matrix,
        )

        assert result["summary"] == "The group discussed remote work with mixed opinions."
        assert result["key_themes"] == ["flexibility", "isolation"]
        assert result["consensus_level"] == "moderate"
        assert len(result["notable_interactions"]) == 1

    async def test_malformed_json_returns_defaults(
        self,
        research_provider: MockLLMProvider,
        assistant: ResearchAssistant,
        stance_matrix: StanceMatrix,
    ):
        """Malformed JSON returns safe default values."""
        research_provider.add_response(r".*", "Not valid JSON, sorry!")

        result = await assistant.generate_question_summary(
            question="What do you think?",
            question_index=0,
            transcript=[{"speaker": "Sarah", "content": "Something."}],
            stance_matrix=stance_matrix,
        )

        assert result["summary"] == "Discussion summary unavailable."
        assert result["key_themes"] == []
        assert result["consensus_level"] == "none"
        assert result["notable_interactions"] == []

    async def test_transcript_included_in_prompt(
        self,
        research_provider: MockLLMProvider,
        assistant: ResearchAssistant,
        stance_matrix: StanceMatrix,
    ):
        """Verify the transcript content appears in the prompt sent to the LLM."""
        valid_response = json.dumps(
            {
                "summary": "Summary.",
                "key_themes": [],
                "consensus_level": "none",
                "notable_interactions": [],
            }
        )
        research_provider.add_response(r".*", valid_response)

        await assistant.generate_question_summary(
            question="Test question?",
            question_index=0,
            transcript=[
                {"speaker": "Sarah Chen", "content": "I support this."},
                {"speaker": "Robert Wilson", "content": "I disagree."},
            ],
            stance_matrix=stance_matrix,
        )

        last_call = research_provider.get_last_call()
        assert last_call is not None
        prompt = last_call["prompt"]
        assert "Sarah Chen: I support this." in prompt
        assert "Robert Wilson: I disagree." in prompt
        assert "Test question?" in prompt

    async def test_empty_transcript_handled(
        self,
        research_provider: MockLLMProvider,
        assistant: ResearchAssistant,
        stance_matrix: StanceMatrix,
    ):
        """An empty transcript is handled gracefully."""
        valid_response = json.dumps(
            {
                "summary": "No discussion occurred.",
                "key_themes": [],
                "consensus_level": "none",
                "notable_interactions": [],
            }
        )
        research_provider.add_response(r".*", valid_response)

        result = await assistant.generate_question_summary(
            question="Test question?",
            question_index=0,
            transcript=[],
            stance_matrix=stance_matrix,
        )

        assert result["summary"] == "No discussion occurred."

        last_call = research_provider.get_last_call()
        assert "(no transcript)" in last_call["prompt"]

    async def test_invalid_consensus_level_defaults_to_none(
        self,
        research_provider: MockLLMProvider,
        assistant: ResearchAssistant,
        stance_matrix: StanceMatrix,
    ):
        """An unrecognized consensus level defaults to 'none'."""
        response = json.dumps(
            {
                "summary": "Summary.",
                "key_themes": [],
                "consensus_level": "extremely_high",
                "notable_interactions": [],
            }
        )
        research_provider.add_response(r".*", response)

        result = await assistant.generate_question_summary(
            question="Test?",
            question_index=0,
            transcript=[],
            stance_matrix=stance_matrix,
        )

        assert result["consensus_level"] == "none"


# =============================================================================
# JSON PARSING
# =============================================================================


class TestJSONParsing:
    """Tests for the internal _extract_json static method."""

    def test_clean_json(self):
        """Plain JSON string is parsed correctly."""
        raw = '{"position": "supportive", "key_points": ["good"], "sentiment_score": 0.9}'
        result = ResearchAssistant._extract_json(raw)
        assert result is not None
        assert result["position"] == "supportive"

    def test_fenced_json_with_label(self):
        """JSON wrapped in ```json ... ``` fences is parsed."""
        raw = '```json\n{"position": "skeptical", "key_points": [], "sentiment_score": 0.2}\n```'
        result = ResearchAssistant._extract_json(raw)
        assert result is not None
        assert result["position"] == "skeptical"

    def test_fenced_json_without_label(self):
        """JSON wrapped in ``` ... ``` fences (no label) is parsed."""
        raw = '```\n{"position": "neutral", "key_points": [], "sentiment_score": 0.5}\n```'
        result = ResearchAssistant._extract_json(raw)
        assert result is not None
        assert result["position"] == "neutral"

    def test_garbage_returns_none(self):
        """Completely invalid input returns None."""
        result = ResearchAssistant._extract_json("This is just random text with no JSON.")
        assert result is None

    def test_empty_string_returns_none(self):
        """Empty string returns None."""
        result = ResearchAssistant._extract_json("")
        assert result is None

    def test_whitespace_only_returns_none(self):
        """Whitespace-only string returns None."""
        result = ResearchAssistant._extract_json("   \n\t  ")
        assert result is None

    def test_json_with_surrounding_text(self):
        """JSON embedded in surrounding text is extracted."""
        raw = (
            "Here is the analysis: "
            '{"position": "mixed", "key_points": ["a"], "sentiment_score": 0.6}'
            " Hope that helps!"
        )
        result = ResearchAssistant._extract_json(raw)
        assert result is not None
        assert result["position"] == "mixed"
