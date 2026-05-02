"""Unit tests for memory integration into existing components.

Covers:
- Participant: init_memory, idempotent init, convenience method no-ops
- ResponseGenerator: prompt branching with/without memory
- MemoryConfig: defaults and SessionConfig integration
- Memory model serialization (model_dump round-trips)
"""

import pytest

from focus_group.core.participant import Participant
from focus_group.core.response_generator import ResponseGenerator
from focus_group.models.config import (
    MemoryConfig,
    OllamaModelConfig,
    SessionConfig,
)
from focus_group.models.memory import (
    ConsensusLevel,
    ParticipantMemory,
    QuestionRecord,
    StanceMatrix,
    StancePosition,
)
from focus_group.models.persona import Persona
from tests.mocks.mock_provider import MockLLMProvider

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def persona() -> Persona:
    """Provide a sample persona for memory tests."""
    return Persona(
        name="Sarah Chen",
        age=34,
        background="Senior software engineer with 10 years experience",
        personality_traits=["analytical", "collaborative"],
        communication_style="Clear and structured",
        demographic_source="Female, 30-40, tech",
    )


@pytest.fixture
def provider() -> MockLLMProvider:
    """Provide a mock provider."""
    return MockLLMProvider()


@pytest.fixture
def participant(persona: Persona, provider: MockLLMProvider) -> Participant:
    """Provide a participant without memory."""
    return Participant(persona=persona, provider=provider)


# =============================================================================
# PARTICIPANT MEMORY INTEGRATION
# =============================================================================


class TestParticipantMemory:
    """Tests for Participant memory integration."""

    def test_memory_is_none_by_default(self, participant: Participant):
        """Participant memory is None when not initialized."""
        assert participant.memory is None

    def test_init_memory_creates_memory(self, participant: Participant):
        """init_memory() creates a ParticipantMemory from the persona."""
        participant.init_memory(focus_group_topic="Remote work")
        assert participant.memory is not None
        assert isinstance(participant.memory, ParticipantMemory)
        assert participant.memory.episodic.participant_name == "Sarah Chen"
        assert participant.memory.episodic.focus_group_topic == "Remote work"

    def test_init_memory_is_idempotent(self, participant: Participant):
        """Calling init_memory() twice does not reset memory."""
        participant.init_memory(focus_group_topic="Topic A")
        memory_id = id(participant.memory)
        participant.init_memory(focus_group_topic="Topic B")
        assert id(participant.memory) == memory_id
        assert participant.memory.episodic.focus_group_topic == "Topic A"

    def test_memory_on_new_question_noop_without_memory(self, participant: Participant):
        """memory_on_new_question is a no-op when memory is None."""
        # Should not raise
        participant.memory_on_new_question("What?", 0)

    def test_memory_on_turn_noop_without_memory(self, participant: Participant):
        """memory_on_turn is a no-op when memory is None."""
        participant.memory_on_turn("Someone", "Something", 1)

    def test_memory_on_question_complete_noop_without_memory(self, participant: Participant):
        """memory_on_question_complete is a no-op when memory is None."""
        participant.memory_on_question_complete(
            summary="Summary",
            key_themes=["theme"],
            consensus_level=ConsensusLevel.HIGH,
            notable_interactions=[],
            turn_count=3,
        )

    def test_memory_convenience_methods_work_with_memory(self, participant: Participant):
        """Convenience methods delegate to memory when initialized."""
        participant.init_memory()
        participant.memory_on_new_question("Test question?", 0)
        assert participant.memory.question.current_question == "Test question?"
        assert participant.memory.question.current_question_index == 0

        participant.memory_on_turn("Sarah Chen", "I think so.", 1)
        assert len(participant.memory.working.recent_turns) == 1
        assert participant.memory.working.recent_turns[0].is_own_turn is True

        participant.memory_on_question_complete(
            summary="Good discussion",
            key_themes=["agreement"],
            consensus_level=ConsensusLevel.MODERATE,
            notable_interactions=[],
            turn_count=1,
        )
        assert len(participant.memory.question.completed_questions) == 1


# =============================================================================
# RESPONSE GENERATOR MEMORY PATH
# =============================================================================


class TestResponseGeneratorMemoryPath:
    """Tests for ResponseGenerator memory-enriched prompt branching."""

    async def test_legacy_path_without_memory(self, persona: Persona, provider: MockLLMProvider):
        """Without memory, generate_response uses raw context (legacy path)."""
        participant = Participant(persona=persona, provider=provider)
        gen = ResponseGenerator(provider)

        result = await gen.generate_response(participant, "Test prompt")
        assert isinstance(result, str)
        assert len(result) > 0

        # Verify the prompt did NOT include memory sections
        last_call = provider.get_last_call()
        assert "=== YOUR IDENTITY ===" not in last_call["prompt"]

    async def test_memory_path_with_memory_and_matrix(
        self, persona: Persona, provider: MockLLMProvider
    ):
        """With memory + stance_matrix, generate_response uses enriched path."""
        participant = Participant(persona=persona, provider=provider)
        participant.init_memory(focus_group_topic="Testing")
        participant.memory_on_new_question("What about tests?", 0)

        gen = ResponseGenerator(provider)
        matrix = StanceMatrix()

        result = await gen.generate_response(participant, "Test prompt", stance_matrix=matrix)
        assert isinstance(result, str)

        # Verify the prompt used the memory-enriched format
        last_call = provider.get_last_call()
        assert "=== YOUR IDENTITY ===" in last_call["prompt"]
        assert "Sarah Chen" in last_call["prompt"]

    async def test_memory_path_falls_back_without_matrix(
        self, persona: Persona, provider: MockLLMProvider
    ):
        """With memory but no stance_matrix, falls back to legacy path."""
        participant = Participant(persona=persona, provider=provider)
        participant.init_memory()

        gen = ResponseGenerator(provider)

        result = await gen.generate_response(participant, "Test prompt")
        assert isinstance(result, str)

        # Should use legacy path (no memory sections)
        last_call = provider.get_last_call()
        assert "=== YOUR IDENTITY ===" not in last_call["prompt"]


# =============================================================================
# MEMORY CONFIG
# =============================================================================


class TestMemoryConfig:
    """Tests for MemoryConfig and SessionConfig integration."""

    def test_memory_config_defaults(self):
        """MemoryConfig has sensible defaults."""
        config = MemoryConfig()
        assert config.working_memory_window == 10
        assert config.enable_stance_tracking is True
        assert config.question_history_limit == 5

    def test_session_config_memory_default_none(self):
        """SessionConfig.memory defaults to None (legacy behavior)."""
        config = SessionConfig(
            demographics=["Test participant"],
            questions=["Test question?"],
            model=OllamaModelConfig(),
        )
        assert config.memory is None

    def test_session_config_with_memory(self):
        """SessionConfig accepts a MemoryConfig."""
        config = SessionConfig(
            demographics=["Test participant"],
            questions=["Test question?"],
            model=OllamaModelConfig(),
            memory=MemoryConfig(working_memory_window=5),
        )
        assert config.memory is not None
        assert config.memory.working_memory_window == 5


# =============================================================================
# MEMORY SERIALIZATION
# =============================================================================


class TestMemorySerialization:
    """Tests for memory model serialization via model_dump()."""

    def test_stance_matrix_serializes(self):
        """StanceMatrix.model_dump() produces a valid dict with enum values as strings."""
        matrix = StanceMatrix()
        matrix.update_stance(
            question_index=0,
            participant_name="Alice",
            position=StancePosition.SUPPORTIVE,
            key_points=["Good point"],
            sentiment_score=0.8,
        )
        dumped = matrix.model_dump()
        assert isinstance(dumped, dict)
        assert "stances" in dumped
        assert 0 in dumped["stances"]
        assert "Alice" in dumped["stances"][0]
        alice_stance = dumped["stances"][0]["Alice"]
        assert alice_stance["position"] == "supportive"
        assert alice_stance["key_points"] == ["Good point"]
        assert alice_stance["contribution_count"] == 1

    def test_question_record_serializes(self):
        """QuestionRecord.model_dump() produces a valid dict."""
        record = QuestionRecord(
            question_index=0,
            question_text="What do you think?",
            summary="Good discussion",
            key_themes=["agreement", "collaboration"],
            consensus_level=ConsensusLevel.MODERATE,
            turn_count=3,
        )
        dumped = record.model_dump()
        assert isinstance(dumped, dict)
        assert dumped["question_index"] == 0
        assert dumped["question_text"] == "What do you think?"
        assert dumped["consensus_level"] == "moderate"
        assert dumped["key_themes"] == ["agreement", "collaboration"]
        assert dumped["turn_count"] == 3
