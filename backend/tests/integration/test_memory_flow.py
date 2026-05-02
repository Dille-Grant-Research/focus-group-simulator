"""Integration tests for the three-tier memory system.

Covers:
- Full session with memory enabled completes without error
- Full session without memory is unchanged (backward compatibility)
- Memory is populated after question processing
- QuestionHandler with memory hooks end-to-end
"""

import pytest

from focus_group.core.nonverbal_generator import NonVerbalGenerator
from focus_group.core.participant import Participant
from focus_group.core.question_handler import QuestionHandler
from focus_group.core.research_assistant import ResearchAssistant
from focus_group.core.response_generator import ResponseGenerator
from focus_group.core.session import FocusGroupSession
from focus_group.core.speaker_selection import (
    FirstSpeakerSelector,
    NextSpeakerSelector,
    SilenceHandler,
)
from focus_group.models.config import (
    MemoryConfig,
    OllamaModelConfig,
    OutputConfig,
    SessionConfig,
)
from focus_group.models.memory import FocusGroupMemory
from focus_group.models.persona import Persona
from tests.mocks.mock_provider import MockLLMProvider

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def provider() -> MockLLMProvider:
    """Provide a mock LLM provider."""
    return MockLLMProvider()


@pytest.fixture
def sample_personas() -> list[Persona]:
    """Provide sample personas."""
    return [
        Persona(
            name="Sarah Chen",
            age=34,
            background="Software engineer with 10 years experience",
            personality_traits=["analytical", "collaborative"],
            communication_style="Clear and structured",
            demographic_source="Female, 30-40, tech",
        ),
        Persona(
            name="Robert Wilson",
            age=57,
            background="Retired teacher with 30 years in education",
            personality_traits=["patient", "reflective"],
            communication_style="Thoughtful and measured",
            demographic_source="Male, 50-60, retired",
        ),
    ]


@pytest.fixture
def participants_with_memory(
    sample_personas: list[Persona], provider: MockLLMProvider
) -> list[Participant]:
    """Create participants with memory initialized."""
    participants = []
    for persona in sample_personas:
        p = Participant(persona=persona, provider=provider)
        p.init_memory(focus_group_topic="Remote work")
        participants.append(p)
    return participants


@pytest.fixture
def participants_without_memory(
    sample_personas: list[Persona], provider: MockLLMProvider
) -> list[Participant]:
    """Create participants without memory (legacy mode)."""
    return [Participant(persona=p, provider=provider) for p in sample_personas]


# =============================================================================
# FULL SESSION TESTS
# =============================================================================


class TestSessionWithMemory:
    """Test full session flow with memory enabled."""

    async def test_session_completes_with_memory(self, provider: MockLLMProvider, tmp_path):
        """A full session with memory enabled runs to completion."""
        config = SessionConfig(
            session_name="memory_test",
            demographics=[
                "Female, 30-40, tech professional",
                "Male, 50-60, retired teacher",
            ],
            questions=["What do you think about remote work?"],
            purpose="Discuss workplace trends",
            model=OllamaModelConfig(),
            output=OutputConfig(
                directory=str(tmp_path / "output"),
                save_debug_log=False,
                save_clean_transcript=False,
            ),
            memory=MemoryConfig(),
            validate_personas=False,
        )

        session = FocusGroupSession(provider=provider, config=config)
        result = await session.run()

        assert result.completed_at is not None
        assert len(result.participants) == 2
        assert len(result.question_results) == 1

    async def test_session_result_contains_memory_dumps(self, provider: MockLLMProvider, tmp_path):
        """SessionResult contains memory dumps when memory is enabled."""
        config = SessionConfig(
            session_name="memory_dump_test",
            demographics=[
                "Female, 30-40, tech professional",
                "Male, 50-60, retired teacher",
            ],
            questions=["What do you think about remote work?"],
            purpose="Discuss workplace trends",
            model=OllamaModelConfig(),
            output=OutputConfig(
                directory=str(tmp_path / "output"),
                save_debug_log=False,
                save_clean_transcript=False,
            ),
            memory=MemoryConfig(),
            validate_personas=False,
        )

        session = FocusGroupSession(provider=provider, config=config)
        result = await session.run()

        # Memory artifacts should be populated
        assert result.stance_matrix_dump is not None
        assert isinstance(result.stance_matrix_dump, dict)
        assert "stances" in result.stance_matrix_dump

        assert result.participant_memory_dumps is not None
        assert len(result.participant_memory_dumps) == 2
        for dump in result.participant_memory_dumps:
            assert "participant_name" in dump
            assert "episodic" in dump
            assert "completed_questions" in dump

    async def test_session_result_memory_none_without_config(
        self, provider: MockLLMProvider, tmp_path
    ):
        """SessionResult has None memory dumps when memory is not configured."""
        config = SessionConfig(
            session_name="no_memory_test",
            demographics=[
                "Female, 30-40, tech professional",
                "Male, 50-60, retired teacher",
            ],
            questions=["What do you think about remote work?"],
            purpose="Discuss workplace trends",
            model=OllamaModelConfig(),
            output=OutputConfig(
                directory=str(tmp_path / "output"),
                save_debug_log=False,
                save_clean_transcript=False,
            ),
            validate_personas=False,
        )

        assert config.memory is None
        session = FocusGroupSession(provider=provider, config=config)
        result = await session.run()

        assert result.stance_matrix_dump is None
        assert result.participant_memory_dumps is None

    async def test_session_completes_without_memory(self, provider: MockLLMProvider, tmp_path):
        """A full session without memory (legacy) runs identically to before."""
        config = SessionConfig(
            session_name="legacy_test",
            demographics=[
                "Female, 30-40, tech professional",
                "Male, 50-60, retired teacher",
            ],
            questions=["What do you think about remote work?"],
            purpose="Discuss workplace trends",
            model=OllamaModelConfig(),
            output=OutputConfig(
                directory=str(tmp_path / "output"),
                save_debug_log=False,
                save_clean_transcript=False,
            ),
            validate_personas=False,
        )

        assert config.memory is None  # Confirm legacy mode

        session = FocusGroupSession(provider=provider, config=config)
        result = await session.run()

        assert result.completed_at is not None
        assert len(result.participants) == 2
        assert len(result.question_results) == 1


# =============================================================================
# QUESTION HANDLER WITH MEMORY
# =============================================================================


class TestQuestionHandlerMemory:
    """Test QuestionHandler with memory hooks."""

    async def test_memory_populated_after_question(
        self,
        provider: MockLLMProvider,
        participants_with_memory: list[Participant],
    ):
        """Memory is populated with turns and question records after processing."""
        fgm = FocusGroupMemory(topic="Remote work")
        ra = ResearchAssistant(provider)

        handler = QuestionHandler(
            provider=provider,
            response_generator=ResponseGenerator(provider),
            nonverbal_generator=NonVerbalGenerator(provider),
            first_speaker_selector=FirstSpeakerSelector(provider),
            next_speaker_selector=NextSpeakerSelector(provider),
            silence_handler=SilenceHandler(provider),
            research_assistant=ra,
            focus_group_memory=fgm,
        )

        result = await handler.process_question(
            participants=participants_with_memory,
            question="What do you think about remote work?",
            question_index=0,
        )

        # Question completed
        assert result.total_turns >= 2  # At least 2 participants spoke

        # Memory was populated
        for p in participants_with_memory:
            mem = p.memory
            assert mem is not None
            # Working memory should have turns
            assert len(mem.working.recent_turns) > 0
            # Question should have been archived
            assert len(mem.question.completed_questions) == 1
            record = mem.question.completed_questions[0]
            assert record.question_text == "What do you think about remote work?"
            assert record.turn_count > 0

        # StanceMatrix should have entries
        stances = fgm.stance_matrix.get_question_summary(0)
        assert len(stances) > 0

        # Turn counters were incremented
        assert fgm.total_turns >= 2
        assert fgm.current_question_turns >= 2

    async def test_no_memory_hooks_without_memory(
        self,
        provider: MockLLMProvider,
        participants_without_memory: list[Participant],
    ):
        """Without memory components, the handler works exactly as before."""
        handler = QuestionHandler(
            provider=provider,
            response_generator=ResponseGenerator(provider),
            nonverbal_generator=NonVerbalGenerator(provider),
            first_speaker_selector=FirstSpeakerSelector(provider),
            next_speaker_selector=NextSpeakerSelector(provider),
            silence_handler=SilenceHandler(provider),
            # No research_assistant or focus_group_memory
        )

        result = await handler.process_question(
            participants=participants_without_memory,
            question="What do you think about remote work?",
        )

        # Should complete normally
        assert result.total_turns >= 2

        # Memory should still be None
        for p in participants_without_memory:
            assert p.memory is None
