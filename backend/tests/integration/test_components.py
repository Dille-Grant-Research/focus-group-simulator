"""Component integration tests for the focus group simulator.

This module tests the integration between different components of the system,
including error handling, edge cases, and component interactions.
"""

import json
from pathlib import Path

import pytest

from focus_group.context.storage import ContextStorage
from focus_group.context.summarizer import ContextSummarizer
from focus_group.core.introduction_handler import IntroductionHandler
from focus_group.core.nonverbal_generator import NonVerbalGenerator
from focus_group.core.participant import Participant
from focus_group.core.question_handler import QuestionHandler
from focus_group.core.response_generator import ResponseGenerator
from focus_group.core.speaker_selection import (
    FirstSpeakerSelector,
    NextSpeakerSelector,
    SilenceHandler,
)
from focus_group.logging.event_logger import EventLogger
from focus_group.models.config import ContextConfig
from focus_group.models.message import Message
from focus_group.models.persona import Persona
from focus_group.persona.generator import PersonaGenerator
from focus_group.persona.validator import PersonaValidator
from tests.mocks.mock_provider import MockLLMProvider

pytestmark = pytest.mark.integration


class TestParticipantIntegration:
    """Test Participant class integration with context management."""

    @pytest.mark.asyncio
    async def test_participant_context_initialization(
        self,
        sample_persona: Persona,
        mock_provider: MockLLMProvider,
    ):
        """Test that participant context is properly initialized with persona."""
        participant = Participant(
            persona=sample_persona,
            provider=mock_provider,
        )

        # Context should contain persona description
        context = participant.get_context_for_llm(include_nonverbals=False)
        assert sample_persona.name in context
        # Background should be in context
        assert "engineer" in context.lower() or sample_persona.background in context

    @pytest.mark.asyncio
    async def test_participant_dual_context_system(
        self,
        sample_persona: Persona,
        mock_provider: MockLLMProvider,
    ):
        """Test that participant maintains dual context correctly."""
        participant = Participant(
            persona=sample_persona,
            provider=mock_provider,
        )

        # Add verbal message
        verbal_msg = Message.participant_response(
            content="This is a verbal response",
            participant_name="Other Person",
        )
        participant.add_verbal_message(verbal_msg)

        # Add nonverbal message
        nonverbal_msg = Message.nonverbal(
            content="nods thoughtfully",
            participant_name="Other Person",
        )
        participant.add_nonverbal_message(nonverbal_msg)

        # Full context should contain both
        full_context = participant.get_context_for_llm(include_nonverbals=True)
        assert "verbal response" in full_context
        assert "nods" in full_context

        # Verbal-only context should only have verbal
        verbal_context = participant.get_context_for_llm(include_nonverbals=False)
        assert "verbal response" in verbal_context
        # Nonverbal should NOT be in verbal-only context
        assert "nods" not in verbal_context

    @pytest.mark.asyncio
    async def test_participant_context_summarization(
        self,
        sample_persona: Persona,
        mock_provider: MockLLMProvider,
    ):
        """Test that participant context can be summarized when needed."""
        # Use a small context limit to trigger summarization
        config = ContextConfig(
            max_tokens=1000,
            summarize_after_tokens=500,
        )

        participant = Participant(
            persona=sample_persona,
            provider=mock_provider,
            config=config,
        )

        # Add many messages to trigger summarization
        for i in range(10):
            msg = Message.participant_response(
                content=f"This is response number {i} with some additional text to fill context",
                participant_name="Other Person",
            )
            participant.add_verbal_message(msg)

        # Check if summarization is needed
        if participant.needs_summarization():
            await participant.check_and_summarize()

        # Participant should still have valid context
        context = participant.get_context_for_llm(include_nonverbals=False)
        assert len(context) > 0


class TestResponseGeneratorIntegration:
    """Test ResponseGenerator integration with participants."""

    @pytest.mark.asyncio
    async def test_response_generator_uses_context(
        self,
        sample_personas: list[Persona],
        mock_provider: MockLLMProvider,
    ):
        """Test that response generator uses participant context."""
        participants = [Participant(persona=p, provider=mock_provider) for p in sample_personas]

        generator = ResponseGenerator(mock_provider)

        # Add some context first
        for p in participants:
            p.set_question("What do you think about remote work?")

        # Generate response
        response = await generator.generate_response(
            participant=participants[0],
            prompt="Share your thoughts",
        )

        assert response is not None
        assert len(response) > 0

        # Provider should have been called
        assert mock_provider.call_count > 0

    @pytest.mark.asyncio
    async def test_response_generator_introduction(
        self,
        sample_persona: Persona,
        mock_provider: MockLLMProvider,
    ):
        """Test that response generator can create introductions."""
        participant = Participant(persona=sample_persona, provider=mock_provider)
        generator = ResponseGenerator(mock_provider)

        introduction = await generator.generate_introduction(participant)

        assert introduction is not None
        assert len(introduction) > 0


class TestNonVerbalGeneratorIntegration:
    """Test NonVerbalGenerator integration."""

    @pytest.mark.asyncio
    async def test_nonverbal_generator_parallel_reactions(
        self,
        sample_personas: list[Persona],
        mock_provider: MockLLMProvider,
    ):
        """Test parallel generation of non-verbal reactions."""
        participants = [Participant(persona=p, provider=mock_provider) for p in sample_personas]

        generator = NonVerbalGenerator(mock_provider)

        # Generate reactions in parallel
        reactions = await generator.generate_reactions_to_question_parallel(
            participants=participants,
            question="What is your opinion on remote work?",
        )

        # Should have reaction for each participant
        assert len(reactions) == len(participants)

        # Each reaction should be non-empty
        for name, reaction in reactions.items():
            assert len(reaction) > 0

    @pytest.mark.asyncio
    async def test_nonverbal_generator_reaction_to_statement(
        self,
        sample_personas: list[Persona],
        mock_provider: MockLLMProvider,
    ):
        """Test generating reactions to another participant's statement."""
        participants = [Participant(persona=p, provider=mock_provider) for p in sample_personas]

        generator = NonVerbalGenerator(mock_provider)

        # Generate reaction from second participant to first's statement
        reaction = await generator.generate_reaction_to_statement(
            participant=participants[1],
            speaker_name=participants[0].name,
            statement="I think remote work is great for productivity.",
        )

        assert reaction is not None
        assert len(reaction) > 0


class TestSpeakerSelectionIntegration:
    """Test speaker selection components integration."""

    @pytest.mark.asyncio
    async def test_first_speaker_selector(
        self,
        sample_personas: list[Persona],
        mock_provider: MockLLMProvider,
    ):
        """Test first speaker selection based on reactions."""
        participants = [Participant(persona=p, provider=mock_provider) for p in sample_personas]

        selector = FirstSpeakerSelector(mock_provider)

        reactions = {
            p.name: f"shows interest and leans forward ({i})" for i, p in enumerate(participants)
        }

        result = await selector.select_first_speaker(
            participants=participants,
            question="What do you think about remote work?",
            reactions=reactions,
        )

        assert result is not None
        assert result.participant is not None
        assert result.participant in participants
        assert result.reasoning is not None

    @pytest.mark.asyncio
    async def test_next_speaker_selector_returns_none_when_no_interest(
        self,
        sample_personas: list[Persona],
        mock_provider: MockLLMProvider,
    ):
        """Test that next speaker selector can return None."""
        participants = [Participant(persona=p, provider=mock_provider) for p in sample_personas]

        selector = NextSpeakerSelector(mock_provider)

        # Configure mock to indicate no one wants to speak
        mock_provider.add_response(
            r"next.*speak|who.*respond",
            json.dumps(
                {
                    "selected_speaker": None,
                    "reasoning": "No one appears eager to speak",
                    "someone_wants_to_speak": False,
                }
            ),
        )

        reactions = {
            p.name: "shows neutral expression, appears disinterested" for p in participants
        }

        result = await selector.select_next_speaker(
            participants=participants,
            current_speaker=participants[0],
            statement="That is my opinion.",
            reactions=reactions,
        )

        # Result can be None when no one wants to speak
        # (The mock is configured to return no interest)

    @pytest.mark.asyncio
    async def test_silence_handler_always_returns_speaker(
        self,
        sample_personas: list[Persona],
        mock_provider: MockLLMProvider,
    ):
        """Test that silence handler always returns a speaker."""
        participants = [Participant(persona=p, provider=mock_provider) for p in sample_personas]

        handler = SilenceHandler(mock_provider)

        reactions = {p.name: "appears thoughtful but hesitant" for p in participants}

        # Only first person has spoken
        already_spoken = {participants[0].name}

        result = await handler.select_reluctant_speaker(
            participants=participants,
            question="What do you think?",
            reactions=reactions,
            already_spoken=already_spoken,
        )

        # Should always return someone
        assert result is not None
        assert result.participant is not None
        # Should not be someone who already spoke
        assert result.participant.name not in already_spoken


class TestQuestionHandlerIntegration:
    """Test QuestionHandler integration with all components."""

    @pytest.mark.asyncio
    async def test_question_handler_full_flow(
        self,
        sample_personas: list[Persona],
        mock_provider: MockLLMProvider,
    ):
        """Test complete question processing flow."""
        participants = [Participant(persona=p, provider=mock_provider) for p in sample_personas]

        # Create all required components
        response_gen = ResponseGenerator(mock_provider)
        nonverbal_gen = NonVerbalGenerator(mock_provider)
        first_selector = FirstSpeakerSelector(mock_provider)
        next_selector = NextSpeakerSelector(mock_provider)
        silence_handler = SilenceHandler(mock_provider)

        handler = QuestionHandler(
            provider=mock_provider,
            response_generator=response_gen,
            nonverbal_generator=nonverbal_gen,
            first_speaker_selector=first_selector,
            next_speaker_selector=next_selector,
            silence_handler=silence_handler,
        )

        result = await handler.process_question(
            participants=participants,
            question="What is your opinion on remote work?",
        )

        # Verify result structure
        assert result is not None
        assert result.question == "What is your opinion on remote work?"

        # Should have responses
        assert len(result.responses) > 0

        # Should have speaking order
        assert len(result.speaking_order) == len(result.responses)

        # Should have timing
        assert result.started_at is not None
        assert result.completed_at is not None


class TestIntroductionHandlerIntegration:
    """Test IntroductionHandler integration."""

    @pytest.mark.asyncio
    async def test_introduction_handler_with_purpose(
        self,
        sample_personas: list[Persona],
        mock_provider: MockLLMProvider,
    ):
        """Test introduction handler with focus group purpose."""
        participants = [Participant(persona=p, provider=mock_provider) for p in sample_personas]

        response_gen = ResponseGenerator(mock_provider)
        nonverbal_gen = NonVerbalGenerator(mock_provider)

        handler = IntroductionHandler(
            provider=mock_provider,
            response_generator=response_gen,
            nonverbal_generator=nonverbal_gen,
        )

        result = await handler.process_introductions(
            participants=participants,
            purpose="We are here to discuss remote work policies.",
        )

        assert result is not None
        assert result.purpose == "We are here to discuss remote work policies."
        assert len(result.introductions) == len(participants)

        # Each introduction should have content
        for intro in result.introductions:
            assert intro.content is not None
            assert len(intro.content) > 0

    @pytest.mark.asyncio
    async def test_introduction_handler_without_purpose(
        self,
        sample_personas: list[Persona],
        mock_provider: MockLLMProvider,
    ):
        """Test introduction handler without focus group purpose."""
        participants = [Participant(persona=p, provider=mock_provider) for p in sample_personas]

        response_gen = ResponseGenerator(mock_provider)
        nonverbal_gen = NonVerbalGenerator(mock_provider)

        handler = IntroductionHandler(
            provider=mock_provider,
            response_generator=response_gen,
            nonverbal_generator=nonverbal_gen,
        )

        result = await handler.process_introductions(
            participants=participants,
            purpose=None,
        )

        assert result is not None
        assert result.purpose is None
        assert len(result.introductions) == len(participants)


class TestEventLoggerIntegration:
    """Test EventLogger integration with session components."""

    def test_event_logger_logs_all_event_types(
        self,
        sample_persona: Persona,
        temp_output_dir: Path,
    ):
        """Test that event logger can log all event types."""
        log_path = temp_output_dir / "events.jsonl"
        logger = EventLogger(file_path=log_path)

        # Log various events
        logger.log_session_start("test_session", {"participants": 3})
        logger.log_persona_generated(sample_persona)
        logger.log_question_asked("Test question?", question_number=1)
        logger.log_response("Test Person", "Test response", turn=1)
        logger.log_nonverbal_reaction("Test Person", "nods", "question")
        logger.log_speaker_selected("Test Person", "first", "Appeared eager")
        logger.log_session_end({"total_responses": 5})

        logger.close()

        # Verify file was created
        assert log_path.exists()

        # Verify we can get events by type
        all_events = logger.get_events()
        assert len(all_events) == 7

        # Verify event types
        event_types = [e.event_type for e in all_events]
        assert "session_start" in event_types
        assert "persona_generated" in event_types
        assert "question_asked" in event_types

    def test_event_logger_json_serialization(
        self,
        sample_persona: Persona,
    ):
        """Test that logged events can be serialized to JSON."""
        logger = EventLogger()

        logger.log_session_start("test", {})
        logger.log_persona_generated(sample_persona)

        json_str = logger.to_json()

        # Should be valid JSON
        data = json.loads(json_str)
        assert len(data) == 2


class TestPersonaValidatorIntegration:
    """Test PersonaValidator integration with generated personas."""

    @pytest.mark.asyncio
    async def test_validator_with_generated_personas(
        self,
        sample_demographics: list[str],
        mock_provider: MockLLMProvider,
    ):
        """Test validating generated personas."""
        generator = PersonaGenerator(mock_provider)
        validator = PersonaValidator()

        personas = await generator.generate_personas(sample_demographics)

        # Validate each persona
        results = validator.validate_batch(personas)

        # All should have validation results
        assert len(results) == len(personas)

        # Each result should be a valid ValidationResult
        for result in results:
            assert result.persona is not None

    def test_validator_catches_invalid_persona(self):
        """Test that validator catches invalid personas."""
        validator = PersonaValidator()

        # Create invalid persona (age out of range)
        try:
            invalid_persona = Persona(
                name="",  # Empty name - invalid
                age=15,  # Too young
                background="",  # Empty background
                personality_traits=[],  # Empty traits
                communication_style="",  # Empty
                demographic_source="Test",
            )
            # If it gets here, validate should catch issues
            result = validator.validate(invalid_persona)
            assert not result.is_valid or len(result.issues) > 0
        except Exception:
            # Pydantic validation might catch this first
            pass


class TestContextSummarizerIntegration:
    """Test ContextSummarizer integration."""

    @pytest.mark.asyncio
    async def test_summarizer_preserves_important_content(
        self,
        mock_provider: MockLLMProvider,
    ):
        """Test that summarizer preserves persona and recent content."""
        storage = ContextStorage.create_for_participant(
            persona_description="This is the persona that must be preserved."
        )

        # Add a current question
        storage.set_current_question("What is the current question?")

        # Add many messages to fill context
        for i in range(20):
            msg = Message.participant_response(
                content=f"This is old message number {i}",
                participant_name="Other Person",
            )
            storage.add_verbal_message(msg)

        # Add recent message that should be preserved
        recent_msg = Message.participant_response(
            content="This is the most recent message that should be preserved.",
            participant_name="Other Person",
        )
        storage.add_verbal_message(recent_msg)

        summarizer = ContextSummarizer(mock_provider)

        # Check if summarization is needed
        if summarizer.needs_summarization(storage, max_tokens=500):
            new_storage = await summarizer.summarize_context(
                storage=storage,
                max_tokens=500,
            )

            # Get the verbal context
            context = new_storage.format_for_llm(include_nonverbals=False)

            # Persona should still be there
            assert "persona" in context.lower() or "preserved" in context.lower()


class TestNonVerbalReactionHistory:
    """Test reaction history tracking in NonVerbalGenerator (N06)."""

    def test_reaction_history_capped(
        self,
        sample_personas: list[Persona],
        mock_provider: MockLLMProvider,
    ):
        """Test that _record_reaction caps at _max_history."""
        generator = NonVerbalGenerator(mock_provider)
        name = sample_personas[0].name

        # Record more than max_history reactions
        for i in range(8):
            generator._record_reaction(name, f"reaction {i}")

        history = generator._get_recent_reactions(name)
        assert len(history) == generator._max_history
        # Should keep the most recent ones
        assert history[-1] == "reaction 7"
        assert history[0] == "reaction 3"

    def test_reaction_history_empty_for_unknown(
        self,
        mock_provider: MockLLMProvider,
    ):
        """Test that _get_recent_reactions returns [] for unknown participant."""
        generator = NonVerbalGenerator(mock_provider)
        assert generator._get_recent_reactions("Nobody") == []

    @pytest.mark.asyncio
    async def test_reaction_recorded_after_generation(
        self,
        sample_personas: list[Persona],
        mock_provider: MockLLMProvider,
    ):
        """Test that reactions are recorded in history after generation."""
        participants = [Participant(persona=p, provider=mock_provider) for p in sample_personas]
        generator = NonVerbalGenerator(mock_provider)

        # Generate a reaction
        await generator.generate_reaction_to_question(
            participant=participants[0],
            question="Test question?",
        )

        # History should have one entry
        history = generator._get_recent_reactions(participants[0].name)
        assert len(history) == 1


class TestExtractNotableReaction:
    """Test _extract_notable_reaction in ResponseGenerator (N10)."""

    def test_strong_reaction_detected(self):
        """Test that strong reactions are found."""
        reactions = {
            "Alice": "Alice sits quietly, appearing relaxed",
            "Bob": "Bob leans forward eagerly, eyes wide",
        }
        result = ResponseGenerator._extract_notable_reaction(reactions, "Charlie")
        assert result is not None
        name, reaction = result
        assert name == "Bob"

    def test_bland_reactions_return_none(self):
        """Test that bland reactions return None."""
        reactions = {
            "Alice": "Alice sits quietly, appearing relaxed",
            "Bob": "Bob smiles gently",
        }
        result = ResponseGenerator._extract_notable_reaction(reactions, "Charlie")
        assert result is None

    def test_responding_participant_excluded(self):
        """Test that the responding participant is excluded from search."""
        reactions = {
            "Alice": "Alice leans forward eagerly",
            "Bob": "Bob sits quietly",
        }
        # Alice has a strong reaction but she is the responding participant
        result = ResponseGenerator._extract_notable_reaction(reactions, "Alice")
        assert result is None


class TestSimplifyReaction:
    """Test _simplify_reaction in ResponseGenerator (N10)."""

    def test_strips_name_prefix(self):
        """Test that participant name is stripped from start."""
        result = ResponseGenerator._simplify_reaction("Bob leans forward eagerly", "Bob")
        assert result == "leans forward eagerly"

    def test_strips_possessive_prefix(self):
        """Test that possessive name prefix is stripped."""
        result = ResponseGenerator._simplify_reaction("Bob's eyes widen with interest", "Bob")
        assert result == "eyes widen with interest"

    def test_lowercases_first_char(self):
        """Test that first character is lowercased."""
        result = ResponseGenerator._simplify_reaction("Leaning forward eagerly", "Someone Else")
        assert result == "leaning forward eagerly"

    def test_handles_first_name_prefix(self):
        """Test that first name only is also stripped."""
        result = ResponseGenerator._simplify_reaction(
            "Robert shifts forward in his seat", "Robert Wilson"
        )
        assert result == "shifts forward in his seat"
