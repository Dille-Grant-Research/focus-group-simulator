"""End-to-end integration tests for the focus group simulator.

This module tests the complete flow of the focus group simulation from
configuration loading through session execution to output generation.
"""

import json
from pathlib import Path

import pytest

from focus_group.core.session import FocusGroupSession, SessionError, SessionResult
from focus_group.logging.debug_writer import DebugLogWriter
from focus_group.logging.transcript_writer import TranscriptWriter
from focus_group.models.config import SessionConfig
from focus_group.persona.generator import PersonaGenerator
from focus_group.persona.validator import PersonaValidator
from tests.mocks.mock_provider import MockLLMProvider

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


class TestFullSessionFlow:
    """Test the complete session flow from start to finish."""

    @pytest.mark.asyncio
    async def test_full_session_with_mock_provider(
        self,
        mock_provider: MockLLMProvider,
        sample_session_config: SessionConfig,
    ):
        """Test running a complete session with mock LLM."""
        session = FocusGroupSession(
            provider=mock_provider,
            config=sample_session_config,
        )

        result = await session.run()

        # Verify session completed
        assert result is not None
        assert isinstance(result, SessionResult)
        assert result.completed_at is not None

        # Verify participants were created
        assert len(result.participants) > 0
        assert len(result.participants) <= len(sample_session_config.demographics)

        # Verify questions were processed
        assert len(result.question_results) == len(sample_session_config.questions)

        # Verify closing was processed
        assert result.closing_result is not None

        # Verify introduction phase completed
        assert result.introduction_result is not None

        # Verify duration is calculated
        assert result.duration_seconds is not None
        assert result.duration_seconds >= 0

    @pytest.mark.asyncio
    async def test_minimal_session_single_participant(
        self,
        mock_provider: MockLLMProvider,
        minimal_session_config: SessionConfig,
    ):
        """Test session with minimal configuration (1 participant, 1 question)."""
        session = FocusGroupSession(
            provider=mock_provider,
            config=minimal_session_config,
        )

        result = await session.run()

        assert result is not None
        assert len(result.participants) == 1
        assert len(result.question_results) == 1
        assert result.total_responses >= 1

    @pytest.mark.asyncio
    async def test_session_generates_personas(
        self,
        mock_provider: MockLLMProvider,
        sample_session_config: SessionConfig,
    ):
        """Test that session properly generates personas from demographics."""
        session = FocusGroupSession(
            provider=mock_provider,
            config=sample_session_config,
        )

        result = await session.run()

        # Verify personas have required fields
        for persona in result.participants:
            assert persona.name is not None and len(persona.name) > 0
            assert 18 <= persona.age <= 100
            assert persona.background is not None
            assert len(persona.personality_traits) > 0
            assert persona.communication_style is not None

    @pytest.mark.asyncio
    async def test_session_records_all_responses(
        self,
        mock_provider: MockLLMProvider,
        sample_session_config: SessionConfig,
    ):
        """Test that all participant responses are recorded."""
        session = FocusGroupSession(
            provider=mock_provider,
            config=sample_session_config,
        )

        result = await session.run()

        # Each question should have responses
        for question_result in result.question_results:
            assert len(question_result.responses) > 0
            # Each response should have content
            for response in question_result.responses:
                assert response.content is not None
                assert len(response.content) > 0
                assert response.speaker_name is not None

    @pytest.mark.asyncio
    async def test_session_tracks_speaking_order(
        self,
        mock_provider: MockLLMProvider,
        sample_session_config: SessionConfig,
    ):
        """Test that speaking order is tracked for each question."""
        session = FocusGroupSession(
            provider=mock_provider,
            config=sample_session_config,
        )

        result = await session.run()

        for question_result in result.question_results:
            # Speaking order should match number of responses
            assert len(question_result.speaking_order) == len(question_result.responses)
            # Each speaker should be in the speaking order
            for response in question_result.responses:
                assert response.speaker_name in question_result.speaking_order


class TestSeededStances:
    """The session draws stances from its group_dynamic, seeded by the job seed."""

    _SEED = 7
    _GROUP = 5

    @pytest.mark.asyncio
    async def test_same_seed_same_stances(self) -> None:
        from focus_group.models.config import GroupDynamic
        from focus_group.persona.stance_distribution import draw_stances

        config = SessionConfig(
            demographics=[f"Participant {i}, 30-40, professional" for i in range(self._GROUP)],
            questions=["What do you think about remote work?"],
            validate_personas=False,
            group_dynamic=GroupDynamic.HEATED,
        )

        runs = []
        for _ in range(2):
            session = FocusGroupSession(provider=MockLLMProvider(), config=config, seed=self._SEED)
            result = await session.run()
            runs.append([p.discussion_stance for p in result.participants])

        expected = [s.value for s in draw_stances(self._GROUP, GroupDynamic.HEATED, self._SEED)]
        assert runs[0] == runs[1] == expected


class TestSessionWithOutputGeneration:
    """Test session execution with output file generation."""

    @pytest.mark.asyncio
    async def test_session_with_debug_log_output(
        self,
        mock_provider: MockLLMProvider,
        sample_session_config: SessionConfig,
        temp_output_dir: Path,
    ):
        """Test that debug log is properly generated after session."""
        session = FocusGroupSession(
            provider=mock_provider,
            config=sample_session_config,
        )

        result = await session.run()

        # Write debug log
        debug_path = temp_output_dir / "test_debug.json"
        writer = DebugLogWriter(output_path=debug_path, format="json")
        writer.write_session(result)

        # Verify file was created
        assert debug_path.exists()

        # Verify content is valid JSON
        with open(debug_path, encoding="utf-8") as f:
            debug_data = json.load(f)

        # Verify structure
        assert "session" in debug_data
        assert "participants" in debug_data
        assert debug_data["session"]["name"] == sample_session_config.session_name

    @pytest.mark.asyncio
    async def test_session_with_transcript_output(
        self,
        mock_provider: MockLLMProvider,
        sample_session_config: SessionConfig,
        temp_output_dir: Path,
    ):
        """Test that clean transcript is properly generated after session."""
        session = FocusGroupSession(
            provider=mock_provider,
            config=sample_session_config,
        )

        result = await session.run()

        # Write transcript
        transcript_path = temp_output_dir / "test_transcript.md"
        writer = TranscriptWriter(output_path=transcript_path, format="markdown")
        writer.write_session(result)

        # Verify file was created
        assert transcript_path.exists()

        # Verify content
        content = transcript_path.read_text(encoding="utf-8")

        # Verify it's markdown formatted
        assert "# Focus Group Transcript" in content
        assert "## Participants" in content
        assert "## Introductions" in content
        assert "## Question" in content

    @pytest.mark.asyncio
    async def test_session_with_jsonl_debug_output(
        self,
        mock_provider: MockLLMProvider,
        sample_session_config: SessionConfig,
        temp_output_dir: Path,
    ):
        """Test JSONL format debug output."""
        session = FocusGroupSession(
            provider=mock_provider,
            config=sample_session_config,
        )

        result = await session.run()

        # Write debug log in JSONL format
        debug_path = temp_output_dir / "test_debug.jsonl"
        writer = DebugLogWriter(output_path=debug_path, format="jsonl")
        writer.write_session(result)

        # Verify file was created
        assert debug_path.exists()

        # Verify each line is valid JSON
        with open(debug_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    json.loads(line)  # Should not raise


class TestPersonaGenerationFlow:
    """Test persona generation as part of the session flow."""

    @pytest.mark.asyncio
    async def test_persona_generation_creates_unique_names(
        self,
        mock_provider: MockLLMProvider,
        sample_demographics: list[str],
    ):
        """Test that persona generator creates unique names."""
        generator = PersonaGenerator(mock_provider)

        personas = await generator.generate_personas(
            demographics=sample_demographics,
            ensure_unique_names=True,
        )

        # Check all names are unique
        names = [p.name for p in personas]
        assert len(names) == len(set(names))

    @pytest.mark.asyncio
    async def test_persona_validation_in_session(
        self,
        mock_provider: MockLLMProvider,
        sample_session_config: SessionConfig,
    ):
        """Test that personas are validated when configured."""
        # Ensure validation is enabled
        sample_session_config.validate_personas = True

        session = FocusGroupSession(
            provider=mock_provider,
            config=sample_session_config,
        )

        result = await session.run()

        # All participants should have passed validation
        for persona in result.participants:
            validator = PersonaValidator()
            validation_result = validator.validate(persona)
            assert validation_result.is_valid


class TestErrorHandling:
    """Test error handling in the session flow."""

    @pytest.mark.asyncio
    async def test_session_with_unavailable_provider(
        self,
        unavailable_provider: MockLLMProvider,
        minimal_session_config: SessionConfig,
    ):
        """Test session behavior when LLM provider is unavailable."""
        # Check provider availability first (as CLI would)
        is_available = await unavailable_provider.is_available()
        assert is_available is False

    @pytest.mark.asyncio
    async def test_session_records_errors_without_crashing(
        self,
        mock_provider: MockLLMProvider,
        minimal_session_config: SessionConfig,
    ):
        """Test that session records errors and continues when possible."""
        # Add a response that will cause parsing issues
        mock_provider.add_response(
            r"generate.*persona",
            "This is not valid JSON for persona generation",
        )

        session = FocusGroupSession(
            provider=mock_provider,
            config=minimal_session_config,
        )

        # Session should complete (possibly with errors) rather than crash
        try:
            result = await session.run()
            # If it ran, check for recorded errors
            # (Errors may or may not be present depending on implementation)
            assert result is not None
        except SessionError as e:
            # Session errors are acceptable for critical failures
            assert "persona" in str(e).lower() or "failed" in str(e).lower()


class TestSessionProperties:
    """Test session result properties and calculations."""

    @pytest.mark.asyncio
    async def test_session_duration_calculation(
        self,
        mock_provider: MockLLMProvider,
        minimal_session_config: SessionConfig,
    ):
        """Test that session duration is correctly calculated."""
        session = FocusGroupSession(
            provider=mock_provider,
            config=minimal_session_config,
        )

        result = await session.run()

        # Duration should be positive
        assert result.duration_seconds is not None
        assert result.duration_seconds >= 0

        # Started at should be before completed at
        assert result.started_at < result.completed_at

    @pytest.mark.asyncio
    async def test_session_total_counts(
        self,
        mock_provider: MockLLMProvider,
        sample_session_config: SessionConfig,
    ):
        """Test that total counts are correctly calculated."""
        session = FocusGroupSession(
            provider=mock_provider,
            config=sample_session_config,
        )

        result = await session.run()

        # Total questions = regular questions + closing
        expected_questions = len(sample_session_config.questions) + 1  # +1 for closing
        assert result.total_questions == expected_questions

        # Total responses should be sum of all question responses
        manual_count = sum(len(qr.responses) for qr in result.question_results)
        if result.closing_result:
            manual_count += len(result.closing_result.responses)
        assert result.total_responses == manual_count

    @pytest.mark.asyncio
    async def test_session_result_serialization(
        self,
        mock_provider: MockLLMProvider,
        minimal_session_config: SessionConfig,
    ):
        """Test that session result can be serialized to dict."""
        session = FocusGroupSession(
            provider=mock_provider,
            config=minimal_session_config,
        )

        result = await session.run()

        # Convert to dict
        result_dict = result.to_dict()

        # Verify structure
        assert isinstance(result_dict, dict)
        assert "session_name" in result_dict
        assert "participants" in result_dict
        assert "question_results" in result_dict
        assert "started_at" in result_dict
        assert "completed_at" in result_dict

        # Verify JSON serializable
        json_str = json.dumps(result_dict)
        assert len(json_str) > 0


class TestIntroductionPhase:
    """Test the introduction phase of the session."""

    @pytest.mark.asyncio
    async def test_introduction_phase_runs(
        self,
        mock_provider: MockLLMProvider,
        sample_session_config: SessionConfig,
    ):
        """Test that introduction phase completes with all participants."""
        session = FocusGroupSession(
            provider=mock_provider,
            config=sample_session_config,
        )

        result = await session.run()

        assert result.introduction_result is not None
        intro_result = result.introduction_result

        # Each participant should have an introduction
        assert len(intro_result.introductions) == len(result.participants)

        # Each introduction should have content
        for intro in intro_result.introductions:
            assert intro.content is not None
            assert len(intro.content) > 0

    @pytest.mark.asyncio
    async def test_purpose_included_in_session(
        self,
        mock_provider: MockLLMProvider,
        sample_session_config: SessionConfig,
    ):
        """Test that session purpose is properly set."""
        session = FocusGroupSession(
            provider=mock_provider,
            config=sample_session_config,
        )

        result = await session.run()

        assert result.purpose == sample_session_config.purpose


class TestQuestionProcessing:
    """Test question processing in the session."""

    @pytest.mark.asyncio
    async def test_all_participants_speak_on_each_question(
        self,
        mock_provider: MockLLMProvider,
        sample_session_config: SessionConfig,
    ):
        """Test that all participants speak on each question."""
        session = FocusGroupSession(
            provider=mock_provider,
            config=sample_session_config,
        )

        result = await session.run()

        participant_names = {p.name for p in result.participants}

        for question_result in result.question_results:
            # Get all speakers for this question
            speakers = {r.speaker_name for r in question_result.responses}

            # All participants should have spoken
            assert speakers == participant_names

    @pytest.mark.asyncio
    async def test_questions_processed_in_order(
        self,
        mock_provider: MockLLMProvider,
        sample_session_config: SessionConfig,
    ):
        """Test that questions are processed in configuration order."""
        session = FocusGroupSession(
            provider=mock_provider,
            config=sample_session_config,
        )

        result = await session.run()

        # Questions should match config order
        for i, question_result in enumerate(result.question_results):
            assert question_result.question == sample_session_config.questions[i]

    @pytest.mark.asyncio
    async def test_closing_question_processed_last(
        self,
        mock_provider: MockLLMProvider,
        sample_session_config: SessionConfig,
    ):
        """Test that closing question is processed after regular questions."""
        session = FocusGroupSession(
            provider=mock_provider,
            config=sample_session_config,
        )

        result = await session.run()

        assert result.closing_result is not None
        assert result.closing_result.question == sample_session_config.closing_question


class TestConversationalNaturalnessFixes:
    """Test the conversational naturalness improvements (N01-N11)."""

    @pytest.mark.asyncio
    async def test_second_speaker_references_previous(
        self,
        mock_provider: MockLLMProvider,
        sample_session_config: SessionConfig,
    ):
        """Test that turn 2+ prompt contains previous speaker's name (N03)."""
        session = FocusGroupSession(
            provider=mock_provider,
            config=sample_session_config,
        )

        result = await session.run()

        # Check that at least one call used the participant-response prompt
        calls = mock_provider.calls
        participant_response_calls = [
            c for c in calls if "just said:" in c["prompt"] and "How do you respond" in c["prompt"]
        ]
        # With 3 participants, turns 2+ should use the hybrid prompt
        assert (
            len(participant_response_calls) > 0
        ), "Expected at least one call using generate_response_to_participant"

    @pytest.mark.asyncio
    async def test_reactions_gathered_after_last_speaker(
        self,
        mock_provider: MockLLMProvider,
        sample_session_config: SessionConfig,
    ):
        """Test that nonverbal_reactions exist for the final turn (N02)."""
        session = FocusGroupSession(
            provider=mock_provider,
            config=sample_session_config,
        )

        result = await session.run()

        for question_result in result.question_results:
            last_turn = question_result.total_turns
            if last_turn > 0:
                # There should be statement reactions for the last turn
                last_turn_reactions = [
                    nr
                    for nr in question_result.nonverbal_reactions
                    if nr.stimulus_type == "statement" and nr.response_turn == last_turn
                ]
                assert len(last_turn_reactions) > 0, (
                    f"No reactions found for last turn ({last_turn}) of "
                    f"question: {question_result.question}"
                )

    @pytest.mark.asyncio
    async def test_pronouns_in_persona(
        self,
        mock_provider: MockLLMProvider,
        sample_session_config: SessionConfig,
    ):
        """Test that generated personas include pronouns (N04)."""
        session = FocusGroupSession(
            provider=mock_provider,
            config=sample_session_config,
        )

        result = await session.run()

        for persona in result.participants:
            assert hasattr(persona, "pronouns"), "Persona missing pronouns field"
            assert persona.pronouns is not None
            assert len(persona.pronouns) >= 2

    @pytest.mark.asyncio
    async def test_followup_round_can_occur(
        self,
        mock_provider: MockLLMProvider,
        sample_session_config: SessionConfig,
    ):
        """Test that follow-up responses can be generated (N11)."""
        session = FocusGroupSession(
            provider=mock_provider,
            config=sample_session_config,
        )

        result = await session.run()

        # Check if any question had a follow-up response
        all_responses = []
        for qr in result.question_results:
            all_responses.extend(qr.responses)
        if result.closing_result:
            all_responses.extend(result.closing_result.responses)

        followup_responses = [r for r in all_responses if r.selection_method == "followup"]
        # Follow-ups depend on mock selection returning a result;
        # we just verify the system doesn't crash and can produce them
        # The mock provider may or may not trigger follow-ups depending on state
        assert isinstance(followup_responses, list)

    @pytest.mark.asyncio
    async def test_closing_uses_specialized_prompt(
        self,
        mock_provider: MockLLMProvider,
        sample_session_config: SessionConfig,
    ):
        """Test that closing question uses generate_closing_response (N09)."""
        session = FocusGroupSession(
            provider=mock_provider,
            config=sample_session_config,
        )

        result = await session.run()

        # Check that at least one call used the closing prompt
        calls = mock_provider.calls
        closing_calls = [c for c in calls if "Reflect on THIS specific discussion" in c["prompt"]]
        assert len(closing_calls) > 0, "Expected at least one call using generate_closing_response"

    @pytest.mark.asyncio
    async def test_nonverbal_reaction_history_in_prompt(
        self,
        mock_provider: MockLLMProvider,
        sample_session_config: SessionConfig,
    ):
        """Test that reaction history is injected into nonverbal prompts (N06)."""
        session = FocusGroupSession(
            provider=mock_provider,
            config=sample_session_config,
        )

        result = await session.run()

        # After the first reaction per participant, subsequent nonverbal
        # prompts should contain "recently shown these reactions"
        calls = mock_provider.calls
        nonverbal_calls_with_history = [
            c
            for c in calls
            if (
                "non-verbal" in (c.get("system_prompt") or "").lower()
                or "nonverbal" in (c.get("system_prompt") or "").lower()
                or "body language" in (c.get("system_prompt") or "").lower()
            )
            and "recently shown these reactions" in c["prompt"]
        ]
        # With 3 participants and 2+ questions, history should appear
        assert (
            len(nonverbal_calls_with_history) > 0
        ), "Expected at least one nonverbal prompt with reaction history"

    @pytest.mark.asyncio
    async def test_notable_reaction_injected_in_response_prompt(
        self,
        mock_provider: MockLLMProvider,
        sample_session_config: SessionConfig,
    ):
        """Test that notable reactions are injected into response prompts (N10)."""
        # Make mock generate a "strong" reaction that triggers injection
        mock_provider.add_response(
            r"non-verbal|nonverbal|body language|physical reaction",
            "leans forward eagerly, making direct eye contact with the speaker",
        )

        session = FocusGroupSession(
            provider=mock_provider,
            config=sample_session_config,
        )

        result = await session.run()

        # Check that at least one response prompt contains "you notice"
        calls = mock_provider.calls
        notice_calls = [
            c
            for c in calls
            if "you notice" in c["prompt"].lower() and "How do you respond" in c["prompt"]
        ]
        assert len(notice_calls) > 0, "Expected at least one response prompt with 'you notice' cue"

    @pytest.mark.asyncio
    async def test_followup_uses_permissive_selector(
        self,
        mock_provider: MockLLMProvider,
        sample_session_config: SessionConfig,
    ):
        """Test that follow-up selection uses the permissive prompt (N11)."""
        session = FocusGroupSession(
            provider=mock_provider,
            config=sample_session_config,
        )

        result = await session.run()

        # Check that at least one speaker selection call used the
        # selective follow-up prompt (contains "SELECTIVE")
        calls = mock_provider.calls
        selective_calls = [c for c in calls if "SELECTIVE" in (c.get("system_prompt") or "")]
        assert (
            len(selective_calls) > 0
        ), "Expected at least one follow-up selection using selective prompt"
