"""Tests for the three-tier memory models.

Comprehensive unit tests covering all memory model classes:
- StancePosition/ConsensusLevel enums
- EpisodicMemory (Tier 1)
- ParticipantStance and StanceMatrix
- QuestionRecord and QuestionMemory (Tier 2)
- DialogueTurn and WorkingMemory (Tier 3)
- ParticipantMemory (unified interface)
- FocusGroupMemory (shared state)
"""

from focus_group.models.memory import (
    ConsensusLevel,
    DialogueTurn,
    EpisodicMemory,
    FocusGroupMemory,
    ParticipantMemory,
    ParticipantStance,
    QuestionMemory,
    QuestionRecord,
    StanceMatrix,
    StancePosition,
    WorkingMemory,
)

# =============================================================================
# ENUMS
# =============================================================================


class TestStancePosition:
    def test_values(self):
        assert StancePosition.SUPPORTIVE == "supportive"
        assert StancePosition.SKEPTICAL == "skeptical"
        assert StancePosition.NEUTRAL == "neutral"
        assert StancePosition.MIXED == "mixed"
        assert StancePosition.NOT_CONTRIBUTED == "not_contributed"

    def test_string_serialization(self):
        assert StancePosition.SUPPORTIVE.value == "supportive"
        assert str(StancePosition.NEUTRAL) == "StancePosition.NEUTRAL"

    def test_from_string(self):
        assert StancePosition("supportive") == StancePosition.SUPPORTIVE


class TestConsensusLevel:
    def test_values(self):
        assert ConsensusLevel.HIGH == "high"
        assert ConsensusLevel.MODERATE == "moderate"
        assert ConsensusLevel.LOW == "low"
        assert ConsensusLevel.NONE == "none"

    def test_from_string(self):
        assert ConsensusLevel("high") == ConsensusLevel.HIGH


# =============================================================================
# TIER 1: EPISODIC MEMORY
# =============================================================================


class TestEpisodicMemory:
    def test_creation_with_defaults(self):
        memory = EpisodicMemory(
            participant_name="Sarah Chen",
            persona_description="A software engineer",
        )
        assert memory.participant_name == "Sarah Chen"
        assert memory.persona_description == "A software engineer"
        assert memory.traits == []
        assert memory.backstory == ""
        assert memory.pronouns == "they/them"
        assert memory.discussion_stance == "balanced"
        assert memory.communication_register == "conversational"
        assert memory.focus_group_topic is None

    def test_creation_with_all_fields(self):
        memory = EpisodicMemory(
            participant_name="Sarah Chen",
            persona_description="A 34-year-old software engineer",
            traits=["analytical", "collaborative"],
            backstory="Senior engineer at a tech company",
            pronouns="she/her",
            discussion_stance="challenging",
            communication_register="formal",
            focus_group_topic="Remote work",
        )
        assert memory.pronouns == "she/her"
        assert memory.discussion_stance == "challenging"
        assert memory.communication_register == "formal"
        assert memory.focus_group_topic == "Remote work"

    def test_to_prompt_context(self):
        memory = EpisodicMemory(
            participant_name="Sarah Chen",
            persona_description="A software engineer",
            traits=["analytical", "collaborative"],
            pronouns="she/her",
            discussion_stance="balanced",
            communication_register="conversational",
        )
        context = memory.to_prompt_context()
        assert "Name: Sarah Chen" in context
        assert "Persona: A software engineer" in context
        assert "Traits: analytical, collaborative" in context
        assert "Pronouns: she/her" in context
        assert "Discussion stance: balanced" in context
        assert "Register: conversational" in context

    def test_to_prompt_context_minimal(self):
        memory = EpisodicMemory(
            participant_name="Test",
            persona_description="A test participant",
        )
        context = memory.to_prompt_context()
        assert "Name: Test" in context
        assert "Persona: A test participant" in context


# =============================================================================
# STANCE TRACKING
# =============================================================================


class TestParticipantStance:
    def test_default_not_contributed(self):
        stance = ParticipantStance()
        assert stance.position == StancePosition.NOT_CONTRIBUTED
        assert stance.key_points == []
        assert stance.sentiment_score == 0.5
        assert stance.contribution_count == 0

    def test_to_brief_not_contributed(self):
        stance = ParticipantStance()
        assert stance.to_brief() == "did not contribute"

    def test_to_brief_with_points(self):
        stance = ParticipantStance(
            position=StancePosition.SUPPORTIVE,
            key_points=["likes flexibility", "values autonomy"],
        )
        brief = stance.to_brief()
        assert "supportive" in brief
        assert "likes flexibility" in brief
        assert "values autonomy" in brief

    def test_to_brief_no_points(self):
        stance = ParticipantStance(position=StancePosition.SKEPTICAL)
        brief = stance.to_brief()
        assert "skeptical" in brief
        assert "no specific points" in brief

    def test_to_brief_truncates_to_two_points(self):
        stance = ParticipantStance(
            position=StancePosition.MIXED,
            key_points=["point1", "point2", "point3"],
        )
        brief = stance.to_brief()
        assert "point1" in brief
        assert "point2" in brief
        assert "point3" not in brief


class TestStanceMatrix:
    def test_empty_matrix(self):
        matrix = StanceMatrix()
        assert matrix.stances == {}

    def test_get_stance_auto_creates(self):
        matrix = StanceMatrix()
        stance = matrix.get_stance(0, "Sarah")
        assert stance.position == StancePosition.NOT_CONTRIBUTED
        assert 0 in matrix.stances
        assert "Sarah" in matrix.stances[0]

    def test_update_stance(self):
        matrix = StanceMatrix()
        matrix.update_stance(
            question_index=0,
            participant_name="Sarah",
            position=StancePosition.SUPPORTIVE,
            key_points=["likes remote work"],
            sentiment_score=0.8,
        )
        stance = matrix.stances[0]["Sarah"]
        assert stance.position == StancePosition.SUPPORTIVE
        assert stance.key_points == ["likes remote work"]
        assert stance.sentiment_score == 0.8
        assert stance.contribution_count == 1

    def test_update_stance_increments_count(self):
        matrix = StanceMatrix()
        matrix.update_stance(0, "Sarah", StancePosition.SUPPORTIVE, ["point1"], 0.7)
        matrix.update_stance(0, "Sarah", StancePosition.SUPPORTIVE, ["point2"], 0.8)
        assert matrix.stances[0]["Sarah"].contribution_count == 2

    def test_get_participant_history(self):
        matrix = StanceMatrix()
        matrix.update_stance(0, "Sarah", StancePosition.SUPPORTIVE, [], 0.8)
        matrix.update_stance(1, "Sarah", StancePosition.SKEPTICAL, [], 0.3)
        matrix.update_stance(0, "Robert", StancePosition.NEUTRAL, [], 0.5)

        history = matrix.get_participant_history("Sarah")
        assert len(history) == 2
        assert 0 in history
        assert 1 in history
        assert history[0].position == StancePosition.SUPPORTIVE
        assert history[1].position == StancePosition.SKEPTICAL

    def test_get_participant_history_empty(self):
        matrix = StanceMatrix()
        history = matrix.get_participant_history("Nobody")
        assert history == {}

    def test_get_question_summary(self):
        matrix = StanceMatrix()
        matrix.update_stance(0, "Sarah", StancePosition.SUPPORTIVE, ["point"], 0.8)
        matrix.update_stance(0, "Robert", StancePosition.SKEPTICAL, ["concern"], 0.3)

        summary = matrix.get_question_summary(0)
        assert "Sarah" in summary
        assert "Robert" in summary
        assert "supportive" in summary["Sarah"]
        assert "skeptical" in summary["Robert"]

    def test_get_question_summary_empty(self):
        matrix = StanceMatrix()
        summary = matrix.get_question_summary(99)
        assert summary == {}


# =============================================================================
# TIER 2: QUESTION MEMORY
# =============================================================================


class TestQuestionRecord:
    def test_creation(self):
        record = QuestionRecord(
            question_index=0,
            question_text="What do you think about remote work?",
            summary="Mixed reactions to remote work.",
            key_themes=["flexibility", "isolation"],
            consensus_level=ConsensusLevel.LOW,
            turn_count=5,
        )
        assert record.question_index == 0
        assert record.consensus_level == ConsensusLevel.LOW
        assert record.turn_count == 5

    def test_to_prompt_context(self):
        record = QuestionRecord(
            question_index=0,
            question_text="What about remote work?",
            summary="People had mixed views.",
        )
        context = record.to_prompt_context()
        assert "Q1:" in context
        assert "What about remote work?" in context
        assert "People had mixed views." in context

    def test_defaults(self):
        record = QuestionRecord(
            question_index=0,
            question_text="Test question",
        )
        assert record.summary == ""
        assert record.key_themes == []
        assert record.consensus_level == ConsensusLevel.NONE
        assert record.notable_interactions == []
        assert record.turn_count == 0


class TestQuestionMemory:
    def test_initial_state(self):
        qm = QuestionMemory()
        assert qm.current_question is None
        assert qm.current_question_index == -1
        assert qm.current_question_summary == ""
        assert qm.completed_questions == []

    def test_set_current_question(self):
        qm = QuestionMemory()
        qm.set_current_question("What about remote work?", 0)
        assert qm.current_question == "What about remote work?"
        assert qm.current_question_index == 0
        assert qm.current_question_summary == ""

    def test_set_current_question_resets_summary(self):
        qm = QuestionMemory()
        qm.set_current_question("Q1?", 0)
        qm.current_question_summary = "Some summary"
        qm.set_current_question("Q2?", 1)
        assert qm.current_question_summary == ""

    def test_archive_current_question(self):
        qm = QuestionMemory()
        qm.set_current_question("What about remote work?", 0)
        qm.archive_current_question(
            summary="Mixed views on remote work.",
            key_themes=["flexibility", "isolation"],
            consensus_level=ConsensusLevel.LOW,
            notable_interactions=["Sarah disagreed with Robert"],
            turn_count=5,
        )
        assert len(qm.completed_questions) == 1
        record = qm.completed_questions[0]
        assert record.question_text == "What about remote work?"
        assert record.question_index == 0
        assert record.summary == "Mixed views on remote work."
        assert record.turn_count == 5

    def test_archive_noop_without_current_question(self):
        qm = QuestionMemory()
        qm.archive_current_question(
            summary="test",
            key_themes=[],
            consensus_level=ConsensusLevel.NONE,
            notable_interactions=[],
            turn_count=0,
        )
        assert len(qm.completed_questions) == 0

    def test_get_question_history_context_empty(self):
        qm = QuestionMemory()
        matrix = StanceMatrix()
        context = qm.get_question_history_context("Sarah", matrix)
        assert context == "No previous questions discussed."

    def test_get_question_history_context_with_stances(self):
        qm = QuestionMemory()
        matrix = StanceMatrix()

        # Simulate completing a question
        qm.set_current_question("What about remote work?", 0)
        matrix.update_stance(0, "Sarah", StancePosition.SUPPORTIVE, ["likes flexibility"], 0.8)
        qm.archive_current_question(
            summary="People discussed remote work.",
            key_themes=["flexibility"],
            consensus_level=ConsensusLevel.MODERATE,
            notable_interactions=[],
            turn_count=3,
        )

        context = qm.get_question_history_context("Sarah", matrix)
        assert "Q1:" in context
        assert "What about remote work?" in context
        assert "supportive" in context
        assert "likes flexibility" in context
        assert "People discussed remote work." in context

    def test_get_question_history_context_respects_max(self):
        qm = QuestionMemory()
        matrix = StanceMatrix()

        for i in range(5):
            qm.set_current_question(f"Question {i}?", i)
            qm.archive_current_question(
                summary=f"Summary {i}",
                key_themes=[],
                consensus_level=ConsensusLevel.NONE,
                notable_interactions=[],
                turn_count=1,
            )

        context = qm.get_question_history_context("Sarah", matrix, max_questions=2)
        assert "Question 3?" in context
        assert "Question 4?" in context
        assert "Question 0?" not in context


# =============================================================================
# TIER 3: WORKING MEMORY
# =============================================================================


class TestDialogueTurn:
    def test_creation(self):
        turn = DialogueTurn(
            speaker="Sarah",
            content="I think remote work is great.",
            turn_index=0,
        )
        assert turn.speaker == "Sarah"
        assert turn.content == "I think remote work is great."
        assert turn.turn_index == 0
        assert turn.is_own_turn is False

    def test_own_turn(self):
        turn = DialogueTurn(
            speaker="Sarah",
            content="My point",
            turn_index=1,
            is_own_turn=True,
        )
        assert turn.is_own_turn is True


class TestWorkingMemory:
    def test_initial_state(self):
        wm = WorkingMemory(participant_name="Sarah")
        assert wm.recent_turns == []
        assert wm.discussion_summary == ""
        assert wm.participant_name == "Sarah"
        assert wm.max_turns == 5

    def test_add_turn(self):
        wm = WorkingMemory(participant_name="Sarah")
        wm.add_turn("Robert", "I disagree.", 1)
        assert len(wm.recent_turns) == 1
        assert wm.recent_turns[0].speaker == "Robert"
        assert wm.recent_turns[0].is_own_turn is False

    def test_add_own_turn(self):
        wm = WorkingMemory(participant_name="Sarah")
        wm.add_turn("Sarah", "I think so too.", 2)
        assert wm.recent_turns[0].is_own_turn is True

    def test_rolling_window(self):
        wm = WorkingMemory(participant_name="Sarah", max_turns=3)
        wm.add_turn("A", "Turn 1", 1)
        wm.add_turn("B", "Turn 2", 2)
        wm.add_turn("C", "Turn 3", 3)
        wm.add_turn("D", "Turn 4", 4)
        assert len(wm.recent_turns) == 3
        assert wm.recent_turns[0].speaker == "B"
        assert wm.recent_turns[2].speaker == "D"

    def test_clear_for_new_question(self):
        wm = WorkingMemory(participant_name="Sarah")
        wm.add_turn("Robert", "Something", 1)
        wm.discussion_summary = "A summary"
        wm.clear_for_new_question()
        assert wm.recent_turns == []
        assert wm.discussion_summary == ""

    def test_update_summary(self):
        wm = WorkingMemory(participant_name="Sarah")
        wm.update_summary("The group discussed remote work.")
        assert wm.discussion_summary == "The group discussed remote work."

    def test_to_prompt_context_empty(self):
        wm = WorkingMemory(participant_name="Sarah")
        context = wm.to_prompt_context()
        assert "No recent exchanges yet." in context

    def test_to_prompt_context_with_turns(self):
        wm = WorkingMemory(participant_name="Sarah")
        wm.add_turn("Robert", "I disagree.", 1)
        wm.add_turn("Sarah", "Well, I think...", 2)
        context = wm.to_prompt_context()
        assert "Robert said: I disagree." in context
        assert "You said: Well, I think..." in context

    def test_to_prompt_context_with_summary(self):
        wm = WorkingMemory(participant_name="Sarah")
        wm.discussion_summary = "Mixed views on remote work."
        wm.add_turn("Robert", "Interesting point.", 1)
        context = wm.to_prompt_context()
        assert "Discussion so far: Mixed views on remote work." in context
        assert "Robert said: Interesting point." in context

    def test_get_others_context(self):
        wm = WorkingMemory(participant_name="Sarah")
        wm.add_turn("Robert", "I disagree.", 1)
        wm.add_turn("Sarah", "Well, I think...", 2)
        wm.add_turn("Jordan", "Good point.", 3)
        context = wm.get_others_context()
        assert "Robert:" in context
        assert "Jordan:" in context
        assert "Sarah" not in context  # Own turns excluded

    def test_get_others_context_with_summary(self):
        wm = WorkingMemory(participant_name="Sarah")
        wm.discussion_summary = "A lively discussion."
        wm.add_turn("Robert", "Interesting.", 1)
        context = wm.get_others_context()
        assert "Discussion so far: A lively discussion." in context


# =============================================================================
# UNIFIED PARTICIPANT MEMORY
# =============================================================================


class TestParticipantMemory:
    def test_from_persona(self, sample_persona):
        memory = ParticipantMemory.from_persona(sample_persona)
        assert memory.episodic.participant_name == sample_persona.name
        assert memory.episodic.traits == list(sample_persona.personality_traits)
        assert memory.episodic.backstory == sample_persona.background
        assert memory.episodic.pronouns == sample_persona.pronouns
        assert memory.working.participant_name == sample_persona.name

    def test_from_persona_maps_stance_and_register(self, sample_persona):
        memory = ParticipantMemory.from_persona(sample_persona)
        assert memory.episodic.discussion_stance == sample_persona.discussion_stance
        assert memory.episodic.communication_register == sample_persona.communication_register

    def test_working_memory_gets_participant_name(self):
        episodic = EpisodicMemory(
            participant_name="Sarah Chen",
            persona_description="A software engineer",
        )
        memory = ParticipantMemory(episodic=episodic)
        assert memory.working.participant_name == "Sarah Chen"

    def test_build_speak_context(self):
        episodic = EpisodicMemory(
            participant_name="Sarah",
            persona_description="A software engineer",
            traits=["analytical"],
        )
        memory = ParticipantMemory(episodic=episodic)
        memory.question.set_current_question("What about remote work?", 0)

        matrix = StanceMatrix()
        context = memory.build_speak_context(matrix)

        assert "episodic_memory" in context
        assert "question" in context
        assert "question_history" in context
        assert "working_memory" in context
        assert context["question"] == "What about remote work?"

    def test_build_concluding_context(self):
        episodic = EpisodicMemory(
            participant_name="Sarah",
            persona_description="A software engineer",
        )
        memory = ParticipantMemory(episodic=episodic)
        matrix = StanceMatrix()

        context = memory.build_concluding_context(matrix)
        assert "episodic_memory" in context
        assert "question_history" in context
        assert "working_memory" in context

    def test_on_new_question(self):
        episodic = EpisodicMemory(
            participant_name="Sarah",
            persona_description="Test",
        )
        memory = ParticipantMemory(episodic=episodic)

        # Add some state first
        memory.working.add_turn("Robert", "Hello", 1)
        memory.working.discussion_summary = "Old summary"

        memory.on_new_question("New question?", 1)
        assert memory.question.current_question == "New question?"
        assert memory.question.current_question_index == 1
        assert memory.working.recent_turns == []
        assert memory.working.discussion_summary == ""

    def test_on_turn(self):
        episodic = EpisodicMemory(
            participant_name="Sarah",
            persona_description="Test",
        )
        memory = ParticipantMemory(episodic=episodic)
        memory.on_turn("Robert", "I think so.", 1)
        assert len(memory.working.recent_turns) == 1
        assert memory.working.recent_turns[0].speaker == "Robert"

    def test_on_question_complete(self):
        episodic = EpisodicMemory(
            participant_name="Sarah",
            persona_description="Test",
        )
        memory = ParticipantMemory(episodic=episodic)
        memory.question.set_current_question("Q1?", 0)

        memory.on_question_complete(
            summary="Discussion completed.",
            key_themes=["theme1"],
            consensus_level=ConsensusLevel.MODERATE,
            notable_interactions=["Sarah agreed with Robert"],
            turn_count=4,
        )
        assert len(memory.question.completed_questions) == 1
        record = memory.question.completed_questions[0]
        assert record.question_text == "Q1?"
        assert record.summary == "Discussion completed."
        assert record.consensus_level == ConsensusLevel.MODERATE

    def test_full_lifecycle(self, sample_persona):
        """Test the complete memory lifecycle: create -> question -> turns -> archive."""
        memory = ParticipantMemory.from_persona(sample_persona)
        matrix = StanceMatrix()

        # Start question
        memory.on_new_question("What about remote work?", 0)

        # Simulate turns
        memory.on_turn("Robert", "I like remote work.", 1)
        memory.on_turn(sample_persona.name, "I agree, but with caveats.", 2)
        memory.on_turn("Jordan", "Interesting perspective.", 3)

        # Update stance
        matrix.update_stance(
            0, sample_persona.name, StancePosition.MIXED, ["agrees with caveats"], 0.6
        )

        # Build context (should work mid-question)
        context = memory.build_speak_context(matrix)
        assert context["question"] == "What about remote work?"

        # Archive
        memory.on_question_complete(
            summary="Mixed views on remote work.",
            key_themes=["flexibility", "isolation"],
            consensus_level=ConsensusLevel.MODERATE,
            notable_interactions=[],
            turn_count=3,
        )

        assert len(memory.question.completed_questions) == 1

        # Start new question
        memory.on_new_question("How do you stay productive?", 1)
        assert memory.working.recent_turns == []

        # History should now include Q1
        history_context = memory.question.get_question_history_context(sample_persona.name, matrix)
        assert "What about remote work?" in history_context
        assert "mixed" in history_context.lower()


# =============================================================================
# GLOBAL STATE
# =============================================================================


class TestFocusGroupMemory:
    def test_initial_state(self):
        fgm = FocusGroupMemory()
        assert fgm.total_turns == 0
        assert fgm.current_question_turns == 0
        assert fgm.topic == ""
        assert isinstance(fgm.stance_matrix, StanceMatrix)

    def test_increment_turn(self):
        fgm = FocusGroupMemory()
        result = fgm.increment_turn()
        assert result == 1
        assert fgm.total_turns == 1
        assert fgm.current_question_turns == 1

    def test_increment_turn_multiple(self):
        fgm = FocusGroupMemory()
        fgm.increment_turn()
        fgm.increment_turn()
        result = fgm.increment_turn()
        assert result == 3
        assert fgm.total_turns == 3
        assert fgm.current_question_turns == 3

    def test_reset_question_turns(self):
        fgm = FocusGroupMemory()
        fgm.increment_turn()
        fgm.increment_turn()
        fgm.reset_question_turns()
        assert fgm.current_question_turns == 0
        assert fgm.total_turns == 2  # Total not reset

    def test_with_topic(self):
        fgm = FocusGroupMemory(topic="Remote work discussion")
        assert fgm.topic == "Remote work discussion"

    def test_stance_matrix_shared(self):
        fgm = FocusGroupMemory()
        fgm.stance_matrix.update_stance(0, "Sarah", StancePosition.SUPPORTIVE, ["point"], 0.8)
        assert fgm.stance_matrix.stances[0]["Sarah"].position == StancePosition.SUPPORTIVE
