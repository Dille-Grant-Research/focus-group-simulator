"""Three-tier memory architecture for focus group participants.

This module implements a structured memory system with:
1. EpisodicMemory (Long-term): Immutable persona identity, traits, pronouns
2. QuestionMemory (Per-question): Stance tracking, question records, summaries
3. WorkingMemory (Turn-level): Recent dialogue turns and discussion summary

The centralized Research Assistant updates the StanceMatrix objectively,
replacing self-reported participant summaries with data-driven analysis.

Reference: Ported from natoli/exp_v1.0/app/memory.py and adapted to kai conventions.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from focus_group.models.persona import Persona


# =============================================================================
# ENUMS
# =============================================================================


class StancePosition(str, Enum):
    """Possible stance positions on a question."""

    SUPPORTIVE = "supportive"
    SKEPTICAL = "skeptical"
    NEUTRAL = "neutral"
    MIXED = "mixed"
    NOT_CONTRIBUTED = "not_contributed"


class ConsensusLevel(str, Enum):
    """Level of group consensus on a question."""

    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    NONE = "none"


# =============================================================================
# TIER 1: EPISODIC MEMORY (Long-term, Immutable)
# =============================================================================


class EpisodicMemory(BaseModel):
    """Long-term, immutable memory for a participant.

    Contains static identity information that never changes during a session.
    This forms the foundation of the participant's character.

    Attributes:
        participant_name: The participant's full name
        persona_description: Full narrative description of the persona
        traits: Key personality characteristics
        backstory: Background/occupation description
        pronouns: Preferred pronouns
        discussion_stance: How they engage in discussion
        communication_register: Formality level of speech
        focus_group_topic: Optional topic of the focus group
    """

    participant_name: str = Field(description="The participant's full name")
    persona_description: str = Field(description="Full narrative persona description")
    traits: list[str] = Field(default_factory=list, description="Personality characteristics")
    backstory: str = Field(default="", description="Background/occupation description")
    pronouns: str = Field(default="they/them", description="Preferred pronouns")
    discussion_stance: str = Field(
        default="balanced",
        description="Discussion stance: agreeable, balanced, challenging, contrarian",
    )
    communication_register: str = Field(
        default="conversational",
        description="Register: formal, casual, academic, folksy, curt, warm, conversational",
    )
    focus_group_topic: str | None = Field(default=None, description="Optional focus group topic")

    def to_prompt_context(self) -> str:
        """Render as identity context for prompt injection.

        Returns:
            A structured string preserving the participant's core identity.
        """
        parts = [
            f"Name: {self.participant_name}",
            f"Persona: {self.persona_description}",
        ]

        if self.traits:
            parts.append(f"Traits: {', '.join(self.traits)}")

        if self.pronouns:
            parts.append(f"Pronouns: {self.pronouns}")

        if self.discussion_stance:
            parts.append(f"Discussion stance: {self.discussion_stance}")

        if self.communication_register:
            parts.append(f"Register: {self.communication_register}")

        return "\n".join(parts)


# =============================================================================
# STANCE MATRIX
# =============================================================================


class ParticipantStance(BaseModel):
    """A single participant's stance on a specific question.

    Updated by the Research Assistant, not self-reported.

    Attributes:
        position: The stance position (supportive, skeptical, etc.)
        key_points: Key arguments or points made
        sentiment_score: 0.0 = negative, 1.0 = positive
        contribution_count: Number of times they spoke on this question
    """

    position: StancePosition = Field(default=StancePosition.NOT_CONTRIBUTED)
    key_points: list[str] = Field(default_factory=list)
    sentiment_score: float = Field(default=0.5, ge=0.0, le=1.0)
    contribution_count: int = Field(default=0, ge=0)

    def to_brief(self) -> str:
        """Brief summary for prompt injection.

        Returns:
            A concise string describing this stance.
        """
        if self.position == StancePosition.NOT_CONTRIBUTED:
            return "did not contribute"

        points_str = "; ".join(self.key_points[:2]) if self.key_points else "no specific points"
        return f"{self.position.value} ({points_str})"


class StanceMatrix(BaseModel):
    """Matrix mapping question indices to participant stances.

    Structure: {question_index: {participant_name: ParticipantStance}}

    This is the central data structure for tracking group dynamics,
    updated objectively by the Research Assistant.

    Attributes:
        stances: Nested dict mapping question index -> participant name -> stance
    """

    stances: dict[int, dict[str, ParticipantStance]] = Field(default_factory=dict)

    def get_stance(self, question_index: int, participant_name: str) -> ParticipantStance:
        """Get a participant's stance on a question, creating default if needed.

        Args:
            question_index: The question number
            participant_name: The participant's name

        Returns:
            The participant's stance (default NOT_CONTRIBUTED if not yet tracked)
        """
        if question_index not in self.stances:
            self.stances[question_index] = {}

        if participant_name not in self.stances[question_index]:
            self.stances[question_index][participant_name] = ParticipantStance()

        return self.stances[question_index][participant_name]

    def update_stance(
        self,
        question_index: int,
        participant_name: str,
        position: StancePosition,
        key_points: list[str],
        sentiment_score: float,
    ) -> None:
        """Update a participant's stance (called by Research Assistant).

        Args:
            question_index: The question number
            participant_name: The participant's name
            position: The analyzed stance position
            key_points: Key arguments extracted from the statement
            sentiment_score: Sentiment score 0.0-1.0
        """
        if question_index not in self.stances:
            self.stances[question_index] = {}

        current = self.stances[question_index].get(participant_name, ParticipantStance())

        self.stances[question_index][participant_name] = ParticipantStance(
            position=position,
            key_points=key_points,
            sentiment_score=sentiment_score,
            contribution_count=current.contribution_count + 1,
        )

    def get_participant_history(self, participant_name: str) -> dict[int, ParticipantStance]:
        """Get all stances for a participant across questions.

        Args:
            participant_name: The participant's name

        Returns:
            Dict mapping question indices to stances for this participant
        """
        return {
            q_idx: stances[participant_name]
            for q_idx, stances in self.stances.items()
            if participant_name in stances
        }

    def get_question_summary(self, question_index: int) -> dict[str, str]:
        """Get brief summary of all stances for a question.

        Args:
            question_index: The question number

        Returns:
            Dict mapping participant names to brief stance descriptions
        """
        if question_index not in self.stances:
            return {}

        return {name: stance.to_brief() for name, stance in self.stances[question_index].items()}


# =============================================================================
# TIER 2: QUESTION MEMORY
# =============================================================================


class QuestionRecord(BaseModel):
    """Record for a single completed question discussion.

    Stores the question text, objective summary (from Research Assistant),
    and metadata about the discussion.

    Attributes:
        question_index: 0-based index of the question
        question_text: The question that was discussed
        summary: Objective summary from Research Assistant
        key_themes: Major themes identified in the discussion
        consensus_level: Level of group agreement
        notable_interactions: Brief descriptions of notable exchanges
        turn_count: Number of speaking turns during this question
    """

    question_index: int = Field(ge=0)
    question_text: str = Field(min_length=1)
    summary: str = Field(default="")
    key_themes: list[str] = Field(default_factory=list)
    consensus_level: ConsensusLevel = Field(default=ConsensusLevel.NONE)
    notable_interactions: list[str] = Field(default_factory=list)
    turn_count: int = Field(default=0, ge=0)

    def to_prompt_context(self) -> str:
        """Render as context for prompt injection.

        Returns:
            A concise string with question and summary.
        """
        return f"Q{self.question_index + 1}: {self.question_text}\nSummary: {self.summary}"


class QuestionMemory(BaseModel):
    """Memory tier for question-level information.

    Contains the current question being discussed and records of completed questions.

    Attributes:
        current_question: Text of the question currently being discussed
        current_question_index: 0-based index of the current question
        current_question_summary: Running summary updated by Research Assistant
        completed_questions: Archived records of finished question discussions
    """

    current_question: str | None = Field(default=None)
    current_question_index: int = Field(default=-1)
    current_question_summary: str = Field(default="")
    completed_questions: list[QuestionRecord] = Field(default_factory=list)

    def set_current_question(self, question: str, index: int) -> None:
        """Set a new current question.

        Args:
            question: The question text
            index: 0-based question index
        """
        self.current_question = question
        self.current_question_index = index
        self.current_question_summary = ""

    def archive_current_question(
        self,
        summary: str,
        key_themes: list[str],
        consensus_level: ConsensusLevel,
        notable_interactions: list[str],
        turn_count: int,
    ) -> None:
        """Archive the current question before advancing.

        Creates a QuestionRecord and appends it to completed_questions.
        No-op if there is no current question set.

        Args:
            summary: Objective discussion summary
            key_themes: Major themes identified
            consensus_level: Level of group agreement
            notable_interactions: Notable exchanges
            turn_count: Number of speaking turns
        """
        if self.current_question is None:
            return

        record = QuestionRecord(
            question_index=self.current_question_index,
            question_text=self.current_question,
            summary=summary,
            key_themes=key_themes,
            consensus_level=consensus_level,
            notable_interactions=notable_interactions,
            turn_count=turn_count,
        )
        self.completed_questions.append(record)

    def get_question_history_context(
        self,
        participant_name: str,
        stance_matrix: StanceMatrix,
        max_questions: int = 3,
    ) -> str:
        """Get question history context for a participant.

        Combines question summaries with the participant's stances
        from the shared StanceMatrix.

        Args:
            participant_name: The participant's name
            stance_matrix: The shared StanceMatrix
            max_questions: Maximum number of past questions to include

        Returns:
            Formatted context string, or "No previous questions discussed."
        """
        if not self.completed_questions:
            return "No previous questions discussed."

        recent = self.completed_questions[-max_questions:]

        parts = []
        for record in recent:
            stance = stance_matrix.get_stance(record.question_index, participant_name)
            stance_str = stance.to_brief()
            parts.append(
                f"Q{record.question_index + 1}: {record.question_text}\n"
                f"  Your position: {stance_str}\n"
                f"  Discussion: {record.summary}"
            )

        return "\n\n".join(parts)


# =============================================================================
# TIER 3: WORKING MEMORY
# =============================================================================


class DialogueTurn(BaseModel):
    """A single turn in the dialogue.

    Attributes:
        speaker: Name of the speaker
        content: What they said
        turn_index: Global turn number
        is_own_turn: Whether this participant was the speaker
    """

    speaker: str = Field(min_length=1)
    content: str = Field(min_length=1)
    turn_index: int = Field(ge=0)
    is_own_turn: bool = Field(default=False)


class WorkingMemory(BaseModel):
    """Short-term working memory for recent dialogue.

    Contains recent dialogue turns (rolling window) and a discussion summary
    updated by the Research Assistant.

    Attributes:
        max_turns: Maximum number of recent turns to retain
        recent_turns: Rolling window of recent dialogue turns
        discussion_summary: Running summary updated by Research Assistant
        participant_name: Name of the participant this memory belongs to
    """

    max_turns: int = Field(default=5, ge=1, le=50)
    recent_turns: list[DialogueTurn] = Field(default_factory=list)
    discussion_summary: str = Field(default="")
    participant_name: str = Field(default="")

    def add_turn(self, speaker: str, content: str, turn_index: int) -> None:
        """Add a new turn, maintaining the max_turns rolling window.

        Args:
            speaker: Name of the speaker
            content: What they said
            turn_index: Global turn number
        """
        is_own = speaker == self.participant_name

        turn = DialogueTurn(
            speaker=speaker,
            content=content,
            turn_index=turn_index,
            is_own_turn=is_own,
        )

        self.recent_turns.append(turn)

        if len(self.recent_turns) > self.max_turns:
            self.recent_turns = self.recent_turns[-self.max_turns :]

    def clear_for_new_question(self) -> None:
        """Clear working memory when advancing to a new question."""
        self.recent_turns = []
        self.discussion_summary = ""

    def update_summary(self, summary: str) -> None:
        """Update the discussion summary (called by Research Assistant).

        Args:
            summary: The updated discussion summary
        """
        self.discussion_summary = summary

    def to_prompt_context(self) -> str:
        """Render working memory for prompt injection.

        Format: Summary first (if exists), then recent raw turns.

        Returns:
            Formatted context string.
        """
        parts = []

        if self.discussion_summary:
            parts.append(f"Discussion so far: {self.discussion_summary}")
            parts.append("")

        if self.recent_turns:
            parts.append("Recent exchanges:")
            for turn in self.recent_turns:
                prefix = "You said" if turn.is_own_turn else f"{turn.speaker} said"
                parts.append(f"  {prefix}: {turn.content}")
        else:
            parts.append("No recent exchanges yet.")

        return "\n".join(parts)

    def get_others_context(self) -> str:
        """Get context excluding own turns (for response generation).

        Returns:
            Formatted context string with only other participants' turns.
        """
        parts = []

        if self.discussion_summary:
            parts.append(f"Discussion so far: {self.discussion_summary}")
            parts.append("")

        others_turns = [t for t in self.recent_turns if not t.is_own_turn]
        if others_turns:
            parts.append("What others have said:")
            for turn in others_turns:
                parts.append(f"  {turn.speaker}: {turn.content}")

        return "\n".join(parts)


# =============================================================================
# UNIFIED PARTICIPANT MEMORY
# =============================================================================


class ParticipantMemory(BaseModel):
    """Unified memory interface combining all three tiers.

    This is the main interface for accessing participant memory,
    providing clean methods for building prompts and updating state.

    Attributes:
        episodic: Long-term identity memory (immutable after creation)
        question: Question-level memory (stance tracking, question records)
        working: Short-term working memory (recent turns, discussion summary)
    """

    episodic: EpisodicMemory
    question: QuestionMemory = Field(default_factory=QuestionMemory)
    working: WorkingMemory = Field(default_factory=WorkingMemory)

    def model_post_init(self, __context: Any) -> None:
        """Initialize working memory with participant name."""
        self.working.participant_name = self.episodic.participant_name

    @classmethod
    def from_persona(cls, persona: Persona) -> ParticipantMemory:
        """Factory method to create memory from a kai Persona object.

        Args:
            persona: A focus_group.models.persona.Persona instance

        Returns:
            A new ParticipantMemory with episodic memory populated from the persona.
        """
        episodic = EpisodicMemory(
            participant_name=persona.name,
            persona_description=persona.to_full_description(),
            traits=list(persona.personality_traits),
            backstory=persona.background,
            pronouns=persona.pronouns,
            discussion_stance=persona.discussion_stance,
            communication_register=persona.communication_register,
        )

        working = WorkingMemory(participant_name=persona.name)

        return cls(
            episodic=episodic,
            question=QuestionMemory(),
            working=working,
        )

    def build_speak_context(self, stance_matrix: StanceMatrix) -> dict[str, str]:
        """Build context dictionary for speaking prompt.

        Returns dict with keys matching prompt template variables:
        - episodic_memory: Long-term identity
        - question: Current question
        - question_history: Past question positions
        - working_memory: Recent dialogue (excluding own turns)

        Args:
            stance_matrix: The shared StanceMatrix for stance lookups

        Returns:
            Dict with context strings for prompt injection.
        """
        return {
            "episodic_memory": self.episodic.to_prompt_context(),
            "question": self.question.current_question or "",
            "question_history": self.question.get_question_history_context(
                self.episodic.participant_name, stance_matrix
            ),
            "working_memory": self.working.get_others_context(),
        }

    def build_concluding_context(self, stance_matrix: StanceMatrix) -> dict[str, str]:
        """Build context for concluding thoughts prompt.

        Args:
            stance_matrix: The shared StanceMatrix

        Returns:
            Dict with context strings including fuller question history.
        """
        return {
            "episodic_memory": self.episodic.to_prompt_context(),
            "question_history": self.question.get_question_history_context(
                self.episodic.participant_name,
                stance_matrix,
                max_questions=5,
            ),
            "working_memory": self.working.to_prompt_context(),
        }

    def on_new_question(self, question: str, index: int) -> None:
        """Called when a new question starts.

        Args:
            question: The question text
            index: 0-based question index
        """
        self.question.set_current_question(question, index)
        self.working.clear_for_new_question()

    def on_turn(self, speaker: str, content: str, turn_index: int) -> None:
        """Called when any participant speaks.

        Args:
            speaker: Name of the speaker
            content: What they said
            turn_index: Global turn number
        """
        self.working.add_turn(speaker, content, turn_index)

    def on_question_complete(
        self,
        summary: str,
        key_themes: list[str],
        consensus_level: ConsensusLevel,
        notable_interactions: list[str],
        turn_count: int,
    ) -> None:
        """Called when advancing to next question.

        Args:
            summary: Objective discussion summary
            key_themes: Major themes identified
            consensus_level: Level of group agreement
            notable_interactions: Notable exchanges
            turn_count: Number of speaking turns
        """
        self.question.archive_current_question(
            summary=summary,
            key_themes=key_themes,
            consensus_level=consensus_level,
            notable_interactions=notable_interactions,
            turn_count=turn_count,
        )


# =============================================================================
# GLOBAL STATE (Shared across participants)
# =============================================================================


class FocusGroupMemory(BaseModel):
    """Global memory state for the entire focus group.

    Contains shared state that needs to be consistent across all participants:
    the global StanceMatrix and turn counters.

    Attributes:
        stance_matrix: Shared stance tracking matrix
        topic: The focus group topic
        total_turns: Total speaking turns across all questions
        current_question_turns: Speaking turns in the current question
    """

    stance_matrix: StanceMatrix = Field(default_factory=StanceMatrix)
    topic: str = Field(default="")
    total_turns: int = Field(default=0, ge=0)
    current_question_turns: int = Field(default=0, ge=0)

    def increment_turn(self) -> int:
        """Increment turn counters.

        Returns:
            The new total turn count.
        """
        self.total_turns += 1
        self.current_question_turns += 1
        return self.total_turns

    def reset_question_turns(self) -> None:
        """Reset turn counter for a new question."""
        self.current_question_turns = 0


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "StancePosition",
    "ConsensusLevel",
    # Tier 1: Episodic
    "EpisodicMemory",
    # Stance tracking
    "ParticipantStance",
    "StanceMatrix",
    # Tier 2: Question
    "QuestionRecord",
    "QuestionMemory",
    # Tier 3: Working
    "DialogueTurn",
    "WorkingMemory",
    # Unified
    "ParticipantMemory",
    # Global
    "FocusGroupMemory",
]
