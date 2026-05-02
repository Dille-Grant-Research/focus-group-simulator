"""Participant model for focus group simulations.

This module defines the Participant class that combines a persona with
dual-context management, representing a single focus group participant.
"""

from __future__ import annotations

from focus_group.context.manager import ContextManager
from focus_group.llm.base import BaseLLMProvider
from focus_group.models.config import ContextConfig
from focus_group.models.memory import ConsensusLevel, ParticipantMemory
from focus_group.models.message import Message
from focus_group.models.persona import Persona


class Participant:
    """Represents a focus group participant with persona and dual context.

    The Participant class combines a Persona (who the participant is) with
    a ContextManager (what they know/have heard). This provides a unified
    interface for managing participant state during a focus group session.

    The dual-context system tracks:
    - Full context: All messages including non-verbal reactions (for generating responses)
    - Verbal context: Only spoken words (what they "heard")

    Attributes:
        persona: The participant's persona defining their identity
        context: The context manager handling dual-context storage

    Example:
        >>> persona = Persona(
        ...     name="Sarah Chen",
        ...     age=34,
        ...     background="Software engineer...",
        ...     personality_traits=["analytical", "introverted"],
        ...     communication_style="Thoughtful and precise",
        ...     demographic_source="Female, 30-40, tech industry"
        ... )
        >>> participant = Participant(
        ...     persona=persona,
        ...     provider=llm_provider,
        ...     config=context_config
        ... )
        >>> print(participant.name)  # "Sarah Chen"
    """

    def __init__(
        self,
        persona: Persona,
        provider: BaseLLMProvider,
        config: ContextConfig | None = None,
    ) -> None:
        """Initialize a participant with persona and context.

        The context is automatically initialized with the persona description
        as the system message.

        Args:
            persona: The participant's persona defining their identity
            provider: LLM provider for context summarization
            config: Optional context configuration (uses defaults if not provided)
        """
        self._persona = persona
        self._provider = provider
        self._config = config
        self._memory: ParticipantMemory | None = None

        # Initialize context with persona description
        self._context = ContextManager(
            persona_description=persona.to_full_description(),
            provider=provider,
            config=config,
        )

    @property
    def name(self) -> str:
        """Get the participant's name.

        Returns:
            The participant's name from their persona
        """
        return self._persona.name

    @property
    def persona(self) -> Persona:
        """Get the participant's full persona.

        Returns:
            The Persona object containing identity information
        """
        return self._persona

    @property
    def context(self) -> ContextManager:
        """Get the participant's context manager.

        Returns:
            The ContextManager handling dual-context storage
        """
        return self._context

    # Delegation methods for common context operations

    def add_verbal_message(self, message: Message) -> None:
        """Add a verbal message to the participant's context.

        Verbal messages are added to both full and verbal-only contexts.

        Args:
            message: The verbal message to add
        """
        self._context.add_verbal_message(message)

    def add_nonverbal_message(self, message: Message) -> None:
        """Add a non-verbal reaction to the participant's context.

        Non-verbal messages are only added to the full context.

        Args:
            message: The non-verbal message to add
        """
        self._context.add_nonverbal_message(message)

    def add_system_message(self, message: Message) -> None:
        """Add a system message to both contexts.

        Args:
            message: The system message to add
        """
        self._context.add_system_message(message)

    def set_question(self, question: str) -> None:
        """Set the current focus group question.

        The current question is preserved during context summarization.

        Args:
            question: The current question text
        """
        self._context.set_question(question)

    def clear_question(self) -> None:
        """Clear the current question when moving to the next one."""
        self._context.clear_question()

    def get_context_for_llm(self, include_nonverbals: bool = True) -> str:
        """Get the formatted context string for LLM input.

        Args:
            include_nonverbals: If True, include non-verbal reactions;
                if False, only include verbal messages

        Returns:
            Formatted context string ready for use in an LLM prompt
        """
        return self._context.get_context_for_llm(include_nonverbals=include_nonverbals)

    async def check_and_summarize(self) -> bool:
        """Check if summarization is needed and perform it if so.

        Should be called periodically to keep context within limits.

        Returns:
            True if summarization was performed, False otherwise
        """
        return await self._context.check_and_summarize()

    def has_spoken(self) -> bool:
        """Check if this participant has spoken in the current context.

        Returns:
            True if the participant has at least one verbal message
        """
        return self._context.has_spoken(self.name)

    def get_last_verbal_message(self) -> Message | None:
        """Get the most recent verbal message in the context.

        Returns:
            The last verbal message, or None if none exist
        """
        return self._context.get_last_verbal_message()

    @property
    def token_count_full(self) -> int:
        """Get the token count for the full context.

        Returns:
            Estimated tokens in the full context
        """
        return self._context.token_count_full

    @property
    def token_count_verbal(self) -> int:
        """Get the token count for the verbal-only context.

        Returns:
            Estimated tokens in the verbal context
        """
        return self._context.token_count_verbal

    def needs_summarization(self) -> bool:
        """Check if the context needs summarization.

        Returns:
            True if the context exceeds the summarization threshold
        """
        return self._context.needs_summarization()

    def get_context_usage_ratio(self) -> float:
        """Get the ratio of used context to maximum allowed.

        Returns:
            Float between 0.0 and 1.0+ indicating context usage
        """
        return self._context.get_context_usage_ratio()

    # ================================================================
    # Memory support (opt-in)
    # ================================================================

    @property
    def memory(self) -> ParticipantMemory | None:
        """Get the participant's structured memory, if initialized.

        Returns:
            The ParticipantMemory instance, or None if memory is not enabled.
        """
        return self._memory

    def init_memory(self, focus_group_topic: str | None = None) -> None:
        """Initialize structured memory from this participant's persona.

        Idempotent — calling multiple times has no effect after the first.

        Args:
            focus_group_topic: Optional topic of the focus group
        """
        if self._memory is not None:
            return
        self._memory = ParticipantMemory.from_persona(self._persona)
        if focus_group_topic:
            self._memory.episodic.focus_group_topic = focus_group_topic

    def memory_on_new_question(self, question: str, index: int) -> None:
        """Notify memory that a new question is starting. No-op if memory is None.

        Args:
            question: The question text
            index: 0-based question index
        """
        if self._memory is not None:
            self._memory.on_new_question(question, index)

    def memory_on_turn(self, speaker: str, content: str, turn_index: int) -> None:
        """Notify memory of a speaking turn. No-op if memory is None.

        Args:
            speaker: Name of the speaker
            content: What they said
            turn_index: Global turn number
        """
        if self._memory is not None:
            self._memory.on_turn(speaker, content, turn_index)

    def memory_on_question_complete(
        self,
        summary: str,
        key_themes: list[str],
        consensus_level: ConsensusLevel,
        notable_interactions: list[str],
        turn_count: int,
    ) -> None:
        """Notify memory that a question discussion is complete. No-op if memory is None.

        Args:
            summary: Objective discussion summary
            key_themes: Major themes identified
            consensus_level: Level of group agreement
            notable_interactions: Notable exchanges
            turn_count: Number of speaking turns
        """
        if self._memory is not None:
            self._memory.on_question_complete(
                summary=summary,
                key_themes=key_themes,
                consensus_level=consensus_level,
                notable_interactions=notable_interactions,
                turn_count=turn_count,
            )

    def __repr__(self) -> str:
        """Get string representation of the participant.

        Returns:
            String representation showing name and basic stats
        """
        return (
            f"Participant(name={self.name!r}, "
            f"age={self._persona.age}, "
            f"tokens_full={self.token_count_full}, "
            f"has_spoken={self.has_spoken()})"
        )

    def __str__(self) -> str:
        """Get human-readable string of the participant.

        Returns:
            Human-readable string with participant name
        """
        return f"{self.name} ({self._persona.age}yo)"

    def to_dict(self) -> dict:
        """Convert participant state to a dictionary.

        Returns:
            Dictionary representation suitable for serialization
        """
        return {
            "persona": self._persona.to_dict(),
            "context": self._context.to_dict(),
            "has_spoken": self.has_spoken(),
        }
