"""Dual-context storage for focus group participants.

This module implements the dual-context system where each participant maintains:
- context_with_nonverbals: Full context including all non-verbal reactions
- context_verbal_only: Only what was spoken (what they "heard")

The verbal context is a subset of the full context. When a verbal message is added,
it goes into both contexts. Non-verbal messages only go into the full context.
"""

from typing import Any

from pydantic import BaseModel, Field, PrivateAttr

from focus_group.models.message import Message, MessageRole
from focus_group.utils.tokens import count_tokens


class ContextStorage(BaseModel):
    """Stores dual context for a focus group participant.

    Each participant has two views of the conversation:
    - Full context (with non-verbals): Used internally for generating responses
    - Verbal-only context: Represents what they actually "heard"

    The storage tracks token counts for context window management and provides
    methods for adding messages to the appropriate contexts.

    Attributes:
        persona_description: The full persona description (never summarized)
        messages_full: All messages including non-verbal reactions
        messages_verbal: Only verbal messages (subset of full)
        current_question: The current focus group question (preserved during summarization)
        token_count_full: Estimated tokens in full context
        token_count_verbal: Estimated tokens in verbal context
    """

    persona_description: str = Field(
        description="Full persona description (preserved during summarization)",
    )
    messages_full: list[Message] = Field(
        default_factory=list,
        description="All messages including non-verbal reactions",
    )
    messages_verbal: list[Message] = Field(
        default_factory=list,
        description="Only verbal messages (what they heard)",
    )
    current_question: str | None = Field(
        default=None,
        description="Current focus group question (preserved during summarization)",
    )

    # Private attributes for caching token counts
    _token_count_full: int = PrivateAttr(default=0)
    _token_count_verbal: int = PrivateAttr(default=0)
    _persona_tokens: int = PrivateAttr(default=0)

    def model_post_init(self, __context: Any) -> None:
        """Initialize token counts after model creation."""
        self._persona_tokens = count_tokens(self.persona_description)
        self._recalculate_token_counts()

    def _recalculate_token_counts(self) -> None:
        """Recalculate token counts for both contexts."""
        self._token_count_full = self._persona_tokens
        self._token_count_verbal = self._persona_tokens

        for msg in self.messages_full:
            self._token_count_full += count_tokens(msg.content)
            # Add overhead for message formatting
            self._token_count_full += 4

        for msg in self.messages_verbal:
            self._token_count_verbal += count_tokens(msg.content)
            self._token_count_verbal += 4

        if self.current_question:
            question_tokens = count_tokens(self.current_question) + 4
            self._token_count_full += question_tokens
            self._token_count_verbal += question_tokens

    @property
    def token_count_full(self) -> int:
        """Get the estimated token count for the full context.

        Returns:
            Estimated tokens in the full context including persona and all messages.
        """
        return self._token_count_full

    @property
    def token_count_verbal(self) -> int:
        """Get the estimated token count for the verbal-only context.

        Returns:
            Estimated tokens in the verbal context including persona and verbal messages.
        """
        return self._token_count_verbal

    def add_verbal_message(self, message: Message) -> None:
        """Add a verbal message to both contexts.

        Verbal messages (from moderator or participants) are added to both the
        full context and the verbal-only context since they represent what was
        actually spoken and heard.

        Args:
            message: The message to add (should be MODERATOR or PARTICIPANT role)

        Raises:
            ValueError: If the message has NONVERBAL role
        """
        if message.role == MessageRole.NONVERBAL:
            raise ValueError(
                "Non-verbal messages should use add_nonverbal_message(), "
                "not add_verbal_message()"
            )

        self.messages_full.append(message)
        self.messages_verbal.append(message)

        msg_tokens = count_tokens(message.content) + 4
        self._token_count_full += msg_tokens
        self._token_count_verbal += msg_tokens

    def add_nonverbal_message(self, message: Message) -> None:
        """Add a non-verbal reaction to the full context only.

        Non-verbal reactions are only visible in the full context (what the
        participant internally perceives) but not in the verbal context
        (what they actually "heard").

        Args:
            message: The non-verbal message to add

        Raises:
            ValueError: If the message does not have NONVERBAL role
        """
        if message.role != MessageRole.NONVERBAL:
            raise ValueError(
                "Only NONVERBAL messages should use add_nonverbal_message(). "
                f"Got role: {message.role}"
            )

        self.messages_full.append(message)
        self._token_count_full += count_tokens(message.content) + 4

    def add_system_message(self, message: Message) -> None:
        """Add a system message to both contexts.

        System messages (instructions, setup) are added to both contexts.

        Args:
            message: The system message to add

        Raises:
            ValueError: If the message does not have SYSTEM role
        """
        if message.role != MessageRole.SYSTEM:
            raise ValueError(
                "Only SYSTEM messages should use add_system_message(). " f"Got role: {message.role}"
            )

        self.messages_full.append(message)
        self.messages_verbal.append(message)

        msg_tokens = count_tokens(message.content) + 4
        self._token_count_full += msg_tokens
        self._token_count_verbal += msg_tokens

    def set_current_question(self, question: str) -> None:
        """Set the current focus group question.

        The current question is tracked separately because it should never be
        summarized - it must always be preserved in full during context
        summarization.

        Args:
            question: The current focus group question text
        """
        # Remove old question tokens if there was one
        if self.current_question:
            old_tokens = count_tokens(self.current_question) + 4
            self._token_count_full -= old_tokens
            self._token_count_verbal -= old_tokens

        self.current_question = question

        # Add new question tokens
        new_tokens = count_tokens(question) + 4
        self._token_count_full += new_tokens
        self._token_count_verbal += new_tokens

    def clear_current_question(self) -> None:
        """Clear the current question (e.g., when moving to next question)."""
        if self.current_question:
            old_tokens = count_tokens(self.current_question) + 4
            self._token_count_full -= old_tokens
            self._token_count_verbal -= old_tokens
            self.current_question = None

    def get_full_context_messages(self) -> list[Message]:
        """Get all messages from the full context.

        Returns:
            List of all messages including non-verbal reactions.
        """
        return list(self.messages_full)

    def get_verbal_context_messages(self) -> list[Message]:
        """Get only verbal messages.

        Returns:
            List of verbal messages only (what was actually spoken).
        """
        return list(self.messages_verbal)

    def get_last_verbal_message(self) -> Message | None:
        """Get the most recent verbal message.

        Returns:
            The last verbal message, or None if no verbal messages exist.
        """
        if not self.messages_verbal:
            return None
        return self.messages_verbal[-1]

    def get_last_message(self) -> Message | None:
        """Get the most recent message from the full context.

        Returns:
            The last message (could be verbal or non-verbal), or None if empty.
        """
        if not self.messages_full:
            return None
        return self.messages_full[-1]

    def format_for_llm(self, include_nonverbals: bool = True) -> str:
        """Format the context as a string for LLM input.

        Creates a formatted string representation of the context suitable for
        use as part of an LLM prompt.

        Args:
            include_nonverbals: If True, use full context; if False, verbal only.

        Returns:
            Formatted string with persona, question, and conversation history.
        """
        parts = []

        # Always include persona
        parts.append(f"You are {self.persona_description}")

        # Include current question if set
        if self.current_question:
            parts.append(f"\nCurrent question: {self.current_question}")

        # Add conversation history
        messages = self.messages_full if include_nonverbals else self.messages_verbal

        if messages:
            parts.append("\nConversation so far:")
            for msg in messages:
                if msg.role == MessageRole.MODERATOR:
                    parts.append(f"\nModerator: {msg.content}")
                elif msg.role == MessageRole.PARTICIPANT:
                    parts.append(f"\n{msg.participant_name}: {msg.content}")
                elif msg.role == MessageRole.NONVERBAL and include_nonverbals:
                    parts.append(f"\n[{msg.participant_name}'s reaction: {msg.content}]")
                elif msg.role == MessageRole.SYSTEM:
                    parts.append(f"\n[System: {msg.content}]")

        return "\n".join(parts)

    def get_nonverbal_messages(self) -> list[Message]:
        """Get only the non-verbal messages.

        Returns:
            List of non-verbal messages only.
        """
        return [msg for msg in self.messages_full if msg.role == MessageRole.NONVERBAL]

    def get_verbal_messages_by_participant(self, name: str) -> list[Message]:
        """Get all verbal messages from a specific participant.

        Args:
            name: The participant name to filter by.

        Returns:
            List of verbal messages from that participant.
        """
        return [
            msg
            for msg in self.messages_verbal
            if msg.role == MessageRole.PARTICIPANT and msg.participant_name == name
        ]

    def has_spoken(self, participant_name: str) -> bool:
        """Check if a participant has spoken in the verbal context.

        Args:
            participant_name: The name to check.

        Returns:
            True if the participant has at least one verbal message.
        """
        return any(
            msg.role == MessageRole.PARTICIPANT and msg.participant_name == participant_name
            for msg in self.messages_verbal
        )

    def message_count(self, include_nonverbals: bool = True) -> int:
        """Get the count of messages in the context.

        Args:
            include_nonverbals: If True, count all messages; if False, verbal only.

        Returns:
            Number of messages in the specified context.
        """
        if include_nonverbals:
            return len(self.messages_full)
        return len(self.messages_verbal)

    def to_dict(self) -> dict[str, Any]:
        """Convert the context storage to a dictionary.

        Returns:
            Dictionary representation suitable for serialization.
        """
        return {
            "persona_description": self.persona_description,
            "current_question": self.current_question,
            "messages_full": [msg.to_dict() for msg in self.messages_full],
            "messages_verbal": [msg.to_dict() for msg in self.messages_verbal],
            "token_count_full": self._token_count_full,
            "token_count_verbal": self._token_count_verbal,
        }

    @classmethod
    def create_for_participant(cls, persona_description: str) -> "ContextStorage":
        """Create a new context storage for a participant.

        This is the preferred way to create a ContextStorage instance.

        Args:
            persona_description: The full persona description for this participant.

        Returns:
            A new ContextStorage initialized with the persona.
        """
        return cls(persona_description=persona_description)
