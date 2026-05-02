"""High-level context manager for focus group participants.

This module provides the ContextManager class that coordinates context storage
and summarization, providing a clean API for managing participant context
throughout a focus group session.
"""

from focus_group.context.storage import ContextStorage
from focus_group.context.summarizer import ContextSummarizer
from focus_group.llm.base import BaseLLMProvider
from focus_group.models.config import ContextConfig
from focus_group.models.message import Message


class ContextManager:
    """Manages context for a focus group participant.

    This class provides a high-level interface that coordinates the
    ContextStorage and ContextSummarizer, automatically handling context
    window management and summarization when needed.

    Attributes:
        storage: The underlying ContextStorage instance
        summarizer: The ContextSummarizer for handling overflow
        config: Context configuration settings
        provider: The LLM provider for summarization
    """

    def __init__(
        self,
        persona_description: str,
        provider: BaseLLMProvider,
        config: ContextConfig | None = None,
    ) -> None:
        """Initialize the context manager.

        Args:
            persona_description: Full description of the participant's persona
            provider: LLM provider for summarization
            config: Context configuration (uses defaults if not provided)
        """
        self._storage = ContextStorage.create_for_participant(persona_description)
        self._provider = provider
        self._config = config or ContextConfig()
        self._summarizer = ContextSummarizer(provider)
        self._needs_summarization_check = False

    @property
    def storage(self) -> ContextStorage:
        """Get the underlying context storage.

        Returns:
            The ContextStorage instance managing the dual contexts
        """
        return self._storage

    @property
    def config(self) -> ContextConfig:
        """Get the context configuration.

        Returns:
            The ContextConfig with token limits and thresholds
        """
        return self._config

    @property
    def token_count_full(self) -> int:
        """Get the token count for the full context.

        Returns:
            Estimated tokens in the full context
        """
        return self._storage.token_count_full

    @property
    def token_count_verbal(self) -> int:
        """Get the token count for the verbal-only context.

        Returns:
            Estimated tokens in the verbal context
        """
        return self._storage.token_count_verbal

    def add_verbal_message(self, message: Message) -> None:
        """Add a verbal message to the context.

        Verbal messages (moderator questions, participant responses) are
        added to both the full and verbal-only contexts.

        Args:
            message: The verbal message to add

        Raises:
            ValueError: If the message has NONVERBAL role
        """
        self._storage.add_verbal_message(message)
        self._needs_summarization_check = True

    def add_nonverbal_message(self, message: Message) -> None:
        """Add a non-verbal reaction to the context.

        Non-verbal messages are only added to the full context, not
        the verbal-only context.

        Args:
            message: The non-verbal message to add

        Raises:
            ValueError: If the message does not have NONVERBAL role
        """
        self._storage.add_nonverbal_message(message)
        self._needs_summarization_check = True

    def add_system_message(self, message: Message) -> None:
        """Add a system message to both contexts.

        Args:
            message: The system message to add

        Raises:
            ValueError: If the message does not have SYSTEM role
        """
        self._storage.add_system_message(message)
        self._needs_summarization_check = True

    def set_question(self, question: str) -> None:
        """Set the current focus group question.

        The current question is tracked separately and preserved during
        any context summarization.

        Args:
            question: The current question text
        """
        self._storage.set_current_question(question)

    def clear_question(self) -> None:
        """Clear the current question (when moving to next question)."""
        self._storage.clear_current_question()

    def get_context_for_llm(self, include_nonverbals: bool = True) -> str:
        """Get the formatted context string for LLM input.

        Args:
            include_nonverbals: If True, include non-verbal reactions;
                if False, only include verbal messages

        Returns:
            Formatted context string ready for use in an LLM prompt
        """
        return self._storage.format_for_llm(include_nonverbals=include_nonverbals)

    def needs_summarization(self) -> bool:
        """Check if the context needs summarization.

        Returns:
            True if the context exceeds the summarization threshold
        """
        return self._storage.token_count_full > self._config.summarize_after_tokens

    async def check_and_summarize(self) -> bool:
        """Check if summarization is needed and perform it if so.

        This method should be called periodically (e.g., after adding
        messages) to keep the context within limits.

        Returns:
            True if summarization was performed, False otherwise
        """
        if not self._needs_summarization_check:
            return False

        self._needs_summarization_check = False

        if not self.needs_summarization():
            return False

        # Perform summarization
        new_storage = await self._summarizer.summarize_context(
            storage=self._storage,
            max_tokens=self._config.max_tokens,
        )

        self._storage = new_storage
        return True

    async def force_summarize(self) -> None:
        """Force context summarization regardless of current size.

        This can be useful when you want to reduce context size proactively.
        """
        new_storage = await self._summarizer.summarize_context(
            storage=self._storage,
            max_tokens=self._config.max_tokens,
        )
        self._storage = new_storage

    def has_spoken(self, participant_name: str) -> bool:
        """Check if a participant has spoken in this context.

        Args:
            participant_name: The name to check

        Returns:
            True if the participant has at least one verbal message
        """
        return self._storage.has_spoken(participant_name)

    def get_last_verbal_message(self) -> Message | None:
        """Get the most recent verbal message.

        Returns:
            The last verbal message, or None if none exist
        """
        return self._storage.get_last_verbal_message()

    def get_message_count(self, include_nonverbals: bool = True) -> int:
        """Get the count of messages in the context.

        Args:
            include_nonverbals: If True, count all messages; if False, verbal only

        Returns:
            Number of messages in the specified context
        """
        return self._storage.message_count(include_nonverbals=include_nonverbals)

    def get_full_context_messages(self) -> list[Message]:
        """Get all messages from the full context.

        Returns:
            List of all messages including non-verbal reactions
        """
        return self._storage.get_full_context_messages()

    def get_verbal_context_messages(self) -> list[Message]:
        """Get only verbal messages.

        Returns:
            List of verbal messages only
        """
        return self._storage.get_verbal_context_messages()

    def get_nonverbal_messages(self) -> list[Message]:
        """Get only non-verbal messages.

        Returns:
            List of non-verbal messages only
        """
        return self._storage.get_nonverbal_messages()

    @property
    def current_question(self) -> str | None:
        """Get the current focus group question.

        Returns:
            The current question text, or None if not set
        """
        return self._storage.current_question

    @property
    def persona_description(self) -> str:
        """Get the persona description.

        Returns:
            The full persona description for this participant
        """
        return self._storage.persona_description

    def get_context_usage_ratio(self) -> float:
        """Get the ratio of used context to maximum allowed.

        Returns:
            Float between 0.0 and 1.0+ indicating context usage
        """
        return self._storage.token_count_full / self._config.max_tokens

    def get_available_tokens(self) -> int:
        """Get the number of tokens available before hitting the limit.

        Returns:
            Number of tokens still available in the context window
        """
        return max(0, self._config.max_tokens - self._storage.token_count_full)

    def to_dict(self) -> dict:
        """Convert the context manager state to a dictionary.

        Returns:
            Dictionary representation suitable for serialization
        """
        return {
            "storage": self._storage.to_dict(),
            "config": self._config.model_dump(),
            "token_count_full": self.token_count_full,
            "token_count_verbal": self.token_count_verbal,
            "needs_summarization": self.needs_summarization(),
        }
