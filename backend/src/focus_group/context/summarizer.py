"""Context summarization for managing context window limits.

This module provides functionality to summarize older portions of conversation
context when it exceeds token limits, while preserving critical information
like persona descriptions, current questions, and recent responses.
"""

from focus_group.context.storage import ContextStorage
from focus_group.llm.base import BaseLLMProvider
from focus_group.models.message import Message, MessageRole
from focus_group.utils.tokens import count_tokens

# System prompt for the summarization task
SUMMARIZATION_SYSTEM_PROMPT = """You are a conversation summarizer for a focus group discussion.
Your task is to create a concise summary of the conversation that captures:
- Main topics and themes discussed
- Key opinions and positions expressed by participants
- Any points of agreement or disagreement
- Important context that would be needed to continue the conversation

Keep the summary brief but informative. Focus on content that would help someone
continue the conversation naturally without losing important context."""


class ContextSummarizer:
    """Summarizes context to stay within token limits.

    When a participant's context grows too large, this class uses an LLM to
    generate a summary of older messages while preserving critical elements:
    - The complete persona description (never summarized)
    - The current focus group question
    - The most recent verbal response

    Attributes:
        provider: The LLM provider used for generating summaries
    """

    def __init__(self, provider: BaseLLMProvider) -> None:
        """Initialize the context summarizer.

        Args:
            provider: The LLM provider to use for generating summaries
        """
        self.provider = provider

    async def summarize_context(
        self,
        storage: ContextStorage,
        max_tokens: int,
        target_tokens: int | None = None,
    ) -> ContextStorage:
        """Summarize the context if it exceeds the token limit.

        This method checks if the context exceeds max_tokens and, if so,
        summarizes older messages to bring it under the limit. Critical
        elements are always preserved.

        Args:
            storage: The context storage to potentially summarize
            max_tokens: Maximum allowed tokens for the context
            target_tokens: Target token count after summarization (defaults to 60% of max)

        Returns:
            A new ContextStorage with summarized content if needed,
            or the original storage if no summarization was required
        """
        if target_tokens is None:
            target_tokens = int(max_tokens * 0.6)

        # Check if summarization is needed
        if storage.token_count_full <= max_tokens:
            return storage

        # Find messages to summarize (exclude the ones we must preserve)
        messages_to_summarize = self._identify_messages_to_summarize(storage)

        if not messages_to_summarize:
            # Nothing to summarize, return as-is
            return storage

        # Generate summary using LLM
        summary_text = await self._generate_summary(messages_to_summarize)

        # Create new storage with summarized content
        new_storage = self._create_summarized_storage(
            original=storage,
            summary_text=summary_text,
            summarized_count=len(messages_to_summarize),
        )

        return new_storage

    def _identify_messages_to_summarize(
        self,
        storage: ContextStorage,
    ) -> list[Message]:
        """Identify which messages can be summarized.

        Messages that must be PRESERVED (never summarized):
        - The most recent verbal response
        - System messages setting up the persona

        Args:
            storage: The context storage to analyze

        Returns:
            List of messages that can be safely summarized
        """
        messages = storage.get_full_context_messages()

        if len(messages) <= 2:
            # Not enough messages to summarize
            return []

        # Find the last verbal message (must preserve)
        last_verbal_idx = -1
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].role in (MessageRole.PARTICIPANT, MessageRole.MODERATOR):
                last_verbal_idx = i
                break

        # We want to summarize everything EXCEPT:
        # - The last verbal message
        # - Any messages after the last verbal message (recent non-verbals)
        # - Keep at least the 2 most recent messages regardless

        if last_verbal_idx <= 1:
            # Not enough older messages to summarize
            return []

        # Summarize from index 0 up to (but not including) last_verbal_idx
        # Leave at least the 2 most recent exchanges
        summarize_up_to = min(last_verbal_idx, len(messages) - 2)

        if summarize_up_to <= 0:
            return []

        return messages[:summarize_up_to]

    async def _generate_summary(self, messages: list[Message]) -> str:
        """Generate a summary of the given messages using the LLM.

        Args:
            messages: List of messages to summarize

        Returns:
            A text summary of the conversation
        """
        # Format messages for summarization
        formatted_messages = self._format_messages_for_summary(messages)

        prompt = f"""Please summarize the following focus group conversation excerpt:

{formatted_messages}

Provide a concise summary (2-4 sentences) that captures the key points discussed."""

        summary = await self.provider.generate(
            prompt=prompt,
            system_prompt=SUMMARIZATION_SYSTEM_PROMPT,
        )

        return summary.strip()

    def _format_messages_for_summary(self, messages: list[Message]) -> str:
        """Format messages as readable text for summarization.

        Args:
            messages: List of messages to format

        Returns:
            Formatted text representation of the messages
        """
        lines = []
        for msg in messages:
            if msg.role == MessageRole.MODERATOR:
                lines.append(f"Moderator: {msg.content}")
            elif msg.role == MessageRole.PARTICIPANT:
                lines.append(f"{msg.participant_name}: {msg.content}")
            elif msg.role == MessageRole.NONVERBAL:
                lines.append(f"[{msg.participant_name} (non-verbal): {msg.content}]")
            elif msg.role == MessageRole.SYSTEM:
                lines.append(f"[System: {msg.content}]")

        return "\n".join(lines)

    def _create_summarized_storage(
        self,
        original: ContextStorage,
        summary_text: str,
        summarized_count: int,
    ) -> ContextStorage:
        """Create a new ContextStorage with summarized content.

        Args:
            original: The original context storage
            summary_text: The generated summary text
            summarized_count: Number of messages that were summarized

        Returns:
            New ContextStorage with the summary replacing old messages
        """
        # Create new storage with same persona
        new_storage = ContextStorage.create_for_participant(
            persona_description=original.persona_description
        )

        # Set current question if exists
        if original.current_question:
            new_storage.set_current_question(original.current_question)

        # Add the summary as a system message
        summary_message = Message.system(f"[Previous conversation summary: {summary_text}]")
        new_storage.add_system_message(summary_message)

        # Add preserved messages (those after the summarized ones)
        all_full_messages = original.get_full_context_messages()
        all_verbal_messages = original.get_verbal_context_messages()

        # Add remaining full context messages
        for msg in all_full_messages[summarized_count:]:
            if msg.role == MessageRole.NONVERBAL:
                new_storage.add_nonverbal_message(msg)
            elif msg.role == MessageRole.SYSTEM:
                new_storage.add_system_message(msg)
            else:
                new_storage.add_verbal_message(msg)

        return new_storage

    async def needs_summarization(
        self,
        storage: ContextStorage,
        max_tokens: int,
        threshold: float = 0.8,
    ) -> bool:
        """Check if the context needs summarization.

        Args:
            storage: The context storage to check
            max_tokens: Maximum allowed tokens
            threshold: Trigger summarization when usage exceeds this fraction

        Returns:
            True if summarization is recommended
        """
        return storage.token_count_full > (max_tokens * threshold)

    def estimate_summary_tokens(self, messages: list[Message]) -> int:
        """Estimate how many tokens a summary of these messages would use.

        This is a rough estimate for planning purposes.

        Args:
            messages: Messages that would be summarized

        Returns:
            Estimated token count for the summary
        """
        # Summaries are typically 10-20% of original length
        original_tokens = sum(count_tokens(msg.content) for msg in messages)
        return max(50, int(original_tokens * 0.15))
