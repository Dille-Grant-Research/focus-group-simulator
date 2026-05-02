"""Token counting utilities for context management.

This module provides simple token estimation for managing context windows.
The estimates are intentionally conservative (slightly high) to ensure we
stay within model limits.
"""


def count_tokens(text: str) -> int:
    """Estimate the number of tokens in the given text.

    This uses a simple heuristic of approximately 4 characters per token,
    which is a reasonable approximation across most LLM tokenizers
    (GPT-3.5/4, Llama, Claude, etc.).

    The estimate is intentionally conservative (rounds up) to ensure we don't
    exceed context limits. Individual LLM providers may override this with
    more accurate counting in their `count_tokens()` implementation.

    Args:
        text: The text to count tokens for

    Returns:
        Estimated number of tokens (rounded up)

    Examples:
        >>> count_tokens("Hello world")
        3
        >>> count_tokens("")
        0
        >>> count_tokens("A" * 100)
        25
    """
    if not text:
        return 0

    # Conservative estimate: ~4 characters per token
    # Add 1 to round up (ensures we don't underestimate)
    char_count = len(text)
    return (char_count + 3) // 4  # Integer division with ceiling


def estimate_tokens_for_messages(messages: list[dict[str, str]]) -> int:
    """Estimate total tokens for a list of messages.

    This is useful for estimating the size of a conversation history.
    Each message should have 'role' and 'content' keys.

    Args:
        messages: List of message dictionaries with 'role' and 'content'

    Returns:
        Estimated total tokens including message formatting overhead

    Examples:
        >>> messages = [
        ...     {"role": "system", "content": "You are helpful"},
        ...     {"role": "user", "content": "Hello"}
        ... ]
        >>> estimate_tokens_for_messages(messages) > 0
        True
    """
    if not messages:
        return 0

    total_tokens = 0

    for message in messages:
        # Count tokens in role and content
        role = message.get("role", "")
        content = message.get("content", "")

        total_tokens += count_tokens(role)
        total_tokens += count_tokens(content)

        # Add overhead for message formatting (typically ~4 tokens per message)
        total_tokens += 4

    # Add overhead for the message list structure
    total_tokens += 3

    return total_tokens
