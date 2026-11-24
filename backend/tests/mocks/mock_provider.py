"""Mock LLM provider for testing.

This module provides a mock implementation of BaseLLMProvider that returns
predefined responses for testing purposes, eliminating the need for a real
LLM backend during tests.
"""

import json
import re
from collections.abc import Callable
from typing import Any

from focus_group.llm.base import BaseLLMProvider
from focus_group.utils.tokens import count_tokens


class MockLLMProvider(BaseLLMProvider):
    """Mock LLM provider for testing.

    Provides deterministic responses based on prompt patterns, making tests
    reproducible and fast. Can be configured with custom response handlers
    or use default mock responses.

    Attributes:
        _model_name: Name of the mock model
        _max_context_tokens: Maximum context size
        _responses: List of (pattern, response) tuples for matching prompts
        _default_response: Fallback response when no pattern matches
        _call_count: Number of generate() calls made
        _calls: List of all prompts received
        _available: Whether the provider should appear available

    Example:
        >>> provider = MockLLMProvider()
        >>> response = await provider.generate("Hello")
        >>> print(response)  # Returns mock response
    """

    def __init__(
        self,
        model_name: str = "mock-model",
        max_context_tokens: int = 8192,
        available: bool = True,
        responses: list[tuple[str, str | Callable[[str], str]]] | None = None,
        default_response: str | None = None,
    ) -> None:
        """Initialize the mock provider.

        Args:
            model_name: Name to return from model_name property
            max_context_tokens: Max context size to simulate
            available: Whether is_available() should return True
            responses: List of (regex_pattern, response) tuples. Response can be
                a string or a callable that takes the prompt and returns a response.
            default_response: Response to return when no pattern matches
        """
        self._model_name = model_name
        self._max_context_tokens = max_context_tokens
        self._available = available
        self._responses: list[tuple[str, str | Callable[[str], str]]] = responses or []
        self._default_response = default_response or "Mock response"
        self._call_count = 0
        self._calls: list[dict[str, Any]] = []

        # Add default response patterns
        self._add_default_patterns()

    def _add_default_patterns(self) -> None:
        """Add default response patterns for common focus group operations."""
        # Pattern for persona generation - returns JSON
        self._responses.append(
            (
                r"generate.*persona|demographic|create.*character",
                self._generate_persona_response,
            )
        )

        # Pattern for introduction
        self._responses.append(
            (
                r"introduce.*yourself|introduction",
                self._generate_introduction_response,
            )
        )

        # Pattern for non-verbal reactions
        self._responses.append(
            (
                r"non-verbal|nonverbal|body language|physical reaction",
                self._generate_nonverbal_response,
            )
        )

        # Pattern for speaker selection - returns JSON
        self._responses.append(
            (
                r"who.*speak|select.*speaker|first.*speak|next.*speak",
                self._generate_speaker_selection_response,
            )
        )

        # Pattern for participant-to-participant responses (must come before generic verbal)
        self._responses.append(
            (
                r"just said:.*How do you respond",
                self._generate_participant_response,
            )
        )

        # Pattern for closing responses
        self._responses.append(
            (
                r"Reflect on THIS specific discussion",
                self._generate_closing_response,
            )
        )

        # Pattern for verbal responses
        self._responses.append(
            (
                r"respond|opinion|think about|feel about|perspective",
                self._generate_verbal_response,
            )
        )

        # Pattern for context summarization
        self._responses.append(
            (
                r"summarize|summary|condense",
                self._generate_summary_response,
            )
        )

    def _generate_persona_response(self, prompt: str) -> str:
        """Generate a mock persona JSON response."""
        # Extract demographic info if present
        demo_match = re.search(r'demographic[:\s]*["\']?([^"\']+)["\']?', prompt, re.IGNORECASE)
        demographic = demo_match.group(1).strip() if demo_match else "Unknown"

        # Generate a unique name based on call count
        names = [
            ("Sarah Chen", 34),
            ("Marcus Johnson", 52),
            ("Elena Rodriguez", 28),
            ("David Kim", 45),
            ("Amanda Foster", 38),
            ("James Wilson", 61),
            ("Priya Patel", 29),
            ("Michael Brown", 47),
            ("Lisa Thompson", 33),
            ("Robert Garcia", 55),
        ]
        name, age = names[self._call_count % len(names)]

        # Infer pronouns from demographic
        demo_lower = demographic.lower()
        if "non-binary" in demo_lower or "nonbinary" in demo_lower:
            pronouns = "they/them"
        elif "female" in demo_lower or "woman" in demo_lower:
            pronouns = "she/her"
        elif "male" in demo_lower or "man" in demo_lower:
            pronouns = "he/him"
        else:
            pronouns = "they/them"

        # Cycle through stances and registers to ensure diversity in tests
        stances = ["agreeable", "balanced", "challenging", "contrarian"]
        registers = ["conversational", "casual", "curt", "folksy", "formal", "academic", "warm"]
        stance = stances[self._call_count % len(stances)]
        register = registers[self._call_count % len(registers)]

        return json.dumps(
            {
                "name": name,
                "age": age,
                "background": f"A professional with extensive experience in their field. {demographic}",
                "personality_traits": ["thoughtful", "articulate", "engaged"],
                "communication_style": "Clear and direct, tends to provide specific examples",
                "pronouns": pronouns,
                "discussion_stance": stance,
                "communication_register": register,
            }
        )

    def _generate_introduction_response(self, prompt: str) -> str:
        """Generate a mock introduction response."""
        return (
            "Hi everyone, my name is [Name] and I'm happy to be here today. "
            "I work in the professional sector and I'm looking forward to sharing "
            "my thoughts on today's topics."
        )

    def _generate_nonverbal_response(self, prompt: str) -> str:
        """Generate a mock non-verbal reaction."""
        reactions = [
            "leans forward with interest, nodding slightly",
            "crosses arms thoughtfully, tilting head to one side",
            "maintains eye contact and nods in agreement",
            "shifts in seat, appears to be considering the point carefully",
            "eyebrows raise slightly, indicating engagement with the topic",
        ]
        return reactions[self._call_count % len(reactions)]

    def _generate_speaker_selection_response(self, prompt: str) -> str:
        """Generate a mock speaker selection JSON response."""
        # Try to extract participant names from the prompt
        name_match = re.search(r"participants?[:\s]*\[([^\]]+)\]", prompt, re.IGNORECASE)
        if name_match:
            names_str = name_match.group(1)
            names = re.findall(r'["\']([^"\']+)["\']', names_str)
            if names:
                selected = names[self._call_count % len(names)]
                return json.dumps(
                    {
                        "selected_speaker": selected,
                        "reasoning": f"{selected} appears most engaged based on their body language",
                        "someone_wants_to_speak": True,
                    }
                )

        # Default response
        return json.dumps(
            {
                "selected_speaker": "Sarah Chen",
                "reasoning": "Sarah appears eager to contribute based on forward lean and eye contact",
                "someone_wants_to_speak": True,
            }
        )

    def _generate_verbal_response(self, prompt: str) -> str:
        """Generate a mock verbal response."""
        responses = [
            "That's an interesting point. From my experience, I've found that "
            "this topic has multiple dimensions worth considering.",
            "I agree with what was said earlier. In my view, the key factor here "
            "is understanding the underlying needs of all stakeholders.",
            "Building on what others have shared, I'd add that we should also "
            "consider the practical implications of this approach.",
            "I have a slightly different perspective. While I understand the points "
            "being made, I think we should also factor in the long-term effects.",
            "Thank you for that insight. It reminds me of a similar situation I "
            "encountered where the outcome was quite positive.",
        ]
        return responses[self._call_count % len(responses)]

    def _generate_participant_response(self, prompt: str) -> str:
        """Generate a mock response to another participant's statement."""
        # Extract the previous speaker's name if possible
        import re as _re

        name_match = _re.search(r"^(.+?) just said:", prompt, _re.MULTILINE)
        speaker = name_match.group(1) if name_match else "the previous speaker"

        responses = [
            f"What {speaker} said about experience matches mine, but the conclusion "
            "is different — I found the opposite to be true in practice.",
            f"I hear {speaker}, but honestly that hasn't been my experience at all. "
            "The way I see it, the real issue is structural.",
            f"Sure, but {speaker} is only looking at one side of this. "
            "There's a practical cost that nobody's talking about.",
            f"I see where {speaker} is coming from, but I've had a different "
            "experience that shaped my view on this.",
        ]
        return responses[self._call_count % len(responses)]

    def _generate_closing_response(self, prompt: str) -> str:
        """Generate a mock closing response that references the discussion."""
        responses = [
            "This has been a really productive conversation. I especially "
            "appreciated hearing different perspectives on the topic.",
            "I'm glad we had this discussion. It's given me a lot to think about, "
            "particularly the points raised earlier.",
            "Thank you for including me. The range of viewpoints shared today "
            "really broadened my understanding.",
        ]
        return responses[self._call_count % len(responses)]

    def _generate_summary_response(self, prompt: str) -> str:
        """Generate a mock context summary."""
        return (
            "[Summary] The discussion has covered several key points including "
            "diverse perspectives on the topic, with participants sharing relevant "
            "experiences and building on each other's ideas."
        )

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        json_schema: dict[str, Any] | None = None,
    ) -> str:
        """Generate a mock response based on prompt patterns.

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt (ignored in mock)

        Returns:
            Mock response string
        """
        self._call_count += 1
        self._calls.append(
            {
                "prompt": prompt,
                "system_prompt": system_prompt,
                "json_schema": json_schema,
                "call_number": self._call_count,
            }
        )

        # Check for matching patterns
        for pattern, response in self._responses:
            if re.search(pattern, prompt, re.IGNORECASE):
                if callable(response):
                    return response(prompt)
                return response

        return self._default_response

    async def count_tokens(self, text: str) -> int:
        """Count tokens using the utility function.

        Args:
            text: Text to count tokens for

        Returns:
            Estimated token count
        """
        return count_tokens(text)

    @property
    def max_context_tokens(self) -> int:
        """Get maximum context tokens."""
        return self._max_context_tokens

    @property
    def model_name(self) -> str:
        """Get model name."""
        return self._model_name

    async def is_available(self) -> bool:
        """Check if provider is available (configurable)."""
        return self._available

    # Test helper methods

    @property
    def call_count(self) -> int:
        """Get the number of generate() calls made."""
        return self._call_count

    @property
    def calls(self) -> list[dict[str, Any]]:
        """Get all recorded calls."""
        return self._calls.copy()

    def get_last_call(self) -> dict[str, Any] | None:
        """Get the most recent call."""
        return self._calls[-1] if self._calls else None

    def reset(self) -> None:
        """Reset call tracking."""
        self._call_count = 0
        self._calls = []

    def add_response(
        self,
        pattern: str,
        response: str | Callable[[str], str],
    ) -> None:
        """Add a custom response pattern.

        Args:
            pattern: Regex pattern to match against prompts
            response: Response string or callable
        """
        # Insert at beginning to take precedence
        self._responses.insert(0, (pattern, response))

    def set_available(self, available: bool) -> None:
        """Set whether the provider should appear available."""
        self._available = available


class SequentialMockProvider(MockLLMProvider):
    """Mock provider that returns responses in sequence.

    Useful for tests that need predictable, ordered responses.

    Example:
        >>> provider = SequentialMockProvider(responses=[
        ...     "First response",
        ...     "Second response",
        ... ])
        >>> await provider.generate("any prompt")  # Returns "First response"
        >>> await provider.generate("any prompt")  # Returns "Second response"
    """

    def __init__(
        self,
        responses: list[str],
        loop: bool = True,
        **kwargs,
    ) -> None:
        """Initialize with a sequence of responses.

        Args:
            responses: List of responses to return in order
            loop: Whether to loop back to beginning when exhausted
            **kwargs: Additional arguments passed to MockLLMProvider
        """
        super().__init__(**kwargs)
        self._sequential_responses = responses
        self._loop = loop
        self._response_index = 0

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        json_schema: dict[str, Any] | None = None,
    ) -> str:
        """Return the next response in sequence."""
        self._call_count += 1
        self._calls.append(
            {
                "prompt": prompt,
                "system_prompt": system_prompt,
                "json_schema": json_schema,
                "call_number": self._call_count,
            }
        )

        if not self._sequential_responses:
            return self._default_response

        if self._response_index >= len(self._sequential_responses):
            if self._loop:
                self._response_index = 0
            else:
                return self._default_response

        response = self._sequential_responses[self._response_index]
        self._response_index += 1
        return response

    def reset(self) -> None:
        """Reset call tracking and response index."""
        super().reset()
        self._response_index = 0
