"""Research Assistant for objective stance analysis and discussion summarization.

This module provides the ResearchAssistant class that uses an LLM to objectively
analyze participant turns and generate discussion summaries. It updates the
StanceMatrix with data-driven analysis instead of relying on self-reported
participant positions.

The Research Assistant is a standalone component that will be wired into the
session pipeline in Sprint 3.
"""

import json
import logging
import re

from focus_group.llm.base import BaseLLMProvider
from focus_group.models.memory import (
    ConsensusLevel,
    ParticipantStance,
    StanceMatrix,
    StancePosition,
)
from focus_group.prompts.analysis import (
    ANALYZE_STANCE_PROMPT,
    SUMMARIZE_DISCUSSION_PROMPT,
)

logger = logging.getLogger(__name__)


class ResearchAssistant:
    """Objectively analyzes participant turns and updates the StanceMatrix.

    The Research Assistant uses an LLM to:
    - Determine each speaker's stance (supportive, skeptical, neutral, mixed)
    - Extract key points from statements
    - Generate discussion summaries
    - Track consensus level across the group

    This replaces unreliable self-reporting where participants describe
    their own positions (which LLMs often get wrong).

    Attributes:
        provider: The LLM provider for analysis calls

    Example:
        >>> assistant = ResearchAssistant(llm_provider)
        >>> stance = await assistant.analyze_turn(
        ...     speaker_name="Sarah",
        ...     statement="I think remote work is great for flexibility.",
        ...     question="What do you think about remote work?",
        ...     question_index=0,
        ...     stance_matrix=matrix,
        ...     participants=["Sarah", "Robert", "Jordan"],
        ... )
        >>> print(stance.position)
        StancePosition.SUPPORTIVE
    """

    def __init__(self, provider: BaseLLMProvider) -> None:
        """Initialize the Research Assistant.

        Args:
            provider: The LLM provider to use for analysis calls
        """
        self._provider = provider

    async def analyze_turn(
        self,
        speaker_name: str,
        statement: str,
        question: str,
        question_index: int,
        stance_matrix: StanceMatrix,
        participants: list[str],
    ) -> ParticipantStance:
        """Analyze a single speaking turn and update the StanceMatrix.

        Sends the speaker's statement to the LLM for objective stance analysis,
        then updates the shared StanceMatrix with the result.

        Args:
            speaker_name: Name of the participant who spoke
            statement: What they said (verbatim)
            question: The question being discussed
            question_index: 0-based index of the current question
            stance_matrix: The shared StanceMatrix to update
            participants: List of all participant names in the focus group

        Returns:
            The analyzed ParticipantStance. On parse failure, returns a
            default stance with position=NEUTRAL.
        """
        participants_json = json.dumps(participants)

        prompt = ANALYZE_STANCE_PROMPT.format(
            speaker_name=speaker_name,
            question=question,
            statement=statement,
            participants_json=participants_json,
        )

        raw_response = await self._provider.generate(
            prompt=prompt,
            system_prompt="You are an objective research assistant. Respond with ONLY valid JSON.",
        )

        parsed = self._parse_stance_json(raw_response)

        if parsed is None:
            logger.warning(
                "Failed to parse stance JSON for %s on question %d. "
                "Using default NEUTRAL stance. Raw response: %s",
                speaker_name,
                question_index,
                raw_response[:200],
            )
            # Still update the matrix with a default so contribution_count increments
            stance_matrix.update_stance(
                question_index=question_index,
                participant_name=speaker_name,
                position=StancePosition.NEUTRAL,
                key_points=[],
                sentiment_score=0.5,
            )
            return stance_matrix.get_stance(question_index, speaker_name)

        # Extract and validate fields
        position_str = parsed.get("position", "neutral").lower()
        try:
            position = StancePosition(position_str)
        except ValueError:
            position = StancePosition.NEUTRAL

        key_points = parsed.get("key_points", [])
        if not isinstance(key_points, list):
            key_points = []
        key_points = [str(p) for p in key_points]

        sentiment_score = parsed.get("sentiment_score", 0.5)
        if not isinstance(sentiment_score, (int, float)):
            sentiment_score = 0.5
        sentiment_score = max(0.0, min(1.0, float(sentiment_score)))

        # Update the shared StanceMatrix
        stance_matrix.update_stance(
            question_index=question_index,
            participant_name=speaker_name,
            position=position,
            key_points=key_points,
            sentiment_score=sentiment_score,
        )

        return stance_matrix.get_stance(question_index, speaker_name)

    async def generate_question_summary(
        self,
        question: str,
        question_index: int,
        transcript: list[dict[str, str]],
        stance_matrix: StanceMatrix,
    ) -> dict[str, str | list[str]]:
        """Generate an objective summary of a question's discussion.

        Args:
            question: The question that was discussed
            question_index: 0-based index of the question
            transcript: List of dicts with "speaker" and "content" keys
            stance_matrix: The shared StanceMatrix for stance context

        Returns:
            Dict with keys: "summary", "key_themes", "consensus_level",
            "notable_interactions". On parse failure, returns safe defaults.
        """
        # Format transcript for the prompt
        transcript_lines = []
        for entry in transcript:
            speaker = entry.get("speaker", "Unknown")
            content = entry.get("content", "")
            transcript_lines.append(f"{speaker}: {content}")
        transcript_text = "\n".join(transcript_lines) if transcript_lines else "(no transcript)"

        # Format participant stances for context
        stances_summary = stance_matrix.get_question_summary(question_index)
        participant_stances_json = json.dumps(stances_summary, indent=2)

        prompt = SUMMARIZE_DISCUSSION_PROMPT.format(
            question=question,
            transcript=transcript_text,
            participant_stances_json=participant_stances_json,
        )

        raw_response = await self._provider.generate(
            prompt=prompt,
            system_prompt="You are an objective research assistant. Respond with ONLY valid JSON.",
        )

        parsed = self._parse_summary_json(raw_response)

        if parsed is None:
            logger.warning(
                "Failed to parse summary JSON for question %d. " "Using defaults. Raw response: %s",
                question_index,
                raw_response[:200],
            )
            return {
                "summary": "Discussion summary unavailable.",
                "key_themes": [],
                "consensus_level": "none",
                "notable_interactions": [],
            }

        # Extract and validate fields with safe defaults
        summary = parsed.get("summary", "Discussion summary unavailable.")
        if not isinstance(summary, str):
            summary = "Discussion summary unavailable."

        key_themes = parsed.get("key_themes", [])
        if not isinstance(key_themes, list):
            key_themes = []
        key_themes = [str(t) for t in key_themes]

        consensus_level = parsed.get("consensus_level", "none").lower()
        valid_levels = {level.value for level in ConsensusLevel}
        if consensus_level not in valid_levels:
            consensus_level = "none"

        notable_interactions = parsed.get("notable_interactions", [])
        if not isinstance(notable_interactions, list):
            notable_interactions = []
        notable_interactions = [str(i) for i in notable_interactions]

        return {
            "summary": summary,
            "key_themes": key_themes,
            "consensus_level": consensus_level,
            "notable_interactions": notable_interactions,
        }

    def _parse_stance_json(self, raw: str) -> dict | None:
        """Parse stance analysis JSON from LLM output.

        Handles common LLM output artifacts:
        - Markdown code fences (```json ... ```)
        - Leading/trailing whitespace
        - Code fences without language label

        Args:
            raw: Raw LLM response string

        Returns:
            Parsed dict if valid JSON found, None otherwise
        """
        return self._extract_json(raw)

    def _parse_summary_json(self, raw: str) -> dict | None:
        """Parse discussion summary JSON from LLM output.

        Handles common LLM output artifacts:
        - Markdown code fences (```json ... ```)
        - Leading/trailing whitespace
        - Code fences without language label

        Args:
            raw: Raw LLM response string

        Returns:
            Parsed dict if valid JSON found, None otherwise
        """
        return self._extract_json(raw)

    @staticmethod
    def _extract_json(raw: str) -> dict | None:
        """Extract and parse JSON from a potentially messy LLM response.

        Tries multiple strategies:
        1. Strip markdown code fences with language label (```json ... ```)
        2. Strip plain code fences (``` ... ```)
        3. Try parsing raw string directly

        Args:
            raw: Raw string that should contain JSON

        Returns:
            Parsed dict if valid JSON found, None otherwise
        """
        if not raw or not raw.strip():
            return None

        cleaned = raw.strip()

        # Strategy 1: Strip ```json ... ``` fences
        fenced_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", cleaned, re.DOTALL)
        if fenced_match:
            try:
                result = json.loads(fenced_match.group(1).strip())
                if isinstance(result, dict):
                    return result
            except (json.JSONDecodeError, ValueError):
                pass

        # Strategy 2: Try parsing the cleaned string directly
        try:
            result = json.loads(cleaned)
            if isinstance(result, dict):
                return result
        except (json.JSONDecodeError, ValueError):
            pass

        # Strategy 3: Try to find JSON object within the string
        brace_match = re.search(r"\{[^{}]*\}", cleaned, re.DOTALL)
        if brace_match:
            try:
                result = json.loads(brace_match.group(0))
                if isinstance(result, dict):
                    return result
            except (json.JSONDecodeError, ValueError):
                pass

        return None
