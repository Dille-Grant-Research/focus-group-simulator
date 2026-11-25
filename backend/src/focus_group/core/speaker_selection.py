"""Speaker selection logic for focus group simulations.

This module provides classes for selecting which participant should speak
based on non-verbal reactions and group dynamics.
"""

import json
import logging
import re
from typing import Any

from focus_group.core.participant import Participant
from focus_group.llm.base import BaseLLMProvider

logger = logging.getLogger(__name__)

_NO_REASONING = "No reasoning provided"
_NOBODY_VALUES = ("null", "none")
_LOGGED_REPLY_CHARS = 200


def _extract_json_object(response: str) -> dict[str, Any] | None:
    """Return the first flat JSON object in the reply, or None if there isn't one."""
    match = re.search(r"\{[^}]+\}", response, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group())
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _str_field(data: dict[str, Any], key: str) -> str:
    """Return data[key] stripped if it is a string, else "" (handles null / numbers)."""
    value = data.get(key)
    return value.strip() if isinstance(value, str) else ""


def _selection_schema(names: list[str], *, allow_nobody: bool) -> dict[str, Any]:
    """JSON Schema for a selection reply, with `selected` limited to the given names.

    allow_nobody adds `someone_wants_to_speak` and lets `selected` be null.
    """
    properties: dict[str, Any] = {}
    required: list[str] = []
    if allow_nobody:
        properties["someone_wants_to_speak"] = {"type": "boolean"}
        required.append("someone_wants_to_speak")
        properties["selected"] = {"enum": [*names, None]}
    else:
        properties["selected"] = {"type": "string", "enum": list(names)}
    properties["reasoning"] = {"type": "string"}
    required += ["selected", "reasoning"]
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


class SpeakerSelectionResult:
    """Result of a speaker selection decision.

    Holds the selected participant along with the reasoning used
    to make the selection, useful for debug logging.

    Attributes:
        participant: The selected participant
        reasoning: Explanation of why this participant was selected
    """

    def __init__(self, participant: Participant, reasoning: str) -> None:
        """Initialize a speaker selection result.

        Args:
            participant: The participant selected to speak
            reasoning: The reasoning behind the selection
        """
        self.participant = participant
        self.reasoning = reasoning

    def __repr__(self) -> str:
        """Get string representation of the result."""
        return (
            f"SpeakerSelectionResult(participant={self.participant.name!r}, "
            f"reasoning={self.reasoning[:50]!r}...)"
        )


class FirstSpeakerSelector:
    """Selects which participant speaks first when a new question is asked.

    This class analyzes non-verbal reactions to determine which participant
    appears most eager or ready to speak first on a new question. The selection
    considers body language cues, personality traits, and communication styles.

    The selection process uses an LLM to interpret the non-verbal reactions
    and make a realistic choice about who would naturally speak first in
    a real focus group setting.

    Attributes:
        provider: The LLM provider used for making selection decisions

    Example:
        >>> selector = FirstSpeakerSelector(llm_provider)
        >>> reactions = {
        ...     "Sarah Chen": "leans forward with interest, eyes bright",
        ...     "John Smith": "crosses arms and looks thoughtful",
        ...     "Maria Garcia": "nods slowly, tapping her chin"
        ... }
        >>> result = await selector.select_first_speaker(
        ...     participants=[sarah, john, maria],
        ...     question="What do you think about remote work?",
        ...     reactions=reactions
        ... )
        >>> print(result.participant.name)  # "Sarah Chen"
        >>> print(result.reasoning)  # "Sarah's forward lean and bright eyes..."
    """

    def __init__(self, provider: BaseLLMProvider) -> None:
        """Initialize the first speaker selector.

        Args:
            provider: The LLM provider to use for selection decisions
        """
        self._provider = provider

    async def select_first_speaker(
        self,
        participants: list[Participant],
        question: str,
        reactions: dict[str, str],
    ) -> SpeakerSelectionResult:
        """Select which participant should speak first on a question.

        Analyzes the non-verbal reactions of all participants to determine
        who appears most eager or ready to speak. The selection considers:
        - Non-verbal cues indicating eagerness (leaning forward, eye contact)
        - Personality traits (extroverts may be more likely to speak first)
        - Communication style preferences

        Args:
            participants: List of participants who could speak
            question: The focus group question that was asked
            reactions: Dictionary mapping participant names to their non-verbal
                reactions to the question

        Returns:
            SpeakerSelectionResult containing the selected participant and reasoning

        Raises:
            ValueError: If participants list is empty or no valid participant found
        """
        if not participants:
            raise ValueError("Cannot select first speaker from empty participant list")

        # If only one participant, select them directly
        if len(participants) == 1:
            return SpeakerSelectionResult(
                participant=participants[0],
                reasoning=f"Only one participant ({participants[0].name}) available to speak.",
            )

        # Build prompts for LLM selection
        system_prompt = self._build_selection_system_prompt()
        user_prompt = self._build_selection_user_prompt(
            participants=participants,
            question=question,
            reactions=reactions,
        )

        # Get LLM decision
        response = await self._provider.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            json_schema=_selection_schema(
                [p.name for p in participants], allow_nobody=False
            ),
        )

        # Parse the response to get selected participant
        selected_name, reasoning = self._parse_selection_response(
            response=response,
            participants=participants,
        )

        # Find the participant object
        selected_participant = self._find_participant_by_name(
            name=selected_name,
            participants=participants,
        )

        return SpeakerSelectionResult(
            participant=selected_participant,
            reasoning=reasoning,
        )

    def _build_selection_system_prompt(self) -> str:
        """Build the system prompt for speaker selection.

        Returns:
            The system prompt establishing the selection task
        """
        return """You are an expert moderator analyzing focus group dynamics.

Your task is to determine which participant would naturally speak FIRST after a question is asked, based on their non-verbal reactions.

In real focus groups, certain body language signals indicate someone is ready or eager to speak:
- Leaning forward, sitting up straight
- Making eye contact with the moderator
- Nodding, showing clear engagement
- Open posture, uncrossed arms
- Visible excitement or strong reaction

Other signals suggest hesitation:
- Leaning back, crossed arms
- Looking down or away
- Frowning, appearing confused
- Settling into seat, getting comfortable to listen

Consider both the non-verbal cues AND the participant's personality traits. Extroverts are generally quicker to speak, while introverts may wait even if interested.

You MUST respond in this exact JSON format:
{
    "selected": "Participant Name",
    "reasoning": "Brief explanation of why this person would speak first based on their body language and personality"
}

Choose based on realistic focus group dynamics. Not everyone is equally eager to speak first."""

    def _build_selection_user_prompt(
        self,
        participants: list[Participant],
        question: str,
        reactions: dict[str, str],
    ) -> str:
        """Build the user prompt for speaker selection.

        Args:
            participants: The participants to choose from
            question: The question that was asked
            reactions: Non-verbal reactions of each participant

        Returns:
            The user prompt containing reactions and participant info
        """
        # Build participant info section
        participant_info = []
        for p in participants:
            persona = p.persona
            traits = ", ".join(persona.personality_traits)
            reaction = reactions.get(p.name, "no visible reaction")

            participant_info.append(
                f"**{p.name}** (Age: {persona.age}, Pronouns: {persona.pronouns})\n"
                f"  - Personality: {traits}\n"
                f"  - Communication style: {persona.communication_style}\n"
                f"  - Non-verbal reaction: {reaction}"
            )

        participants_section = "\n\n".join(participant_info)

        return f"""The moderator just asked the focus group:

"{question}"

Here are the participants and their non-verbal reactions:

{participants_section}

Based on these non-verbal cues and personality traits, who would naturally speak FIRST?

Remember to respond in JSON format with "selected" and "reasoning" fields."""

    def _parse_selection_response(
        self,
        response: str,
        participants: list[Participant],
    ) -> tuple[str, str]:
        """Parse the LLM response to extract selected participant and reasoning.

        Args:
            response: The raw LLM response
            participants: List of valid participants

        Returns:
            Tuple of (selected_name, reasoning)

        Raises:
            ValueError: If response cannot be parsed or contains invalid participant
        """
        data = _extract_json_object(response)
        if data is not None:
            selected = _str_field(data, "selected")
            reasoning = _str_field(data, "reasoning") or _NO_REASONING
            if selected:
                # Verify the selected name matches a participant
                matched_name = self._match_participant_name(selected, participants)
                if matched_name:
                    return matched_name, reasoning

        logger.warning(
            "First-speaker selection could not use the LLM reply, falling back: %r",
            response[:_LOGGED_REPLY_CHARS],
        )

        # Fallback: try to find a participant name mentioned in the response
        for p in participants:
            if p.name.lower() in response.lower():
                return p.name, f"Selected based on mention in response: {response[:200]}"

        # Last resort: select the first participant
        return participants[0].name, "Fallback selection (could not parse LLM response)"

    def _match_participant_name(
        self,
        name: str,
        participants: list[Participant],
    ) -> str | None:
        """Match a name string to a participant (handles variations).

        Args:
            name: The name to match (may be partial or slightly different)
            participants: List of participants to match against

        Returns:
            The matched participant name, or None if no match
        """
        name_lower = name.lower().strip()

        # Try exact match first
        for p in participants:
            if p.name.lower() == name_lower:
                return p.name

        # Try partial match (first name or last name)
        for p in participants:
            name_parts = p.name.lower().split()
            if name_lower in name_parts:
                return p.name
            # Check if input contains any part of the name
            for part in name_parts:
                if part in name_lower or name_lower in part:
                    return p.name

        return None

    def _find_participant_by_name(
        self,
        name: str,
        participants: list[Participant],
    ) -> Participant:
        """Find a participant by name.

        Args:
            name: The participant's name
            participants: List of participants to search

        Returns:
            The matching Participant object

        Raises:
            ValueError: If no participant with that name is found
        """
        for p in participants:
            if p.name == name:
                return p

        # Should not happen if _parse_selection_response worked correctly
        raise ValueError(f"Participant '{name}' not found in participant list")


class NextSpeakerSelector:
    """Selects the next speaker during a discussion based on non-verbal reactions.

    This class analyzes non-verbal reactions to determine which participant
    wants to respond to what was just said. Unlike FirstSpeakerSelector, this
    class can return None when no one appears eager to speak, and it excludes
    the current speaker from selection.

    The selection considers:
    - Strong reactions (agreement, disagreement, surprise) suggesting desire to speak
    - Body language indicating desire to interject (raised hand, leaning forward)
    - Personality traits (some people are more likely to respond)

    Attributes:
        provider: The LLM provider used for making selection decisions

    Example:
        >>> selector = NextSpeakerSelector(llm_provider)
        >>> reactions = {
        ...     "John Smith": "nods vigorously and raises hand slightly",
        ...     "Maria Garcia": "leans back, appearing satisfied"
        ... }
        >>> result = await selector.select_next_speaker(
        ...     participants=[sarah, john, maria],
        ...     current_speaker=sarah,
        ...     statement="I think remote work improved my work-life balance.",
        ...     reactions=reactions
        ... )
        >>> if result:
        ...     print(result.participant.name)  # "John Smith"
        ... else:
        ...     print("No one wants to speak")
    """

    def __init__(self, provider: BaseLLMProvider) -> None:
        """Initialize the next speaker selector.

        Args:
            provider: The LLM provider to use for selection decisions
        """
        self._provider = provider

    async def select_next_speaker(
        self,
        participants: list[Participant],
        current_speaker: Participant,
        statement: str,
        reactions: dict[str, str],
        already_spoken: set[str] | None = None,
    ) -> SpeakerSelectionResult | None:
        """Select which participant should speak next based on reactions.

        Analyzes the non-verbal reactions of participants (excluding the current
        speaker and those who have already spoken) to determine if anyone wants
        to respond. Returns None if no one appears eager to speak.

        Args:
            participants: List of all participants in the focus group
            current_speaker: The participant who just spoke (excluded from selection)
            statement: What the current speaker just said
            reactions: Dictionary mapping participant names to their non-verbal
                reactions to the statement (should not include current speaker)
            already_spoken: Optional set of participant names who have already
                spoken on this question (excluded from selection to ensure balance)

        Returns:
            SpeakerSelectionResult if someone wants to speak, None otherwise

        Raises:
            ValueError: If participants list is empty or only contains current speaker
        """
        # Filter out the current speaker AND those who have already spoken
        eligible_participants = [
            p
            for p in participants
            if p.name != current_speaker.name
            and (already_spoken is None or p.name not in already_spoken)
        ]

        if not eligible_participants:
            raise ValueError("No eligible participants to select from (all excluded)")

        # If only one eligible participant, check if they want to speak
        if len(eligible_participants) == 1:
            participant = eligible_participants[0]
            reaction = reactions.get(participant.name, "")
            if self._reaction_suggests_desire_to_speak(reaction):
                return SpeakerSelectionResult(
                    participant=participant,
                    reasoning=f"Only eligible participant ({participant.name}) shows desire to speak.",
                )
            return None

        # Build prompts for LLM selection
        system_prompt = self._build_selection_system_prompt()
        user_prompt = self._build_selection_user_prompt(
            eligible_participants=eligible_participants,
            current_speaker=current_speaker,
            statement=statement,
            reactions=reactions,
        )

        # Get LLM decision
        response = await self._provider.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            json_schema=_selection_schema(
                [p.name for p in eligible_participants], allow_nobody=True
            ),
        )

        # Parse the response
        return self._parse_selection_response(
            response=response,
            eligible_participants=eligible_participants,
        )

    async def select_followup_speaker(
        self,
        participants: list[Participant],
        current_speaker: Participant,
        statement: str,
        reactions: dict[str, str],
    ) -> SpeakerSelectionResult | None:
        """Select a follow-up speaker using a permissive assessment.

        Used after everyone has spoken once to determine if anyone wants
        to add a follow-up thought. Uses more generous criteria than
        select_next_speaker to encourage natural follow-up rounds.

        Args:
            participants: List of all participants in the focus group
            current_speaker: The participant who just spoke (excluded from selection)
            statement: What the current speaker just said
            reactions: Dictionary mapping participant names to their non-verbal
                reactions to the statement

        Returns:
            SpeakerSelectionResult if someone wants to follow up, None otherwise
        """
        eligible_participants = [p for p in participants if p.name != current_speaker.name]

        if not eligible_participants:
            return None

        if len(eligible_participants) == 1:
            participant = eligible_participants[0]
            reaction = reactions.get(participant.name, "")
            if self._reaction_suggests_desire_to_speak(reaction):
                return SpeakerSelectionResult(
                    participant=participant,
                    reasoning=f"{participant.name} shows engagement for a follow-up.",
                )
            return None

        system_prompt = self._build_followup_system_prompt()
        user_prompt = self._build_selection_user_prompt(
            eligible_participants=eligible_participants,
            current_speaker=current_speaker,
            statement=statement,
            reactions=reactions,
        )

        response = await self._provider.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            json_schema=_selection_schema(
                [p.name for p in eligible_participants], allow_nobody=True
            ),
        )

        return self._parse_selection_response(
            response=response,
            eligible_participants=eligible_participants,
        )

    def _build_followup_system_prompt(self) -> str:
        """Build a permissive system prompt for follow-up speaker selection.

        Returns:
            The system prompt with generous assessment criteria
        """
        return """You are an expert moderator analyzing focus group dynamics.

Everyone has now spoken once on this question. Your task is to determine if anyone would naturally want to add a follow-up thought based on their non-verbal reactions.

Be SELECTIVE — only select someone who shows a genuinely strong reaction, not just polite engagement. Say "no one" unless someone looks like they would interrupt in a real conversation. Mild nodding or relaxed posture = no follow-up needed.

Signs someone GENUINELY wants to follow up:
- Leaning forward urgently, raised hand, or opening mouth to speak
- Visible disagreement: shaking head, frowning, crossing arms after hearing something
- Strong surprise or excitement that demands a verbal response
- Fidgeting impatiently as if they can barely wait to respond

Signs that do NOT warrant a follow-up:
- Gentle nodding, relaxed posture, satisfied expression
- Thoughtful processing, looking contemplative
- Polite engagement without urgency

You MUST respond in this exact JSON format:
{
    "someone_wants_to_speak": true or false,
    "selected": "Participant Name" or null,
    "reasoning": "Brief explanation based on body language"
}"""

    def _reaction_suggests_desire_to_speak(self, reaction: str) -> bool:
        """Check if a reaction suggests the participant wants to speak.

        Simple heuristic check for common indicators of desire to speak.

        Args:
            reaction: The non-verbal reaction description

        Returns:
            True if the reaction suggests desire to speak
        """
        if not reaction:
            return False

        reaction_lower = reaction.lower()

        # Positive indicators (wanting to speak)
        positive_indicators = [
            "leans forward",
            "raises hand",
            "nods vigorously",
            "opens mouth",
            "makes eye contact",
            "sits up",
            "gestures",
            "eager",
            "animated",
            "excited",
            "wants to speak",
            "ready to respond",
        ]

        # Negative indicators (not wanting to speak)
        negative_indicators = [
            "leans back",
            "crosses arms",
            "looks away",
            "satisfied",
            "settles",
            "relaxes",
            "passive",
            "contemplative",
            "listening",
        ]

        positive_count = sum(1 for ind in positive_indicators if ind in reaction_lower)
        negative_count = sum(1 for ind in negative_indicators if ind in reaction_lower)

        return positive_count > negative_count

    def _build_selection_system_prompt(self) -> str:
        """Build the system prompt for next speaker selection.

        Returns:
            The system prompt establishing the selection task
        """
        return """You are an expert moderator analyzing focus group dynamics.

Your task is to determine if anyone wants to speak next after hearing another participant's statement, based on their non-verbal reactions.

IMPORTANT: Not everyone will want to respond to every statement. It's common and realistic for there to be pauses where no one immediately jumps in.

Body language signals suggesting someone WANTS to speak:
- Leaning forward, raising hand slightly
- Nodding vigorously (especially while disagreeing or having something to add)
- Making eye contact with the speaker or moderator
- Opening mouth as if about to speak
- Visible excitement, surprise, or strong disagreement
- Shifting in seat as if preparing to speak

Body language signals suggesting someone does NOT want to speak yet:
- Leaning back, appearing satisfied or contemplative
- Nodding gently in agreement (agreeing doesn't mean wanting to add)
- Looking thoughtful or processing
- Crossed arms, passive posture
- Looking down at notes
- Relaxed, settled posture

You MUST respond in this exact JSON format:
{
    "someone_wants_to_speak": true or false,
    "selected": "Participant Name" or null,
    "reasoning": "Brief explanation based on body language"
}

If no one shows clear desire to speak, set "someone_wants_to_speak" to false and "selected" to null.
Be realistic - in real focus groups, there are natural pauses between speakers."""

    def _build_selection_user_prompt(
        self,
        eligible_participants: list[Participant],
        current_speaker: Participant,
        statement: str,
        reactions: dict[str, str],
    ) -> str:
        """Build the user prompt for next speaker selection.

        Args:
            eligible_participants: Participants who could speak (excluding current)
            current_speaker: The participant who just spoke
            statement: What they said
            reactions: Non-verbal reactions of eligible participants

        Returns:
            The user prompt containing the statement and reactions
        """
        # Build participant info section
        participant_info = []
        for p in eligible_participants:
            persona = p.persona
            traits = ", ".join(persona.personality_traits)
            reaction = reactions.get(p.name, "no visible reaction")

            participant_info.append(
                f"**{p.name}** (Age: {persona.age}, Pronouns: {persona.pronouns})\n"
                f"  - Personality: {traits}\n"
                f"  - Communication style: {persona.communication_style}\n"
                f"  - Non-verbal reaction: {reaction}"
            )

        participants_section = "\n\n".join(participant_info)

        return f"""{current_speaker.name} just said:

"{statement}"

Here are the other participants and their non-verbal reactions:

{participants_section}

Based on these non-verbal cues, does anyone want to speak next? If so, who?

Remember:
- Set "someone_wants_to_speak" to false if no one shows clear desire to respond
- Only select someone if their body language clearly indicates they want to speak
- Respond in JSON format"""

    def _parse_selection_response(
        self,
        response: str,
        eligible_participants: list[Participant],
    ) -> SpeakerSelectionResult | None:
        """Parse the LLM response to determine if/who should speak next.

        Args:
            response: The raw LLM response
            eligible_participants: List of valid participants to select from

        Returns:
            SpeakerSelectionResult if someone selected, None if no one wants to speak
        """
        data = _extract_json_object(response)
        if data is not None and "someone_wants_to_speak" in data:
            # An explicit "no one" is a valid answer, not a fallback
            if not data["someone_wants_to_speak"]:
                return None

            raw_selected = data.get("selected")
            if raw_selected is None or (
                isinstance(raw_selected, str)
                and raw_selected.strip().lower() in ("", *_NOBODY_VALUES)
            ):
                return None

            selected = _str_field(data, "selected")
            reasoning = _str_field(data, "reasoning") or _NO_REASONING

            # Verify the selected name matches an eligible participant
            if selected:
                matched_name = self._match_participant_name(selected, eligible_participants)
                if matched_name:
                    participant = self._find_participant_by_name(
                        matched_name, eligible_participants
                    )
                    return SpeakerSelectionResult(
                        participant=participant,
                        reasoning=reasoning,
                    )

        logger.warning(
            "Next-speaker selection could not use the LLM reply, falling back: %r",
            response[:_LOGGED_REPLY_CHARS],
        )

        # Fallback: check if response clearly indicates no one wants to speak
        response_lower = response.lower()
        no_speaker_indicators = [
            "no one",
            "nobody",
            "none of",
            "doesn't want",
            "don't want",
            "not eager",
            "false",
        ]
        if any(indicator in response_lower for indicator in no_speaker_indicators):
            return None

        # Try to find a participant name mentioned in the response
        for p in eligible_participants:
            if p.name.lower() in response_lower:
                return SpeakerSelectionResult(
                    participant=p,
                    reasoning=f"Selected based on mention in response: {response[:200]}",
                )

        # Default: no one wants to speak (conservative choice)
        return None

    def _match_participant_name(
        self,
        name: str,
        participants: list[Participant],
    ) -> str | None:
        """Match a name string to a participant (handles variations).

        Args:
            name: The name to match (may be partial or slightly different)
            participants: List of participants to match against

        Returns:
            The matched participant name, or None if no match
        """
        name_lower = name.lower().strip()

        # Try exact match first
        for p in participants:
            if p.name.lower() == name_lower:
                return p.name

        # Try partial match (first name or last name)
        for p in participants:
            name_parts = p.name.lower().split()
            if name_lower in name_parts:
                return p.name
            # Check if input contains any part of the name
            for part in name_parts:
                if part in name_lower or name_lower in part:
                    return p.name

        return None

    def _find_participant_by_name(
        self,
        name: str,
        participants: list[Participant],
    ) -> Participant:
        """Find a participant by name.

        Args:
            name: The participant's name
            participants: List of participants to search

        Returns:
            The matching Participant object

        Raises:
            ValueError: If no participant with that name is found
        """
        for p in participants:
            if p.name == name:
                return p

        raise ValueError(f"Participant '{name}' not found in participant list")


class SilenceHandler:
    """Handles situations when no one wants to speak, selecting a reluctant speaker.

    This class is used when NextSpeakerSelector returns None (no one volunteers)
    but not everyone has spoken on the current question yet. It selects someone
    who hasn't spoken, considering who seems most likely to contribute after
    enough silence has passed.

    Unlike NextSpeakerSelector, this class ALWAYS returns a speaker - it never
    returns None. The selection prioritizes:
    1. Participants who haven't spoken on this question
    2. Non-verbal cues suggesting they have something to say but are hesitant
    3. Personality traits (introverts may need encouragement)

    Attributes:
        provider: The LLM provider used for making selection decisions

    Example:
        >>> handler = SilenceHandler(llm_provider)
        >>> already_spoken = {"Sarah Chen", "John Smith"}
        >>> reactions = {
        ...     "Maria Garcia": "shifts slightly, glances at notes",
        ...     "Alex Kim": "looks thoughtful, taps pen"
        ... }
        >>> result = await handler.select_reluctant_speaker(
        ...     participants=[sarah, john, maria, alex],
        ...     question="What do you think about remote work?",
        ...     reactions=reactions,
        ...     already_spoken=already_spoken
        ... )
        >>> print(result.participant.name)  # "Maria Garcia" or "Alex Kim"
    """

    def __init__(self, provider: BaseLLMProvider) -> None:
        """Initialize the silence handler.

        Args:
            provider: The LLM provider to use for selection decisions
        """
        self._provider = provider

    async def select_reluctant_speaker(
        self,
        participants: list[Participant],
        question: str,
        reactions: dict[str, str],
        already_spoken: set[str],
    ) -> SpeakerSelectionResult:
        """Select a speaker when no one volunteers, prioritizing those who haven't spoken.

        This method is called when there's silence after a response and no one
        seems eager to speak, but not everyone has contributed yet. It selects
        someone who hasn't spoken on this question to ensure everyone participates.

        The selection considers:
        - Who hasn't spoken yet on this question (highest priority)
        - Non-verbal cues suggesting hesitation vs. disengagement
        - Personality traits (some need more encouragement)
        - Who seems most ready/least uncomfortable

        Args:
            participants: List of all participants in the focus group
            question: The current focus group question being discussed
            reactions: Dictionary mapping participant names to their current
                non-verbal state/reaction
            already_spoken: Set of participant names who have already spoken
                on this question

        Returns:
            SpeakerSelectionResult containing the selected participant and reasoning.
            ALWAYS returns a valid result (never None).

        Raises:
            ValueError: If participants list is empty or all have already spoken
        """
        if not participants:
            raise ValueError("Cannot select speaker from empty participant list")

        # Filter to those who haven't spoken
        not_spoken = [p for p in participants if p.name not in already_spoken]

        if not not_spoken:
            raise ValueError(
                "All participants have already spoken on this question. "
                "Use this handler only when some participants haven't spoken."
            )

        # If only one person hasn't spoken, select them directly
        if len(not_spoken) == 1:
            participant = not_spoken[0]
            return SpeakerSelectionResult(
                participant=participant,
                reasoning=(
                    f"{participant.name} is the only participant who hasn't spoken "
                    f"on this question yet."
                ),
            )

        # Build prompts for LLM selection
        system_prompt = self._build_selection_system_prompt()
        user_prompt = self._build_selection_user_prompt(
            not_spoken=not_spoken,
            question=question,
            reactions=reactions,
        )

        # Get LLM decision
        response = await self._provider.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            json_schema=_selection_schema([p.name for p in not_spoken], allow_nobody=False),
        )

        # Parse the response to get selected participant
        selected_name, reasoning = self._parse_selection_response(
            response=response,
            not_spoken=not_spoken,
        )

        # Find the participant object
        selected_participant = self._find_participant_by_name(
            name=selected_name,
            participants=not_spoken,
        )

        return SpeakerSelectionResult(
            participant=selected_participant,
            reasoning=reasoning,
        )

    def _build_selection_system_prompt(self) -> str:
        """Build the system prompt for reluctant speaker selection.

        Returns:
            The system prompt establishing the selection task
        """
        return """You are an expert moderator analyzing focus group dynamics.

The discussion has reached a pause where no one is immediately volunteering to speak, but not everyone has shared their thoughts yet. Your task is to determine which participant would be MOST LIKELY to speak after a few moments of silence, among those who haven't spoken.

In real focus groups, moderators recognize subtle cues that indicate someone has something to say but is hesitant:
- Shifting in seat, appearing to want to contribute
- Glancing at notes or the moderator
- Subtle gestures suggesting internal thought
- Opening mouth slightly as if considering speaking
- Looking engaged but uncertain

Some participants need more time or encouragement:
- Introverts may have valuable insights but wait for a natural opening
- Those who communicate thoughtfully may be formulating their response
- Quiet personalities may be waiting to be invited

You MUST select someone - the goal is to ensure everyone participates. Consider:
1. Who seems to have something on their mind (even if hesitant)
2. Who would be most comfortable being called upon
3. Personality traits that suggest they'd respond well to encouragement

You MUST respond in this exact JSON format:
{
    "selected": "Participant Name",
    "reasoning": "Brief explanation of why this person seems most ready to speak with encouragement"
}

IMPORTANT: You must always select someone. Never say no one should speak."""

    def _build_selection_user_prompt(
        self,
        not_spoken: list[Participant],
        question: str,
        reactions: dict[str, str],
    ) -> str:
        """Build the user prompt for reluctant speaker selection.

        Args:
            not_spoken: Participants who haven't spoken on this question
            question: The current question being discussed
            reactions: Current non-verbal states of participants

        Returns:
            The user prompt containing participant info and reactions
        """
        # Build participant info section
        participant_info = []
        for p in not_spoken:
            persona = p.persona
            traits = ", ".join(persona.personality_traits)
            reaction = reactions.get(p.name, "no visible reaction")

            participant_info.append(
                f"**{p.name}** (Age: {persona.age}, Pronouns: {persona.pronouns})\n"
                f"  - Personality: {traits}\n"
                f"  - Communication style: {persona.communication_style}\n"
                f"  - Current non-verbal state: {reaction}"
            )

        participants_section = "\n\n".join(participant_info)

        return f"""The focus group is discussing:

"{question}"

There's been a pause in the conversation. No one is immediately volunteering to speak, but the following participants haven't shared their thoughts on this question yet:

{participants_section}

Based on their personalities and current non-verbal states, which participant would be most likely to speak if given a moment of silence and gentle encouragement?

You MUST select one of these participants. Respond in JSON format with "selected" and "reasoning" fields."""

    def _parse_selection_response(
        self,
        response: str,
        not_spoken: list[Participant],
    ) -> tuple[str, str]:
        """Parse the LLM response to extract selected participant and reasoning.

        Args:
            response: The raw LLM response
            not_spoken: List of valid participants who haven't spoken

        Returns:
            Tuple of (selected_name, reasoning)
        """
        data = _extract_json_object(response)
        if data is not None:
            selected = _str_field(data, "selected")
            reasoning = _str_field(data, "reasoning") or _NO_REASONING
            if selected:
                # Verify the selected name matches a participant
                matched_name = self._match_participant_name(selected, not_spoken)
                if matched_name:
                    return matched_name, reasoning

        logger.warning(
            "Reluctant-speaker selection could not use the LLM reply, falling back: %r",
            response[:_LOGGED_REPLY_CHARS],
        )

        # Fallback: try to find a participant name mentioned in the response
        for p in not_spoken:
            if p.name.lower() in response.lower():
                return p.name, f"Selected based on mention in response: {response[:200]}"

        # Last resort: select based on personality (prefer those with more
        # outgoing traits, as they're more likely to respond to encouragement)
        return self._select_most_likely_to_respond(not_spoken)

    def _select_most_likely_to_respond(
        self,
        participants: list[Participant],
    ) -> tuple[str, str]:
        """Select the participant most likely to respond to encouragement.

        Used as a fallback when LLM response cannot be parsed.
        Prefers participants with more outgoing personality traits.

        Args:
            participants: List of participants to choose from

        Returns:
            Tuple of (selected_name, reasoning)
        """
        # Traits that suggest openness to being called upon
        positive_traits = {
            "friendly",
            "outgoing",
            "sociable",
            "talkative",
            "warm",
            "expressive",
            "engaging",
            "confident",
            "articulate",
            "direct",
            "open",
            "enthusiastic",
            "communicative",
        }

        best_match = participants[0]
        best_score = 0

        for p in participants:
            score = sum(
                1
                for trait in p.persona.personality_traits
                if any(pos in trait.lower() for pos in positive_traits)
            )
            if score > best_score:
                best_score = score
                best_match = p

        return (
            best_match.name,
            "Fallback selection based on personality traits suggesting openness to encouragement.",
        )

    def _match_participant_name(
        self,
        name: str,
        participants: list[Participant],
    ) -> str | None:
        """Match a name string to a participant (handles variations).

        Args:
            name: The name to match (may be partial or slightly different)
            participants: List of participants to match against

        Returns:
            The matched participant name, or None if no match
        """
        name_lower = name.lower().strip()

        # Try exact match first
        for p in participants:
            if p.name.lower() == name_lower:
                return p.name

        # Try partial match (first name or last name)
        for p in participants:
            name_parts = p.name.lower().split()
            if name_lower in name_parts:
                return p.name
            # Check if input contains any part of the name
            for part in name_parts:
                if part in name_lower or name_lower in part:
                    return p.name

        return None

    def _find_participant_by_name(
        self,
        name: str,
        participants: list[Participant],
    ) -> Participant:
        """Find a participant by name.

        Args:
            name: The participant's name
            participants: List of participants to search

        Returns:
            The matching Participant object

        Raises:
            ValueError: If no participant with that name is found
        """
        for p in participants:
            if p.name == name:
                return p

        # Should not happen if _parse_selection_response worked correctly
        raise ValueError(f"Participant '{name}' not found in participant list")
