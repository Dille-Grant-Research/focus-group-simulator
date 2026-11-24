"""Non-verbal reaction generator for focus group participants.

This module provides the NonVerbalGenerator class that uses an LLM to generate
non-verbal reactions (body language, facial expressions) from participants
in response to questions and statements.
"""

import asyncio
import logging

from focus_group.core.participant import Participant
from focus_group.llm.base import BaseLLMProvider


logger = logging.getLogger(__name__)


class NonVerbalGenerator:
    """Generates non-verbal reactions from focus group participants.

    This class uses an LLM to generate realistic non-verbal cues based on
    the participant's persona and their reaction to stimuli (questions or
    statements from others). Non-verbal reactions are added to the full
    context but not the verbal-only context.

    Non-verbal reactions include:
    - Facial expressions (smiling, frowning, raised eyebrows)
    - Body language (leaning forward, crossing arms, nodding)
    - Gestures and movements (shifting in seat, making eye contact)

    Attributes:
        provider: The LLM provider used for generating reactions

    Example:
        >>> generator = NonVerbalGenerator(llm_provider)
        >>> reaction = await generator.generate_reaction_to_question(
        ...     participant=sarah,
        ...     question="What do you think about remote work?"
        ... )
        >>> print(reaction)
        "Sarah leans forward slightly, her eyes brightening with interest."
    """

    def __init__(self, provider: BaseLLMProvider) -> None:
        """Initialize the non-verbal generator.

        Args:
            provider: The LLM provider to use for generating reactions
        """
        self._provider = provider
        self._reaction_history: dict[str, list[str]] = {}
        self._max_history: int = 5

    def _record_reaction(self, participant_name: str, reaction: str) -> None:
        """Record a reaction in the participant's history.

        Args:
            participant_name: The participant who reacted
            reaction: The reaction text to record
        """
        if participant_name not in self._reaction_history:
            self._reaction_history[participant_name] = []
        self._reaction_history[participant_name].append(reaction)
        # Cap at max_history
        if len(self._reaction_history[participant_name]) > self._max_history:
            self._reaction_history[participant_name] = self._reaction_history[participant_name][
                -self._max_history :
            ]

    def _get_recent_reactions(self, participant_name: str) -> list[str]:
        """Get recent reactions for a participant.

        Args:
            participant_name: The participant to look up

        Returns:
            List of recent reaction strings (empty if none)
        """
        return self._reaction_history.get(participant_name, [])

    async def generate_reaction_to_question(
        self,
        participant: Participant,
        question: str,
    ) -> str:
        """Generate a non-verbal reaction to a focus group question.

        This method generates the participant's non-verbal reaction upon
        hearing a new question from the moderator. The reaction reflects
        their personality and feelings about the topic.

        Args:
            participant: The participant reacting to the question
            question: The focus group question being asked

        Returns:
            A brief description of the non-verbal reaction
        """
        system_prompt = self._build_reaction_system_prompt(participant)
        user_prompt = self._build_question_reaction_prompt(participant, question)

        response = await self._provider.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
        )

        cleaned = self._clean_reaction(response, participant.name)
        self._record_reaction(participant.name, cleaned)
        return cleaned

    async def generate_reaction_to_statement(
        self,
        participant: Participant,
        speaker_name: str,
        statement: str,
    ) -> str:
        """Generate a non-verbal reaction to another participant's statement.

        This method generates the participant's non-verbal reaction when
        hearing something another participant said. The reaction reflects
        their agreement, disagreement, interest, or other emotions.

        Args:
            participant: The participant reacting to the statement
            speaker_name: The name of the person who made the statement
            statement: What the other participant said

        Returns:
            A brief description of the non-verbal reaction
        """
        system_prompt = self._build_reaction_system_prompt(participant)
        user_prompt = self._build_statement_reaction_prompt(participant, speaker_name, statement)

        response = await self._provider.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
        )

        cleaned = self._clean_reaction(response, participant.name)
        self._record_reaction(participant.name, cleaned)
        return cleaned

    async def generate_reactions_parallel(
        self,
        participants: list[Participant],
        stimulus: str,
        stimulus_type: str,
        speaker_name: str | None = None,
    ) -> dict[str, str]:
        """Generate non-verbal reactions for multiple participants in parallel.

        This method efficiently generates reactions for all participants
        simultaneously using asyncio.gather.

        Args:
            participants: List of participants to generate reactions for
            stimulus: The question or statement to react to
            stimulus_type: Either "question" or "statement"
            speaker_name: Name of the speaker (required if stimulus_type is "statement")

        Returns:
            Dictionary mapping participant names to their reactions

        Raises:
            ValueError: If stimulus_type is "statement" but speaker_name is not provided
        """
        if stimulus_type == "statement" and speaker_name is None:
            raise ValueError("speaker_name is required when stimulus_type is 'statement'")

        # Create tasks for parallel execution
        if stimulus_type == "question":
            tasks = [self.generate_reaction_to_question(p, stimulus) for p in participants]
        else:
            tasks = [
                self.generate_reaction_to_statement(p, speaker_name, stimulus) for p in participants
            ]

        # Execute all tasks in parallel. A failed reaction becomes "" (callers skip
        # empty reactions) so one dead request doesn't wipe out the whole question.
        outcomes = await asyncio.gather(*tasks, return_exceptions=True)

        reactions: dict[str, str] = {}
        for participant, outcome in zip(participants, outcomes):
            if isinstance(outcome, BaseException):
                if not isinstance(outcome, Exception):
                    raise outcome  # cancellation and other control flow must propagate
                logger.warning(
                    "Non-verbal reaction failed for %s (%s): %s",
                    participant.name,
                    stimulus_type,
                    outcome,
                )
                reactions[participant.name] = ""
            else:
                reactions[participant.name] = outcome
        return reactions

    async def generate_reactions_to_question_parallel(
        self,
        participants: list[Participant],
        question: str,
    ) -> dict[str, str]:
        """Generate reactions to a question for multiple participants in parallel.

        Convenience method for generating question reactions in parallel.

        Args:
            participants: List of participants to generate reactions for
            question: The focus group question

        Returns:
            Dictionary mapping participant names to their reactions
        """
        return await self.generate_reactions_parallel(
            participants=participants,
            stimulus=question,
            stimulus_type="question",
        )

    async def generate_reactions_to_statement_parallel(
        self,
        participants: list[Participant],
        speaker_name: str,
        statement: str,
    ) -> dict[str, str]:
        """Generate reactions to a statement for multiple participants in parallel.

        Convenience method for generating statement reactions in parallel.

        Args:
            participants: List of participants to generate reactions for
            speaker_name: Name of the person who made the statement
            statement: What the other participant said

        Returns:
            Dictionary mapping participant names to their reactions
        """
        return await self.generate_reactions_parallel(
            participants=participants,
            stimulus=statement,
            stimulus_type="statement",
            speaker_name=speaker_name,
        )

    def _build_reaction_system_prompt(self, participant: Participant) -> str:
        """Build the system prompt for generating non-verbal reactions.

        Args:
            participant: The participant generating the reaction

        Returns:
            The system prompt establishing persona and behavior
        """
        persona = participant.persona

        return f"""You are describing the non-verbal reactions of {persona.name} (pronouns: {persona.pronouns}), a {persona.age}-year-old focus group participant.

About {persona.name}:
- Background: {persona.background}
- Personality: {', '.join(persona.personality_traits)}
- Communication style: {persona.communication_style}

Your task is to describe their non-verbal reaction - body language, facial expressions, and gestures that others in the room would observe.

IMPORTANT INSTRUCTIONS:
- ALWAYS use {persona.pronouns} pronouns when referring to {persona.name}
- Describe only observable, non-verbal behaviors (NOT thoughts or internal feelings)
- Keep reactions brief (1-2 sentences maximum)
- Write in third person (e.g., "{persona.name} leans forward...")
- Reactions should reflect their personality (e.g., introverts may be more subtle)
- Use DIVERSE physical reactions. AVOID overused defaults: do NOT repeatedly use "raises an eyebrow", "tilts head to the side", "leans forward", or "crosses arms"
- Choose from a wider vocabulary: tapping fingers on table, adjusting glasses, shifting weight in chair, pressing lips together, drumming on armrest, tucking hair behind ear, rubbing hands together, looking up at the ceiling, fidgeting with a pen, exhaling audibly, resting chin on hand, glancing sideways at another participant, pursing lips, uncrossing legs, picking at a fingernail, straightening in seat, rubbing the back of their neck
- Match reaction intensity to personality: introverts show subtle reactions, extroverts show bigger ones
- Do NOT include any spoken words or dialogue
- Do NOT include internal thoughts or feelings that couldn't be observed
- CRITICAL: Only reference facts explicitly stated in the persona description above
- Do NOT invent or assume details about the participant's family, children, relationships, hobbies, or experiences unless they are EXPLICITLY mentioned in their background
- If the participant's background does not mention children, do NOT describe reactions involving children or parenting
- Base ALL reactions strictly on what is described in the persona"""

    def _build_question_reaction_prompt(
        self,
        participant: Participant,
        question: str,
    ) -> str:
        """Build the prompt for generating a reaction to a question.

        Args:
            participant: The participant reacting
            question: The question being asked

        Returns:
            The user prompt for question reaction
        """
        recent = self._get_recent_reactions(participant.name)
        history_block = ""
        if recent:
            recent_list = "\n".join(f"  - {r}" for r in recent[-3:])
            history_block = (
                f"\n\nIMPORTANT: {participant.name} has recently shown these reactions:\n"
                f"{recent_list}\n"
                f"Generate a COMPLETELY DIFFERENT physical reaction this time.\n"
                f"Do NOT repeat any of the gestures or postures listed above."
            )

        return f"""The moderator just asked the focus group this question:

"{question}"

Describe {participant.name}'s immediate non-verbal reaction upon hearing this question. What do they do with their body, face, or hands? How do they physically respond?

Choose a reaction that feels SPECIFIC to this person and this moment — avoid generic defaults.

Remember: Only describe observable behaviors, no spoken words or thoughts.{history_block}"""

    def _build_statement_reaction_prompt(
        self,
        participant: Participant,
        speaker_name: str,
        statement: str,
    ) -> str:
        """Build the prompt for generating a reaction to a statement.

        Args:
            participant: The participant reacting
            speaker_name: Who made the statement
            statement: What they said

        Returns:
            The user prompt for statement reaction
        """
        recent = self._get_recent_reactions(participant.name)
        history_block = ""
        if recent:
            recent_list = "\n".join(f"  - {r}" for r in recent[-3:])
            history_block = (
                f"\n\nIMPORTANT: {participant.name} has recently shown these reactions:\n"
                f"{recent_list}\n"
                f"Generate a COMPLETELY DIFFERENT physical reaction this time.\n"
                f"Do NOT repeat any of the gestures or postures listed above."
            )

        return f"""{speaker_name} just said:

"{statement}"

Describe {participant.name}'s non-verbal reaction to hearing this. Do they agree? Disagree? Feel surprised, interested, or uncomfortable? Show this through their body language and facial expression.

Choose a reaction that feels SPECIFIC to this person's opinion of what was said — avoid generic defaults.

Occasionally, if the participant has a strong opinion, show physical signs of wanting to add something — such as shifting forward, opening mouth slightly, or making direct eye contact with the speaker.

Remember: Only describe observable behaviors, no spoken words or thoughts.{history_block}"""

    def _clean_reaction(self, response: str, participant_name: str) -> str:
        """Clean and normalize the LLM response.

        Ensures the reaction is properly formatted and refers to the
        participant in third person.

        Args:
            response: The raw response from the LLM
            participant_name: The participant's name for reference

        Returns:
            The cleaned reaction description
        """
        cleaned = response.strip()

        # Remove surrounding quotation marks if present
        if cleaned.startswith('"') and cleaned.endswith('"'):
            cleaned = cleaned[1:-1].strip()
        if cleaned.startswith("'") and cleaned.endswith("'"):
            cleaned = cleaned[1:-1].strip()

        # Remove common prefixes
        prefixes_to_remove = [
            "Non-verbal reaction: ",
            "Reaction: ",
            "Response: ",
            f"{participant_name}'s reaction: ",
        ]
        for prefix in prefixes_to_remove:
            if cleaned.lower().startswith(prefix.lower()):
                cleaned = cleaned[len(prefix) :].strip()

        # Ensure the reaction starts with the participant's name if it doesn't
        # already reference them in some way
        first_name = participant_name.split()[0]
        if not any(
            name.lower() in cleaned.lower()[:50]
            for name in [participant_name, first_name, "they", "their", "she", "he"]
        ):
            cleaned = f"{participant_name} {cleaned[0].lower()}{cleaned[1:]}"

        return cleaned
