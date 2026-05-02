"""Introduction handler for focus group participants.

This module provides the IntroductionHandler class that orchestrates the
introduction phase where each participant introduces themselves to the group.
"""

from dataclasses import dataclass, field
from datetime import datetime

from focus_group.core.nonverbal_generator import NonVerbalGenerator
from focus_group.core.participant import Participant
from focus_group.core.response_generator import ResponseGenerator
from focus_group.llm.base import BaseLLMProvider
from focus_group.models.message import Message


@dataclass
class ParticipantIntroduction:
    """A single participant's introduction.

    Attributes:
        speaker_name: Name of the participant who introduced themselves
        content: What they said (their introduction)
        order: The order in which they introduced themselves (1-indexed)
        timestamp: When this introduction was generated
    """

    speaker_name: str
    content: str
    order: int
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "speaker_name": self.speaker_name,
            "content": self.content,
            "order": self.order,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class IntroductionReaction:
    """A non-verbal reaction to a participant's introduction.

    Attributes:
        participant_name: Name of the participant reacting
        reaction: Description of their non-verbal reaction
        reacting_to: Name of the person who just introduced themselves
        timestamp: When this reaction was generated
    """

    participant_name: str
    reaction: str
    reacting_to: str
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "participant_name": self.participant_name,
            "reaction": self.reaction,
            "reacting_to": self.reacting_to,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class IntroductionResult:
    """Result of processing the introduction phase.

    Contains all introductions, reactions, and debug information from
    the introduction phase of the focus group.

    Attributes:
        purpose: The focus group purpose/introductory statement (if provided)
        introductions: List of all participant introductions in order
        reactions: List of all non-verbal reactions to introductions
        introduction_order: Names of participants in the order they introduced
        started_at: When processing started
        completed_at: When processing completed
    """

    purpose: str | None = None
    introductions: list[ParticipantIntroduction] = field(default_factory=list)
    reactions: list[IntroductionReaction] = field(default_factory=list)
    introduction_order: list[str] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "purpose": self.purpose,
            "introductions": [i.to_dict() for i in self.introductions],
            "reactions": [r.to_dict() for r in self.reactions],
            "introduction_order": self.introduction_order,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


class IntroductionHandler:
    """Orchestrates the introduction phase of a focus group session.

    This class implements Step 4 of the focus group workflow specification,
    handling the phase where each participant introduces themselves to the group.

    The introduction phase:
    1. If a purpose/introductory statement is provided, add it to everyone's context
    2. For each participant (in order):
       a. Generate their self-introduction
       b. Add introduction to everyone's verbal context
       c. Gather non-verbal reactions from others (parallel)

    Attributes:
        provider: The LLM provider for all operations
        response_generator: Generates verbal introductions
        nonverbal_generator: Generates non-verbal reactions

    Example:
        >>> handler = IntroductionHandler(
        ...     provider=llm_provider,
        ...     response_generator=response_gen,
        ...     nonverbal_generator=nonverbal_gen,
        ... )
        >>> result = await handler.process_introductions(
        ...     participants=[sarah, john, maria],
        ...     purpose="We're here to discuss remote work policies."
        ... )
        >>> for intro in result.introductions:
        ...     print(f"{intro.speaker_name}: {intro.content}")
    """

    def __init__(
        self,
        provider: BaseLLMProvider,
        response_generator: ResponseGenerator,
        nonverbal_generator: NonVerbalGenerator,
    ) -> None:
        """Initialize the introduction handler.

        Args:
            provider: The LLM provider for all operations
            response_generator: Generator for verbal introductions
            nonverbal_generator: Generator for non-verbal reactions
        """
        self._provider = provider
        self._response_generator = response_generator
        self._nonverbal_generator = nonverbal_generator

    async def process_introductions(
        self,
        participants: list[Participant],
        purpose: str | None = None,
    ) -> IntroductionResult:
        """Process the introduction phase for all participants.

        Implements the introduction phase from the workflow specification,
        having each participant introduce themselves in order while others
        react non-verbally.

        Args:
            participants: List of all focus group participants
            purpose: Optional introductory statement/purpose for the focus group

        Returns:
            IntroductionResult containing all introductions, reactions, and debug info

        Raises:
            ValueError: If participants list is empty
        """
        if not participants:
            raise ValueError("Cannot process introductions with no participants")

        result = IntroductionResult(purpose=purpose)

        # Step 4.1: If a purpose is provided, add it to everyone's context
        if purpose:
            await self._add_purpose_to_contexts(participants, purpose)

        # Add initial system message about introductions
        await self._add_introduction_prompt_to_contexts(participants)

        # Step 4.2: Each participant introduces themselves in order
        for order, participant in enumerate(participants, start=1):
            # Generate introduction
            introduction_content = await self._response_generator.generate_introduction(
                participant=participant,
            )

            # Record the introduction
            introduction = ParticipantIntroduction(
                speaker_name=participant.name,
                content=introduction_content,
                order=order,
            )
            result.introductions.append(introduction)
            result.introduction_order.append(participant.name)

            # Add introduction to everyone's verbal context
            await self._add_introduction_to_contexts(
                participants=participants,
                speaker_name=participant.name,
                introduction=introduction_content,
            )

            # Gather non-verbal reactions from others (not from the person introducing)
            other_participants = [p for p in participants if p.name != participant.name]
            if other_participants:
                await self._gather_introduction_reactions(
                    participants=other_participants,
                    speaker_name=participant.name,
                    introduction=introduction_content,
                    result=result,
                )

        # Mark completion
        result.completed_at = datetime.now()

        return result

    async def _add_purpose_to_contexts(
        self,
        participants: list[Participant],
        purpose: str,
    ) -> None:
        """Add the focus group purpose to everyone's context.

        Args:
            participants: All participants in the focus group
            purpose: The focus group purpose/introductory statement
        """
        purpose_message = Message.moderator(f"Welcome to this focus group. {purpose}")

        for participant in participants:
            participant.add_verbal_message(purpose_message)

    async def _add_introduction_prompt_to_contexts(
        self,
        participants: list[Participant],
    ) -> None:
        """Add the moderator's introduction request to everyone's context.

        Args:
            participants: All participants in the focus group
        """
        intro_prompt = Message.moderator(
            "Let's start by having everyone introduce themselves briefly."
        )

        for participant in participants:
            participant.add_verbal_message(intro_prompt)

    async def _add_introduction_to_contexts(
        self,
        participants: list[Participant],
        speaker_name: str,
        introduction: str,
    ) -> None:
        """Add a participant's introduction to everyone's verbal context.

        Args:
            participants: All participants in the focus group
            speaker_name: Name of the participant who introduced themselves
            introduction: What they said
        """
        introduction_message = Message.participant_response(introduction, speaker_name)

        for participant in participants:
            participant.add_verbal_message(introduction_message)

    async def _gather_introduction_reactions(
        self,
        participants: list[Participant],
        speaker_name: str,
        introduction: str,
        result: IntroductionResult,
    ) -> dict[str, str]:
        """Gather non-verbal reactions to an introduction.

        Runs reaction generation in parallel for efficiency.

        Args:
            participants: Participants to gather reactions from (excludes speaker)
            speaker_name: Name of the participant who just introduced themselves
            introduction: What they said
            result: The IntroductionResult to record reactions in

        Returns:
            Dictionary mapping participant names to their reactions
        """
        if not participants:
            return {}

        reactions = await self._nonverbal_generator.generate_reactions_to_statement_parallel(
            participants=participants,
            speaker_name=speaker_name,
            statement=introduction,
        )

        # Record reactions and add to participant contexts
        for participant in participants:
            reaction = reactions.get(participant.name, "")
            if reaction:
                # Record in result
                result.reactions.append(
                    IntroductionReaction(
                        participant_name=participant.name,
                        reaction=reaction,
                        reacting_to=speaker_name,
                    )
                )
                # Add to participant's non-verbal context
                nonverbal_message = Message.nonverbal(reaction, participant.name)
                participant.add_nonverbal_message(nonverbal_message)

        return reactions
