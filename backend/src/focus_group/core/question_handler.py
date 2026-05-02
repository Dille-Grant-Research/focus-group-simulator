"""Question handler for processing focus group questions.

This module provides the QuestionHandler class that orchestrates the full
speaking loop for a single focus group question, coordinating response
generation, non-verbal reactions, and speaker selection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

from focus_group.core.nonverbal_generator import NonVerbalGenerator
from focus_group.core.participant import Participant
from focus_group.core.response_generator import ResponseGenerator
from focus_group.core.speaker_selection import (
    FirstSpeakerSelector,
    NextSpeakerSelector,
    SilenceHandler,
)
from focus_group.llm.base import BaseLLMProvider
from focus_group.models.memory import ConsensusLevel, FocusGroupMemory
from focus_group.models.message import Message

if TYPE_CHECKING:
    from focus_group.core.research_assistant import ResearchAssistant

logger = logging.getLogger(__name__)


@dataclass
class ParticipantResponse:
    """A single participant's response during the question discussion.

    Attributes:
        speaker_name: Name of the participant who spoke
        content: What they said (verbal response)
        turn_number: The order in which they spoke (1-indexed)
        selection_method: How they were selected ("first_speaker", "eager", "reluctant")
        selection_reasoning: The reasoning behind their selection
        timestamp: When this response was generated
    """

    speaker_name: str
    content: str
    turn_number: int
    selection_method: str
    selection_reasoning: str
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "speaker_name": self.speaker_name,
            "content": self.content,
            "turn_number": self.turn_number,
            "selection_method": self.selection_method,
            "selection_reasoning": self.selection_reasoning,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class NonVerbalReaction:
    """A non-verbal reaction from a participant.

    Attributes:
        participant_name: Name of the participant reacting
        reaction: Description of their non-verbal reaction
        stimulus_type: What they reacted to ("question" or "statement")
        stimulus_source: Who/what triggered the reaction (moderator or speaker name)
        timestamp: When this reaction was generated
        response_turn: For statement reactions, the turn number this reaction is for
    """

    participant_name: str
    reaction: str
    stimulus_type: str
    stimulus_source: str
    timestamp: datetime = field(default_factory=datetime.now)
    response_turn: int | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "participant_name": self.participant_name,
            "reaction": self.reaction,
            "stimulus_type": self.stimulus_type,
            "stimulus_source": self.stimulus_source,
            "timestamp": self.timestamp.isoformat(),
            "response_turn": self.response_turn,
        }


@dataclass
class QuestionResult:
    """Result of processing a single focus group question.

    Contains all responses, reactions, and debug information from
    processing one question through the full speaking loop.

    Attributes:
        question: The focus group question that was processed
        responses: List of all participant responses in order
        nonverbal_reactions: List of all non-verbal reactions generated
        speaking_order: Names of participants in the order they spoke
        total_turns: Total number of speaking turns
        started_at: When processing started
        completed_at: When processing completed
    """

    question: str
    responses: list[ParticipantResponse] = field(default_factory=list)
    nonverbal_reactions: list[NonVerbalReaction] = field(default_factory=list)
    speaking_order: list[str] = field(default_factory=list)
    total_turns: int = 0
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "question": self.question,
            "responses": [r.to_dict() for r in self.responses],
            "nonverbal_reactions": [nr.to_dict() for nr in self.nonverbal_reactions],
            "speaking_order": self.speaking_order,
            "total_turns": self.total_turns,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


class QuestionHandler:
    """Orchestrates the full speaking loop for a single focus group question.

    This class implements Step 5 of the focus group workflow specification,
    coordinating all components to process a question through the speaking loop
    until all participants have contributed.

    The speaking loop:
    1. Add question to everyone's verbal context
    2. Gather non-verbal reactions to question (parallel)
    3. Select first speaker
    4. Speaking loop until everyone has spoken:
       a. Get verbal response from selected speaker
       b. Add response to everyone's verbal context
       c. Gather non-verbal reactions to response (parallel)
       d. Check if anyone wants to speak next
       e. If no one, use SilenceHandler to select from those who haven't spoken

    Attributes:
        provider: The LLM provider for all operations
        response_generator: Generates verbal responses
        nonverbal_generator: Generates non-verbal reactions
        first_speaker_selector: Selects who speaks first
        next_speaker_selector: Selects who speaks next (can return None)
        silence_handler: Handles when no one volunteers

    Example:
        >>> handler = QuestionHandler(
        ...     provider=llm_provider,
        ...     response_generator=response_gen,
        ...     nonverbal_generator=nonverbal_gen,
        ...     first_speaker_selector=first_selector,
        ...     next_speaker_selector=next_selector,
        ...     silence_handler=silence_handler,
        ... )
        >>> result = await handler.process_question(
        ...     participants=[sarah, john, maria],
        ...     question="What do you think about remote work?"
        ... )
        >>> for response in result.responses:
        ...     print(f"{response.speaker_name}: {response.content}")
    """

    def __init__(
        self,
        provider: BaseLLMProvider,
        response_generator: ResponseGenerator,
        nonverbal_generator: NonVerbalGenerator,
        first_speaker_selector: FirstSpeakerSelector,
        next_speaker_selector: NextSpeakerSelector,
        silence_handler: SilenceHandler,
        research_assistant: ResearchAssistant | None = None,
        focus_group_memory: FocusGroupMemory | None = None,
    ) -> None:
        """Initialize the question handler.

        Args:
            provider: The LLM provider for all operations
            response_generator: Generator for verbal responses
            nonverbal_generator: Generator for non-verbal reactions
            first_speaker_selector: Selector for first speaker
            next_speaker_selector: Selector for next speaker (can return None)
            silence_handler: Handler for when no one volunteers
            research_assistant: Optional Research Assistant for stance analysis
            focus_group_memory: Optional shared memory state (enables memory hooks)
        """
        self._provider = provider
        self._response_generator = response_generator
        self._nonverbal_generator = nonverbal_generator
        self._first_speaker_selector = first_speaker_selector
        self._next_speaker_selector = next_speaker_selector
        self._silence_handler = silence_handler
        self._research_assistant = research_assistant
        self._focus_group_memory = focus_group_memory

    async def process_question(
        self,
        participants: list[Participant],
        question: str,
        is_closing: bool = False,
        allow_followup: bool = True,
        question_index: int = 0,
    ) -> QuestionResult:
        """Process a question through the full speaking loop.

        Implements the complete speaking loop from the workflow specification,
        ensuring all participants have a chance to speak on the question.

        Args:
            participants: List of all focus group participants
            question: The focus group question to discuss
            is_closing: Whether this is the closing question
            allow_followup: Whether to allow one follow-up response
            question_index: 0-based index of the question (used by memory)

        Returns:
            QuestionResult containing all responses, reactions, and debug info

        Raises:
            ValueError: If participants list is empty
        """
        if not participants:
            raise ValueError("Cannot process question with no participants")

        result = QuestionResult(question=question)
        # Convenience references for memory components
        fgm = self._focus_group_memory
        ra = self._research_assistant
        stance_matrix = fgm.stance_matrix if fgm else None

        # Step 5.1: Add question to everyone's verbal context
        await self._add_question_to_contexts(participants, question)

        # Hook A: Memory — notify participants of new question, reset turn counter
        if fgm is not None:
            fgm.reset_question_turns()
            for p in participants:
                p.memory_on_new_question(question, question_index)

        # Step 5.2: Gather non-verbal reactions to question (parallel)
        question_reactions = await self._gather_question_reactions(
            participants=participants,
            question=question,
            result=result,
        )

        # Step 5.3: Select first speaker
        first_speaker_result = await self._first_speaker_selector.select_first_speaker(
            participants=participants,
            question=question,
            reactions=question_reactions,
        )

        current_speaker = first_speaker_result.participant
        selection_method = "first_speaker"
        selection_reasoning = first_speaker_result.reasoning

        # Track who has spoken on this question
        already_spoken: set[str] = set()
        turn_number = 0
        previous_speaker_name: str | None = None
        previous_response: str | None = None
        statement_reactions: dict[str, str] = {}

        # Step 5.4: Speaking loop until everyone has spoken
        while len(already_spoken) < len(participants):
            turn_number += 1

            # Step 5.4.1: Get verbal response from selected speaker
            # Hook D: pass stance_matrix to response generation calls
            if is_closing and previous_speaker_name is None:
                # Closing first-speaker: use specialized closing prompt
                response_content = await self._response_generator.generate_closing_response(
                    participant=current_speaker,
                    closing_question=question,
                    stance_matrix=stance_matrix,
                )
            elif previous_speaker_name is not None and previous_response is not None:
                # Turn 2+: respond to what the previous speaker said
                response_content = await self._response_generator.generate_response_to_participant(
                    participant=current_speaker,
                    speaker_name=previous_speaker_name,
                    speaker_statement=previous_response,
                    question=question,
                    reactions=statement_reactions,
                    stance_matrix=stance_matrix,
                )
            else:
                # Turn 1: respond to the moderator's question
                response_content = await self._response_generator.generate_response_to_question(
                    participant=current_speaker,
                    question=question,
                    stance_matrix=stance_matrix,
                )

            # Record the response
            participant_response = ParticipantResponse(
                speaker_name=current_speaker.name,
                content=response_content,
                turn_number=turn_number,
                selection_method=selection_method,
                selection_reasoning=selection_reasoning,
            )
            result.responses.append(participant_response)
            result.speaking_order.append(current_speaker.name)
            already_spoken.add(current_speaker.name)

            # Track previous speaker for response routing
            previous_speaker_name = current_speaker.name
            previous_response = response_content

            # Step 5.4.2: Add response to everyone's verbal context
            await self._add_response_to_contexts(
                participants=participants,
                speaker_name=current_speaker.name,
                response=response_content,
            )

            # Hook B: Memory — record turn, analyze stance
            if fgm is not None:
                global_turn = fgm.increment_turn()
                for p in participants:
                    p.memory_on_turn(current_speaker.name, response_content, global_turn)
                # Analyze the speaker's stance via Research Assistant
                if ra is not None:
                    try:
                        await ra.analyze_turn(
                            speaker_name=current_speaker.name,
                            statement=response_content,
                            question=question,
                            question_index=question_index,
                            stance_matrix=fgm.stance_matrix,
                            participants=[p.name for p in participants],
                        )
                    except Exception as e:
                        logger.warning("Research Assistant analyze_turn failed: %s", e)

            # Step 5.4.3: Gather non-verbal reactions to response (parallel)
            # Always gather reactions, even after the last speaker
            other_participants = [p for p in participants if p.name != current_speaker.name]
            statement_reactions = {}
            if other_participants:
                statement_reactions = await self._gather_statement_reactions(
                    participants=other_participants,
                    speaker_name=current_speaker.name,
                    statement=response_content,
                    result=result,
                    turn_number=turn_number,
                )

            # Check if everyone has spoken - if so, exit the loop
            if len(already_spoken) >= len(participants):
                break

            # Step 5.4.4: Check if anyone wants to speak next
            next_speaker_result = await self._next_speaker_selector.select_next_speaker(
                participants=participants,
                current_speaker=current_speaker,
                statement=response_content,
                reactions=statement_reactions,
                already_spoken=already_spoken,
            )

            if next_speaker_result is not None:
                # Someone wants to speak
                current_speaker = next_speaker_result.participant
                selection_method = "eager"
                selection_reasoning = next_speaker_result.reasoning
            else:
                # No one wants to speak - check if everyone has spoken
                if len(already_spoken) >= len(participants):
                    # Everyone has spoken, exit loop
                    break
                else:
                    # Use SilenceHandler to select from those who haven't spoken
                    silence_result = await self._silence_handler.select_reluctant_speaker(
                        participants=participants,
                        question=question,
                        reactions=statement_reactions,
                        already_spoken=already_spoken,
                    )
                    current_speaker = silence_result.participant
                    selection_method = "reluctant"
                    selection_reasoning = silence_result.reasoning

        # Optional follow-up: allow one voluntary response after everyone has spoken
        if allow_followup and statement_reactions:
            try:
                followup_result = await self._next_speaker_selector.select_followup_speaker(
                    participants=participants,
                    current_speaker=current_speaker,
                    statement=previous_response or response_content,
                    reactions=statement_reactions,
                )
                if followup_result is not None:
                    turn_number += 1
                    followup_speaker = followup_result.participant
                    followup_content = (
                        await self._response_generator.generate_response_to_participant(
                            participant=followup_speaker,
                            speaker_name=previous_speaker_name or current_speaker.name,
                            speaker_statement=previous_response or response_content,
                            question=question,
                            reactions=statement_reactions,
                            stance_matrix=stance_matrix,
                        )
                    )
                    # Record the follow-up
                    followup_response = ParticipantResponse(
                        speaker_name=followup_speaker.name,
                        content=followup_content,
                        turn_number=turn_number,
                        selection_method="followup",
                        selection_reasoning=followup_result.reasoning,
                    )
                    result.responses.append(followup_response)
                    result.speaking_order.append(followup_speaker.name)
                    # Add to contexts
                    await self._add_response_to_contexts(
                        participants=participants,
                        speaker_name=followup_speaker.name,
                        response=followup_content,
                    )
                    # Hook B (follow-up): record turn in memory
                    if fgm is not None:
                        global_turn = fgm.increment_turn()
                        for p in participants:
                            p.memory_on_turn(followup_speaker.name, followup_content, global_turn)
                        if ra is not None:
                            try:
                                await ra.analyze_turn(
                                    speaker_name=followup_speaker.name,
                                    statement=followup_content,
                                    question=question,
                                    question_index=question_index,
                                    stance_matrix=fgm.stance_matrix,
                                    participants=[p.name for p in participants],
                                )
                            except Exception as e:
                                logger.warning(
                                    "Research Assistant analyze_turn (followup) failed: %s", e
                                )
            except (ValueError, Exception):
                # If follow-up selection fails, gracefully skip
                pass

        # Hook C: Memory — generate question summary and archive
        if fgm is not None and ra is not None:
            try:
                transcript = [
                    {"speaker": r.speaker_name, "content": r.content} for r in result.responses
                ]
                summary_data = await ra.generate_question_summary(
                    question=question,
                    question_index=question_index,
                    transcript=transcript,
                    stance_matrix=fgm.stance_matrix,
                )
                consensus_str = summary_data.get("consensus_level", "none")
                try:
                    consensus = ConsensusLevel(consensus_str)
                except ValueError:
                    consensus = ConsensusLevel.NONE

                for p in participants:
                    p.memory_on_question_complete(
                        summary=summary_data.get("summary", ""),
                        key_themes=summary_data.get("key_themes", []),
                        consensus_level=consensus,
                        notable_interactions=summary_data.get("notable_interactions", []),
                        turn_count=turn_number,
                    )
            except Exception as e:
                logger.warning("Research Assistant question summary failed: %s", e)

        # Mark completion
        result.total_turns = turn_number
        result.completed_at = datetime.now()

        return result

    async def _add_question_to_contexts(
        self,
        participants: list[Participant],
        question: str,
    ) -> None:
        """Add the moderator's question to everyone's verbal context.

        Args:
            participants: All participants in the focus group
            question: The question being asked
        """
        moderator_message = Message.moderator(f"Question: {question}")

        for participant in participants:
            participant.add_verbal_message(moderator_message)
            participant.set_question(question)

    async def _gather_question_reactions(
        self,
        participants: list[Participant],
        question: str,
        result: QuestionResult,
    ) -> dict[str, str]:
        """Gather non-verbal reactions to the question from all participants.

        Runs reaction generation in parallel for efficiency.

        Args:
            participants: All participants to gather reactions from
            question: The question being asked
            result: The QuestionResult to record reactions in

        Returns:
            Dictionary mapping participant names to their reactions
        """
        reactions = await self._nonverbal_generator.generate_reactions_to_question_parallel(
            participants=participants,
            question=question,
        )

        # Record reactions and add to participant contexts
        for participant in participants:
            reaction = reactions.get(participant.name, "")
            if reaction:
                # Record in result
                result.nonverbal_reactions.append(
                    NonVerbalReaction(
                        participant_name=participant.name,
                        reaction=reaction,
                        stimulus_type="question",
                        stimulus_source="moderator",
                    )
                )
                # Add to participant's non-verbal context
                nonverbal_message = Message.nonverbal(reaction, participant.name)
                participant.add_nonverbal_message(nonverbal_message)

        return reactions

    async def _add_response_to_contexts(
        self,
        participants: list[Participant],
        speaker_name: str,
        response: str,
    ) -> None:
        """Add a participant's response to everyone's verbal context.

        Args:
            participants: All participants in the focus group
            speaker_name: Name of the participant who spoke
            response: What they said
        """
        response_message = Message.participant_response(response, speaker_name)

        for participant in participants:
            participant.add_verbal_message(response_message)

    async def _gather_statement_reactions(
        self,
        participants: list[Participant],
        speaker_name: str,
        statement: str,
        result: QuestionResult,
        turn_number: int,
    ) -> dict[str, str]:
        """Gather non-verbal reactions to a statement from participants.

        Runs reaction generation in parallel for efficiency.

        Args:
            participants: Participants to gather reactions from (excludes speaker)
            speaker_name: Name of the participant who spoke
            statement: What they said
            result: The QuestionResult to record reactions in
            turn_number: The turn number this statement corresponds to

        Returns:
            Dictionary mapping participant names to their reactions
        """
        if not participants:
            return {}

        reactions = await self._nonverbal_generator.generate_reactions_to_statement_parallel(
            participants=participants,
            speaker_name=speaker_name,
            statement=statement,
        )

        # Record reactions and add to participant contexts
        for participant in participants:
            reaction = reactions.get(participant.name, "")
            if reaction:
                # Record in result
                result.nonverbal_reactions.append(
                    NonVerbalReaction(
                        participant_name=participant.name,
                        reaction=reaction,
                        stimulus_type="statement",
                        stimulus_source=speaker_name,
                        response_turn=turn_number,
                    )
                )
                # Add to participant's non-verbal context
                nonverbal_message = Message.nonverbal(reaction, participant.name)
                participant.add_nonverbal_message(nonverbal_message)

        return reactions
