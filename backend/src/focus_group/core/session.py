"""Focus group session orchestrator.

This module provides the FocusGroupSession class that orchestrates the entire
focus group simulation, coordinating persona generation, introductions,
question discussions, and closing.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from focus_group.core.introduction_handler import IntroductionHandler, IntroductionResult
from focus_group.core.nonverbal_generator import NonVerbalGenerator
from focus_group.core.participant import Participant
from focus_group.core.question_handler import QuestionHandler, QuestionResult
from focus_group.core.research_assistant import ResearchAssistant
from focus_group.core.response_generator import ResponseGenerator
from focus_group.core.speaker_selection import (
    FirstSpeakerSelector,
    NextSpeakerSelector,
    SilenceHandler,
)
from focus_group.llm.base import BaseLLMProvider
from focus_group.models.config import SessionConfig
from focus_group.models.memory import FocusGroupMemory
from focus_group.models.persona import Persona
from focus_group.persona.generator import PersonaGenerationError, PersonaGenerator
from focus_group.persona.stance_distribution import draw_stances
from focus_group.persona.validator import PersonaValidator, ValidationResult


class SessionError(Exception):
    """Raised when a session error occurs."""

    pass


@dataclass
class SessionResult:
    """Result of a complete focus group session.

    Contains all data from the session including participants, introductions,
    question results, and timing information.

    Attributes:
        session_name: Optional name for the session
        purpose: The focus group purpose/introductory statement (if provided)
        participants: List of Persona objects for all participants
        introduction_result: Results from the introduction phase
        question_results: Results from each question discussion
        closing_result: Results from the closing question (uses QuestionResult)
        demographics: Original demographic specifications
        questions: Original questions list
        started_at: When the session started
        completed_at: When the session completed
        errors: List of any errors that occurred during the session
    """

    session_name: str | None = None
    purpose: str | None = None
    participants: list[Persona] = field(default_factory=list)
    introduction_result: IntroductionResult | None = None
    question_results: list[QuestionResult] = field(default_factory=list)
    closing_result: QuestionResult | None = None
    demographics: list[str] = field(default_factory=list)
    questions: list[str] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime | None = None
    errors: list[str] = field(default_factory=list)

    # Memory artifacts (populated when memory system is enabled, None otherwise)
    stance_matrix_dump: dict | None = None
    participant_memory_dumps: list[dict] | None = None

    @property
    def total_questions(self) -> int:
        """Total number of questions processed (including closing)."""
        count = len(self.question_results)
        if self.closing_result:
            count += 1
        return count

    @property
    def total_responses(self) -> int:
        """Total number of verbal responses across all questions."""
        count = sum(len(qr.responses) for qr in self.question_results)
        if self.closing_result:
            count += len(self.closing_result.responses)
        return count

    @property
    def duration_seconds(self) -> float | None:
        """Duration of the session in seconds, or None if not completed."""
        if self.completed_at is None:
            return None
        return (self.completed_at - self.started_at).total_seconds()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "session_name": self.session_name,
            "purpose": self.purpose,
            "participants": [p.to_dict() for p in self.participants],
            "introduction_result": (
                self.introduction_result.to_dict() if self.introduction_result else None
            ),
            "question_results": [qr.to_dict() for qr in self.question_results],
            "closing_result": (self.closing_result.to_dict() if self.closing_result else None),
            "demographics": self.demographics,
            "questions": self.questions,
            "started_at": self.started_at.isoformat(),
            "completed_at": (self.completed_at.isoformat() if self.completed_at else None),
            "errors": self.errors,
            "total_questions": self.total_questions,
            "total_responses": self.total_responses,
            "duration_seconds": self.duration_seconds,
            "stance_matrix_dump": self.stance_matrix_dump,
            "participant_memory_dumps": self.participant_memory_dumps,
        }


class FocusGroupSession:
    """Orchestrates the entire focus group simulation.

    This class is the main entry point for running a focus group session.
    It coordinates all components to execute the complete workflow:

    1. Generate personas from demographic specifications
    2. Create Participant objects with dual context
    3. Run introduction phase
    4. Process each question through the speaking loop
    5. Handle closing question
    6. Generate output/result

    Attributes:
        provider: The LLM provider for all operations
        config: Session configuration

    Example:
        >>> config = SessionConfig(
        ...     demographics=["Female, 30-40, tech", "Male, 50-60, retired"],
        ...     questions=["What do you think about remote work?"],
        ...     purpose="Discuss workplace trends",
        ... )
        >>> session = FocusGroupSession(provider=llm_provider, config=config)
        >>> result = await session.run()
        >>> print(f"Session completed with {len(result.participants)} participants")
    """

    def __init__(
        self,
        provider: BaseLLMProvider,
        config: SessionConfig,
        seed: int | None = None,
    ) -> None:
        """Initialize the focus group session.

        Args:
            provider: The LLM provider for all operations
            config: Session configuration including demographics, questions, etc.
            seed: Seed for the participants' stance draw (normally the job's sampling
                seed), so the same seed gives the same group. None draws unseeded.
        """
        self._provider = provider
        self._config = config
        self._seed = seed

        # Component instances (created lazily)
        self._persona_generator: PersonaGenerator | None = None
        self._persona_validator: PersonaValidator | None = None
        self._response_generator: ResponseGenerator | None = None
        self._nonverbal_generator: NonVerbalGenerator | None = None
        self._first_speaker_selector: FirstSpeakerSelector | None = None
        self._next_speaker_selector: NextSpeakerSelector | None = None
        self._silence_handler: SilenceHandler | None = None
        self._introduction_handler: IntroductionHandler | None = None
        self._question_handler: QuestionHandler | None = None

        # Memory components (only created when memory config is present)
        self._research_assistant: ResearchAssistant | None = None
        self._focus_group_memory: FocusGroupMemory | None = None

    def _init_components(self) -> None:
        """Initialize all sub-components.

        Called once at the start of run() to create all required components.
        """
        # Persona components
        self._persona_generator = PersonaGenerator(self._provider)
        self._persona_validator = PersonaValidator()

        # Core generators
        self._response_generator = ResponseGenerator(self._provider)
        self._nonverbal_generator = NonVerbalGenerator(self._provider)

        # Speaker selection
        self._first_speaker_selector = FirstSpeakerSelector(self._provider)
        self._next_speaker_selector = NextSpeakerSelector(self._provider)
        self._silence_handler = SilenceHandler(self._provider)

        # Memory components (only when memory config is present)
        if self._config.memory is not None:
            self._research_assistant = ResearchAssistant(self._provider)
            self._focus_group_memory = FocusGroupMemory(
                topic=self._config.purpose or "",
            )

        # Handlers
        self._introduction_handler = IntroductionHandler(
            provider=self._provider,
            response_generator=self._response_generator,
            nonverbal_generator=self._nonverbal_generator,
        )
        self._question_handler = QuestionHandler(
            provider=self._provider,
            response_generator=self._response_generator,
            nonverbal_generator=self._nonverbal_generator,
            first_speaker_selector=self._first_speaker_selector,
            next_speaker_selector=self._next_speaker_selector,
            silence_handler=self._silence_handler,
            research_assistant=self._research_assistant,
            focus_group_memory=self._focus_group_memory,
        )

    async def run(self) -> SessionResult:
        """Run the complete focus group session.

        Executes all steps of the focus group workflow from CLAUDE.md:
        1. Generate personas from demographics
        2. Create Participant objects
        3. Initialize contexts (automatic via Participant)
        4. Run introduction phase
        5. Process each question through speaking loop
        6. Handle closing question
        7. Return structured results

        Returns:
            SessionResult containing all session data

        Raises:
            SessionError: If a critical error occurs that prevents continuation
        """
        result = SessionResult(
            session_name=self._config.session_name,
            purpose=self._config.purpose,
            demographics=self._config.demographics,
            questions=self._config.questions,
        )

        try:
            # Initialize all components
            self._init_components()

            # Step 1: Generate personas from demographics
            personas = await self._generate_personas(result)
            if not personas:
                raise SessionError("Failed to generate any valid personas")

            result.participants = personas

            # Step 2: Create Participant objects
            participants = await self._create_participants(personas)
            if not participants:
                raise SessionError("Failed to create any participants")

            # Step 3: Context initialization is automatic via Participant constructor

            # Step 4: Run introduction phase
            introduction_result = await self._run_introductions(
                participants=participants,
                result=result,
            )
            result.introduction_result = introduction_result

            # Step 5: Process each question through speaking loop
            for q_idx, question in enumerate(self._config.questions):
                question_result = await self._process_question(
                    participants=participants,
                    question=question,
                    result=result,
                    question_index=q_idx,
                )
                result.question_results.append(question_result)

            # Step 6: Handle closing question
            closing_question_index = len(self._config.questions)
            closing_result = await self._process_question(
                participants=participants,
                question=self._config.closing_question,
                result=result,
                is_closing=True,
                question_index=closing_question_index,
            )
            result.closing_result = closing_result

            # Serialize memory artifacts if memory system was enabled
            if self._focus_group_memory is not None:
                result.stance_matrix_dump = self._focus_group_memory.stance_matrix.to_display_dict(
                    questions=self._config.questions,
                    closing_question=self._config.closing_question,
                )
                result.participant_memory_dumps = [
                    {
                        "participant_name": p.name,
                        "episodic": p.memory.episodic.model_dump(),
                        "completed_questions": [
                            qr.model_dump() for qr in p.memory.question.completed_questions
                        ],
                        "working_memory_summary": p.memory.working.discussion_summary,
                    }
                    for p in participants
                    if p.memory is not None
                ]

        except SessionError:
            # Re-raise session errors
            raise
        except Exception as e:
            # Log unexpected errors but don't crash
            error_msg = f"Unexpected error during session: {e}"
            result.errors.append(error_msg)

        # Step 7: Mark completion
        result.completed_at = datetime.now()

        return result

    async def _generate_personas(self, result: SessionResult) -> list[Persona]:
        """Generate personas from demographic specifications.

        Implements Step 1 of the workflow: Persona Generation.

        Args:
            result: The SessionResult to record errors in

        Returns:
            List of generated Persona objects
        """
        if not self._persona_generator:
            raise SessionError("PersonaGenerator not initialized")

        try:
            personas = await self._persona_generator.generate_personas(
                demographics=self._config.demographics,
                ensure_unique_names=True,
                name_overrides=self._config.participant_names,
                stances=draw_stances(
                    len(self._config.demographics), self._config.group_dynamic, self._seed
                ),
            )
        except PersonaGenerationError as e:
            error_msg = f"Failed to generate personas: {e}"
            result.errors.append(error_msg)
            return []

        # Validate personas if configured
        if self._config.validate_personas and self._persona_validator:
            valid_personas: list[Persona] = []
            for persona in personas:
                validation_result: ValidationResult = self._persona_validator.validate(persona)
                if validation_result.is_valid:
                    valid_personas.append(persona)
                else:
                    issues = "; ".join(str(i) for i in validation_result.issues)
                    result.errors.append(f"Persona '{persona.name}' failed validation: {issues}")
            return valid_personas

        return personas

    async def _create_participants(self, personas: list[Persona]) -> list[Participant]:
        """Create Participant objects from personas.

        Implements Step 2 of the workflow: Participant Object Creation.
        Step 3 (Context Initialization) happens automatically in Participant constructor.

        Args:
            personas: List of Persona objects to create participants from

        Returns:
            List of Participant objects
        """
        participants: list[Participant] = []

        for persona in personas:
            participant = Participant(
                persona=persona,
                provider=self._provider,
                config=self._config.context,
            )
            # Initialize memory if configured
            if self._config.memory is not None:
                participant.init_memory(focus_group_topic=self._config.purpose)
            participants.append(participant)

        return participants

    async def _run_introductions(
        self,
        participants: list[Participant],
        result: SessionResult,
    ) -> IntroductionResult | None:
        """Run the introduction phase.

        Implements Step 4 of the workflow: Introduction Phase.

        Args:
            participants: All session participants
            result: The SessionResult to record errors in

        Returns:
            IntroductionResult if successful, None if failed
        """
        if not self._introduction_handler:
            raise SessionError("IntroductionHandler not initialized")

        try:
            introduction_result = await self._introduction_handler.process_introductions(
                participants=participants,
                purpose=self._config.purpose,
            )
            return introduction_result
        except Exception as e:
            error_msg = f"Introduction phase failed: {e}"
            result.errors.append(error_msg)
            return None

    async def _process_question(
        self,
        participants: list[Participant],
        question: str,
        result: SessionResult,
        is_closing: bool = False,
        question_index: int = 0,
    ) -> QuestionResult:
        """Process a single question through the speaking loop.

        Implements Step 5 of the workflow: Question Loop.

        Args:
            participants: All session participants
            question: The question to discuss
            result: The SessionResult to record errors in
            is_closing: Whether this is the closing question
            question_index: 0-based index of the question (used by memory)

        Returns:
            QuestionResult containing all responses and reactions
        """
        if not self._question_handler:
            raise SessionError("QuestionHandler not initialized")

        try:
            question_result = await self._question_handler.process_question(
                participants=participants,
                question=question,
                is_closing=is_closing,
                question_index=question_index,
            )
            return question_result
        except Exception as e:
            error_msg = f"Question processing failed for '{question}': {e}"
            result.errors.append(error_msg)
            # Return empty result on failure
            return QuestionResult(question=question)

    @property
    def config(self) -> SessionConfig:
        """Get the session configuration."""
        return self._config

    @property
    def provider(self) -> BaseLLMProvider:
        """Get the LLM provider."""
        return self._provider
