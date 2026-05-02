"""Core module containing the main simulation logic.

This module provides the core components for running focus group simulations,
including participant management, session orchestration, and response generation.
"""

from focus_group.core.introduction_handler import (
    IntroductionHandler,
    IntroductionReaction,
    IntroductionResult,
    ParticipantIntroduction,
)
from focus_group.core.nonverbal_generator import NonVerbalGenerator
from focus_group.core.participant import Participant
from focus_group.core.question_handler import (
    NonVerbalReaction,
    ParticipantResponse,
    QuestionHandler,
    QuestionResult,
)
from focus_group.core.response_generator import ResponseGenerator
from focus_group.core.session import (
    FocusGroupSession,
    SessionError,
    SessionResult,
)
from focus_group.core.speaker_selection import (
    FirstSpeakerSelector,
    NextSpeakerSelector,
    SilenceHandler,
    SpeakerSelectionResult,
)

__all__ = [
    "FirstSpeakerSelector",
    "FocusGroupSession",
    "IntroductionHandler",
    "IntroductionReaction",
    "IntroductionResult",
    "NextSpeakerSelector",
    "NonVerbalGenerator",
    "NonVerbalReaction",
    "Participant",
    "ParticipantIntroduction",
    "ParticipantResponse",
    "QuestionHandler",
    "QuestionResult",
    "ResponseGenerator",
    "SessionError",
    "SessionResult",
    "SilenceHandler",
    "SpeakerSelectionResult",
]
