"""Data models for conversation messages and events.

This module defines the message and event structures used throughout
the focus group simulation for tracking conversation flow and logging.
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    """Roles for different types of messages in the conversation.

    - SYSTEM: System-level instructions and prompts
    - MODERATOR: Focus group moderator/facilitator
    - PARTICIPANT: Verbal responses from participants
    - NONVERBAL: Non-verbal reactions and cues
    """

    SYSTEM = "system"
    MODERATOR = "moderator"
    PARTICIPANT = "participant"
    NONVERBAL = "nonverbal"


class Message(BaseModel):
    """Represents a single message in the focus group conversation.

    Messages can come from the system, moderator, or participants. They can
    be verbal responses or non-verbal reactions depending on the role.

    Attributes:
        role: Who is sending this message
        content: The text content of the message
        participant_name: Name of the participant (required for PARTICIPANT/NONVERBAL)
        timestamp: When this message was created
    """

    role: MessageRole = Field(
        description="Who is sending this message",
    )
    content: str = Field(
        description="The text content of the message",
        min_length=1,
    )
    participant_name: str | None = Field(
        default=None,
        description="Name of the participant (for PARTICIPANT/NONVERBAL roles)",
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When this message was created",
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert this message to a dictionary for serialization.

        Returns:
            Dictionary representation of the message
        """
        return {
            "role": self.role.value,
            "content": self.content,
            "participant_name": self.participant_name,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def system(cls, content: str) -> "Message":
        """Create a system message.

        Args:
            content: The system message content

        Returns:
            A new Message with SYSTEM role
        """
        return cls(role=MessageRole.SYSTEM, content=content)

    @classmethod
    def moderator(cls, content: str) -> "Message":
        """Create a moderator message (e.g., asking a question).

        Args:
            content: The moderator's message

        Returns:
            A new Message with MODERATOR role
        """
        return cls(role=MessageRole.MODERATOR, content=content)

    @classmethod
    def participant_response(cls, content: str, participant_name: str) -> "Message":
        """Create a participant's verbal response.

        Args:
            content: What the participant said
            participant_name: Name of the participant

        Returns:
            A new Message with PARTICIPANT role
        """
        return cls(
            role=MessageRole.PARTICIPANT,
            content=content,
            participant_name=participant_name,
        )

    @classmethod
    def nonverbal(cls, content: str, participant_name: str) -> "Message":
        """Create a non-verbal reaction.

        Args:
            content: Description of the non-verbal reaction
            participant_name: Name of the participant

        Returns:
            A new Message with NONVERBAL role
        """
        return cls(
            role=MessageRole.NONVERBAL,
            content=content,
            participant_name=participant_name,
        )


class ConversationEvent(BaseModel):
    """Represents a logged event during the focus group session.

    Events are used for detailed logging and debugging, tracking things like
    speaker selection, question transitions, context summarization, etc.

    Attributes:
        event_type: Type of event (e.g., "question_asked", "participant_spoke")
        timestamp: When the event occurred
        data: Event-specific data as a dictionary
    """

    event_type: str = Field(
        description="Type of event (e.g., 'question_asked', 'participant_spoke')",
        min_length=1,
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When the event occurred",
    )
    data: dict[str, Any] = Field(
        default_factory=dict,
        description="Event-specific data",
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert this event to a dictionary for serialization.

        Returns:
            Dictionary representation of the event
        """
        return {
            "event_type": self.event_type,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
        }

    @classmethod
    def question_asked(cls, question: str, question_number: int) -> "ConversationEvent":
        """Create an event for when a question is asked.

        Args:
            question: The question text
            question_number: The index of this question

        Returns:
            A new ConversationEvent
        """
        return cls(
            event_type="question_asked",
            data={"question": question, "question_number": question_number},
        )

    @classmethod
    def participant_spoke(
        cls,
        participant_name: str,
        response: str,
        turn_number: int,
    ) -> "ConversationEvent":
        """Create an event for when a participant speaks.

        Args:
            participant_name: Name of the participant
            response: What they said
            turn_number: The turn number in this question

        Returns:
            A new ConversationEvent
        """
        return cls(
            event_type="participant_spoke",
            data={
                "participant_name": participant_name,
                "response": response,
                "turn_number": turn_number,
            },
        )

    @classmethod
    def speaker_selected(
        cls,
        participant_name: str,
        selection_reason: str,
    ) -> "ConversationEvent":
        """Create an event for when a speaker is selected.

        Args:
            participant_name: Name of the selected speaker
            selection_reason: Why they were selected (e.g., "first_speaker", "eager")

        Returns:
            A new ConversationEvent
        """
        return cls(
            event_type="speaker_selected",
            data={
                "participant_name": participant_name,
                "selection_reason": selection_reason,
            },
        )
