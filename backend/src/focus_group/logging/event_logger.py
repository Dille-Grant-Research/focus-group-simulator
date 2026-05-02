"""Event logger for focus group session events.

This module provides the EventLogger class that logs all session events
with timestamps, supporting both in-memory storage and file output.
"""

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

from focus_group.models.message import ConversationEvent
from focus_group.models.persona import Persona


class EventLogger:
    """Central logging facility for all focus group session events.

    Logs events with timestamps, event types, and event-specific data.
    Supports both in-memory storage (for programmatic access) and
    optional file output (for persistence).

    Thread-safe for concurrent logging from multiple coroutines.

    Attributes:
        events: List of all logged events (read-only access via get_events())
        file_path: Path to the log file, if file logging is enabled

    Example:
        >>> logger = EventLogger()
        >>> logger.log_session_start("session_001", {"participants": 3})
        >>> logger.log_persona_generated(persona)
        >>> logger.log_response("Sarah Chen", "I think...", turn=1)
        >>> events = logger.get_events()

        >>> # With file logging
        >>> logger = EventLogger(file_path="logs/session.jsonl")
        >>> logger.log_session_start("my_session", {})
        >>> logger.close()  # Flush and close file
    """

    def __init__(
        self,
        file_path: str | Path | None = None,
        auto_flush: bool = True,
    ) -> None:
        """Initialize the event logger.

        Args:
            file_path: Optional path to write events to a file (JSONL format).
                       If None, events are only stored in memory.
            auto_flush: If True, flush to file after each event (default True).
                        Set to False for better performance with manual flush.
        """
        self._events: list[ConversationEvent] = []
        self._lock = threading.Lock()
        self._file_path: Path | None = Path(file_path) if file_path else None
        self._file_handle: TextIO | None = None
        self._auto_flush = auto_flush

        # Open file if path provided
        if self._file_path:
            self._file_path.parent.mkdir(parents=True, exist_ok=True)
            self._file_handle = open(self._file_path, "a", encoding="utf-8")

    def _log_event(self, event: ConversationEvent) -> None:
        """Internal method to log an event (thread-safe).

        Args:
            event: The ConversationEvent to log
        """
        with self._lock:
            self._events.append(event)

            # Write to file if enabled
            if self._file_handle:
                json_line = json.dumps(event.to_dict(), default=str)
                self._file_handle.write(json_line + "\n")
                if self._auto_flush:
                    self._file_handle.flush()

    def log_session_start(
        self,
        session_name: str,
        config: dict[str, Any],
    ) -> None:
        """Log the start of a focus group session.

        Args:
            session_name: Name or identifier for the session
            config: Session configuration dictionary
        """
        event = ConversationEvent(
            event_type="session_start",
            data={
                "session_name": session_name,
                "config": config,
            },
        )
        self._log_event(event)

    def log_session_end(
        self,
        session_name: str | None = None,
        total_questions: int = 0,
        total_responses: int = 0,
        duration_seconds: float | None = None,
        error_count: int = 0,
    ) -> None:
        """Log the end of a focus group session.

        Args:
            session_name: Name or identifier for the session
            total_questions: Total number of questions processed
            total_responses: Total number of verbal responses
            duration_seconds: Session duration in seconds
            error_count: Number of errors that occurred
        """
        event = ConversationEvent(
            event_type="session_end",
            data={
                "session_name": session_name,
                "total_questions": total_questions,
                "total_responses": total_responses,
                "duration_seconds": duration_seconds,
                "error_count": error_count,
            },
        )
        self._log_event(event)

    def log_persona_generated(
        self,
        persona: Persona,
        demographic_source: str | None = None,
    ) -> None:
        """Log a generated persona.

        Args:
            persona: The Persona object that was generated
            demographic_source: Optional demographic specification used
        """
        event = ConversationEvent(
            event_type="persona_generated",
            data={
                "name": persona.name,
                "age": persona.age,
                "background": persona.background,
                "personality_traits": persona.personality_traits,
                "communication_style": persona.communication_style,
                "demographic_source": demographic_source or persona.demographic_source,
            },
        )
        self._log_event(event)

    def log_introduction(
        self,
        speaker: str,
        content: str,
        order: int | None = None,
    ) -> None:
        """Log a participant's self-introduction.

        Args:
            speaker: Name of the participant who introduced themselves
            content: The content of their introduction
            order: Optional order in which they introduced (1-indexed)
        """
        event = ConversationEvent(
            event_type="introduction",
            data={
                "speaker": speaker,
                "content": content,
                "order": order,
            },
        )
        self._log_event(event)

    def log_question_asked(
        self,
        question: str,
        question_number: int | None = None,
        is_closing: bool = False,
    ) -> None:
        """Log when a question is asked by the moderator.

        Args:
            question: The question text
            question_number: Optional question number (1-indexed)
            is_closing: Whether this is the closing question
        """
        event = ConversationEvent(
            event_type="question_asked",
            data={
                "question": question,
                "question_number": question_number,
                "is_closing": is_closing,
            },
        )
        self._log_event(event)

    def log_response(
        self,
        speaker: str,
        content: str,
        turn: int,
        question: str | None = None,
    ) -> None:
        """Log a participant's verbal response.

        Args:
            speaker: Name of the participant who spoke
            content: What they said
            turn: The turn number for this question (1-indexed)
            question: Optional question being discussed
        """
        event = ConversationEvent(
            event_type="participant_spoke",
            data={
                "participant_name": speaker,
                "response": content,
                "turn_number": turn,
                "question": question,
            },
        )
        self._log_event(event)

    def log_nonverbal_reaction(
        self,
        participant: str,
        reaction: str,
        stimulus: str,
        stimulus_type: str = "statement",
        stimulus_source: str | None = None,
    ) -> None:
        """Log a participant's non-verbal reaction.

        Args:
            participant: Name of the participant reacting
            reaction: Description of their non-verbal reaction
            stimulus: The question or statement they reacted to
            stimulus_type: Type of stimulus ("question" or "statement")
            stimulus_source: Who/what triggered the reaction
        """
        event = ConversationEvent(
            event_type="nonverbal_reaction",
            data={
                "participant": participant,
                "reaction": reaction,
                "stimulus": stimulus,
                "stimulus_type": stimulus_type,
                "stimulus_source": stimulus_source,
            },
        )
        self._log_event(event)

    def log_speaker_selected(
        self,
        speaker: str,
        method: str,
        reasoning: str,
        question: str | None = None,
    ) -> None:
        """Log when a speaker is selected.

        Args:
            speaker: Name of the selected speaker
            method: Selection method ("first_speaker", "eager", "reluctant")
            reasoning: The reasoning behind the selection
            question: Optional question being discussed
        """
        event = ConversationEvent(
            event_type="speaker_selected",
            data={
                "participant_name": speaker,
                "selection_method": method,
                "selection_reason": reasoning,
                "question": question,
            },
        )
        self._log_event(event)

    def log_context_summarized(
        self,
        participant: str,
        original_tokens: int,
        summarized_tokens: int,
    ) -> None:
        """Log when a participant's context is summarized.

        Args:
            participant: Name of the participant whose context was summarized
            original_tokens: Token count before summarization
            summarized_tokens: Token count after summarization
        """
        event = ConversationEvent(
            event_type="context_summarized",
            data={
                "participant": participant,
                "original_tokens": original_tokens,
                "summarized_tokens": summarized_tokens,
                "tokens_saved": original_tokens - summarized_tokens,
            },
        )
        self._log_event(event)

    def log_error(
        self,
        error_type: str,
        message: str,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Log an error that occurred during the session.

        Args:
            error_type: Type of error (e.g., "generation_error", "validation_error")
            message: Error message
            context: Optional additional context about the error
        """
        event = ConversationEvent(
            event_type="error",
            data={
                "error_type": error_type,
                "message": message,
                "context": context or {},
            },
        )
        self._log_event(event)

    def log_custom(
        self,
        event_type: str,
        data: dict[str, Any],
    ) -> None:
        """Log a custom event with arbitrary data.

        Use this for events not covered by the specific log methods.

        Args:
            event_type: Type identifier for the event
            data: Event-specific data dictionary
        """
        event = ConversationEvent(
            event_type=event_type,
            data=data,
        )
        self._log_event(event)

    def get_events(self) -> list[ConversationEvent]:
        """Get all logged events.

        Returns a copy of the events list to prevent external modification.

        Returns:
            List of all ConversationEvent objects logged so far
        """
        with self._lock:
            return list(self._events)

    def get_events_by_type(self, event_type: str) -> list[ConversationEvent]:
        """Get all events of a specific type.

        Args:
            event_type: The event type to filter by

        Returns:
            List of ConversationEvent objects matching the type
        """
        with self._lock:
            return [e for e in self._events if e.event_type == event_type]

    def get_events_since(self, since: datetime) -> list[ConversationEvent]:
        """Get all events since a specific timestamp.

        Args:
            since: Only return events after this time

        Returns:
            List of ConversationEvent objects after the timestamp
        """
        with self._lock:
            return [e for e in self._events if e.timestamp >= since]

    def clear(self) -> None:
        """Clear all logged events from memory.

        Note: This does not remove events from the file if file logging is enabled.
        """
        with self._lock:
            self._events.clear()

    def flush(self) -> None:
        """Flush any buffered data to the file.

        Only has effect if file logging is enabled and auto_flush is False.
        """
        with self._lock:
            if self._file_handle:
                self._file_handle.flush()

    def close(self) -> None:
        """Close the logger and release resources.

        Flushes and closes the file handle if file logging was enabled.
        Should be called when done logging to ensure all data is written.
        """
        with self._lock:
            if self._file_handle:
                self._file_handle.flush()
                self._file_handle.close()
                self._file_handle = None

    def to_dict_list(self) -> list[dict[str, Any]]:
        """Convert all events to a list of dictionaries.

        Useful for serialization to JSON or other formats.

        Returns:
            List of dictionaries representing all logged events
        """
        with self._lock:
            return [e.to_dict() for e in self._events]

    def to_json(self, indent: int | None = 2) -> str:
        """Convert all events to a JSON string.

        Args:
            indent: Indentation level for pretty printing (None for compact)

        Returns:
            JSON string representing all logged events
        """
        return json.dumps(self.to_dict_list(), indent=indent, default=str)

    @property
    def event_count(self) -> int:
        """Get the total number of logged events."""
        with self._lock:
            return len(self._events)

    @property
    def file_path(self) -> Path | None:
        """Get the file path if file logging is enabled."""
        return self._file_path

    def __enter__(self) -> "EventLogger":
        """Support context manager protocol."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Close the logger when exiting context."""
        self.close()

    def __repr__(self) -> str:
        """String representation of the logger."""
        file_info = f", file={self._file_path}" if self._file_path else ""
        return f"EventLogger(events={len(self._events)}{file_info})"
