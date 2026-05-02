"""Debug log writer for comprehensive focus group session logs.

This module provides the DebugLogWriter class that writes full debug logs
including all non-verbal reactions for detailed analysis and debugging.
"""

import json
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, TextIO

try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

from focus_group.models.message import ConversationEvent


class OutputFormat(Enum):
    """Supported output formats for debug logs."""

    JSONL = "jsonl"  # JSON Lines - one JSON object per line (streaming-friendly)
    JSON = "json"  # Single JSON object (complete session)
    YAML = "yaml"  # YAML format (human-readable)


class DebugLogWriter:
    """Writer for comprehensive debug logs with all session details.

    Writes full debug logs including ALL non-verbal reactions, speaker
    selection reasoning, and timing information. This is distinct from
    the transcript writer which only includes verbal responses.

    Supports multiple output formats:
    - JSONL: One JSON object per line, ideal for streaming and large sessions
    - JSON: Single JSON object containing the entire session
    - YAML: Human-readable format for manual inspection

    Can be used in two modes:
    - Batch mode: Write entire SessionResult after session completes
    - Streaming mode: Write events in real-time during session

    Attributes:
        output_path: Path where the debug log is written
        format: The output format being used

    Example:
        >>> # Batch mode - write after session completes
        >>> writer = DebugLogWriter(
        ...     output_path="logs/session_debug.jsonl",
        ...     format="jsonl"
        ... )
        >>> writer.write_session(result=session_result)

        >>> # Streaming mode - write events in real-time
        >>> writer = DebugLogWriter("logs/live_debug.jsonl", format="jsonl")
        >>> for event in events:
        ...     writer.write_event(event)
        >>> writer.close()
    """

    def __init__(
        self,
        output_path: str | Path,
        format: str = "jsonl",
        pretty: bool = True,
    ) -> None:
        """Initialize the debug log writer.

        Args:
            output_path: Where to write the debug log
            format: Output format - "jsonl", "json", or "yaml"
            pretty: If True, format output for readability (indentation, etc.)

        Raises:
            ValueError: If format is not supported
            ImportError: If YAML format requested but PyYAML not installed
        """
        self._output_path = Path(output_path)
        self._pretty = pretty

        # Validate and set format
        try:
            self._format = OutputFormat(format.lower())
        except ValueError:
            valid_formats = [f.value for f in OutputFormat]
            raise ValueError(f"Unsupported format '{format}'. Valid formats: {valid_formats}")

        # Check YAML availability
        if self._format == OutputFormat.YAML and not YAML_AVAILABLE:
            raise ImportError(
                "PyYAML is required for YAML output format. " "Install with: pip install pyyaml"
            )

        # Ensure parent directory exists
        self._output_path.parent.mkdir(parents=True, exist_ok=True)

        # File handle for streaming mode
        self._file_handle: TextIO | None = None
        self._events_written = 0

    def _open_file(self, mode: str = "w") -> TextIO:
        """Open the output file.

        Args:
            mode: File mode ("w" for write, "a" for append)

        Returns:
            File handle
        """
        if self._file_handle is None:
            self._file_handle = open(self._output_path, mode, encoding="utf-8")
        return self._file_handle

    def _format_timestamp(self, dt: datetime | None) -> str | None:
        """Format a datetime for output.

        Args:
            dt: Datetime to format, or None

        Returns:
            ISO format string or None
        """
        return dt.isoformat() if dt else None

    def _serialize_value(self, obj: Any) -> Any:
        """Serialize a value for JSON/YAML output.

        Handles datetime objects and nested structures.

        Args:
            obj: Value to serialize

        Returns:
            Serializable value
        """
        if isinstance(obj, datetime):
            return obj.isoformat()
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        if isinstance(obj, dict):
            return {k: self._serialize_value(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._serialize_value(item) for item in obj]
        return obj

    def write_session(
        self,
        result: Any,  # SessionResult - avoiding circular import
        events: list[ConversationEvent] | None = None,
    ) -> None:
        """Write a complete session to the debug log.

        This is the primary method for batch mode writing after a session
        completes. Writes comprehensive debug information including all
        non-verbal reactions, speaker selection reasoning, and timing.

        Args:
            result: The SessionResult object from a completed session
            events: Optional list of ConversationEvent objects from EventLogger
                   for additional detail

        Raises:
            IOError: If writing to file fails
        """
        # Build the debug log structure
        debug_log = self._build_session_log(result, events)

        # Write based on format
        if self._format == OutputFormat.JSONL:
            self._write_session_jsonl(debug_log)
        elif self._format == OutputFormat.JSON:
            self._write_session_json(debug_log)
        elif self._format == OutputFormat.YAML:
            self._write_session_yaml(debug_log)

        self.close()

    def _build_session_log(
        self,
        result: Any,
        events: list[ConversationEvent] | None,
    ) -> dict[str, Any]:
        """Build the complete debug log structure from session result.

        Args:
            result: The SessionResult object
            events: Optional EventLogger events

        Returns:
            Dictionary containing the complete debug log
        """
        log = {
            "meta": {
                "log_type": "debug",
                "generated_at": datetime.now().isoformat(),
                "format_version": "1.0",
            },
            "session": {
                "name": result.session_name,
                "purpose": result.purpose,
                "started_at": self._format_timestamp(result.started_at),
                "completed_at": self._format_timestamp(result.completed_at),
                "duration_seconds": result.duration_seconds,
                "total_questions": result.total_questions,
                "total_responses": result.total_responses,
                "error_count": len(result.errors),
            },
            "configuration": {
                "demographics": result.demographics,
                "questions": result.questions,
            },
            "participants": self._build_participants_section(result.participants),
        }

        # Introduction phase
        if result.introduction_result:
            log["introduction_phase"] = self._build_introduction_section(result.introduction_result)

        # Question discussions
        log["question_discussions"] = [
            self._build_question_section(qr, i + 1) for i, qr in enumerate(result.question_results)
        ]

        # Closing question
        if result.closing_result:
            log["closing_question"] = self._build_question_section(
                result.closing_result,
                question_number=None,
                is_closing=True,
            )

        # Errors
        if result.errors:
            log["errors"] = result.errors

        # Memory artifacts (when memory system is enabled)
        if hasattr(result, "stance_matrix_dump") and result.stance_matrix_dump is not None:
            log["stance_matrix"] = result.stance_matrix_dump
        if (
            hasattr(result, "participant_memory_dumps")
            and result.participant_memory_dumps is not None
        ):
            log["participant_memories"] = result.participant_memory_dumps

        # Raw events if provided
        if events:
            log["raw_events"] = [self._serialize_value(e.to_dict()) for e in events]

        return log

    def _build_participants_section(self, participants: list) -> list[dict[str, Any]]:
        """Build the participants section of the debug log.

        Args:
            participants: List of Persona objects

        Returns:
            List of participant dictionaries
        """
        return [
            {
                "name": p.name,
                "age": p.age,
                "background": p.background,
                "personality_traits": p.personality_traits,
                "communication_style": p.communication_style,
                "pronouns": getattr(p, "pronouns", None),
                "discussion_stance": getattr(p, "discussion_stance", None),
                "communication_register": getattr(p, "communication_register", None),
                "demographic_source": p.demographic_source,
                "full_description": p.to_full_description(),
            }
            for p in participants
        ]

    def _build_introduction_section(self, intro_result: Any) -> dict[str, Any]:
        """Build the introduction phase section.

        Args:
            intro_result: IntroductionResult object

        Returns:
            Dictionary with introduction details and ALL reactions
        """
        section = {
            "purpose": intro_result.purpose,
            "started_at": self._format_timestamp(intro_result.started_at),
            "completed_at": self._format_timestamp(intro_result.completed_at),
            "introduction_order": intro_result.introduction_order,
            "introductions": [],
        }

        # Add each introduction with its associated reactions
        for intro in intro_result.introductions:
            intro_entry = {
                "speaker": intro.speaker_name,
                "content": intro.content,
                "order": intro.order,
                "timestamp": self._format_timestamp(intro.timestamp),
                # Include non-verbal reactions to this introduction
                "reactions_received": [
                    {
                        "from": r.participant_name,
                        "reaction": r.reaction,
                        "timestamp": self._format_timestamp(r.timestamp),
                    }
                    for r in intro_result.reactions
                    if r.reacting_to == intro.speaker_name
                ],
            }
            section["introductions"].append(intro_entry)

        return section

    def _build_question_section(
        self,
        question_result: Any,
        question_number: int | None,
        is_closing: bool = False,
    ) -> dict[str, Any]:
        """Build a question discussion section.

        Args:
            question_result: QuestionResult object
            question_number: Question number (1-indexed) or None for closing
            is_closing: Whether this is the closing question

        Returns:
            Dictionary with complete question discussion details
        """
        section = {
            "question": question_result.question,
            "question_number": question_number,
            "is_closing": is_closing,
            "started_at": self._format_timestamp(question_result.started_at),
            "completed_at": self._format_timestamp(question_result.completed_at),
            "speaking_order": question_result.speaking_order,
            "total_turns": question_result.total_turns,
        }

        # Initial reactions to the question (before anyone speaks)
        question_reactions = [
            r for r in question_result.nonverbal_reactions if r.stimulus_type == "question"
        ]
        section["initial_reactions_to_question"] = [
            {
                "participant": r.participant_name,
                "reaction": r.reaction,
                "timestamp": self._format_timestamp(r.timestamp),
            }
            for r in question_reactions
        ]

        # Build conversation flow with responses and reactions
        section["conversation_flow"] = []

        for response in question_result.responses:
            turn_entry = {
                "turn": response.turn_number,
                "speaker": response.speaker_name,
                "content": response.content,
                "selection_method": response.selection_method,
                "selection_reasoning": response.selection_reasoning,
                "timestamp": self._format_timestamp(response.timestamp),
                # Non-verbal reactions to this specific response (matched by turn number)
                "reactions_to_response": [
                    {
                        "from": r.participant_name,
                        "reaction": r.reaction,
                        "timestamp": self._format_timestamp(r.timestamp),
                    }
                    for r in question_result.nonverbal_reactions
                    if (
                        r.stimulus_type == "statement"
                        and r.stimulus_source == response.speaker_name
                        and getattr(r, "response_turn", None) == response.turn_number
                    )
                ],
            }
            section["conversation_flow"].append(turn_entry)

        return section

    def _write_session_jsonl(self, log: dict[str, Any]) -> None:
        """Write session log in JSONL format.

        Writes each major section as a separate line for easy streaming/parsing.

        Args:
            log: The complete debug log dictionary
        """
        with open(self._output_path, "w", encoding="utf-8") as f:
            # Write meta
            f.write(json.dumps({"type": "meta", "data": log["meta"]}) + "\n")

            # Write session info
            f.write(json.dumps({"type": "session", "data": log["session"]}) + "\n")

            # Write configuration
            f.write(json.dumps({"type": "configuration", "data": log["configuration"]}) + "\n")

            # Write each participant
            for p in log["participants"]:
                f.write(json.dumps({"type": "participant", "data": p}) + "\n")

            # Write introduction phase
            if "introduction_phase" in log:
                f.write(
                    json.dumps({"type": "introduction_phase", "data": log["introduction_phase"]})
                    + "\n"
                )

            # Write each question discussion
            for qd in log["question_discussions"]:
                f.write(json.dumps({"type": "question_discussion", "data": qd}) + "\n")

            # Write closing question
            if "closing_question" in log:
                f.write(
                    json.dumps({"type": "closing_question", "data": log["closing_question"]}) + "\n"
                )

            # Write errors
            if "errors" in log:
                f.write(json.dumps({"type": "errors", "data": log["errors"]}) + "\n")

            # Write memory artifacts (when memory system is enabled)
            if "stance_matrix" in log:
                f.write(json.dumps({"type": "stance_matrix", "data": log["stance_matrix"]}) + "\n")
            if "participant_memories" in log:
                for pm in log["participant_memories"]:
                    f.write(json.dumps({"type": "participant_memory", "data": pm}) + "\n")

            # Write raw events if present
            if "raw_events" in log:
                for event in log["raw_events"]:
                    f.write(json.dumps({"type": "raw_event", "data": event}) + "\n")

    def _write_session_json(self, log: dict[str, Any]) -> None:
        """Write session log as a single JSON object.

        Args:
            log: The complete debug log dictionary
        """
        indent = 2 if self._pretty else None
        with open(self._output_path, "w", encoding="utf-8") as f:
            json.dump(log, f, indent=indent, default=str)

    def _write_session_yaml(self, log: dict[str, Any]) -> None:
        """Write session log in YAML format.

        Args:
            log: The complete debug log dictionary
        """
        with open(self._output_path, "w", encoding="utf-8") as f:
            yaml.dump(
                log,
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
                width=120,
            )

    def write_event(self, event: ConversationEvent) -> None:
        """Write a single event to the debug log (streaming mode).

        Use this method for real-time logging during a session.
        Only supported for JSONL format.

        Args:
            event: The ConversationEvent to write

        Raises:
            ValueError: If format is not JSONL (streaming not supported)
        """
        if self._format != OutputFormat.JSONL:
            raise ValueError(
                f"Streaming mode (write_event) only supported for JSONL format, "
                f"not {self._format.value}"
            )

        f = self._open_file(mode="a")
        json_line = json.dumps(
            {
                "type": "event",
                "data": self._serialize_value(event.to_dict()),
            }
        )
        f.write(json_line + "\n")
        f.flush()
        self._events_written += 1

    def write_header(
        self,
        session_name: str | None = None,
        purpose: str | None = None,
        demographics: list[str] | None = None,
        questions: list[str] | None = None,
    ) -> None:
        """Write session header information (streaming mode).

        Call this at the start of a session when streaming events.
        Only supported for JSONL format.

        Args:
            session_name: Optional session name
            purpose: Optional focus group purpose
            demographics: List of demographic specifications
            questions: List of questions

        Raises:
            ValueError: If format is not JSONL
        """
        if self._format != OutputFormat.JSONL:
            raise ValueError(
                f"Streaming mode only supported for JSONL format, " f"not {self._format.value}"
            )

        f = self._open_file(mode="w")  # Start fresh

        # Write meta
        meta = {
            "log_type": "debug",
            "generated_at": datetime.now().isoformat(),
            "format_version": "1.0",
            "streaming": True,
        }
        f.write(json.dumps({"type": "meta", "data": meta}) + "\n")

        # Write session header
        session_header = {
            "name": session_name,
            "purpose": purpose,
            "started_at": datetime.now().isoformat(),
        }
        f.write(json.dumps({"type": "session_start", "data": session_header}) + "\n")

        # Write configuration if provided
        if demographics or questions:
            config = {
                "demographics": demographics or [],
                "questions": questions or [],
            }
            f.write(json.dumps({"type": "configuration", "data": config}) + "\n")

        f.flush()

    def write_footer(
        self,
        total_questions: int = 0,
        total_responses: int = 0,
        duration_seconds: float | None = None,
        errors: list[str] | None = None,
    ) -> None:
        """Write session footer information (streaming mode).

        Call this at the end of a session when streaming events.
        Only supported for JSONL format.

        Args:
            total_questions: Total questions processed
            total_responses: Total verbal responses
            duration_seconds: Session duration
            errors: List of errors that occurred

        Raises:
            ValueError: If format is not JSONL
        """
        if self._format != OutputFormat.JSONL:
            raise ValueError(
                f"Streaming mode only supported for JSONL format, " f"not {self._format.value}"
            )

        f = self._open_file(mode="a")

        footer = {
            "completed_at": datetime.now().isoformat(),
            "total_questions": total_questions,
            "total_responses": total_responses,
            "duration_seconds": duration_seconds,
            "error_count": len(errors) if errors else 0,
            "events_written": self._events_written,
        }
        f.write(json.dumps({"type": "session_end", "data": footer}) + "\n")

        if errors:
            f.write(json.dumps({"type": "errors", "data": errors}) + "\n")

        f.flush()

    def flush(self) -> None:
        """Flush any buffered data to the file."""
        if self._file_handle:
            self._file_handle.flush()

    def close(self) -> None:
        """Close the writer and release resources.

        Should be called when done writing to ensure all data is flushed.
        """
        if self._file_handle:
            self._file_handle.flush()
            self._file_handle.close()
            self._file_handle = None

    @property
    def output_path(self) -> Path:
        """Get the output file path."""
        return self._output_path

    @property
    def format(self) -> OutputFormat:
        """Get the output format."""
        return self._format

    @property
    def events_written(self) -> int:
        """Get the number of events written in streaming mode."""
        return self._events_written

    def __enter__(self) -> "DebugLogWriter":
        """Support context manager protocol."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Close the writer when exiting context."""
        self.close()

    def __repr__(self) -> str:
        """String representation of the writer."""
        return f"DebugLogWriter(path={self._output_path}, format={self._format.value})"
