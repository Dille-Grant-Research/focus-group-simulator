"""Transcript writer for clean verbal-only focus group transcripts.

This module provides the TranscriptWriter class that writes clean transcripts
containing only verbal content, excluding all non-verbal reactions and
debugging information.
"""

import json
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, TextIO


class TranscriptFormat(Enum):
    """Supported output formats for transcripts."""

    TEXT = "text"  # Plain text with simple formatting
    MARKDOWN = "markdown"  # Markdown formatted for easy reading
    JSONL = "jsonl"  # Structured JSONL with verbal-only content


class TranscriptWriter:
    """Writer for clean verbal-only focus group transcripts.

    Writes transcripts that contain ONLY verbal content, excluding:
    - All non-verbal reactions
    - Speaker selection reasoning
    - Debug timing information

    This is the clean output suitable for analysis or sharing, in contrast
    to DebugLogWriter which includes all debug information.

    Supports multiple output formats:
    - TEXT: Plain text with simple formatting
    - MARKDOWN: Markdown formatted for easy reading
    - JSONL: Structured JSONL with verbal-only content

    Attributes:
        output_path: Path where the transcript is written
        format: The output format being used
        include_timestamps: Whether to include timestamps for responses

    Example:
        >>> writer = TranscriptWriter(
        ...     output_path="transcripts/session.md",
        ...     format="markdown"
        ... )
        >>> writer.write_session(result=session_result)

    Output Example (Markdown):
        # Focus Group Transcript

        **Purpose**: Discuss remote work policies

        ## Participants
        - Sarah Chen (34) - Software Engineer
        - John Smith (52) - Retired Teacher

        ## Introductions

        **Sarah Chen**: Hi everyone, I'm Sarah...

        ## Question 1: What do you think about remote work?

        **Sarah Chen**: I think remote work has really changed...

        **John Smith**: From my perspective...
    """

    def __init__(
        self,
        output_path: str | Path,
        format: str = "markdown",
        include_timestamps: bool = False,
    ) -> None:
        """Initialize the transcript writer.

        Args:
            output_path: Where to write the transcript
            format: Output format - "text", "markdown", or "jsonl"
            include_timestamps: If True, include timestamps for each response

        Raises:
            ValueError: If format is not supported
        """
        self._output_path = Path(output_path)
        self._include_timestamps = include_timestamps

        # Validate and set format
        try:
            self._format = TranscriptFormat(format.lower())
        except ValueError:
            valid_formats = [f.value for f in TranscriptFormat]
            raise ValueError(f"Unsupported format '{format}'. Valid formats: {valid_formats}")

        # Ensure parent directory exists
        self._output_path.parent.mkdir(parents=True, exist_ok=True)

        # File handle
        self._file_handle: TextIO | None = None

    def write_session(
        self,
        result: Any,  # SessionResult - avoiding circular import
    ) -> None:
        """Write a complete session transcript.

        This is the primary method for generating transcripts. Writes ONLY
        verbal content, excluding all non-verbal reactions and debug info.

        Args:
            result: The SessionResult object from a completed session

        Raises:
            IOError: If writing to file fails
        """
        if self._format == TranscriptFormat.TEXT:
            self._write_session_text(result)
        elif self._format == TranscriptFormat.MARKDOWN:
            self._write_session_markdown(result)
        elif self._format == TranscriptFormat.JSONL:
            self._write_session_jsonl(result)

        self.close()

    def _write_session_text(self, result: Any) -> None:
        """Write session transcript in plain text format.

        Args:
            result: The SessionResult object
        """
        lines: list[str] = []

        # Header
        lines.append("=" * 60)
        lines.append("FOCUS GROUP TRANSCRIPT")
        lines.append("=" * 60)
        lines.append("")

        # Session info
        if result.session_name:
            lines.append(f"Session: {result.session_name}")
        if result.purpose:
            lines.append(f"Purpose: {result.purpose}")
        if result.started_at:
            lines.append(f"Date: {result.started_at.strftime('%Y-%m-%d %H:%M')}")
        lines.append("")

        # Participants
        lines.append("-" * 40)
        lines.append("PARTICIPANTS")
        lines.append("-" * 40)
        for p in result.participants:
            lines.append(f"  - {p.name} ({p.age}) - {self._get_brief_background(p)}")
        lines.append("")

        # Introductions
        if result.introduction_result and result.introduction_result.introductions:
            lines.append("-" * 40)
            lines.append("INTRODUCTIONS")
            lines.append("-" * 40)
            lines.append("")
            for intro in result.introduction_result.introductions:
                if self._include_timestamps:
                    ts = intro.timestamp.strftime("%H:%M:%S")
                    lines.append(f"[{ts}] {intro.speaker_name}:")
                else:
                    lines.append(f"{intro.speaker_name}:")
                lines.append(f"  {intro.content}")
                lines.append("")

        # Questions
        for i, qr in enumerate(result.question_results, start=1):
            lines.append("-" * 40)
            lines.append(f"QUESTION {i}: {qr.question}")
            lines.append("-" * 40)
            lines.append("")
            for response in qr.responses:
                if self._include_timestamps:
                    ts = response.timestamp.strftime("%H:%M:%S")
                    lines.append(f"[{ts}] {response.speaker_name}:")
                else:
                    lines.append(f"{response.speaker_name}:")
                lines.append(f"  {response.content}")
                lines.append("")

        # Closing
        if result.closing_result and result.closing_result.responses:
            lines.append("-" * 40)
            lines.append("CLOSING")
            lines.append("-" * 40)
            lines.append("")
            for response in result.closing_result.responses:
                if self._include_timestamps:
                    ts = response.timestamp.strftime("%H:%M:%S")
                    lines.append(f"[{ts}] {response.speaker_name}:")
                else:
                    lines.append(f"{response.speaker_name}:")
                lines.append(f"  {response.content}")
                lines.append("")

        # Footer
        lines.append("=" * 60)
        lines.append("END OF TRANSCRIPT")
        lines.append("=" * 60)

        with open(self._output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def _write_session_markdown(self, result: Any) -> None:
        """Write session transcript in Markdown format.

        Args:
            result: The SessionResult object
        """
        lines: list[str] = []

        # Header
        lines.append("# Focus Group Transcript")
        lines.append("")

        # Session info
        if result.session_name:
            lines.append(f"**Session**: {result.session_name}")
            lines.append("")
        if result.purpose:
            lines.append(f"**Purpose**: {result.purpose}")
            lines.append("")
        if result.started_at:
            lines.append(f"**Date**: {result.started_at.strftime('%Y-%m-%d %H:%M')}")
            lines.append("")

        # Participants
        lines.append("## Participants")
        lines.append("")
        for p in result.participants:
            lines.append(f"- **{p.name}** ({p.age}) - {self._get_brief_background(p)}")
        lines.append("")

        # Introductions
        if result.introduction_result and result.introduction_result.introductions:
            lines.append("## Introductions")
            lines.append("")
            for intro in result.introduction_result.introductions:
                self._append_verbal_entry_md(
                    lines=lines,
                    speaker=intro.speaker_name,
                    content=intro.content,
                    timestamp=intro.timestamp if self._include_timestamps else None,
                )

        # Questions
        for i, qr in enumerate(result.question_results, start=1):
            lines.append(f"## Question {i}: {qr.question}")
            lines.append("")
            for response in qr.responses:
                self._append_verbal_entry_md(
                    lines=lines,
                    speaker=response.speaker_name,
                    content=response.content,
                    timestamp=response.timestamp if self._include_timestamps else None,
                )

        # Closing
        if result.closing_result and result.closing_result.responses:
            lines.append("## Closing")
            lines.append("")
            for response in result.closing_result.responses:
                self._append_verbal_entry_md(
                    lines=lines,
                    speaker=response.speaker_name,
                    content=response.content,
                    timestamp=response.timestamp if self._include_timestamps else None,
                )

        # Footer
        lines.append("---")
        lines.append("")
        lines.append("*Transcript generated by AI Focus Group Simulator*")

        with open(self._output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def _write_session_jsonl(self, result: Any) -> None:
        """Write session transcript in JSONL format.

        Each line is a JSON object representing a verbal entry (introduction
        or response). Excludes all non-verbal content.

        Args:
            result: The SessionResult object
        """
        with open(self._output_path, "w", encoding="utf-8") as f:
            # Header record
            header = {
                "type": "header",
                "session_name": result.session_name,
                "purpose": result.purpose,
                "started_at": (result.started_at.isoformat() if result.started_at else None),
                "participant_count": len(result.participants),
            }
            f.write(json.dumps(header) + "\n")

            # Participants
            for p in result.participants:
                participant = {
                    "type": "participant",
                    "name": p.name,
                    "age": p.age,
                    "background": self._get_brief_background(p),
                }
                f.write(json.dumps(participant) + "\n")

            # Introductions
            if result.introduction_result:
                for intro in result.introduction_result.introductions:
                    entry = {
                        "type": "introduction",
                        "speaker": intro.speaker_name,
                        "content": intro.content,
                        "order": intro.order,
                    }
                    if self._include_timestamps:
                        entry["timestamp"] = intro.timestamp.isoformat()
                    f.write(json.dumps(entry) + "\n")

            # Question responses
            for i, qr in enumerate(result.question_results, start=1):
                # Question marker
                q_marker = {
                    "type": "question",
                    "number": i,
                    "question": qr.question,
                }
                f.write(json.dumps(q_marker) + "\n")

                # Responses
                for response in qr.responses:
                    entry = {
                        "type": "response",
                        "question_number": i,
                        "speaker": response.speaker_name,
                        "content": response.content,
                        "turn": response.turn_number,
                    }
                    if self._include_timestamps:
                        entry["timestamp"] = response.timestamp.isoformat()
                    f.write(json.dumps(entry) + "\n")

            # Closing
            if result.closing_result and result.closing_result.responses:
                q_marker = {
                    "type": "closing_question",
                    "question": result.closing_result.question,
                }
                f.write(json.dumps(q_marker) + "\n")

                for response in result.closing_result.responses:
                    entry = {
                        "type": "closing_response",
                        "speaker": response.speaker_name,
                        "content": response.content,
                        "turn": response.turn_number,
                    }
                    if self._include_timestamps:
                        entry["timestamp"] = response.timestamp.isoformat()
                    f.write(json.dumps(entry) + "\n")

            # Footer
            footer = {
                "type": "footer",
                "completed_at": (result.completed_at.isoformat() if result.completed_at else None),
                "total_questions": result.total_questions,
                "total_responses": result.total_responses,
            }
            f.write(json.dumps(footer) + "\n")

    def _append_verbal_entry_md(
        self,
        lines: list[str],
        speaker: str,
        content: str,
        timestamp: datetime | None = None,
    ) -> None:
        """Append a verbal entry in Markdown format.

        Args:
            lines: The list of lines to append to
            speaker: Name of the speaker
            content: What they said
            timestamp: Optional timestamp to include
        """
        if timestamp:
            ts = timestamp.strftime("%H:%M:%S")
            lines.append(f"**{speaker}** *({ts})*: {content}")
        else:
            lines.append(f"**{speaker}**: {content}")
        lines.append("")

    def _get_brief_background(self, persona: Any) -> str:
        """Get a brief background description from a persona.

        Args:
            persona: A Persona object

        Returns:
            A brief description (first sentence or truncated)
        """
        background = persona.background
        # Get first sentence or truncate
        if ". " in background:
            return background.split(". ")[0]
        if len(background) > 60:
            return background[:57] + "..."
        return background

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
    def format(self) -> TranscriptFormat:
        """Get the output format."""
        return self._format

    @property
    def include_timestamps(self) -> bool:
        """Whether timestamps are included in the output."""
        return self._include_timestamps

    def __enter__(self) -> "TranscriptWriter":
        """Support context manager protocol."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Close the writer when exiting context."""
        self.close()

    def __repr__(self) -> str:
        """String representation of the writer."""
        return (
            f"TranscriptWriter(path={self._output_path}, "
            f"format={self._format.value}, timestamps={self._include_timestamps})"
        )
