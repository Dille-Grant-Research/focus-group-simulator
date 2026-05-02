"""Logging and transcript management module.

This module provides facilities for logging session events and
generating transcripts in various formats.

Exports:
    EventLogger: Central logging facility for all session events
    DebugLogWriter: Writer for comprehensive debug logs with all non-verbals
    OutputFormat: Enum of supported output formats for debug logs
    TranscriptWriter: Writer for clean verbal-only transcripts
    TranscriptFormat: Enum of supported output formats for transcripts
"""

from focus_group.logging.debug_writer import DebugLogWriter, OutputFormat
from focus_group.logging.event_logger import EventLogger
from focus_group.logging.transcript_writer import TranscriptFormat, TranscriptWriter

__all__ = [
    "DebugLogWriter",
    "EventLogger",
    "OutputFormat",
    "TranscriptFormat",
    "TranscriptWriter",
]
