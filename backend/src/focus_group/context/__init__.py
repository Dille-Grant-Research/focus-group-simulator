"""Context management module for handling participant conversation context.

This module provides the dual-context system where each participant maintains:
- A full context with all messages including non-verbal reactions
- A verbal-only context representing what was actually spoken/heard

It also provides context summarization for managing token limits, and a
high-level manager that coordinates storage and summarization.
"""

from focus_group.context.manager import ContextManager
from focus_group.context.storage import ContextStorage
from focus_group.context.summarizer import ContextSummarizer

__all__ = ["ContextManager", "ContextStorage", "ContextSummarizer"]
