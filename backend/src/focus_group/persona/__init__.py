"""Persona generation and validation module for focus group participants.

This module provides functionality to:
- Generate realistic personas from demographic specifications using LLM
- Validate personas for quality and completeness
"""

from focus_group.persona.generator import PersonaGenerationError, PersonaGenerator
from focus_group.persona.validator import (
    PersonaValidator,
    ValidationIssue,
    ValidationResult,
    ValidationSeverity,
)

__all__ = [
    "PersonaGenerationError",
    "PersonaGenerator",
    "PersonaValidator",
    "ValidationIssue",
    "ValidationResult",
    "ValidationSeverity",
]
