"""Persona validation for quality assurance.

This module provides validation functionality to ensure generated personas
meet quality standards before being used in focus group simulations.
"""

from dataclasses import dataclass, field
from enum import Enum

from focus_group.models.persona import Persona


class ValidationSeverity(str, Enum):
    """Severity levels for validation issues."""

    ERROR = "error"  # Must be fixed before use
    WARNING = "warning"  # Should be reviewed but usable
    INFO = "info"  # Minor suggestion


@dataclass
class ValidationIssue:
    """Represents a single validation issue.

    Attributes:
        field: The persona field with the issue
        message: Description of the issue
        severity: How serious the issue is
    """

    field: str
    message: str
    severity: ValidationSeverity


@dataclass
class ValidationResult:
    """Result of persona validation.

    Attributes:
        is_valid: True if no errors were found
        issues: List of validation issues found
        persona: The persona that was validated
    """

    is_valid: bool
    issues: list[ValidationIssue] = field(default_factory=list)
    persona: Persona | None = None

    @property
    def errors(self) -> list[ValidationIssue]:
        """Get only error-level issues.

        Returns:
            List of issues with ERROR severity
        """
        return [i for i in self.issues if i.severity == ValidationSeverity.ERROR]

    @property
    def warnings(self) -> list[ValidationIssue]:
        """Get only warning-level issues.

        Returns:
            List of issues with WARNING severity
        """
        return [i for i in self.issues if i.severity == ValidationSeverity.WARNING]

    def has_errors(self) -> bool:
        """Check if there are any error-level issues.

        Returns:
            True if there are errors
        """
        return len(self.errors) > 0

    def summary(self) -> str:
        """Get a summary of validation results.

        Returns:
            Human-readable summary string
        """
        if self.is_valid and not self.issues:
            return "Persona is valid with no issues."

        parts = []
        if self.is_valid:
            parts.append("Persona is valid.")
        else:
            parts.append("Persona has validation errors.")

        error_count = len(self.errors)
        warning_count = len(self.warnings)

        if error_count:
            parts.append(f"{error_count} error(s)")
        if warning_count:
            parts.append(f"{warning_count} warning(s)")

        return " ".join(parts)


class PersonaValidator:
    """Validates focus group personas for quality and completeness.

    This validator checks that personas meet minimum quality standards
    including field completeness, content quality, and consistency.
    """

    # Minimum lengths for text fields
    MIN_BACKGROUND_LENGTH = 50
    MIN_COMMUNICATION_STYLE_LENGTH = 30
    MIN_NAME_LENGTH = 3

    # Trait requirements
    MIN_TRAITS = 2
    MAX_TRAITS = 8

    def __init__(
        self,
        strict: bool = False,
    ) -> None:
        """Initialize the validator.

        Args:
            strict: If True, warnings are treated as errors
        """
        self.strict = strict

    def validate(self, persona: Persona) -> ValidationResult:
        """Validate a persona for quality and completeness.

        Args:
            persona: The persona to validate

        Returns:
            ValidationResult with validation status and any issues found
        """
        issues: list[ValidationIssue] = []

        # Validate each field
        issues.extend(self._validate_name(persona.name))
        issues.extend(self._validate_age(persona.age))
        issues.extend(self._validate_background(persona.background))
        issues.extend(self._validate_personality_traits(persona.personality_traits))
        issues.extend(self._validate_communication_style(persona.communication_style))
        issues.extend(self._validate_demographic_source(persona.demographic_source))

        # Check for cross-field consistency
        issues.extend(self._validate_consistency(persona))

        # Determine if valid
        has_errors = any(i.severity == ValidationSeverity.ERROR for i in issues)
        has_warnings = any(i.severity == ValidationSeverity.WARNING for i in issues)

        is_valid = not has_errors
        if self.strict and has_warnings:
            is_valid = False

        return ValidationResult(
            is_valid=is_valid,
            issues=issues,
            persona=persona,
        )

    def _validate_name(self, name: str) -> list[ValidationIssue]:
        """Validate the persona name."""
        issues = []

        if len(name) < self.MIN_NAME_LENGTH:
            issues.append(
                ValidationIssue(
                    field="name",
                    message=f"Name is too short (minimum {self.MIN_NAME_LENGTH} characters)",
                    severity=ValidationSeverity.ERROR,
                )
            )

        # Check for placeholder-looking names
        placeholder_patterns = ["test", "example", "sample", "person", "user"]
        name_lower = name.lower()
        for pattern in placeholder_patterns:
            if pattern in name_lower and len(name) < 10:
                issues.append(
                    ValidationIssue(
                        field="name",
                        message=f"Name appears to be a placeholder (contains '{pattern}')",
                        severity=ValidationSeverity.WARNING,
                    )
                )
                break

        return issues

    def _validate_age(self, age: int) -> list[ValidationIssue]:
        """Validate the persona age."""
        issues = []

        # Pydantic already validates 18-100, but check for edge cases
        if age < 18 or age > 100:
            issues.append(
                ValidationIssue(
                    field="age",
                    message="Age must be between 18 and 100",
                    severity=ValidationSeverity.ERROR,
                )
            )

        return issues

    def _validate_background(self, background: str) -> list[ValidationIssue]:
        """Validate the persona background."""
        issues = []

        if len(background) < self.MIN_BACKGROUND_LENGTH:
            issues.append(
                ValidationIssue(
                    field="background",
                    message=f"Background is too brief (minimum {self.MIN_BACKGROUND_LENGTH} characters)",
                    severity=ValidationSeverity.ERROR,
                )
            )

        # Check for generic/placeholder content
        generic_phrases = [
            "generic background",
            "typical person",
            "average individual",
        ]
        background_lower = background.lower()
        for phrase in generic_phrases:
            if phrase in background_lower:
                issues.append(
                    ValidationIssue(
                        field="background",
                        message=f"Background contains generic phrasing ('{phrase}')",
                        severity=ValidationSeverity.WARNING,
                    )
                )
                break

        return issues

    def _validate_personality_traits(
        self,
        traits: list[str],
    ) -> list[ValidationIssue]:
        """Validate the personality traits."""
        issues = []

        if len(traits) < self.MIN_TRAITS:
            issues.append(
                ValidationIssue(
                    field="personality_traits",
                    message=f"Too few personality traits (minimum {self.MIN_TRAITS})",
                    severity=ValidationSeverity.ERROR,
                )
            )

        if len(traits) > self.MAX_TRAITS:
            issues.append(
                ValidationIssue(
                    field="personality_traits",
                    message=f"Too many personality traits (maximum {self.MAX_TRAITS})",
                    severity=ValidationSeverity.WARNING,
                )
            )

        # Check for duplicate traits
        unique_traits = set(t.lower().strip() for t in traits)
        if len(unique_traits) < len(traits):
            issues.append(
                ValidationIssue(
                    field="personality_traits",
                    message="Personality traits contain duplicates",
                    severity=ValidationSeverity.WARNING,
                )
            )

        # Check for very short traits
        for trait in traits:
            if len(trait.strip()) < 3:
                issues.append(
                    ValidationIssue(
                        field="personality_traits",
                        message=f"Trait '{trait}' is too short",
                        severity=ValidationSeverity.WARNING,
                    )
                )
                break

        return issues

    def _validate_communication_style(
        self,
        communication_style: str,
    ) -> list[ValidationIssue]:
        """Validate the communication style description."""
        issues = []

        if len(communication_style) < self.MIN_COMMUNICATION_STYLE_LENGTH:
            issues.append(
                ValidationIssue(
                    field="communication_style",
                    message=f"Communication style is too brief (minimum {self.MIN_COMMUNICATION_STYLE_LENGTH} characters)",
                    severity=ValidationSeverity.ERROR,
                )
            )

        return issues

    def _validate_demographic_source(
        self,
        demographic_source: str,
    ) -> list[ValidationIssue]:
        """Validate the demographic source field."""
        issues = []

        if not demographic_source or not demographic_source.strip():
            issues.append(
                ValidationIssue(
                    field="demographic_source",
                    message="Demographic source is empty",
                    severity=ValidationSeverity.ERROR,
                )
            )

        return issues

    def _validate_consistency(self, persona: Persona) -> list[ValidationIssue]:
        """Check for cross-field consistency issues."""
        issues = []

        # Check if background mentions the age approximately
        age_str = str(persona.age)
        age_decade = f"{(persona.age // 10) * 10}s"

        background_lower = persona.background.lower()
        has_age_reference = (
            age_str in persona.background
            or age_decade in background_lower
            or "year" in background_lower
        )

        if not has_age_reference:
            issues.append(
                ValidationIssue(
                    field="background",
                    message="Background doesn't reference the persona's age or life stage",
                    severity=ValidationSeverity.INFO,
                )
            )

        return issues

    def validate_batch(
        self,
        personas: list[Persona],
    ) -> list[ValidationResult]:
        """Validate multiple personas.

        Args:
            personas: List of personas to validate

        Returns:
            List of ValidationResults for each persona
        """
        return [self.validate(persona) for persona in personas]

    def all_valid(self, personas: list[Persona]) -> bool:
        """Check if all personas in a batch are valid.

        Args:
            personas: List of personas to check

        Returns:
            True if all personas are valid
        """
        results = self.validate_batch(personas)
        return all(r.is_valid for r in results)
