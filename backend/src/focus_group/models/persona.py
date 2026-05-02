"""Data models for participant personas.

This module defines the Persona model that represents a focus group participant's
identity, including their background, personality, and communication style.
"""

from typing import Any

from pydantic import BaseModel, Field, field_validator


class Persona(BaseModel):
    """Represents a focus group participant's persona.

    A persona is generated from demographic specifications and includes all
    the characteristics needed to simulate a realistic participant in the
    focus group discussion.

    Attributes:
        name: The participant's full name
        age: Age of the persona (18-100)
        background: Full background description including occupation, experience, etc.
        personality_traits: List of key personality characteristics
        communication_style: Description of how they communicate and interact
        demographic_source: Original demographic specification used to generate this
    """

    name: str = Field(
        description="The participant's full name",
        min_length=1,
    )
    age: int = Field(
        description="Age of the persona",
        ge=18,
        le=100,
    )
    background: str = Field(
        description="Full background description including occupation, experience, etc.",
        min_length=10,
    )
    personality_traits: list[str] = Field(
        description="List of key personality characteristics",
        min_length=1,
    )
    communication_style: str = Field(
        description="Description of how they communicate and interact",
        min_length=10,
    )
    pronouns: str = Field(
        default="they/them",
        description="Preferred pronouns (e.g., she/her, he/him, they/them)",
        min_length=2,
    )
    discussion_stance: str = Field(
        default="balanced",
        description="Discussion stance: agreeable, balanced, challenging, or contrarian",
    )
    communication_register: str = Field(
        default="conversational",
        description="Communication register: formal, casual, academic, folksy, curt, warm, or conversational",
    )
    demographic_source: str = Field(
        description="Original demographic specification used to generate this persona",
        min_length=1,
    )

    @field_validator("personality_traits")
    @classmethod
    def validate_traits(cls, v: list[str]) -> list[str]:
        """Ensure personality traits are non-empty strings.

        Args:
            v: List of personality traits

        Returns:
            Validated list of traits

        Raises:
            ValueError: If any trait is empty or whitespace-only
        """
        if not v:
            raise ValueError("At least one personality trait is required")

        for trait in v:
            if not trait or not trait.strip():
                raise ValueError("Personality traits cannot be empty")

        return v

    @field_validator("discussion_stance")
    @classmethod
    def validate_stance(cls, v: str) -> str:
        """Normalize discussion_stance to a known value."""
        valid = {"agreeable", "balanced", "challenging", "contrarian"}
        if v.lower().strip() in valid:
            return v.lower().strip()
        return "balanced"

    @field_validator("communication_register")
    @classmethod
    def validate_register(cls, v: str) -> str:
        """Normalize communication_register to a known value."""
        valid = {"formal", "casual", "academic", "folksy", "curt", "warm", "conversational"}
        if v.lower().strip() in valid:
            return v.lower().strip()
        return "conversational"

    def to_full_description(self) -> str:
        """Generate a full narrative description of this persona.

        This creates a human-readable description that can be used as the
        system prompt to establish the participant's identity for the LLM.

        Returns:
            A formatted narrative description of the persona
        """
        traits_str = ", ".join(self.personality_traits)

        stance_descriptions = {
            "agreeable": "You tend to find common ground and build on what others say.",
            "balanced": "You weigh different sides fairly before forming your opinion.",
            "challenging": "You instinctively probe assumptions and push back on weak arguments.",
            "contrarian": "You instinctively look for the other side of any argument and voice unpopular positions.",
        }
        stance_line = stance_descriptions.get(
            self.discussion_stance, stance_descriptions["balanced"]
        )

        description = f"""You are {self.name} (pronouns: {self.pronouns}), a {self.age}-year-old participant in this focus group.

Background: {self.background}

Personality: You are {traits_str}.

Communication Style: {self.communication_style}

Discussion Stance: {stance_line}

Remember to stay in character throughout the discussion. Respond as {self.name} would, \
drawing on your background and personality. Your responses should reflect your unique \
perspective and communication style."""

        return description

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Persona":
        """Create a Persona instance from a dictionary.

        This is a convenience method for deserializing personas from JSON
        or other dictionary-based formats.

        Args:
            data: Dictionary containing persona fields

        Returns:
            A new Persona instance

        Raises:
            ValidationError: If the data doesn't match the Persona schema
        """
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        """Convert this persona to a dictionary.

        Returns:
            Dictionary representation of the persona
        """
        return self.model_dump()
