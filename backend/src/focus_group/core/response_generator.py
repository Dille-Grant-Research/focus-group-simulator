"""Response generator for focus group participants.

This module provides the ResponseGenerator class that uses an LLM to generate
verbal responses from participants based on their persona and conversation context.
"""

from focus_group.core.participant import Participant
from focus_group.llm.base import BaseLLMProvider
from focus_group.models.memory import StanceMatrix

# Keywords indicating a "strong" non-verbal reaction worth highlighting.
# Includes both the original detection phrases and vocabulary explicitly
# encouraged by the nonverbal generator prompt (gerund + present tense).
_STRONG_REACTION_KEYWORDS = [
    # Original detection phrases
    "leans forward",
    "shakes head",
    "nods vigorously",
    "eyes widen",
    "crosses arms",
    "frowns",
    "raises eyebrows",
    "sits up straight",
    "shifts forward",
    "opens mouth",
    "looks skeptical",
    "jaw drops",
    "scoffs",
    "rolls eyes",
    "gasps",
    "winces",
    "makes direct eye contact",
    # From nonverbal generator vocabulary (gerund + present tense)
    "tapping fingers",
    "taps fingers",
    "adjusting glasses",
    "adjusts glasses",
    "shifting weight",
    "shifts weight",
    "pressing lips",
    "presses lips",
    "drumming on",
    "drums on",
    "tucking hair",
    "tucks hair",
    "rubbing hands",
    "rubs hands",
    "looking up at the ceiling",
    "looks up at the ceiling",
    "fidgeting with",
    "fidgets with",
    "exhaling audibly",
    "exhales audibly",
    "resting chin",
    "rests chin",
    "glancing sideways",
    "glances sideways",
    "pursing lips",
    "purses lips",
    "uncrossing legs",
    "uncrosses legs",
    "picking at",
    "picks at",
    "straightening in seat",
    "straightens in seat",
    "rubbing the back of",
    "rubs the back of",
    # Additional strong-reaction cues
    "leaning in",
    "leans in",
    "opens mouth slightly",
    "thin line",
    "bridge of nose",
    "tracing patterns",
    "rotates pen",
    "rotating pen",
]

# Body-part nouns used as a fallback when no keyword matches.
# Catches creative LLM-generated reactions that describe physical actions.
_BODY_PART_INDICATORS = frozenset(
    {
        "fingers",
        "lips",
        "eyebrows",
        "jaw",
        "eyes",
        "mouth",
        "chin",
        "hands",
        "glasses",
        "fist",
        "brow",
        "neck",
        "shoulders",
        "forehead",
        "pen",
        "armrest",
        "temple",
    }
)


class ResponseGenerator:
    """Generates verbal responses from focus group participants.

    This class uses an LLM to generate in-character responses based on
    the participant's persona and their full conversation context
    (including non-verbal reactions they've observed).

    The generator handles two main scenarios:
    1. Introductions - Brief self-introductions at the start of the focus group
    2. Discussion responses - Responses to questions or other participants

    Attributes:
        provider: The LLM provider used for generating responses

    Example:
        >>> generator = ResponseGenerator(llm_provider)
        >>> intro = await generator.generate_introduction(participant)
        >>> print(intro)
        "Hi everyone, I'm Sarah Chen. I work as a software engineer..."
    """

    def __init__(self, provider: BaseLLMProvider) -> None:
        """Initialize the response generator.

        Args:
            provider: The LLM provider to use for generating responses
        """
        self._provider = provider

    async def generate_response(
        self,
        participant: Participant,
        prompt: str,
        stance_matrix: StanceMatrix | None = None,
    ) -> str:
        """Generate a verbal response from a participant.

        This method generates what the participant would actually say
        in response to a discussion prompt. The response is based on:
        - The participant's persona (personality, background, communication style)
        - The full conversation context (including non-verbal reactions)
        - The specific prompt/question being responded to

        When the participant has structured memory and a stance_matrix is
        provided, the prompt is enriched with memory-tier context instead
        of raw conversation history.

        Args:
            participant: The participant generating the response
            prompt: The discussion prompt or question to respond to
            stance_matrix: Optional shared StanceMatrix for memory-enriched prompts

        Returns:
            The participant's verbal response (what they say out loud)

        Note:
            This method will check if context summarization is needed
            before generating the response.
        """
        # Check and summarize context if needed
        await participant.check_and_summarize()

        # Build the system prompt from persona
        system_prompt = self._build_response_system_prompt(participant)

        # Memory-enriched path: structured three-tier context
        if participant.memory is not None and stance_matrix is not None:
            user_prompt = self._build_memory_enriched_prompt(participant, prompt, stance_matrix)
        else:
            # Legacy path: raw conversation context
            context = participant.get_context_for_llm(include_nonverbals=True)
            user_prompt = self._build_response_user_prompt(context, prompt)

        # Generate the response
        response = await self._provider.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
        )

        return self._clean_response(response)

    async def generate_introduction(self, participant: Participant) -> str:
        """Generate a self-introduction from a participant.

        This method generates a brief introduction that the participant
        would give at the start of a focus group session. The introduction
        should reflect their personality and communication style.

        Args:
            participant: The participant introducing themselves

        Returns:
            The participant's introduction (what they say out loud)
        """
        # Check and summarize context if needed
        await participant.check_and_summarize()

        # Build the system prompt from persona
        system_prompt = self._build_introduction_system_prompt(participant)

        # Build the user prompt for introduction
        user_prompt = self._build_introduction_user_prompt(participant)

        # Generate the introduction
        response = await self._provider.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
        )

        return self._clean_response(response)

    def _build_response_system_prompt(self, participant: Participant) -> str:
        """Build the system prompt for generating a response.

        Args:
            participant: The participant generating the response

        Returns:
            The system prompt establishing persona and behavior
        """
        persona = participant.persona

        # Stance-aware engagement instruction
        stance = getattr(persona, "discussion_stance", "balanced")
        stance_instructions = {
            "agreeable": "You tend to find common ground, but if something conflicts with your experience, say so",
            "balanced": "If you agree, explain WHY from your experience. If you disagree, say so directly",
            "challenging": "Look for flaws in arguments. Do NOT default to agreement. Question assumptions",
            "contrarian": "Find the counterargument. Play devil's advocate. Voice unpopular positions",
        }
        stance_line = stance_instructions.get(stance, stance_instructions["balanced"])

        # Register-aware style instruction
        register = getattr(persona, "communication_register", "conversational")
        register_instructions = {
            "formal": "Use complete sentences, proper grammar, measured tone",
            "casual": "Talk like you would to friends. Contractions, slang, incomplete sentences OK",
            "academic": "Reference frameworks, use precise vocabulary, qualify claims carefully",
            "folksy": "Use homespun expressions, anecdotes, plain language",
            "curt": "Keep it SHORT. 1-2 sentences max. No pleasantries. Get to the point",
            "warm": "Be encouraging and personal, but stay substantive",
            "conversational": "Natural spoken rhythm — contractions, mid-sentence corrections, trailing thoughts are fine",
        }
        register_line = register_instructions.get(register, register_instructions["conversational"])

        return f"""You are {persona.name} (pronouns: {persona.pronouns}), a {persona.age}-year-old participant in a focus group discussion.

Background: {persona.background}

Personality traits: {', '.join(persona.personality_traits)}

Communication style: {persona.communication_style}

IMPORTANT INSTRUCTIONS:
- Stay completely in character as {persona.name}
- Respond naturally as you would in a real focus group discussion
- Your response should be what you actually SAY out loud (verbal only)
- Keep your response concise and conversational (1-3 sentences typically)
- Draw on your background and personality when forming your opinion
- {stance_line}
- NEVER use: "great point", "that's interesting", "I completely agree", "well said", "absolutely right". If you agree, say WHY — don't just validate.
- Speaking style: {register_line}
- Do NOT include any stage directions, actions, or meta-commentary
- Do NOT start with phrases like "As {persona.name}..." or "I would say..."
- Just provide your natural spoken response
- Your personality traits ({', '.join(persona.personality_traits)}) should be CLEARLY EVIDENT in your tone, word choices, and perspective
- If you are "humble," keep your language modest and understated — avoid effusive or self-congratulatory phrasing
- If you are "analytical," focus on reasoning and evidence
- If you are "playful," let humor come through naturally
- Match the LENGTH and FORMALITY of your response to your communication style
- VARY your sentence openers. Do NOT repeatedly start responses with "Honestly,", "For me,", "I think", "Well,", "You know," or other filler phrases
- Look at what you have already said in the conversation above — choose a DIFFERENT opening style each time
- Be direct and specific rather than hedging with filler language
- Pay attention to non-verbal reactions shown in [brackets] in the conversation
- These bracketed entries represent body language you can see in the room
- When prompted that you notice specific body language, weave a brief, natural acknowledgment into your response (e.g., "I can see you're nodding, David..." or "You look skeptical, Maya...")
- Keep any body language references brief and natural - a short aside, not the focus of your response"""

    def _build_response_user_prompt(self, context: str, prompt: str) -> str:
        """Build the user prompt for generating a response.

        Args:
            context: The formatted conversation context
            prompt: The specific prompt to respond to

        Returns:
            The user prompt with context and prompt
        """
        return f"""Here is the conversation so far in this focus group:

{context}

Note: Entries in [brackets] like [Name's reaction: ...] are body language you can see in the room.

---

Now respond to the following:
{prompt}

Remember: Provide only your verbal response - what you would actually say out loud."""

    def _build_memory_enriched_prompt(
        self,
        participant: Participant,
        prompt: str,
        stance_matrix: StanceMatrix,
    ) -> str:
        """Build a prompt enriched with three-tier memory context.

        Instead of dumping the raw conversation, this method constructs a
        targeted prompt using:
        - Episodic memory (identity)
        - Question history with stance positions
        - Working memory (recent exchanges)

        Args:
            participant: The participant generating the response
            prompt: The specific discussion prompt
            stance_matrix: The shared StanceMatrix for stance lookups

        Returns:
            The enriched user prompt string
        """
        memory = participant.memory
        assert memory is not None  # Caller guarantees this

        ctx = memory.build_speak_context(stance_matrix)

        parts = []

        # Identity context
        parts.append("=== YOUR IDENTITY ===")
        parts.append(ctx["episodic_memory"])
        parts.append("")

        # Question history (if any)
        if (
            ctx["question_history"]
            and ctx["question_history"] != "No previous questions discussed."
        ):
            parts.append("=== PREVIOUS QUESTIONS ===")
            parts.append(ctx["question_history"])
            parts.append("")

        # Current question
        if ctx["question"]:
            parts.append(f"=== CURRENT QUESTION ===\n{ctx['question']}")
            parts.append("")

        # Working memory
        if ctx["working_memory"]:
            parts.append("=== RECENT CONVERSATION ===")
            parts.append(ctx["working_memory"])
            parts.append("")

        # Also include raw non-verbal context for body language cues
        context = participant.get_context_for_llm(include_nonverbals=True)
        # Extract only the bracketed non-verbal lines
        nonverbal_lines = [
            line.strip()
            for line in context.split("\n")
            if line.strip().startswith("[") and "reaction:" in line.lower()
        ]
        if nonverbal_lines:
            parts.append("=== BODY LANGUAGE YOU CAN SEE ===")
            parts.extend(nonverbal_lines[-5:])  # Last 5 reactions
            parts.append("")

        parts.append("---")
        parts.append("")
        parts.append(f"Now respond to the following:\n{prompt}")
        parts.append("")
        parts.append(
            "Remember: Provide only your verbal response " "- what you would actually say out loud."
        )

        return "\n".join(parts)

    def _build_introduction_system_prompt(self, participant: Participant) -> str:
        """Build the system prompt for generating an introduction.

        Args:
            participant: The participant introducing themselves

        Returns:
            The system prompt for introduction
        """
        persona = participant.persona

        return f"""You are {persona.name} (pronouns: {persona.pronouns}), a {persona.age}-year-old participant in a focus group.

Background: {persona.background}

Personality traits: {', '.join(persona.personality_traits)}

Communication style: {persona.communication_style}

You are about to introduce yourself to the other focus group participants. Your introduction should:
- Be natural and conversational
- Reflect your personality and communication style
- Be brief (2-4 sentences)
- Share relevant background without oversharing
- Set the tone for how you'll participate in the discussion

Do NOT include stage directions or actions. Just provide what you would actually say."""

    def _build_introduction_user_prompt(self, participant: Participant) -> str:
        """Build the user prompt for generating an introduction.

        Args:
            participant: The participant introducing themselves

        Returns:
            The user prompt for introduction
        """
        return f"""The moderator has asked everyone to briefly introduce themselves to the group.

Please introduce yourself as {participant.name}. Share a bit about who you are in a natural, conversational way.

Remember: Provide only your verbal introduction - what you would actually say out loud."""

    def _clean_response(self, response: str) -> str:
        """Clean and normalize the LLM response.

        Removes common artifacts like quotation marks, stage directions,
        and ensures the response is clean spoken text.

        Args:
            response: The raw response from the LLM

        Returns:
            The cleaned response
        """
        cleaned = response.strip()

        # Remove surrounding quotation marks if present
        if cleaned.startswith('"') and cleaned.endswith('"'):
            cleaned = cleaned[1:-1].strip()
        if cleaned.startswith("'") and cleaned.endswith("'"):
            cleaned = cleaned[1:-1].strip()

        # Remove common prefixes that LLMs sometimes add
        prefixes_to_remove = [
            "I would say: ",
            "I'd say: ",
            "Response: ",
            "My response: ",
        ]
        for prefix in prefixes_to_remove:
            if cleaned.lower().startswith(prefix.lower()):
                cleaned = cleaned[len(prefix) :].strip()

        return cleaned

    async def generate_response_to_question(
        self,
        participant: Participant,
        question: str,
        stance_matrix: StanceMatrix | None = None,
    ) -> str:
        """Generate a response to a focus group question.

        This is a convenience method specifically for responding to
        moderator questions (as opposed to responding to other participants).

        Args:
            participant: The participant responding
            question: The focus group question
            stance_matrix: Optional StanceMatrix for memory-enriched prompts

        Returns:
            The participant's verbal response to the question
        """
        prompt = f'The moderator asked: "{question}"\n\nWhat is your response?'
        return await self.generate_response(participant, prompt, stance_matrix=stance_matrix)

    @staticmethod
    def _extract_notable_reaction(
        reactions: dict[str, str],
        responding_participant_name: str,
    ) -> tuple[str, str] | None:
        """Find a strong non-verbal reaction worth highlighting to the speaker.

        Scans reactions (excluding the responding participant) using two passes:
        1. Primary: keyword phrase match against _STRONG_REACTION_KEYWORDS
        2. Fallback: body-part noun match against _BODY_PART_INDICATORS

        The fallback catches creative LLM-generated reactions that use vocabulary
        not in the keyword list but still describe observable physical actions.

        Args:
            reactions: Dict mapping participant names to reaction descriptions
            responding_participant_name: Name of the participant about to speak
                (excluded from search)

        Returns:
            Tuple of (reactor_name, reaction_text) if a strong reaction is found,
            None otherwise
        """
        for name, reaction in reactions.items():
            if name == responding_participant_name:
                continue
            reaction_lower = reaction.lower()
            # Primary: keyword phrase match
            for keyword in _STRONG_REACTION_KEYWORDS:
                if keyword in reaction_lower:
                    return (name, reaction)
            # Fallback: body-part indicator match
            for part in _BODY_PART_INDICATORS:
                if part in reaction_lower:
                    return (name, reaction)
        return None

    @staticmethod
    def _simplify_reaction(reaction: str, participant_name: str) -> str:
        """Simplify a reaction string for injection into a response prompt.

        Strips the leading participant name and lowercases the first character
        to produce phrasing suitable for "you notice X is [result]".

        Args:
            reaction: The raw reaction text
            participant_name: The reactor's name to strip from the start

        Returns:
            Simplified reaction string
        """
        simplified = reaction.strip()
        # Strip leading "Name " or "Name's "
        for prefix in [f"{participant_name}'s ", f"{participant_name} "]:
            if simplified.startswith(prefix):
                simplified = simplified[len(prefix) :]
                break
        # Also try first name only
        first_name = participant_name.split()[0]
        for prefix in [f"{first_name}'s ", f"{first_name} "]:
            if simplified.startswith(prefix):
                simplified = simplified[len(prefix) :]
                break
        # Lowercase the first character
        if simplified and simplified[0].isupper():
            simplified = simplified[0].lower() + simplified[1:]
        return simplified

    async def generate_response_to_participant(
        self,
        participant: Participant,
        speaker_name: str,
        speaker_statement: str,
        question: str,
        reactions: dict[str, str] | None = None,
        stance_matrix: StanceMatrix | None = None,
    ) -> str:
        """Generate a response to another participant's statement.

        This is a convenience method for when a participant is responding
        to something another participant said, while keeping the original
        question in context.

        Args:
            participant: The participant responding
            speaker_name: The name of the person who just spoke
            speaker_statement: What they said
            question: The focus group question being discussed
            reactions: Optional dict of current non-verbal reactions from
                other participants. Used to inject notable body language
                cues into the prompt.
            stance_matrix: Optional StanceMatrix for memory-enriched prompts

        Returns:
            The participant's verbal response
        """
        prompt = (
            f'The group is discussing: "{question}"\n\n'
            f'{speaker_name} just said: "{speaker_statement}"\n\n'
            f"How do you respond? You can build on their point, respectfully disagree, "
            f"share a contrasting experience, or take the conversation in a new direction.\n\n"
            f"Do NOT start by validating what they said. Jump straight to your own point."
        )

        # Inject notable reaction cue if available
        if reactions:
            notable = self._extract_notable_reaction(reactions, participant.name)
            if notable:
                reactor_name, reaction_text = notable
                simplified = self._simplify_reaction(reaction_text, reactor_name)
                prompt += (
                    f"\n\nAs you prepare to speak, you notice {reactor_name} is "
                    f"{simplified}. You may naturally acknowledge what you observe."
                )

        return await self.generate_response(participant, prompt, stance_matrix=stance_matrix)

    async def generate_closing_response(
        self,
        participant: Participant,
        closing_question: str,
        stance_matrix: StanceMatrix | None = None,
    ) -> str:
        """Generate a closing response that reflects on the actual discussion.

        This method generates a response to the closing question that
        references actual points made during the conversation rather than
        generating generic focus group closing statements.

        Args:
            participant: The participant giving closing thoughts
            closing_question: The closing question from the moderator
            stance_matrix: Optional StanceMatrix for memory-enriched prompts

        Returns:
            The participant's closing verbal response
        """
        persona = participant.persona
        prompt = (
            f'The moderator asked: "{closing_question}"\n\n'
            f"You are {persona.name}. Speak in FIRST person. "
            f"Do NOT refer to yourself by name or in third person.\n\n"
            f"Reflect on THIS specific discussion you just had. Reference "
            f"actual points made by other participants by name. Only describe "
            f"agreements or disagreements that ACTUALLY happened in the conversation "
            f"above. Do not claim tension or disagreement if the conversation was "
            f"mostly agreeable, and vice versa.\n\n"
            f"What are your closing thoughts?"
        )
        return await self.generate_response(participant, prompt, stance_matrix=stance_matrix)
