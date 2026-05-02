"""Prompt templates for Research Assistant analysis.

These prompts instruct the LLM to objectively analyze participant turns
and generate discussion summaries. Both prompts require JSON-only output.
"""

ANALYZE_STANCE_PROMPT = """\
You are a research assistant objectively analyzing a focus group discussion.

A participant just spoke. Analyze their statement and determine their stance.

**Speaker:** {speaker_name}
**Question being discussed:** {question}
**What they said:** "{statement}"
**All participants in this focus group:** {participants_json}

Analyze the speaker's stance on the question. Consider:
- Are they supportive, skeptical, neutral, or mixed?
- What are the key points they made? (list 1-3 concise bullet points)
- What is their overall sentiment? (0.0 = very negative, 0.5 = neutral, 1.0 = very positive)

IMPORTANT: Be objective. Do NOT infer what they did not say. Only analyze what was actually stated.

Respond with ONLY valid JSON in this exact format (no markdown, no explanation):
{{"position": "supportive|skeptical|neutral|mixed", \
"key_points": ["point 1", "point 2"], "sentiment_score": 0.7}}
"""

SUMMARIZE_DISCUSSION_PROMPT = """\
You are a research assistant summarizing a focus group discussion.

**Question discussed:** {question}
**Transcript:**
{transcript}

**Participant stances so far:**
{participant_stances_json}

Generate an objective summary of this discussion. Consider:
- What were the main themes?
- How much did participants agree or disagree?
- Were there any notable interactions or exchanges?

IMPORTANT: Be factual and objective. Only describe what actually happened in the discussion.

Respond with ONLY valid JSON in this exact format (no markdown, no explanation):
{{"summary": "A concise 2-3 sentence summary.", \
"key_themes": ["theme 1", "theme 2"], \
"consensus_level": "high|moderate|low|none", \
"notable_interactions": ["interaction 1"]}}
"""

__all__ = [
    "ANALYZE_STANCE_PROMPT",
    "SUMMARIZE_DISCUSSION_PROMPT",
]
