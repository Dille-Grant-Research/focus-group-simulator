# SessionConfig Reference

**Audience:** C# frontend developers building the job submission form, and researchers who need to understand available configuration options.

This document describes every field in the `SessionConfig` object that is submitted to `POST /api/v1/jobs`. Fields are grouped by the level of expertise required — most researchers only need the Core Fields section.

---

## Minimal Example

The smallest valid configuration:

```json
{
  "demographics": [
    "Female, age 34, suburban Minnesota, works in healthcare",
    "Male, age 52, rural Minnesota, retired farmer"
  ],
  "questions": [
    "What are your biggest concerns about adopting telehealth services?"
  ]
}
```

All other fields have sensible defaults and can be omitted.

---

## Core Fields

These are the fields researchers interact with directly through the submission form.

### `demographics`

**Type:** array of strings  
**Required:** Yes  
**Constraints:** 1 to 15 items

A list of participant profiles. Each string is a free-text demographic description that the LLM uses to generate a realistic persona. Write them as you would describe a research participant to a colleague.

```json
"demographics": [
  "Female, age 34, suburban Minnesota, healthcare worker, moderate political views",
  "Male, age 52, rural Minnesota, retired farmer, skeptical of technology",
  "Non-binary, age 27, Minneapolis, graduate student in public health"
]
```

**Guidelines:**
- Include age, location, occupation, and any relevant attitudes or background
- Be specific — more detail produces more distinct personas
- The system validates that generated personas match the demographic spec

---

### `questions`

**Type:** array of strings  
**Required:** Yes  
**Constraints:** 1 to 20 items

The focus group questions, in the order they will be asked. The simulation works through each question before moving to the next.

```json
"questions": [
  "What are your first impressions of telehealth?",
  "Have you or someone you know used telehealth? What was that experience like?",
  "What would need to change for you to use telehealth regularly?"
]
```

---

### `purpose`

**Type:** string or null  
**Required:** No  
**Default:** null

An optional introductory context statement shown to all participants at the start of the session. Use this to frame the research topic before the first question.

```json
"purpose": "We are gathering perspectives on telehealth adoption in rural and suburban Minnesota communities."
```

---

### `closing_question`

**Type:** string  
**Required:** No  
**Default:** `"Do you have any concluding thoughts about participating in this focus group?"`

The final question asked after all `questions` have been discussed.

```json
"closing_question": "Is there anything else about telehealth that we haven't discussed that you'd like to share?"
```

---

### `session_name`

**Type:** string or null  
**Required:** No  
**Default:** null  
**Constraints:** Max 100 characters; alphanumeric, spaces, and hyphens only

An optional label for the session used in server-side output filenames. Does not affect simulation behavior.

```json
"session_name": "Telehealth Study Pilot-1"
```

---

## Advanced Fields

These fields control simulation behavior. Most researchers can leave them at defaults. Frontend forms may expose them in a collapsible "Advanced settings" section.

---

### `model`

**Type:** object  
**Required:** No  
**Default:** Ollama defaults (see below)

Controls the LLM used when running the engine directly. **When submitted through the API, the server ignores this connection info** and always uses the inference service configured by the `LLM_PROVIDER`, `LLM_BASE_URL` and `LLM_MODEL` environment variables (in production: the MSUM cruncher, an OpenAI-compatible endpoint).

**Sub-fields:**

| Field | Type | Default | Range | Description |
|-------|------|---------|-------|-------------|
| `model_name` | string | `"llama3.1:8b-instruct-q4_K_M"` | — | Ollama model identifier |
| `base_url` | string | `"http://localhost:11434"` | — | Ollama API URL (set from env in production) |
| `temperature` | float | `0.7` | 0.0 – 2.0 | Response randomness. Higher = more varied. |
| `top_p` | float | `0.9` | 0.0 – 1.0 | Nucleus sampling threshold |
| `num_ctx` | integer | `8192` | 1024 – 131072 | Context window size in tokens |
| `num_gpu` | integer | `-1` | — | GPU layers to use (-1 = all available) |

Example — custom temperature:
```json
"model": {
  "temperature": 0.9
}
```

---

### `context`

**Type:** object  
**Required:** No  
**Default:** See sub-fields below

Controls how the simulation manages token budgets for each participant's context window.

| Field | Type | Default | Range | Description |
|-------|------|---------|-------|-------------|
| `max_tokens` | integer | `6000` | 1000 – 128000 | Total token budget per participant context |
| `summarize_after_tokens` | integer | `4000` | 500 – 128000 | Token threshold that triggers context summarization |
| `reserved_for_persona` | integer | `500` | 100 – 2000 | Tokens always reserved for persona description |
| `reserved_for_question` | integer | `200` | 50 – 500 | Tokens always reserved for current question |
| `reserved_for_recent_response` | integer | `1000` | 200 – 4000 | Tokens always reserved for most recent response |

---

### `memory`

**Type:** object or null  
**Required:** No  
**Default:** null

When set to `null` (the default), the simulation runs in baseline mode without stance tracking. When provided, enables the `ResearchAssistant` component that tracks each participant's stance per topic and produces a structured analysis in the result.

| Field | Type | Default | Range | Description |
|-------|------|---------|-------|-------------|
| `enable_stance_tracking` | boolean | `true` | — | Enable the ResearchAssistant stance analysis |
| `working_memory_window` | integer | `10` | 1 – 50 | Recent turns included in each participant's working memory |
| `question_history_limit` | integer | `5` | 1 – 20 | Past questions included in prompt context |

Enable with defaults:
```json
"memory": {}
```

Enable with custom settings:
```json
"memory": {
  "enable_stance_tracking": true,
  "working_memory_window": 15
}
```

When `memory` is provided, the result's `analysis.stance_matrix` field will be populated.

---

### `output`

**Type:** object  
**Required:** No  
**Default:** See sub-fields below

Controls server-side file artifacts. These files are written to the server's filesystem and are separate from the API result JSON. **The C# frontend does not need to set this** — it only affects what gets written to disk on the server.

| Field | Type | Default | Options | Description |
|-------|------|---------|---------|-------------|
| `directory` | string | `"./output"` | — | Server-side output directory |
| `save_debug_log` | boolean | `true` | — | Save full debug log (includes non-verbals) |
| `save_clean_transcript` | boolean | `true` | — | Save verbal-only clean transcript |
| `log_format` | string | `"jsonl"` | `"jsonl"`, `"json"`, `"yaml"` | File format for log output |

---

### `validate_personas`

**Type:** boolean  
**Required:** No  
**Default:** `true`

When `true`, generated personas are validated against their demographic spec before the simulation begins. Set to `false` only if persona generation is failing and you want to debug the raw output.

---

### `group_dynamic`

**Type:** string, one of `"natural"`, `"cooperative"`, `"heated"`  
**Required:** No  
**Default:** `"natural"`

How participants tend to engage with each other. Each participant's *discussion stance* (how they argue, not what they think about the topic) is drawn from this preset's distribution:

| Preset | agreeable | balanced | challenging | contrarian | Limits (groups of 3+) |
|---|---|---|---|---|---|
| `natural` | 25% | 40% | 25% | 10% | at least 1 challenger; at most 40% challenging or contrarian |
| `cooperative` | 40% | 45% | 15% | 0% | at least 1 challenger; at most 25% |
| `heated` | 10% | 30% | 40% | 20% | at least 1 challenger and 1 non-challenger; at most 60% |

- The draw is seeded by the job's seed (`metadata.run.params.seed`), so resubmitting with the same `"model": {"seed": ...}` gives the same mix. The preset used is recorded in `metadata.run.group_dynamic`.
- Groups of 1–2 people get a plain draw with no limits.
- What participants *think* about the topic comes from their demographic (e.g. "skeptical perspective"). A skeptic can still be polite, and a supporter can be combative.
- The weights are a starting point. The research team owns them: they live in `backend/src/focus_group/persona/stance_distribution.py`.

---

## Fully-Specified Example

```json
{
  "session_name": "Telehealth Study Pilot-1",
  "purpose": "We are gathering perspectives on telehealth adoption in rural and suburban Minnesota communities.",
  "demographics": [
    "Female, age 34, suburban Fargo, works as a registered nurse, moderate political views, uses technology daily",
    "Male, age 52, rural Fergus Falls MN, retired farmer, skeptical of technology, strong community ties",
    "Non-binary, age 27, Minneapolis, graduate student in public health, progressive, early adopter of technology",
    "Female, age 67, suburban St. Cloud, retired teacher, health conscious, moderate comfort with technology"
  ],
  "questions": [
    "What are your first impressions of telehealth services?",
    "Have you or someone you care about used telehealth? What was that experience like?",
    "What barriers, if any, would prevent you from using telehealth regularly?"
  ],
  "closing_question": "Is there anything about telehealth access in Minnesota that we haven't touched on that you'd like to share?",
  "model": {
    "temperature": 0.8,
    "num_ctx": 8192
  },
  "memory": {
    "enable_stance_tracking": true,
    "working_memory_window": 12
  },
  "validate_personas": true,
  "group_dynamic": "natural"
}
```

---

## Field Summary Table

| Field | Required | Type | Default | Key Constraint |
|-------|----------|------|---------|----------------|
| `demographics` | Yes | string[] | — | 1–15 items |
| `questions` | Yes | string[] | — | 1–20 items |
| `purpose` | No | string / null | null | — |
| `closing_question` | No | string | *(see above)* | — |
| `session_name` | No | string / null | null | Max 100 chars |
| `model` | No | object | Ollama defaults | temperature 0–2 |
| `context` | No | object | *(see above)* | — |
| `memory` | No | object / null | null | Enables stance tracking |
| `output` | No | object | *(see above)* | Server-side only |
| `validate_personas` | No | boolean | true | — |
| `group_dynamic` | No | string | `"natural"` | `natural`, `cooperative` or `heated` |
