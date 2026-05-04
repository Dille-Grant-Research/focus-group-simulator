# Mock Data — Example Inputs and Outputs

**Audience:** C# frontend developers who need realistic API payloads to build and test the UI before the backend is reachable, or before a live simulation has been run.

All data on this page is synthetic but representative of real API behavior.

---

## 1. Minimal Job Submission Request

The smallest valid payload for `POST /api/v1/jobs`:

```json
{
  "demographics": [
    "Female, age 41, suburban Moorhead MN, elementary school teacher",
    "Male, age 58, rural Perham MN, small business owner"
  ],
  "questions": [
    "What role do you think local schools should play in addressing food insecurity among students?"
  ]
}
```

---

## 2. Full Job Submission Request

A fully-specified payload with 4 participants, 3 questions, custom model settings, and stance tracking enabled:

```json
{
  "session_name": "Food Security Community Study - Round 2",
  "purpose": "This focus group explores community perspectives on food insecurity and the role of local institutions in addressing it in the greater Fargo-Moorhead area.",
  "demographics": [
    "Female, age 41, suburban Moorhead MN, elementary school teacher, two children, involved in PTA, moderate political views",
    "Male, age 58, rural Perham MN, owner of a small hardware store, conservative, skeptical of government programs",
    "Female, age 29, Fargo ND, social worker at a non-profit food bank, progressive, personally experienced food insecurity in childhood",
    "Male, age 67, Dilworth MN, retired city councilman, civic-minded, believes in local solutions over federal programs"
  ],
  "questions": [
    "What role do you think local schools should play in addressing food insecurity among students?",
    "How do you feel about government-funded meal programs compared to community-organized food drives?",
    "What would it take for your community to treat food access as a shared responsibility rather than a personal one?"
  ],
  "closing_question": "Is there anything about food insecurity in our communities that we haven't talked about yet that feels important to you?",
  "model": {
    "temperature": 0.8,
    "num_ctx": 8192
  },
  "memory": {
    "enable_stance_tracking": true,
    "working_memory_window": 12
  },
  "validate_personas": true
}
```

---

## 3. Job Submission Response

Response from `POST /api/v1/jobs` — HTTP 202 Accepted:

```json
{
  "job_id": "7c3a1f2e-b849-4d7e-9c0a-5f1d8e2b3a4c",
  "status": "queued",
  "estimated_position": 1
}
```

---

## 4. Status Poll Response (Running)

Response from `GET /api/v1/jobs/7c3a1f2e-b849-4d7e-9c0a-5f1d8e2b3a4c/status` while the simulation is in progress:

```json
{
  "job_id": "7c3a1f2e-b849-4d7e-9c0a-5f1d8e2b3a4c",
  "status": "running",
  "created_at": "2026-05-02T14:00:00Z",
  "started_at": "2026-05-02T14:00:45Z",
  "completed_at": null,
  "expires_at": "2026-06-01T14:00:00Z",
  "queue_position": null
}
```

---

## 5. Completed Result Response

Response from `GET /api/v1/jobs/7c3a1f2e-b849-4d7e-9c0a-5f1d8e2b3a4c/result` after completion — HTTP 200:

```json
{
  "job_id": "7c3a1f2e-b849-4d7e-9c0a-5f1d8e2b3a4c",
  "status": "completed",
  "config": {
    "session_name": "Food Security Community Study - Round 2",
    "purpose": "This focus group explores community perspectives on food insecurity and the role of local institutions in addressing it in the greater Fargo-Moorhead area.",
    "demographics": [
      "Female, age 41, suburban Moorhead MN, elementary school teacher, two children, involved in PTA, moderate political views",
      "Male, age 58, rural Perham MN, owner of a small hardware store, conservative, skeptical of government programs",
      "Female, age 29, Fargo ND, social worker at a non-profit food bank, progressive, personally experienced food insecurity in childhood",
      "Male, age 67, Dilworth MN, retired city councilman, civic-minded, believes in local solutions over federal programs"
    ],
    "questions": [
      "What role do you think local schools should play in addressing food insecurity among students?",
      "How do you feel about government-funded meal programs compared to community-organized food drives?",
      "What would it take for your community to treat food access as a shared responsibility rather than a personal one?"
    ],
    "closing_question": "Is there anything about food insecurity in our communities that we haven't talked about yet that feels important to you?"
  },
  "result": {
    "transcript": [
      {
        "turn": 1,
        "speaker": "Participant 3",
        "verbal": "I see kids every week at the food bank who are also students. Schools are one of the few places these kids can reliably get a meal. I think schools have not just a role but an obligation — breakfast and lunch programs save lives, not just grades.",
        "nonverbals": {
          "Participant 1": "nods firmly, makes eye contact with Participant 3",
          "Participant 2": "shifts in seat, arms loosely crossed",
          "Participant 4": "leans back, listening carefully, neutral expression"
        },
        "timestamp": "2026-05-02T14:01:15Z"
      },
      {
        "turn": 2,
        "speaker": "Participant 2",
        "verbal": "I don't disagree that hungry kids can't learn. But I do wonder if we're letting parents off the hook. Schools can't solve every problem that starts at home. At some point families have to take responsibility.",
        "nonverbals": {
          "Participant 1": "slight frown, glances down briefly",
          "Participant 3": "jaw tightens, takes a breath before responding",
          "Participant 4": "raises an eyebrow, appears to consider the comment"
        },
        "timestamp": "2026-05-02T14:02:40Z"
      },
      {
        "turn": 3,
        "speaker": "Participant 1",
        "verbal": "I hear what you're saying, but in my experience a lot of these families are working two jobs and still can't make ends meet. It's not about responsibility — it's about capacity. Schools are a safety net that these kids need.",
        "nonverbals": {
          "Participant 2": "slight nod, expression softens marginally",
          "Participant 3": "visible relief, sits forward",
          "Participant 4": "taps finger on table, thinking"
        },
        "timestamp": "2026-05-02T14:04:05Z"
      },
      {
        "turn": 4,
        "speaker": "Participant 4",
        "verbal": "I've sat on city council long enough to know that federal programs come with strings and delays. What actually worked in our town was when the Lions Club partnered with the school district. Local people, local accountability. The community owned it.",
        "nonverbals": {
          "Participant 1": "thoughtful nod",
          "Participant 2": "noticeably more engaged, leans forward",
          "Participant 3": "mixed expression — acknowledgment but some hesitation"
        },
        "timestamp": "2026-05-02T14:05:30Z"
      },
      {
        "turn": 5,
        "speaker": "Participant 2",
        "verbal": "That's exactly the kind of thing I can get behind. Community-driven, not government-mandated. I'd volunteer for something like that in a heartbeat.",
        "nonverbals": {
          "Participant 1": "smiles",
          "Participant 3": "nods, though with less enthusiasm than Participants 1 and 2",
          "Participant 4": "satisfied expression"
        },
        "timestamp": "2026-05-02T14:06:55Z"
      }
    ],
    "analysis": {
      "stance_matrix": {
        "Participant 1": {
          "school_role": "strongly_supportive",
          "government_programs": "supportive",
          "community_responsibility": "supportive"
        },
        "Participant 2": {
          "school_role": "cautious",
          "government_programs": "opposed",
          "community_responsibility": "conditionally_supportive"
        },
        "Participant 3": {
          "school_role": "strongly_supportive",
          "government_programs": "strongly_supportive",
          "community_responsibility": "supportive"
        },
        "Participant 4": {
          "school_role": "supportive",
          "government_programs": "skeptical",
          "community_responsibility": "strongly_supportive"
        }
      },
      "themes": [
        "Schools as essential safety nets",
        "Tension between personal responsibility and systemic capacity",
        "Preference for local and community-led solutions over federal programs",
        "Practical experience with food insecurity shapes advocacy intensity"
      ],
      "consensus_level": "moderate",
      "notable_interactions": [
        "Participant 2 shifted from skepticism toward engagement after Participant 4 introduced the community-partnership framing",
        "Participant 3 maintained a consistent pro-intervention stance throughout, rooted in direct personal and professional experience",
        "Participant 4 served as a bridging voice between Participants 2 and 3, reframing government vs. community as a question of mechanism rather than values"
      ]
    }
  },
  "metadata": {
    "participant_count": 4,
    "total_turns": 22,
    "consensus_level": "moderate",
    "created_at": "2026-05-02T14:00:00Z",
    "started_at": "2026-05-02T14:00:45Z",
    "completed_at": "2026-05-02T14:09:12Z",
    "expires_at": "2026-06-01T14:00:00Z"
  }
}
```

---

## 6. Job List Response

Response from `GET /api/v1/jobs` — HTTP 200:

```json
[
  {
    "job_id": "7c3a1f2e-b849-4d7e-9c0a-5f1d8e2b3a4c",
    "status": "completed",
    "created_at": "2026-05-02T14:00:00Z",
    "expires_at": "2026-06-01T14:00:00Z",
    "participant_count": 4,
    "total_turns": 22,
    "consensus_level": "moderate"
  },
  {
    "job_id": "a4b7c2d1-e0f9-4321-8765-abcdef012345",
    "status": "completed",
    "created_at": "2026-04-28T09:15:00Z",
    "expires_at": "2026-05-28T09:15:00Z",
    "participant_count": 3,
    "total_turns": 18,
    "consensus_level": "high"
  },
  {
    "job_id": "f1e2d3c4-b5a6-9870-fedc-ba9876543210",
    "status": "failed",
    "created_at": "2026-05-01T11:30:00Z",
    "expires_at": "2026-05-31T11:30:00Z",
    "participant_count": null,
    "total_turns": null,
    "consensus_level": null
  }
]
```

Note: `participant_count`, `total_turns`, and `consensus_level` are `null` for jobs that did not complete successfully.

---

## 7. Error Response Example

A `429 RATE_LIMIT_EXCEEDED` response from `POST /api/v1/jobs`:

```json
{
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Maximum 5 simulations per hour. Try again later or contact an administrator to raise your limit.",
    "details": {
      "limit": 5,
      "window_hours": 1
    }
  }
}
```

See [error_responses.md](api/error_responses.md) for the full error code reference.
