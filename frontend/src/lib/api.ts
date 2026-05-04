export type JobStatus = "queued" | "running" | "completed" | "failed";

export interface JobSubmitResponse {
  job_id: string;
  status: JobStatus;
  estimated_position: number | null;
}

export interface JobStatusResponse {
  job_id: string;
  status: JobStatus;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
  expires_at: string;
  queue_position: number | null;
}

export interface JobSummary {
  job_id: string;
  status: JobStatus;
  created_at: string;
  expires_at: string;
  participant_count: number | null;
  total_turns: number | null;
  consensus_level: string | null;
  session_name: string | null;
}

export interface ParticipantResponse {
  speaker_name: string;
  content: string;
  turn_number: number;
  selection_method: string;
  selection_reasoning: string;
  timestamp: string;
}

export interface NonVerbalReaction {
  participant_name: string;
  reaction: string;
  stimulus_type: string;
  stimulus_source: string;
  timestamp: string;
  response_turn: number | null;
}

export interface QuestionResult {
  question: string;
  responses: ParticipantResponse[];
  nonverbal_reactions: NonVerbalReaction[];
}

export interface SessionResultData {
  session_name: string | null;
  purpose: string | null;
  participants: Array<{ name: string; [key: string]: unknown }>;
  question_results: QuestionResult[];
  closing_result: QuestionResult | null;
  demographics: string[];
  questions: string[];
  started_at: string;
  completed_at: string | null;
  errors: string[];
  total_questions: number;
  total_responses: number;
  duration_seconds: number | null;
  stance_matrix_dump: Record<string, unknown> | null;
  participant_memory_dumps: Array<Record<string, unknown>> | null;
}

export interface JobResultResponse {
  job_id: string;
  status: JobStatus;
  config: Record<string, unknown>;
  result: SessionResultData;
  metadata: {
    participant_count: number | null;
    total_turns: number | null;
    consensus_level: string | null;
    created_at: string;
    started_at: string | null;
    completed_at: string | null;
    expires_at: string;
  };
}

export interface ApiError extends Error {
  status?: number;
  code?: string;
}

export interface GenerateRequest {
  research_topic: string;
  num_participants?: number;
  num_questions?: number;
}

export interface GenerateResponse {
  session_name: string;
  purpose: string;
  demographics: string[];
  questions: string[];
}

function handleUnauthorized() {
  if (typeof window !== "undefined") {
    window.location.href = "/api/v1/auth/login";
  }
}

async function apiFetch<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`/api/v1${path}`, {
    credentials: "include",
    ...options,
  });
  if (res.status === 401) {
    handleUnauthorized();
    const err = new Error("Unauthorized") as ApiError;
    err.status = 401;
    throw err;
  }
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    const err = new Error(
      body?.error?.message ?? `Request failed with status ${res.status}`
    ) as ApiError;
    err.status = res.status;
    err.code = body?.error?.code;
    throw err;
  }
  if (res.status === 204) return undefined as T;
  return res.json() as Promise<T>;
}

export const api = {
  generate: (params: GenerateRequest) =>
    apiFetch<GenerateResponse>("/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params),
    }),
  jobs: {
    create: (config: Record<string, unknown>) =>
      apiFetch<JobSubmitResponse>("/jobs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ config }),
      }),
    list: () => apiFetch<JobSummary[]>("/jobs"),
    getStatus: (jobId: string) =>
      apiFetch<JobStatusResponse>(`/jobs/${jobId}/status`),
    getResult: (jobId: string) =>
      apiFetch<JobResultResponse>(`/jobs/${jobId}/result`),
    delete: (jobId: string) =>
      apiFetch<void>(`/jobs/${jobId}`, { method: "DELETE" }),
  },
};
