"use client";

import { useState } from "react";
import type { ReactNode } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { ChevronLeft, Plus, Minus, Check, X, Sparkles, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardAction,
  CardContent,
  CardFooter,
} from "@/components/ui/card";
import { api, type ApiError, type GenerateResponse } from "@/lib/api";

// ── Constants ─────────────────────────────────────────────────────────────────

const DEFAULT_CLOSING =
  "Do you have any concluding thoughts about participating in this focus group?";

const GENDER_OPTIONS = ["Female", "Male", "Non-binary", "Prefer not to say"] as const;
const LOCATION_OPTIONS = [
  "Twin Cities Metro",
  "Fargo-Moorhead Area",
  "Suburban MN",
  "Rural MN",
  "Out of State",
] as const;
const STANCE_OPTIONS = [
  "Balanced",
  "Progressive",
  "Conservative",
  "Skeptical",
  "Moderate",
] as const;
const REGISTER_OPTIONS = ["Professional", "Formal", "Casual", "Academic"] as const;

const STEP_LABELS = ["Session", "Participants", "Discussion", "Review"] as const;

// ── Types ─────────────────────────────────────────────────────────────────────

interface Participant {
  id: string;
  name: string;
  gender: string;
  age: string;
  location: string;
  occupation: string;
  background: string;
  stance: string;
  register: string;
}

interface FormState {
  research_topic: string;
  session_name: string;
  purpose: string;
  participants: Participant[];
  questions: string[];
  closing_question: string;
  temperature: string;
  num_ctx: string;
  enable_memory: boolean;
  working_memory_window: string;
  validate_personas: boolean;
  showAdvanced: boolean;
}

type ParticipantErrors = Record<string, Partial<Record<keyof Participant, string>>>;

// ── Utilities ─────────────────────────────────────────────────────────────────

function newParticipant(): Participant {
  return {
    id: crypto.randomUUID(),
    name: "",
    gender: "",
    age: "",
    location: "",
    occupation: "",
    background: "",
    stance: "",
    register: "",
  };
}

function serializeParticipant(p: Participant): string {
  const parts = [p.gender, `age ${p.age}`, p.location, p.occupation];
  if (p.background.trim()) parts.push(p.background.trim());
  if (p.stance && p.stance !== "Balanced")
    parts.push(`${p.stance.toLowerCase()} perspective`);
  if (p.register && p.register !== "Professional")
    parts.push(`${p.register.toLowerCase()} communicator`);
  return parts.filter(Boolean).join(", ");
}

function participantLabel(p: Participant): string {
  if (p.name.trim()) return p.name.trim();
  return (
    [p.gender, p.age ? `age ${p.age}` : null, p.occupation || null]
      .filter(Boolean)
      .join(", ") || "Incomplete"
  );
}

function deserializeParticipant(raw: string): Participant {
  const tokens = raw.split(", ").map((t) => t.trim()).filter(Boolean);
  const p = newParticipant();

  for (const token of tokens) {
    const lower = token.toLowerCase();

    if (GENDER_OPTIONS.some((g) => g.toLowerCase() === lower)) {
      p.gender = GENDER_OPTIONS.find((g) => g.toLowerCase() === lower) ?? "";
      continue;
    }

    const ageMatch = token.match(/^age\s+(\d+)$/i);
    if (ageMatch) {
      p.age = ageMatch[1];
      continue;
    }

    if (LOCATION_OPTIONS.some((l) => l.toLowerCase() === lower)) {
      p.location = LOCATION_OPTIONS.find((l) => l.toLowerCase() === lower) ?? "";
      continue;
    }

    const stanceMatch = STANCE_OPTIONS.find(
      (s) => lower === `${s.toLowerCase()} perspective`
    );
    if (stanceMatch) {
      p.stance = stanceMatch;
      continue;
    }

    const registerMatch = REGISTER_OPTIONS.find(
      (r) => lower === `${r.toLowerCase()} communicator`
    );
    if (registerMatch) {
      p.register = registerMatch;
      continue;
    }

    if (!p.occupation) {
      p.occupation = token;
    } else if (!p.background) {
      p.background = token;
    }
  }

  return p;
}

// ── Main component ────────────────────────────────────────────────────────────

export default function NewSimulationPage() {
  const router = useRouter();
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [step, setStep] = useState<1 | 2 | 3 | 4>(1);
  const [stepErrors, setStepErrors] = useState<Record<number, string>>({});
  const [participantCardErrors, setParticipantCardErrors] =
    useState<ParticipantErrors>({});
  const [generatingParticipants, setGeneratingParticipants] = useState(false);
  const [generatingQuestions, setGeneratingQuestions] = useState(false);
  const [generateError, setGenerateError] = useState<string | null>(null);

  const [form, setForm] = useState<FormState>({
    research_topic: "",
    session_name: "",
    purpose: "",
    participants: [newParticipant(), newParticipant()],
    questions: [""],
    closing_question: DEFAULT_CLOSING,
    temperature: "0.7",
    num_ctx: "8192",
    enable_memory: false,
    working_memory_window: "10",
    validate_personas: true,
    showAdvanced: false,
  });

  function setField<K extends keyof FormState>(key: K, value: FormState[K]) {
    setForm((prev) => ({ ...prev, [key]: value }));
  }

  function updateParticipant(
    id: string,
    field: keyof Participant,
    value: string
  ) {
    setForm((prev) => ({
      ...prev,
      participants: prev.participants.map((p) =>
        p.id === id ? { ...p, [field]: value } : p
      ),
    }));
    if (participantCardErrors[id]?.[field]) {
      setParticipantCardErrors((prev) => ({
        ...prev,
        [id]: { ...prev[id], [field]: undefined },
      }));
    }
  }

  function addParticipant() {
    if (form.participants.length >= 15) return;
    setForm((prev) => ({
      ...prev,
      participants: [...prev.participants, newParticipant()],
    }));
  }

  function removeParticipant(id: string) {
    if (form.participants.length <= 1) return;
    setForm((prev) => ({
      ...prev,
      participants: prev.participants.filter((p) => p.id !== id),
    }));
  }

  function updateQuestion(index: number, value: string) {
    setForm((prev) => {
      const arr = [...prev.questions];
      arr[index] = value;
      return { ...prev, questions: arr };
    });
  }

  function addQuestion() {
    if (form.questions.length >= 20) return;
    setForm((prev) => ({ ...prev, questions: [...prev.questions, ""] }));
  }

  function removeQuestion(index: number) {
    if (form.questions.length <= 1) return;
    setForm((prev) => {
      const arr = [...prev.questions];
      arr.splice(index, 1);
      return { ...prev, questions: arr };
    });
  }

  function applyGenerateResponse(data: GenerateResponse, target: "participants" | "questions" | "both") {
    setGenerateError(null);
    setForm((prev) => {
      const next = { ...prev };
      if (target === "participants" || target === "both") {
        if (data.demographics.length > 0) {
          next.participants = data.demographics.map(deserializeParticipant);
        }
        if (data.session_name && !prev.session_name) next.session_name = data.session_name;
        if (data.purpose && !prev.purpose) next.purpose = data.purpose;
      }
      if (target === "questions" || target === "both") {
        if (data.questions.length > 0) {
          next.questions = data.questions;
        }
      }
      return next;
    });
  }

  async function handleGenerateParticipants() {
    if (!form.research_topic.trim()) return;
    setGeneratingParticipants(true);
    setGenerateError(null);
    try {
      const data = await api.generate({
        research_topic: form.research_topic.trim(),
        num_participants: form.participants.length,
        num_questions: 1,
      });
      applyGenerateResponse(data, "participants");
    } catch (err) {
      setGenerateError((err as Error).message ?? "Auto-generation failed. Try again.");
    } finally {
      setGeneratingParticipants(false);
    }
  }

  async function handleGenerateQuestions() {
    if (!form.research_topic.trim()) return;
    setGeneratingQuestions(true);
    setGenerateError(null);
    try {
      const data = await api.generate({
        research_topic: form.research_topic.trim(),
        num_participants: 1,
        num_questions: form.questions.length,
      });
      applyGenerateResponse(data, "questions");
    } catch (err) {
      setGenerateError((err as Error).message ?? "Auto-generation failed. Try again.");
    } finally {
      setGeneratingQuestions(false);
    }
  }

  function validateAndAdvance() {
    setStepErrors({});

    if (step === 2) {
      const errors: ParticipantErrors = {};
      const required: (keyof Participant)[] = [
        "gender",
        "age",
        "location",
        "occupation",
      ];
      let hasError = false;

      for (const p of form.participants) {
        const pErrors: Partial<Record<keyof Participant, string>> = {};
        for (const field of required) {
          if (!p[field].trim()) {
            pErrors[field] = "Required";
            hasError = true;
          }
        }
        if (p.age.trim()) {
          const age = parseInt(p.age, 10);
          if (isNaN(age) || age < 18 || age > 80) {
            pErrors.age = "Must be 18–80";
            hasError = true;
          }
        }
        if (Object.keys(pErrors).length) errors[p.id] = pErrors;
      }

      if (hasError) {
        setParticipantCardErrors(errors);
        setStepErrors({
          2: "Please complete all required fields for each participant.",
        });
        return;
      }
      setParticipantCardErrors({});
    }

    if (step === 3) {
      const nonEmpty = form.questions.filter((q) => q.trim());
      if (nonEmpty.length === 0) {
        setStepErrors({ 3: "At least one question is required." });
        return;
      }
    }

    setStep((prev) => (prev + 1) as typeof prev);
  }

  async function handleSubmit() {
    setSubmitError(null);
    const demographics = form.participants.map(serializeParticipant);
    const nonEmptyQuestions = form.questions.filter((q) => q.trim());

    const participantNames = form.participants.map((p) => p.name.trim() || null);
    const hasAnyName = participantNames.some(Boolean);

    const config: Record<string, unknown> = {
      demographics,
      questions: nonEmptyQuestions,
    };
    if (hasAnyName) config.participant_names = participantNames;

    if (form.session_name.trim()) config.session_name = form.session_name.trim();
    if (form.purpose.trim()) config.purpose = form.purpose.trim();
    if (form.closing_question.trim() !== DEFAULT_CLOSING) {
      config.closing_question = form.closing_question.trim();
    }

    const temp = parseFloat(form.temperature);
    const ctx = parseInt(form.num_ctx, 10);
    if (!isNaN(temp) || !isNaN(ctx)) {
      const model: Record<string, unknown> = {};
      if (!isNaN(temp) && temp >= 0 && temp <= 2) model.temperature = temp;
      if (!isNaN(ctx) && ctx >= 1024) model.num_ctx = ctx;
      if (Object.keys(model).length > 0) config.model = model;
    }

    if (form.enable_memory) {
      const memWindow = parseInt(form.working_memory_window, 10);
      config.memory = {
        enable_stance_tracking: true,
        working_memory_window: !isNaN(memWindow) ? memWindow : 10,
      };
    }

    config.validate_personas = form.validate_personas;

    setSubmitting(true);
    try {
      const res = await api.jobs.create(config);
      router.push(`/jobs/${res.job_id}`);
    } catch (err) {
      const apiErr = err as ApiError;
      if (apiErr.status !== 401) {
        setSubmitError(apiErr.message ?? "Submission failed.");
      }
    } finally {
      setSubmitting(false);
    }
  }

  const nonEmptyQuestions = form.questions.filter((q) => q.trim());
  const hasNonDefaultAdvanced =
    form.temperature !== "0.7" ||
    form.num_ctx !== "8192" ||
    form.enable_memory ||
    !form.validate_personas;

  return (
    <div className="max-w-6xl">
      <Link
        href="/"
        className="inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground mb-6 transition-colors"
      >
        <ChevronLeft className="h-4 w-4" />
        Back to dashboard
      </Link>

      <h1 className="text-2xl font-semibold mb-6">New Simulation</h1>

      <StepIndicator currentStep={step} />

      {stepErrors[step] && (
        <div className="border border-destructive/30 bg-destructive/10 text-destructive px-4 py-3 text-sm mb-6">
          {stepErrors[step]}
        </div>
      )}

      {/* ── Step 1: Session Details ── */}
      {step === 1 && (
        <div className="space-y-8">
          <Section title="Session Details">
            <Field
              label="Research topic"
              hint="Used to auto-generate participants and questions in later steps"
            >
              <Input
                value={form.research_topic}
                onChange={(e) => setField("research_topic", e.target.value)}
                placeholder="e.g. Telehealth adoption in rural Minnesota"
                maxLength={500}
              />
            </Field>
            <Field label="Session name" hint="Optional — max 100 characters">
              <Input
                value={form.session_name}
                onChange={(e) => setField("session_name", e.target.value)}
                placeholder="e.g. Telehealth Study Pilot-1"
                maxLength={100}
              />
            </Field>
            <Field
              label="Research purpose"
              hint="Optional introductory statement shown to all participants"
            >
              <Textarea
                value={form.purpose}
                onChange={(e) => setField("purpose", e.target.value)}
                placeholder="e.g. We are gathering perspectives on telehealth adoption in rural Minnesota."
                rows={3}
              />
            </Field>
          </Section>

          <div className="flex gap-3">
            <Button type="button" onClick={validateAndAdvance}>
              Next →
            </Button>
          </div>
        </div>
      )}

      {/* ── Step 2: Participants ── */}
      {step === 2 && (
        <div className="space-y-4">
          <div className="flex items-center justify-between mb-2">
            <p className="text-sm text-muted-foreground">
              Add 1–15 focus group participants with demographic details.
            </p>
            <div className="flex items-center gap-3 shrink-0 ml-4">
              <span className="text-sm font-medium">
                {form.participants.length} participant
                {form.participants.length !== 1 ? "s" : ""}
              </span>
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={handleGenerateParticipants}
                disabled={generatingParticipants || !form.research_topic.trim()}
                title={!form.research_topic.trim() ? "Enter a research topic in Step 1 first" : undefined}
              >
                {generatingParticipants ? (
                  <><Loader2 className="h-3.5 w-3.5 mr-1.5 animate-spin" />Generating…</>
                ) : (
                  <><Sparkles className="h-3.5 w-3.5 mr-1.5" />Auto-generate</>
                )}
              </Button>
            </div>
          </div>

          {generateError && step === 2 && (
            <div className="border border-destructive/30 bg-destructive/10 text-destructive px-4 py-3 text-sm">
              {generateError}
            </div>
          )}

          {form.participants.map((p, i) => (
            <ParticipantCard
              key={p.id}
              participant={p}
              index={i}
              canRemove={form.participants.length > 1}
              onChange={(field, value) => updateParticipant(p.id, field, value)}
              onRemove={() => removeParticipant(p.id)}
              errors={participantCardErrors[p.id] ?? {}}
            />
          ))}

          {form.participants.length < 15 && (
            <Button
              type="button"
              variant="outline"
              onClick={addParticipant}
              className="w-full"
            >
              <Plus className="h-4 w-4" /> Add Participant
            </Button>
          )}

          <div className="flex gap-3 pt-2">
            <Button
              type="button"
              variant="outline"
              onClick={() => setStep(1)}
            >
              ← Back
            </Button>
            <Button type="button" onClick={validateAndAdvance}>
              Next →
            </Button>
          </div>
        </div>
      )}

      {/* ── Step 3: Discussion ── */}
      {step === 3 && (
        <div className="space-y-8">
          {generateError && (
            <div className="border border-destructive/30 bg-destructive/10 text-destructive px-4 py-3 text-sm">
              {generateError}
            </div>
          )}
          <Section
            title={`Questions (${nonEmptyQuestions.length}/20)`}
            action={
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={handleGenerateQuestions}
                disabled={generatingQuestions || !form.research_topic.trim()}
                title={!form.research_topic.trim() ? "Enter a research topic in Step 1 first" : undefined}
              >
                {generatingQuestions ? (
                  <><Loader2 className="h-3.5 w-3.5 mr-1.5 animate-spin" />Generating…</>
                ) : (
                  <><Sparkles className="h-3.5 w-3.5 mr-1.5" />Auto-generate</>
                )}
              </Button>
            }
          >
            <div className="space-y-2">
              {form.questions.map((q, i) => (
                <div key={i} className="flex gap-2 items-start">
                  <div className="flex-1 flex gap-2 items-start">
                    <span className="text-xs text-muted-foreground w-5 pt-2 shrink-0 text-right">
                      {i + 1}.
                    </span>
                    <Textarea
                      value={q}
                      onChange={(e) => updateQuestion(i, e.target.value)}
                      placeholder={`Question ${i + 1}`}
                      rows={2}
                      className="flex-1 text-sm"
                    />
                  </div>
                  {form.questions.length > 1 && (
                    <button
                      type="button"
                      onClick={() => removeQuestion(i)}
                      className="mt-1 text-muted-foreground hover:text-destructive transition-colors"
                      aria-label="Remove question"
                    >
                      <Minus className="h-4 w-4" />
                    </button>
                  )}
                </div>
              ))}
            </div>
            {form.questions.length < 20 && (
              <button
                type="button"
                onClick={addQuestion}
                className="mt-2 text-sm text-primary hover:underline inline-flex items-center gap-1"
              >
                <Plus className="h-3.5 w-3.5" /> Add question
              </button>
            )}
          </Section>

          <Section title="Closing question">
            <Field hint="Asked to all participants after all main questions">
              <Input
                value={form.closing_question}
                onChange={(e) => setField("closing_question", e.target.value)}
              />
            </Field>
          </Section>

          <div>
            <button
              type="button"
              onClick={() => setField("showAdvanced", !form.showAdvanced)}
              className="text-sm text-muted-foreground hover:text-foreground transition-colors flex items-center gap-1"
            >
              <ChevronLeft
                className={`h-4 w-4 transition-transform ${
                  form.showAdvanced ? "-rotate-90" : "rotate-180"
                }`}
              />
              Advanced settings
            </button>

            {form.showAdvanced && (
              <div className="mt-4 pl-4 border-l border-border space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <Field label="Temperature" hint="0.0 – 2.0 (default 0.7)">
                    <Input
                      type="number"
                      min={0}
                      max={2}
                      step={0.1}
                      value={form.temperature}
                      onChange={(e) => setField("temperature", e.target.value)}
                    />
                  </Field>
                  <Field label="Context window" hint="Tokens (default 8192)">
                    <Input
                      type="number"
                      min={1024}
                      max={131072}
                      step={1024}
                      value={form.num_ctx}
                      onChange={(e) => setField("num_ctx", e.target.value)}
                    />
                  </Field>
                </div>

                <div className="flex items-center gap-3">
                  <input
                    id="enable-memory"
                    type="checkbox"
                    checked={form.enable_memory}
                    onChange={(e) =>
                      setField("enable_memory", e.target.checked)
                    }
                    className="h-4 w-4 accent-primary"
                  />
                  <label htmlFor="enable-memory" className="text-sm">
                    Enable stance tracking (ResearchAssistant)
                  </label>
                </div>

                {form.enable_memory && (
                  <Field
                    label="Working memory window"
                    hint="Recent turns in memory (1–50, default 10)"
                  >
                    <Input
                      type="number"
                      min={1}
                      max={50}
                      value={form.working_memory_window}
                      onChange={(e) =>
                        setField("working_memory_window", e.target.value)
                      }
                      className="w-32"
                    />
                  </Field>
                )}

                <div className="flex items-center gap-3">
                  <input
                    id="validate-personas"
                    type="checkbox"
                    checked={form.validate_personas}
                    onChange={(e) =>
                      setField("validate_personas", e.target.checked)
                    }
                    className="h-4 w-4 accent-primary"
                  />
                  <label htmlFor="validate-personas" className="text-sm">
                    Validate generated personas
                  </label>
                </div>
              </div>
            )}
          </div>

          <div className="flex gap-3">
            <Button
              type="button"
              variant="outline"
              onClick={() => setStep(2)}
            >
              ← Back
            </Button>
            <Button type="button" onClick={validateAndAdvance}>
              Next →
            </Button>
          </div>
        </div>
      )}

      {/* ── Step 4: Review & Submit ── */}
      {step === 4 && (
        <div className="space-y-6">
          <Card>
            <CardHeader className="border-b">
              <CardTitle>
                {form.session_name.trim() || "Simulation Summary"}
              </CardTitle>
              {form.purpose.trim() && (
                <CardDescription>{form.purpose.trim()}</CardDescription>
              )}
            </CardHeader>

            <CardContent className="space-y-6 pt-5">
              <ReviewSection
                title={`Participants (${form.participants.length})`}
              >
                <ol className="space-y-1 list-decimal list-inside">
                  {form.participants.map((p) => (
                    <li key={p.id} className="text-sm text-muted-foreground">
                      {participantLabel(p)}
                    </li>
                  ))}
                </ol>
              </ReviewSection>

              <ReviewSection
                title={`Questions (${nonEmptyQuestions.length})`}
              >
                <ol className="space-y-1 list-decimal list-inside">
                  {nonEmptyQuestions.map((q, i) => (
                    <li
                      key={i}
                      className="text-sm text-muted-foreground line-clamp-2"
                    >
                      {q}
                    </li>
                  ))}
                </ol>
                <p className="text-xs text-muted-foreground mt-2">
                  Closing:{" "}
                  <span className="italic">{form.closing_question}</span>
                </p>
              </ReviewSection>

              {hasNonDefaultAdvanced && (
                <ReviewSection title="Advanced Settings">
                  {form.temperature !== "0.7" && (
                    <ReviewKV label="Temperature" value={form.temperature} />
                  )}
                  {form.num_ctx !== "8192" && (
                    <ReviewKV
                      label="Context window"
                      value={`${form.num_ctx} tokens`}
                    />
                  )}
                  {form.enable_memory && (
                    <ReviewKV label="Stance tracking" value="Enabled" />
                  )}
                  {!form.validate_personas && (
                    <ReviewKV label="Validate personas" value="Disabled" />
                  )}
                </ReviewSection>
              )}
            </CardContent>

            <CardFooter className="flex-col items-stretch gap-3">
              {submitError && (
                <div className="border border-destructive/30 bg-destructive/10 text-destructive px-4 py-3 text-sm w-full">
                  {submitError}
                </div>
              )}
              <div className="flex gap-3">
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => setStep(3)}
                  disabled={submitting}
                >
                  ← Back
                </Button>
                <Button
                  type="button"
                  onClick={handleSubmit}
                  disabled={submitting}
                  className="flex-1"
                  size="lg"
                >
                  {submitting ? "Submitting…" : "Submit Simulation"}
                </Button>
              </div>
            </CardFooter>
          </Card>
        </div>
      )}
    </div>
  );
}

// ── Local components ──────────────────────────────────────────────────────────

function StepIndicator({ currentStep }: { currentStep: number }) {
  return (
    <nav aria-label="Form steps" className="flex items-center mb-8">
      {STEP_LABELS.map((label, i) => {
        const stepNum = i + 1;
        const isComplete = stepNum < currentStep;
        const isCurrent = stepNum === currentStep;

        return (
          <div
            key={label}
            className="flex items-center flex-1 last:flex-none"
          >
            <div className="flex flex-col items-center gap-1 shrink-0">
              <div
                className={[
                  "flex h-7 w-7 items-center justify-center border-2 text-xs font-semibold transition-colors",
                  isComplete
                    ? "border-primary bg-primary text-primary-foreground"
                    : isCurrent
                    ? "border-primary bg-background text-primary"
                    : "border-border bg-background text-muted-foreground",
                ].join(" ")}
                aria-current={isCurrent ? "step" : undefined}
              >
                {isComplete ? <Check className="h-3.5 w-3.5" /> : stepNum}
              </div>
              <span
                className={`text-xs ${
                  isCurrent
                    ? "text-foreground font-medium"
                    : "text-muted-foreground"
                }`}
              >
                {label}
              </span>
            </div>
            {i < STEP_LABELS.length - 1 && (
              <div
                className={`flex-1 h-px mx-3 mb-4 transition-colors ${
                  isComplete ? "bg-primary" : "bg-border"
                }`}
              />
            )}
          </div>
        );
      })}
    </nav>
  );
}

interface ParticipantCardProps {
  participant: Participant;
  index: number;
  canRemove: boolean;
  onChange: (field: keyof Participant, value: string) => void;
  onRemove: () => void;
  errors: Partial<Record<keyof Participant, string>>;
}

function ParticipantCard({
  participant,
  index,
  canRemove,
  onChange,
  onRemove,
  errors,
}: ParticipantCardProps) {
  const hasLabel =
    participant.name || participant.gender || participant.age || participant.occupation;

  return (
    <Card>
      <CardHeader className="border-b">
        <CardTitle className="text-sm font-semibold font-sans">
          Participant {index + 1}
        </CardTitle>
        {hasLabel && (
          <CardDescription>{participantLabel(participant)}</CardDescription>
        )}
        {canRemove && (
          <CardAction>
            <Button
              type="button"
              variant="ghost"
              size="icon-sm"
              onClick={onRemove}
              aria-label="Remove participant"
            >
              <X className="h-4 w-4" />
            </Button>
          </CardAction>
        )}
      </CardHeader>

      <CardContent className="pt-4">
        <div className="mb-4">
          <Field label="Name" hint="Optional — leave blank to autogenerate">
            <Input
              value={participant.name}
              onChange={(e) => onChange("name", e.target.value)}
              placeholder="e.g. Jordan Lee"
            />
          </Field>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <Field label="Gender" error={errors.gender}>
            <Select
              value={participant.gender}
              onValueChange={(v) => onChange("gender", v)}
            >
              <SelectTrigger className="w-full" aria-invalid={!!errors.gender}>
                <SelectValue placeholder="Select gender" />
              </SelectTrigger>
              <SelectContent>
                {GENDER_OPTIONS.map((o) => (
                  <SelectItem key={o} value={o}>
                    {o}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </Field>

          <Field label="Age" hint="18 – 80" error={errors.age}>
            <Input
              type="number"
              min={18}
              max={80}
              value={participant.age}
              onChange={(e) => onChange("age", e.target.value)}
              placeholder="e.g. 34"
              aria-invalid={!!errors.age}
            />
          </Field>

          <Field label="Location" error={errors.location}>
            <Select
              value={participant.location}
              onValueChange={(v) => onChange("location", v)}
            >
              <SelectTrigger
                className="w-full"
                aria-invalid={!!errors.location}
              >
                <SelectValue placeholder="Select region" />
              </SelectTrigger>
              <SelectContent>
                {LOCATION_OPTIONS.map((o) => (
                  <SelectItem key={o} value={o}>
                    {o}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </Field>

          <Field label="Occupation" error={errors.occupation}>
            <Input
              value={participant.occupation}
              onChange={(e) => onChange("occupation", e.target.value)}
              placeholder="e.g. healthcare worker"
              aria-invalid={!!errors.occupation}
            />
          </Field>

          <Field label="Perspective" hint="Optional — defaults to Balanced">
            <Select
              value={participant.stance}
              onValueChange={(v) => onChange("stance", v)}
            >
              <SelectTrigger className="w-full">
                <SelectValue placeholder="Balanced" />
              </SelectTrigger>
              <SelectContent>
                {STANCE_OPTIONS.map((o) => (
                  <SelectItem key={o} value={o}>
                    {o}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </Field>

          <Field
            label="Communication register"
            hint="Optional — defaults to Professional"
          >
            <Select
              value={participant.register}
              onValueChange={(v) => onChange("register", v)}
            >
              <SelectTrigger className="w-full">
                <SelectValue placeholder="Professional" />
              </SelectTrigger>
              <SelectContent>
                {REGISTER_OPTIONS.map((o) => (
                  <SelectItem key={o} value={o}>
                    {o}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </Field>

          <Field
            label="Additional background"
            hint="Optional — e.g. chronic illness, caregiver role"
            className="sm:col-span-2"
          >
            <Textarea
              value={participant.background}
              onChange={(e) => onChange("background", e.target.value)}
              rows={2}
              className="text-sm"
              placeholder="Any additional context about this participant"
            />
          </Field>
        </div>
      </CardContent>
    </Card>
  );
}

function Section({
  title,
  action,
  children,
}: {
  title: string;
  action?: ReactNode;
  children: ReactNode;
}) {
  return (
    <div>
      <div className="flex items-center justify-between mb-3 pb-2 border-b border-border">
        <h2 className="text-sm font-semibold text-foreground uppercase tracking-wide">
          {title}
        </h2>
        {action}
      </div>
      <div className="space-y-4">{children}</div>
    </div>
  );
}

function Field({
  label,
  hint,
  error,
  className,
  children,
}: {
  label?: string;
  hint?: string;
  error?: string;
  className?: string;
  children: ReactNode;
}) {
  return (
    <div className={className}>
      {label && (
        <label className="block text-sm font-medium text-foreground mb-1">
          {label}
        </label>
      )}
      {children}
      {error ? (
        <p className="text-xs text-destructive mt-1">{error}</p>
      ) : hint ? (
        <p className="text-xs text-muted-foreground mt-1">{hint}</p>
      ) : null}
    </div>
  );
}

function ReviewSection({
  title,
  children,
}: {
  title: string;
  children: ReactNode;
}) {
  return (
    <div>
      <h3 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground mb-2">
        {title}
      </h3>
      {children}
    </div>
  );
}

function ReviewKV({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex gap-2 text-sm">
      <span className="text-muted-foreground min-w-36">{label}</span>
      <span>{value}</span>
    </div>
  );
}
