"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { ChevronLeft, Plus, Minus } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { api, type ApiError } from "@/lib/api";

interface FormState {
  session_name: string;
  purpose: string;
  demographics: string[];
  questions: string[];
  closing_question: string;
  temperature: string;
  num_ctx: string;
  enable_memory: boolean;
  working_memory_window: string;
  validate_personas: boolean;
  showAdvanced: boolean;
}

const DEFAULT_CLOSING =
  "Do you have any concluding thoughts about participating in this focus group?";

export default function NewSimulationPage() {
  const router = useRouter();
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);

  const [form, setForm] = useState<FormState>({
    session_name: "",
    purpose: "",
    demographics: ["", ""],
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

  function updateArray(key: "demographics" | "questions", index: number, value: string) {
    setForm((prev) => {
      const arr = [...prev[key]];
      arr[index] = value;
      return { ...prev, [key]: arr };
    });
  }

  function addItem(key: "demographics" | "questions", max: number) {
    setForm((prev) => {
      if (prev[key].length >= max) return prev;
      return { ...prev, [key]: [...prev[key], ""] };
    });
  }

  function removeItem(key: "demographics" | "questions", index: number, min: number) {
    setForm((prev) => {
      if (prev[key].length <= min) return prev;
      const arr = [...prev[key]];
      arr.splice(index, 1);
      return { ...prev, [key]: arr };
    });
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setSubmitError(null);

    const nonEmptyDemographics = form.demographics.filter((d) => d.trim());
    const nonEmptyQuestions = form.questions.filter((q) => q.trim());

    if (nonEmptyDemographics.length < 1) {
      setSubmitError("At least one participant is required.");
      return;
    }
    if (nonEmptyDemographics.length > 15) {
      setSubmitError("Maximum 15 participants.");
      return;
    }
    if (nonEmptyQuestions.length < 1) {
      setSubmitError("At least one question is required.");
      return;
    }
    if (nonEmptyQuestions.length > 20) {
      setSubmitError("Maximum 20 questions.");
      return;
    }

    const config: Record<string, unknown> = {
      demographics: nonEmptyDemographics,
      questions: nonEmptyQuestions,
    };

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
      const window = parseInt(form.working_memory_window, 10);
      config.memory = {
        enable_stance_tracking: true,
        working_memory_window: !isNaN(window) ? window : 10,
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

  return (
    <div className="max-w-2xl">
      <Link
        href="/"
        className="inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground mb-6 transition-colors"
      >
        <ChevronLeft className="h-4 w-4" />
        Back to dashboard
      </Link>

      <h1 className="text-2xl font-semibold mb-6">New Simulation</h1>

      <form onSubmit={handleSubmit} className="space-y-8">
        {/* Core fields */}
        <Section title="Session">
          <Field label="Session name" hint="Optional — max 100 characters">
            <Input
              value={form.session_name}
              onChange={(e) => setField("session_name", e.target.value)}
              placeholder="e.g. Telehealth Study Pilot-1"
              maxLength={100}
            />
          </Field>
          <Field label="Research purpose" hint="Optional introductory statement shown to all participants">
            <Textarea
              value={form.purpose}
              onChange={(e) => setField("purpose", e.target.value)}
              placeholder="e.g. We are gathering perspectives on telehealth adoption in rural Minnesota."
              rows={2}
            />
          </Field>
        </Section>

        <Section title={`Participants (${form.demographics.filter(Boolean).length}/15)`}>
          <div className="space-y-2">
            {form.demographics.map((d, i) => (
              <div key={i} className="flex gap-2 items-start">
                <Textarea
                  value={d}
                  onChange={(e) => updateArray("demographics", i, e.target.value)}
                  placeholder={`Participant ${i + 1}: e.g. Female, age 34, suburban Minnesota, healthcare worker`}
                  rows={2}
                  className="flex-1 text-sm"
                />
                {form.demographics.length > 1 && (
                  <button
                    type="button"
                    onClick={() => removeItem("demographics", i, 1)}
                    className="mt-1 text-muted-foreground hover:text-destructive transition-colors"
                    aria-label="Remove participant"
                  >
                    <Minus className="h-4 w-4" />
                  </button>
                )}
              </div>
            ))}
          </div>
          {form.demographics.length < 15 && (
            <button
              type="button"
              onClick={() => addItem("demographics", 15)}
              className="mt-2 text-sm text-primary hover:underline inline-flex items-center gap-1"
            >
              <Plus className="h-3.5 w-3.5" /> Add participant
            </button>
          )}
        </Section>

        <Section title={`Questions (${form.questions.filter(Boolean).length}/20)`}>
          <div className="space-y-2">
            {form.questions.map((q, i) => (
              <div key={i} className="flex gap-2 items-start">
                <div className="flex-1 flex gap-2 items-start">
                  <span className="text-xs text-muted-foreground w-5 pt-2 shrink-0 text-right">
                    {i + 1}.
                  </span>
                  <Textarea
                    value={q}
                    onChange={(e) => updateArray("questions", i, e.target.value)}
                    placeholder={`Question ${i + 1}`}
                    rows={2}
                    className="flex-1 text-sm"
                  />
                </div>
                {form.questions.length > 1 && (
                  <button
                    type="button"
                    onClick={() => removeItem("questions", i, 1)}
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
              onClick={() => addItem("questions", 20)}
              className="mt-2 text-sm text-primary hover:underline inline-flex items-center gap-1"
            >
              <Plus className="h-3.5 w-3.5" /> Add question
            </button>
          )}
        </Section>

        <Section title="Closing question">
          <Field hint="Asked at the end of all questions">
            <Input
              value={form.closing_question}
              onChange={(e) => setField("closing_question", e.target.value)}
            />
          </Field>
        </Section>

        {/* Advanced settings toggle */}
        <div>
          <button
            type="button"
            onClick={() => setField("showAdvanced", !form.showAdvanced)}
            className="text-sm text-muted-foreground hover:text-foreground transition-colors flex items-center gap-1"
          >
            <ChevronLeft
              className={`h-4 w-4 transition-transform ${form.showAdvanced ? "-rotate-90" : "rotate-180"}`}
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
                  onChange={(e) => setField("enable_memory", e.target.checked)}
                  className="h-4 w-4 accent-primary"
                />
                <label htmlFor="enable-memory" className="text-sm">
                  Enable stance tracking (ResearchAssistant)
                </label>
              </div>

              {form.enable_memory && (
                <Field label="Working memory window" hint="Recent turns in memory (1–50, default 10)">
                  <Input
                    type="number"
                    min={1}
                    max={50}
                    value={form.working_memory_window}
                    onChange={(e) => setField("working_memory_window", e.target.value)}
                    className="w-32"
                  />
                </Field>
              )}

              <div className="flex items-center gap-3">
                <input
                  id="validate-personas"
                  type="checkbox"
                  checked={form.validate_personas}
                  onChange={(e) => setField("validate_personas", e.target.checked)}
                  className="h-4 w-4 accent-primary"
                />
                <label htmlFor="validate-personas" className="text-sm">
                  Validate generated personas
                </label>
              </div>
            </div>
          )}
        </div>

        {submitError && (
          <div className="border border-destructive/30 bg-destructive/10 text-destructive rounded px-4 py-3 text-sm">
            {submitError}
          </div>
        )}

        <div className="flex gap-3">
          <Button type="submit" disabled={submitting} className="min-w-32">
            {submitting ? "Submitting…" : "Run Simulation"}
          </Button>
          <Button asChild variant="outline">
            <Link href="/">Cancel</Link>
          </Button>
        </div>
      </form>
    </div>
  );
}

function Section({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div>
      <h2 className="text-sm font-semibold text-foreground uppercase tracking-wide mb-3 pb-2 border-b border-border">
        {title}
      </h2>
      <div className="space-y-4">{children}</div>
    </div>
  );
}

function Field({
  label,
  hint,
  children,
}: {
  label?: string;
  hint?: string;
  children: React.ReactNode;
}) {
  return (
    <div>
      {label && (
        <label className="block text-sm font-medium text-foreground mb-1">{label}</label>
      )}
      {children}
      {hint && <p className="text-xs text-muted-foreground mt-1">{hint}</p>}
    </div>
  );
}
