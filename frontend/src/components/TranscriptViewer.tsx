import { Separator } from "@/components/ui/separator";
import type { SessionResultData } from "@/lib/api";

interface Props {
  result: SessionResultData;
  metadata: {
    participant_count: number | null;
    total_turns: number | null;
    completed_at: string | null;
  };
}

export default function TranscriptViewer({ result, metadata }: Props) {
  const allQuestions = [
    ...result.question_results,
    ...(result.closing_result ? [result.closing_result] : []),
  ];

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <Stat label="Participants" value={String(metadata.participant_count ?? "—")} />
        <Stat label="Total Turns" value={String(metadata.total_turns ?? "—")} />
        <Stat label="Questions" value={String(result.total_questions)} />
        <Stat
          label="Duration"
          value={
            result.duration_seconds != null
              ? `${Math.floor(result.duration_seconds / 60)}m ${Math.round(result.duration_seconds % 60)}s`
              : "—"
          }
        />
      </div>

      {result.purpose && (
        <blockquote className="border-l-4 border-primary pl-4 py-1 text-sm text-muted-foreground italic">
          {result.purpose}
        </blockquote>
      )}

      <Separator />

      {allQuestions.map((qr, qi) => (
        <QuestionBlock
          key={qi}
          number={qi + 1}
          isClosing={result.closing_result != null && qi === allQuestions.length - 1}
          questionResult={qr}
        />
      ))}

      {result.stance_matrix_dump && (
        <>
          <Separator />
          <div>
            <h3 className="text-sm font-semibold mb-2 text-foreground">Stance Analysis</h3>
            <pre className="text-xs bg-secondary rounded p-4 overflow-auto max-h-64 text-muted-foreground">
              {JSON.stringify(result.stance_matrix_dump, null, 2)}
            </pre>
          </div>
        </>
      )}

      {result.errors.length > 0 && (
        <div className="rounded border border-destructive/30 bg-destructive/10 text-destructive px-4 py-3 text-sm">
          <p className="font-medium mb-1">Session errors ({result.errors.length})</p>
          <ul className="list-disc list-inside space-y-0.5 text-xs">
            {result.errors.map((e, i) => (
              <li key={i}>{e}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-secondary rounded p-3">
      <p className="text-xs text-muted-foreground">{label}</p>
      <p className="text-xl font-semibold mt-0.5">{value}</p>
    </div>
  );
}

interface QuestionResult {
  question: string;
  responses: Array<{
    speaker_name: string;
    content: string;
    turn_number: number;
  }>;
  nonverbal_reactions: Array<{
    participant_name: string;
    reaction: string;
    response_turn: number | null;
  }>;
}

function QuestionBlock({
  number,
  isClosing,
  questionResult,
}: {
  number: number;
  isClosing: boolean;
  questionResult: QuestionResult;
}) {
  return (
    <div className="space-y-3">
      <div className="flex items-start gap-3">
        <span className="mt-0.5 text-xs font-semibold text-primary uppercase tracking-wide shrink-0">
          {isClosing ? "Closing" : `Q${number}`}
        </span>
        <p className="text-sm font-medium text-foreground leading-snug">
          {questionResult.question}
        </p>
      </div>

      <div className="pl-10 space-y-4">
        {questionResult.responses.map((response) => {
          const reactions = questionResult.nonverbal_reactions.filter(
            (r) => r.response_turn === response.turn_number
          );
          return (
            <div key={response.turn_number} className="border-l-2 border-border pl-3">
              <div className="flex items-baseline gap-2 mb-1">
                <span className="text-sm font-semibold text-foreground">
                  {response.speaker_name}
                </span>
                <span className="text-xs text-muted-foreground">turn {response.turn_number}</span>
              </div>
              <p className="text-sm text-foreground leading-relaxed">{response.content}</p>
              {reactions.length > 0 && (
                <p className="mt-1.5 text-xs text-muted-foreground italic">
                  {reactions.map((r) => `[${r.participant_name}: ${r.reaction}]`).join("  ")}
                </p>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
