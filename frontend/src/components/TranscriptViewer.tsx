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
          <StanceMatrixViewer data={result.stance_matrix_dump} />
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

interface ParticipantStanceData {
  position: string;
  key_points: string[];
  sentiment_score: number;
  contribution_count: number;
}

const POSITION_STYLES: Record<string, string> = {
  supportive: "bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400",
  skeptical: "bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400",
  neutral: "bg-zinc-100 text-zinc-700 dark:bg-zinc-800 dark:text-zinc-300",
  mixed: "bg-amber-100 text-amber-800 dark:bg-amber-900/30 dark:text-amber-400",
  not_contributed: "bg-slate-100 text-slate-500 dark:bg-slate-800 dark:text-slate-400",
};

function StanceMatrixViewer({ data }: { data: Record<string, unknown> }) {
  const entries = Object.entries(data);
  return (
    <div>
      <h3 className="text-sm font-semibold mb-3 text-foreground">Stance Analysis</h3>
      <div className="space-y-3">
        {entries.map(([question, participants]) => (
          <QuestionStanceTable
            key={question}
            question={question}
            participants={participants as Record<string, ParticipantStanceData>}
          />
        ))}
      </div>
    </div>
  );
}

function QuestionStanceTable({
  question,
  participants,
}: {
  question: string;
  participants: Record<string, ParticipantStanceData>;
}) {
  const rows = Object.entries(participants ?? {});
  return (
    <details open className="rounded border border-border overflow-hidden">
      <summary className="px-4 py-2.5 bg-secondary text-xs font-semibold text-foreground cursor-pointer select-none hover:bg-secondary/80 transition-colors list-none flex items-center justify-between">
        <span>{question}</span>
        <span className="text-muted-foreground font-normal">{rows.length} participant{rows.length !== 1 ? "s" : ""}</span>
      </summary>
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-border bg-background">
              <th className="text-left px-4 py-2 font-medium text-muted-foreground">Participant</th>
              <th className="text-left px-4 py-2 font-medium text-muted-foreground">Position</th>
              <th className="text-left px-4 py-2 font-medium text-muted-foreground w-24">Sentiment</th>
              <th className="text-center px-4 py-2 font-medium text-muted-foreground">Turns</th>
              <th className="text-left px-4 py-2 font-medium text-muted-foreground">Key Points</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border">
            {rows.map(([name, stance]) => {
              const posStyle = POSITION_STYLES[stance.position] ?? POSITION_STYLES.neutral;
              return (
                <tr key={name} className="hover:bg-secondary/30 transition-colors">
                  <td className="px-4 py-2.5 font-medium text-foreground whitespace-nowrap">{name}</td>
                  <td className="px-4 py-2.5">
                    <span className={`inline-block px-2 py-0.5 rounded text-[11px] font-medium capitalize ${posStyle}`}>
                      {(stance.position ?? "—").replace(/_/g, " ")}
                    </span>
                  </td>
                  <td className="px-4 py-2.5 whitespace-nowrap">
                    <div className="flex items-center gap-2">
                      <div className="w-14 h-1.5 bg-border rounded-full overflow-hidden">
                        <div
                          className="h-full bg-primary rounded-full"
                          style={{ width: `${Math.round((stance.sentiment_score ?? 0) * 100)}%` }}
                        />
                      </div>
                      <span className="text-muted-foreground">{(stance.sentiment_score ?? 0).toFixed(1)}</span>
                    </div>
                  </td>
                  <td className="px-4 py-2.5 text-center text-muted-foreground">{stance.contribution_count ?? 0}</td>
                  <td className="px-4 py-2.5 text-muted-foreground">
                    {stance.key_points?.length > 0
                      ? stance.key_points.join("; ")
                      : <span className="italic">—</span>}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </details>
  );
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
