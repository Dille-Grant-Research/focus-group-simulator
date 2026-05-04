"use client";

import { useState } from "react";
import Link from "next/link";
import { Eye, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import StatusBadge from "@/components/StatusBadge";
import type { JobSummary } from "@/lib/api";

interface Props {
  job: JobSummary;
  onDelete: (jobId: string) => void;
}

function fmt(iso: string): string {
  return new Date(iso).toLocaleString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

const STATUS_STRIPE: Record<string, string> = {
  queued: "border-l-4 border-l-zinc-300",
  running: "border-l-4 border-l-yellow-400",
  completed: "border-l-4 border-l-green-500",
  failed: "border-l-4 border-l-red-500",
};

export default function JobCard({ job, onDelete }: Props) {
  const shortId = job.job_id.split("-")[0];
  const [confirming, setConfirming] = useState(false);

  function handleDeleteClick() {
    if (!confirming) {
      setConfirming(true);
      return;
    }
    onDelete(job.job_id);
  }

  return (
    <tr className="hover:bg-secondary/40 transition-colors">
      <td className={`px-4 py-3 ${STATUS_STRIPE[job.status] ?? ""}`}>
        <Link
          href={`/jobs/${job.job_id}`}
          className="font-mono text-xs text-muted-foreground hover:text-foreground transition-colors"
          title={job.job_id}
        >
          {shortId}…
        </Link>
      </td>
      <td className="px-4 py-3">
        <StatusBadge status={job.status} />
      </td>
      <td className="px-4 py-3 text-sm text-muted-foreground hidden sm:table-cell">
        {job.participant_count ?? "—"}
      </td>
      <td className="px-4 py-3 text-sm text-muted-foreground hidden md:table-cell">
        {job.total_turns ?? "—"}
      </td>
      <td className="px-4 py-3 text-xs text-muted-foreground hidden md:table-cell">
        {fmt(job.created_at)}
      </td>
      <td className="px-4 py-3 text-right">
        {confirming ? (
          <div className="flex items-center justify-end gap-2">
            <span className="text-xs text-destructive font-medium">Delete?</span>
            <Button
              variant="destructive"
              size="xs"
              onClick={handleDeleteClick}
            >
              Yes
            </Button>
            <Button
              variant="ghost"
              size="xs"
              onClick={() => setConfirming(false)}
            >
              Cancel
            </Button>
          </div>
        ) : (
          <div className="flex items-center justify-end gap-1">
            <Button asChild variant="ghost" size="icon-sm">
              <Link href={`/jobs/${job.job_id}`} aria-label="View job">
                <Eye className="h-3.5 w-3.5" />
              </Link>
            </Button>
            {job.status !== "running" && (
              <Button
                variant="ghost"
                size="icon-sm"
                className="text-destructive hover:text-destructive hover:bg-destructive/10"
                onClick={handleDeleteClick}
                aria-label="Delete job"
              >
                <Trash2 className="h-3.5 w-3.5" />
              </Button>
            )}
          </div>
        )}
      </td>
    </tr>
  );
}
