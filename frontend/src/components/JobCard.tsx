"use client";

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

export default function JobCard({ job, onDelete }: Props) {
  const shortId = job.job_id.split("-")[0];

  return (
    <tr className="hover:bg-secondary/40 transition-colors">
      <td className="px-4 py-3">
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
        <div className="flex items-center justify-end gap-1">
          <Button asChild variant="ghost" size="sm" className="h-7 w-7 p-0">
            <Link href={`/jobs/${job.job_id}`} aria-label="View job">
              <Eye className="h-3.5 w-3.5" />
            </Link>
          </Button>
          {job.status !== "running" && (
            <Button
              variant="ghost"
              size="sm"
              className="h-7 w-7 p-0 text-destructive hover:text-destructive hover:bg-destructive/10"
              onClick={() => onDelete(job.job_id)}
              aria-label="Delete job"
            >
              <Trash2 className="h-3.5 w-3.5" />
            </Button>
          )}
        </div>
      </td>
    </tr>
  );
}
