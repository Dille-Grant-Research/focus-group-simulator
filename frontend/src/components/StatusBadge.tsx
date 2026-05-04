import { cn } from "@/lib/utils";
import type { JobStatus } from "@/lib/api";

const styles: Record<JobStatus, string> = {
  queued: "bg-zinc-100 text-zinc-700 border border-zinc-200",
  running: "bg-yellow-100 text-yellow-800 border border-yellow-200",
  completed: "bg-green-100 text-green-800 border border-green-200",
  failed: "bg-red-100 text-red-800 border border-red-200",
};

const labels: Record<JobStatus, string> = {
  queued: "Queued",
  running: "Running",
  completed: "Completed",
  failed: "Failed",
};

export default function StatusBadge({
  status,
  className,
}: {
  status: JobStatus;
  className?: string;
}) {
  return (
    <span
      className={cn(
        "inline-flex items-center px-2 py-0.5 text-xs font-medium",
        styles[status] ?? "bg-zinc-100 text-zinc-700 border border-zinc-200",
        className
      )}
    >
      {labels[status] ?? status}
    </span>
  );
}
