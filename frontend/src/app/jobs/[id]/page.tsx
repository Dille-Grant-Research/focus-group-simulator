"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import Link from "next/link";
import { ChevronLeft, RefreshCw, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { Separator } from "@/components/ui/separator";
import StatusBadge from "@/components/StatusBadge";
import TranscriptViewer from "@/components/TranscriptViewer";
import { api, type JobStatus, type JobStatusResponse, type JobResultResponse, type ApiError } from "@/lib/api";

const POLL_INTERVAL: Record<string, number> = {
  queued: 10_000,
  running: 5_000,
};

export default function JobDetailPage() {
  const { id } = useParams<{ id: string }>();
  const router = useRouter();
  const [status, setStatus] = useState<JobStatusResponse | null>(null);
  const [result, setResult] = useState<JobResultResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [deleteConfirm, setDeleteConfirm] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const handleDelete = async () => {
    setDeleting(true);
    try {
      await api.jobs.delete(id);
      router.push("/");
    } catch (err) {
      setError((err as Error).message);
      setDeleting(false);
      setDeleteConfirm(false);
    }
  };

  const fetchStatus = useCallback(async () => {
    try {
      const data = await api.jobs.getStatus(id);
      setStatus(data);

      if (data.status === "completed") {
        const res = await api.jobs.getResult(id);
        setResult(res);
      }
    } catch (err) {
      const apiErr = err as ApiError;
      if (apiErr.status !== 401) {
        setError(apiErr.message ?? "Failed to load job.");
      }
    } finally {
      setLoading(false);
    }
  }, [id]);

  useEffect(() => {
    fetchStatus();
  }, [fetchStatus]);

  useEffect(() => {
    if (!status) return;
    const interval = POLL_INTERVAL[status.status];
    if (!interval) return;

    timerRef.current = setTimeout(() => {
      fetchStatus();
    }, interval);

    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [status, fetchStatus]);

  return (
    <div className="max-w-6xl">
      <div className="flex items-center justify-between mb-6">
        <Link
          href="/"
          className="inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground transition-colors"
        >
          <ChevronLeft className="h-4 w-4" />
          Back to dashboard
        </Link>
        {status && (
          <div className="flex items-center gap-2">
            {status.status !== "running" && (
              deleteConfirm ? (
                <div className="flex items-center gap-2">
                  <span className="text-xs text-destructive font-medium">Delete this job?</span>
                  <Button
                    variant="destructive"
                    size="sm"
                    onClick={handleDelete}
                    disabled={deleting}
                  >
                    {deleting ? "Deleting…" : "Yes, delete"}
                  </Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setDeleteConfirm(false)}
                    disabled={deleting}
                  >
                    Cancel
                  </Button>
                </div>
              ) : (
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setDeleteConfirm(true)}
                  className="text-destructive hover:text-destructive hover:bg-destructive/10"
                >
                  <Trash2 className="h-3.5 w-3.5 mr-1" />
                  Delete
                </Button>
              )
            )}
            <Button
              variant="ghost"
              size="sm"
              onClick={fetchStatus}
              className="text-muted-foreground"
            >
              <RefreshCw className="h-3.5 w-3.5 mr-1" />
              Refresh
            </Button>
          </div>
        )}
      </div>

      {loading && (
        <div className="space-y-3">
          <Skeleton className="h-8 w-48" />
          <Skeleton className="h-4 w-72" />
        </div>
      )}

      {!loading && error && (
        <div className="border border-destructive/30 bg-destructive/10 text-destructive rounded px-4 py-3 text-sm">
          {error}
        </div>
      )}

      {!loading && status && (
        <div className="space-y-6">
          <div className="flex items-center gap-3">
            <StatusBadge status={status.status as JobStatus} />
            <span className="font-mono text-xs text-muted-foreground">{id}</span>
          </div>

          <JobMeta status={status} />

          <Separator />

          <StatusContent status={status} result={result} />
        </div>
      )}
    </div>
  );
}

function JobMeta({ status }: { status: JobStatusResponse }) {
  const rows: [string, string | null][] = [
    ["Submitted", fmt(status.created_at)],
    ["Started", status.started_at ? fmt(status.started_at) : null],
    ["Completed", status.completed_at ? fmt(status.completed_at) : null],
    ["Expires", fmt(status.expires_at)],
  ];
  return (
    <dl className="grid grid-cols-2 sm:grid-cols-4 gap-3">
      {rows.map(([label, value]) =>
        value ? (
          <div key={label}>
            <dt className="text-xs text-muted-foreground">{label}</dt>
            <dd className="text-xs font-medium mt-0.5">{value}</dd>
          </div>
        ) : null
      )}
    </dl>
  );
}

function StatusContent({
  status,
  result,
}: {
  status: JobStatusResponse;
  result: JobResultResponse | null;
}) {
  if (status.status === "queued") {
    return (
      <div className="text-sm text-muted-foreground space-y-1">
        <p>
          Your simulation is waiting in the queue.
          {status.queue_position != null && (
            <> You are <strong className="text-foreground">#{status.queue_position}</strong> in line.</>
          )}
        </p>
        <p>This page will update automatically every 10 seconds.</p>
      </div>
    );
  }

  if (status.status === "running") {
    return (
      <div className="text-sm text-muted-foreground space-y-2">
        <div className="flex items-center gap-2">
          <span className="h-2 w-2 rounded-full bg-yellow-400 animate-pulse" />
          <p>Simulation is running. This page updates every 5 seconds.</p>
        </div>
        <p className="text-xs">Full simulations typically take 3–10 minutes depending on participants and questions.</p>
      </div>
    );
  }

  if (status.status === "failed") {
    return (
      <div className="border border-destructive/30 bg-destructive/10 text-destructive rounded px-4 py-3 text-sm">
        <p className="font-medium">Simulation failed.</p>
        <p className="text-xs mt-1 opacity-80">
          Check your configuration and try again. Contact the research team if the problem persists.
        </p>
      </div>
    );
  }

  if (status.status === "completed" && result) {
    return (
      <div>
        <h2 className="text-base font-semibold mb-4">
          Transcript
          {result.result.session_name && (
            <span className="font-normal text-muted-foreground ml-2">
              — {result.result.session_name}
            </span>
          )}
        </h2>
        <TranscriptViewer result={result.result} metadata={result.metadata} />
      </div>
    );
  }

  return null;
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
