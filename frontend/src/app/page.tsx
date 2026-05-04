"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { api, type JobSummary, type ApiError } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import JobCard from "@/components/JobCard";

export default function DashboardPage() {
  const [jobs, setJobs] = useState<JobSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchJobs = async () => {
    try {
      const data = await api.jobs.list();
      setJobs(
        [...data].sort(
          (a, b) =>
            new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
        )
      );
    } catch (err) {
      if ((err as ApiError).status !== 401) {
        setError((err as Error).message);
      }
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async (jobId: string) => {
    try {
      await api.jobs.delete(jobId);
      setJobs((prev) => prev.filter((j) => j.job_id !== jobId));
    } catch (err) {
      setError((err as Error).message);
    }
  };

  useEffect(() => {
    fetchJobs();
  }, []);

  return (
    <div>
      <div className="flex items-start justify-between mb-6">
        <div>
          <h1 className="text-2xl font-semibold text-foreground">My Simulations</h1>
          <p className="text-sm text-muted-foreground mt-1">
            Simulation results are available for 30 days after completion.
          </p>
        </div>
        <Button asChild>
          <Link href="/jobs/new">New Simulation</Link>
        </Button>
      </div>

      {loading && (
        <div className="space-y-2">
          {[1, 2, 3].map((i) => (
            <Skeleton key={i} className="h-12 w-full" />
          ))}
        </div>
      )}

      {!loading && error && (
        <div className="border border-destructive/30 bg-destructive/10 text-destructive rounded px-4 py-3 text-sm">
          {error}
        </div>
      )}

      {!loading && !error && jobs.length === 0 && (
        <div className="border border-dashed border-border rounded py-16 text-center">
          <p className="text-muted-foreground text-sm">No simulations yet.</p>
          <Button asChild variant="outline" className="mt-4">
            <Link href="/jobs/new">Run your first simulation</Link>
          </Button>
        </div>
      )}

      {!loading && !error && jobs.length > 0 && (
        <div className="border border-border rounded overflow-hidden">
          <table className="w-full text-sm">
            <thead className="bg-secondary text-secondary-foreground">
              <tr>
                <th className="text-left px-4 py-3 font-medium">Job ID</th>
                <th className="text-left px-4 py-3 font-medium">Status</th>
                <th className="text-left px-4 py-3 font-medium hidden sm:table-cell">
                  Participants
                </th>
                <th className="text-left px-4 py-3 font-medium hidden md:table-cell">
                  Turns
                </th>
                <th className="text-left px-4 py-3 font-medium hidden md:table-cell">
                  Submitted
                </th>
                <th className="text-right px-4 py-3 font-medium">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {jobs.map((job) => (
                <JobCard key={job.job_id} job={job} onDelete={handleDelete} />
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
