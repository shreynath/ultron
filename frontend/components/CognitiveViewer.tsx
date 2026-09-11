"use client";

import { useState } from "react";
import type { CognitiveExecution } from "@/lib/api";

export function CognitiveViewer({ result, compact = false }: { result: CognitiveExecution; compact?: boolean }) {
  const [expanded, setExpanded] = useState(!compact);
  const assessment = result.assessment;
  const statusColor = result.status === "completed" ? "is-completed" : result.status === "failed" ? "is-failed" : "is-pending";
  return <div className="telemetry-card card liquid-target text-sm">
    <div className="flex flex-wrap items-center justify-between gap-2 border-b border-zinc-800 pb-3">
      <div><span className="font-semibold text-zinc-200">Cognitive telemetry</span><span className="ml-2 text-xs text-zinc-500">{result.route}</span></div>
      <div className="flex items-center gap-2"><span className={`badge telemetry-status ${statusColor}`}>{result.status}</span>{compact && <button type="button" onClick={() => setExpanded((value) => !value)} className="text-xs text-red-300 hover:text-red-200">{expanded ? "Collapse" : "Inspect"}</button>}</div>
    </div>
    {expanded && <div className="mt-4 space-y-4 text-xs">
      <div className="flex flex-wrap gap-2"><span className="badge telemetry-emphasis">Task: {assessment.task_type}</span><span className="badge">Complexity: {assessment.complexity}</span><span className="badge">Budget: {assessment.reasoning_budget}</span>{assessment.used_fallback && <span className="badge telemetry-fallback">Fallback assessment</span>}</div>
      <div className="rounded-lg border border-zinc-800 bg-zinc-900/50 p-3 text-zinc-300"><p><strong className="text-zinc-100">Goal:</strong> {assessment.user_goal}</p><p className="mt-1 text-zinc-400"><strong className="text-zinc-300">Rationale:</strong> {assessment.rationale}</p></div>
      {result.required_capabilities.length > 0 && <p className="text-amber-200">Required but unavailable: {result.required_capabilities.join(", ")}</p>}
      {result.plan && <p className="text-zinc-400">Planner status: <span className="text-zinc-200">{result.plan.status}</span>{result.plan.failure_category ? ` (${result.plan.failure_category})` : ""}</p>}
      {result.capability_name && <p className="text-zinc-400">Capability: <span className="text-zinc-200">{result.capability_name}.{result.capability_operation}</span> · action {result.action_status} · observation {result.observation_outcome}</p>}
      {result.response_generation_status && <p className="text-zinc-400">Response generation: <span className="text-zinc-200">{result.response_generation_status}</span></p>}
      {result.verification && <p className="text-zinc-400">Verification: <span className="text-zinc-200">{result.verification.status}</span> · {result.verification.claims.length} claims · {result.verification.supporting_evidence.length} supporting · {result.verification.contradicting_evidence.length} contradicting · {result.verification.latency_ms.toFixed(1)} ms</p>}
      <p className="text-zinc-500">This view does not perform verification or autonomous coordination.</p>
    </div>}
  </div>;
}
