"use client";

import { useEffect, useState } from "react";
import { CognitiveViewer } from "@/components/CognitiveViewer";
import { getBrainStatus, runBrainTask, type BrainStatus, type CognitiveExecution } from "@/lib/api";

export default function BrainPage() {
  const [brainStatus, setBrainStatus] = useState<BrainStatus | null>(null);
  const [prompt, setPrompt] = useState("");
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<CognitiveExecution[]>([]);
  useEffect(() => { void getBrainStatus().then(setBrainStatus).catch(() => undefined); }, []);
  async function run() {
    const message = prompt.trim();
    if (!message || running) return;
    setRunning(true); setError(null);
    try { const result = await runBrainTask({ message }); setResults((previous) => [result, ...previous].slice(0, 10)); }
    catch (err) { setError(err instanceof Error ? err.message : "Cognitive diagnostic failed"); }
    finally { setRunning(false); }
  }
  return <div className="brain-page page-stack">
    <section className="card liquid-target"><p className="eyebrow">Engineering surface</p><h2 className="mt-2 text-3xl font-semibold">Brain telemetry</h2><p className="mt-2 max-w-2xl text-sm text-zinc-400">Inspect structured Executive and Runtime output. Normal conversations belong on the Conversation page.</p>{brainStatus && <p className="mt-4 text-xs text-zinc-500">Runtime ready · model {brainStatus.brain_model}</p>}</section>
    <section className="card liquid-target"><label className="label" htmlFor="brain-prompt">Diagnostic prompt</label><div className="brain-runner"><input id="brain-prompt" className="input" placeholder="Run a diagnostic prompt…" value={prompt} onChange={(event) => setPrompt(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter") void run(); }} /><button className="btn-primary" type="button" onClick={() => void run()} disabled={running || !prompt.trim()}>{running ? "Processing…" : "Run"}</button></div>{error && <p className="error-copy mt-3">{error}</p>}</section>
    {results.map((result, index) => <CognitiveViewer key={`${result.status}-${index}`} result={result} compact={index > 0} />)}
  </div>;
}
