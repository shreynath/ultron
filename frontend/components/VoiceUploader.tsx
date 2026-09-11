"use client";

import { useState } from "react";
import { cloneFromFile, cloneFromUrl } from "@/lib/api";
import { ENGINE_OPTIONS } from "@/components/VoiceProfilePicker";

type Step = "idle" | "downloading" | "extracting" | "transcribing" | "creating" | "done" | "error";

export function VoiceUploader({ onCreated }: { onCreated?: () => void }) {
  const [name, setName] = useState("");
  const [url, setUrl] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [engine, setEngine] = useState("qwen");
  const [step, setStep] = useState<Step>("idle");
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<string | null>(null);

  async function handleSubmit(event: React.FormEvent) {
    event.preventDefault();
    setError(null);
    setResult(null);

    if (!name.trim()) {
      setError("Enter a voice name");
      return;
    }
    if (!url.trim() && !file) {
      setError("Provide a YouTube URL or upload an audio/video file");
      return;
    }

    try {
      setStep("downloading");
      setStep("extracting");
      setStep("transcribing");

      const response = url.trim()
        ? await cloneFromUrl({ url: url.trim(), name: name.trim(), engine })
        : file
          ? await cloneFromFile({ file, name: name.trim(), engine })
          : (() => {
              throw new Error("Provide a YouTube URL or upload an audio/video file");
            })();

      setStep("creating");
      setStep("done");
      setResult(
        `Created voice "${response.name}" with ${response.samples_added} sample(s).${
          response.delivery
            ? ` Tuned pace to ${Math.round(response.delivery.speed * 100)}%, bass ${response.delivery.bass_boost} dB.`
            : ""
        }${
          response.quality_warning ? ` ${response.quality_warning}` : ""
        } Use "Reference match" style in Studio.`
      );
      setName("");
      setUrl("");
      setFile(null);
      onCreated?.();
    } catch (err) {
      setStep("error");
      setError(err instanceof Error ? err.message : "Clone failed");
    }
  }

  return (
    <form className="card space-y-5 liquid-target" onSubmit={handleSubmit}>
      <div>
        <h2 className="text-xl font-semibold">Clone a Voice</h2>
        <p className="mt-1 text-sm text-[color:var(--text-secondary)]">
          Paste a long YouTube clip (e.g. 10 min Ultron scenes). Ultron automatically finds solo
          monologue segments, rejects other characters, and tunes pace and depth to match the
          reference.
        </p>
      </div>

      <div>
        <label className="label" htmlFor="voice-name">
          Voice name
        </label>
        <input
          id="voice-name"
          className="input"
          placeholder="e.g. Morgan"
          value={name}
          onChange={(event) => setName(event.target.value)}
        />
      </div>

      <div>
        <label className="label" htmlFor="voice-engine">
          Cloning engine
        </label>
        <select
          id="voice-engine"
          className="input"
          value={engine}
          onChange={(event) => setEngine(event.target.value)}
        >
          {ENGINE_OPTIONS.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
      </div>

      <div>
        <label className="label" htmlFor="voice-url">
          YouTube or media URL
        </label>
        <input
          id="voice-url"
          className="input"
          placeholder="https://youtube.com/watch?v=..."
          value={url}
          onChange={(event) => {
            setUrl(event.target.value);
            if (event.target.value.trim()) setFile(null);
          }}
        />
      </div>

      <div>
        <label className="label" htmlFor="voice-file">
          Or upload file
        </label>
        <label className="dropzone" htmlFor="voice-file">
          <input id="voice-file" type="file" accept="audio/*,video/*,.mp3,.mp4,.wav,.m4a" onChange={(event) => {
            setFile(event.target.files?.[0] ?? null);
            if (event.target.files?.[0]) setUrl("");
          }} />
          <span>{file ? `◌  ${file.name}` : "Drop an audio or video file here, or click to browse"}</span>
        </label>
      </div>

      {step !== "idle" && step !== "done" && step !== "error" && (
        <div className="scan-progress">{step === "downloading" && "Retrieving reference media…"}{step === "extracting" && "Extracting monologue segments…"}{step === "transcribing" && "Analyzing reference speech…"}{step === "creating" && "Creating voice profile…"}</div>
      )}

      {error && <p className="error-copy">⚠ {error}</p>}
      {result && <p className="success-copy">{result}</p>}

      <button
        className="btn-primary"
        type="submit"
        disabled={step !== "idle" && step !== "done" && step !== "error"}
      >
        Clone Voice
      </button>
    </form>
  );
}
