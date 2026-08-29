"use client";

import { useState } from "react";
import { cloneFromFile, cloneFromUrl } from "@/lib/api";

type Step = "idle" | "downloading" | "extracting" | "transcribing" | "creating" | "done" | "error";

const steps: Step[] = ["downloading", "extracting", "transcribing", "creating"];

export function VoiceUploader({ onCreated }: { onCreated?: () => void }) {
  const [name, setName] = useState("");
  const [url, setUrl] = useState("");
  const [file, setFile] = useState<File | null>(null);
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
      await new Promise((resolve) => setTimeout(resolve, 300));
      setStep("extracting");
      await new Promise((resolve) => setTimeout(resolve, 300));
      setStep("transcribing");

      const response = file
        ? await cloneFromFile({ file, name: name.trim() })
        : await cloneFromUrl({ url: url.trim(), name: name.trim() });

      setStep("creating");
      setStep("done");
      setResult(`Created voice "${response.name}" with ${response.samples_added} sample(s).`);
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
    <form className="card space-y-5" onSubmit={handleSubmit}>
      <div>
        <h2 className="text-xl font-semibold">Clone a Voice</h2>
        <p className="mt-1 text-sm text-zinc-400">
          Paste a YouTube URL or upload MP3/MP4/WAV. Ultron extracts clean speech clips and
          creates a zero-shot voice profile.
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
        <label className="label" htmlFor="voice-url">
          YouTube or media URL
        </label>
        <input
          id="voice-url"
          className="input"
          placeholder="https://youtube.com/watch?v=..."
          value={url}
          onChange={(event) => setUrl(event.target.value)}
        />
      </div>

      <div>
        <label className="label" htmlFor="voice-file">
          Or upload file
        </label>
        <input
          id="voice-file"
          type="file"
          accept="audio/*,video/*,.mp3,.mp4,.wav,.m4a"
          className="block w-full text-sm text-zinc-300 file:mr-4 file:rounded-lg file:border-0 file:bg-zinc-800 file:px-4 file:py-2 file:text-sm file:text-zinc-100"
          onChange={(event) => setFile(event.target.files?.[0] ?? null)}
        />
      </div>

      {step !== "idle" && step !== "done" && step !== "error" && (
        <div className="rounded-xl border border-zinc-800 bg-zinc-950 p-4 text-sm">
          <p className="mb-3 text-zinc-300">Processing...</p>
          <div className="space-y-2">
            {steps.map((item) => (
              <p
                key={item}
                className={item === step ? "text-violet-300" : "text-zinc-500"}
              >
                {item === "downloading" && "Downloading source audio"}
                {item === "extracting" && "Extracting best speech clips"}
                {item === "transcribing" && "Transcribing reference samples"}
                {item === "creating" && "Creating voice profile"}
              </p>
            ))}
          </div>
        </div>
      )}

      {error && <p className="text-sm text-red-400">{error}</p>}
      {result && <p className="text-sm text-emerald-400">{result}</p>}

      <button className="btn-primary" type="submit" disabled={step !== "idle" && step !== "done" && step !== "error"}>
        Clone Voice
      </button>
    </form>
  );
}
