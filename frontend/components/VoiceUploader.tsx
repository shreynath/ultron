"use client";

import { useState } from "react";
import { cloneFromFile, cloneFromUrl } from "@/lib/api";
import { ENGINE_OPTIONS } from "@/components/VoiceProfilePicker";

type Step = "idle" | "downloading" | "extracting" | "transcribing" | "creating" | "done" | "error";

const steps: Step[] = ["downloading", "extracting", "transcribing", "creating"];

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
    <form className="card space-y-5" onSubmit={handleSubmit}>
      <div>
        <h2 className="text-xl font-semibold">Clone a Voice</h2>
        <p className="mt-1 text-sm text-zinc-400">
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
        <input
          id="voice-file"
          type="file"
          accept="audio/*,video/*,.mp3,.mp4,.wav,.m4a"
          className="block w-full text-sm text-zinc-300 file:mr-4 file:rounded-lg file:border-0 file:bg-zinc-800 file:px-4 file:py-2 file:text-sm file:text-zinc-100"
          onChange={(event) => {
            setFile(event.target.files?.[0] ?? null);
            if (event.target.files?.[0]) setUrl("");
          }}
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
