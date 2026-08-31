"use client";

import { useRef, useState } from "react";
import { refineTranscript, speakText, transcribeAudio } from "@/lib/api";
import { VoiceProfilePicker } from "@/components/VoiceProfilePicker";
import {
  DEFAULT_VOICE_DELIVERY,
  VoiceCharacterControls,
  type VoiceDelivery,
} from "@/components/VoiceCharacterControls";

export function DictationMic() {
  const [recording, setRecording] = useState(false);
  const [transcript, setTranscript] = useState("");
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  async function startRecording() {
    setError(null);
    setStatus(null);
    setCopied(false);
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const recorder = new MediaRecorder(stream);
    chunksRef.current = [];
    recorder.ondataavailable = (event) => {
      if (event.data.size > 0) chunksRef.current.push(event.data);
    };
    recorder.onstop = async () => {
      stream.getTracks().forEach((track) => track.stop());
      const blob = new Blob(chunksRef.current, { type: "audio/webm" });
      setStatus("Transcribing...");
      try {
        const text = await transcribeAudio(blob);
        setTranscript(text);
        setStatus("Done");
      } catch (err) {
        setError(err instanceof Error ? err.message : "Transcription failed");
        setStatus(null);
      }
    };
    mediaRecorderRef.current = recorder;
    recorder.start();
    setRecording(true);
  }

  function stopRecording() {
    mediaRecorderRef.current?.stop();
    setRecording(false);
  }

  async function handleRefine() {
    if (!transcript.trim()) return;
    setStatus("Refining with Ollama...");
    setError(null);
    try {
      const refined = await refineTranscript(transcript);
      setTranscript(refined);
      setStatus("Refined");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Refinement failed");
      setStatus(null);
    }
  }

  async function handleCopy() {
    if (!transcript.trim()) return;
    await navigator.clipboard.writeText(transcript);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }

  return (
    <div className="card space-y-4">
      <div>
        <h2 className="text-xl font-semibold">Dictation</h2>
        <p className="mt-1 text-sm text-zinc-400">
          Record speech and transcribe it with Voicebox Whisper.
        </p>
      </div>

      <div className="flex flex-wrap gap-3">
        {!recording ? (
          <button className="btn-primary" type="button" onClick={startRecording}>
            Start Recording
          </button>
        ) : (
          <button className="btn-secondary" type="button" onClick={stopRecording}>
            Stop Recording
          </button>
        )}
        <button
          className="btn-secondary"
          type="button"
          onClick={handleRefine}
          disabled={!transcript.trim()}
        >
          Refine with Ollama
        </button>
        <button
          className="btn-secondary"
          type="button"
          onClick={handleCopy}
          disabled={!transcript.trim()}
        >
          {copied ? "Copied!" : "Copy Transcript"}
        </button>
      </div>

      {status && <p className="text-sm text-violet-300">{status}</p>}
      {error && <p className="text-sm text-red-400">{error}</p>}

      <textarea
        className="input min-h-40"
        placeholder="Transcript will appear here..."
        value={transcript}
        onChange={(event) => setTranscript(event.target.value)}
      />
    </div>
  );
}

export function TTSPlayer() {
  const [profileId, setProfileId] = useState("");
  const [engine, setEngine] = useState("qwen");
  const [text, setText] = useState("Hello from Ultron. This is a cloned voice speaking new words.");
  const [audioSrc, setAudioSrc] = useState<string | null>(null);
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [delivery, setDelivery] = useState<VoiceDelivery>(DEFAULT_VOICE_DELIVERY);
  const [generating, setGenerating] = useState(false);
  const generatingRef = useRef(false);

  async function handleSpeak() {
    if (!profileId || !text.trim() || generatingRef.current) return;
    generatingRef.current = true;
    setGenerating(true);
    setError(null);
    setStatus("Shaping speech in theater mode...");
    try {
      const src = await speakText({ text, profile_id: profileId, engine, ...delivery });
      setAudioSrc(src);
      const base64 = src.split(",")[1];
      const binary = atob(base64);
      const bytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
      setAudioBlob(new Blob([bytes], { type: "audio/wav" }));
      setStatus("Ready");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Speech generation failed");
      setStatus(null);
    } finally {
      generatingRef.current = false;
      setGenerating(false);
    }
  }

  function handleDownload() {
    if (!audioBlob) return;
    const url = URL.createObjectURL(audioBlob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "ultron-speech.wav";
    anchor.click();
    URL.revokeObjectURL(url);
  }

  return (
    <div className="card space-y-4">
      <div>
        <h2 className="text-xl font-semibold">Text to Speech</h2>
        <p className="mt-1 text-sm text-zinc-400">
          Type any text and speak it in your cloned voice.
        </p>
      </div>

      <VoiceProfilePicker
        value={profileId}
        onChange={setProfileId}
        engine={engine}
        onEngineChange={setEngine}
        showEngine
      />

      <VoiceCharacterControls value={delivery} onChange={setDelivery} id="tts-voice-character" />

      <textarea
        className="input min-h-32"
        value={text}
        onChange={(event) => setText(event.target.value)}
      />

      <div className="flex flex-wrap gap-3">
        <button
          className="btn-primary"
          type="button"
          onClick={handleSpeak}
          disabled={!profileId || generating}
        >
          {generating ? "Generating..." : "Generate Speech"}
        </button>
        <button className="btn-secondary" type="button" onClick={handleDownload} disabled={!audioBlob || generating}>
          Download WAV
        </button>
      </div>

      {status && <p className="text-sm text-violet-300">{status}</p>}
      {error && <p className="text-sm text-red-400">{error}</p>}
      {audioSrc && <audio className="w-full" controls src={audioSrc} />}
    </div>
  );
}
