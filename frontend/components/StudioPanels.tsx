"use client";

import { useEffect, useRef, useState } from "react";
import { listVoices, refineTranscript, transcribeAudio, type VoiceProfile } from "@/lib/api";

export function DictationMic() {
  const [recording, setRecording] = useState(false);
  const [transcript, setTranscript] = useState("");
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  async function startRecording() {
    setError(null);
    setStatus(null);
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

  return (
    <div className="card space-y-4">
      <div>
        <h2 className="text-xl font-semibold">Dictation</h2>
        <p className="mt-1 text-sm text-zinc-400">
          Record speech and transcribe it with Voicebox Whisper.
        </p>
      </div>

      <div className="flex gap-3">
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
  const [profiles, setProfiles] = useState<VoiceProfile[]>([]);
  const [profileId, setProfileId] = useState("");
  const [text, setText] = useState("Hello from Ultron. This is a cloned voice speaking new words.");
  const [audioSrc, setAudioSrc] = useState<string | null>(null);
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    listVoices()
      .then((data) => {
        setProfiles(data);
        if (data[0]) setProfileId(data[0].id);
      })
      .catch(() => setError("Could not load voice profiles"));
  }, []);

  async function handleSpeak() {
    if (!profileId || !text.trim()) return;
    setError(null);
    setStatus("Generating speech...");
    try {
      const { speakText } = await import("@/lib/api");
      const src = await speakText({ text, profile_id: profileId });
      setAudioSrc(src);
      setStatus("Ready");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Speech generation failed");
      setStatus(null);
    }
  }

  return (
    <div className="card space-y-4">
      <div>
        <h2 className="text-xl font-semibold">Text to Speech</h2>
        <p className="mt-1 text-sm text-zinc-400">
          Type any text and speak it in your cloned voice.
        </p>
      </div>

      <div>
        <label className="label" htmlFor="voice-select">
          Voice profile
        </label>
        <select
          id="voice-select"
          className="input"
          value={profileId}
          onChange={(event) => setProfileId(event.target.value)}
        >
          {profiles.map((profile) => (
            <option key={profile.id} value={profile.id}>
              {profile.name}
            </option>
          ))}
        </select>
      </div>

      <textarea
        className="input min-h-32"
        value={text}
        onChange={(event) => setText(event.target.value)}
      />

      <button className="btn-primary" type="button" onClick={handleSpeak} disabled={!profileId}>
        Generate Speech
      </button>

      {status && <p className="text-sm text-violet-300">{status}</p>}
      {error && <p className="text-sm text-red-400">{error}</p>}
      {audioSrc && <audio className="w-full" controls src={audioSrc} />}
    </div>
  );
}
