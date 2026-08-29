"use client";

import { useEffect, useRef, useState } from "react";
import { chatWithVoice, voiceChat } from "@/lib/api";
import { VoiceProfilePicker } from "@/components/VoiceProfilePicker";
import {
  DEFAULT_VOICE_DELIVERY,
  VoiceCharacterControls,
  type VoiceDelivery,
} from "@/components/VoiceCharacterControls";

type Message = {
  role: "user" | "assistant";
  text: string;
  audioSrc?: string;
};

export function TalkMode() {
  const [profileId, setProfileId] = useState("");
  const [engine, setEngine] = useState("qwen");
  const [recording, setRecording] = useState(false);
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [textInput, setTextInput] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [delivery, setDelivery] = useState<VoiceDelivery>(DEFAULT_VOICE_DELIVERY);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  useEffect(() => {
    return () => {
      if (audioRef.current) {
        audioRef.current.pause();
      }
    };
  }, []);

  function playAudio(src: string) {
    if (audioRef.current) {
      audioRef.current.pause();
    }
    const audio = new Audio(src);
    audioRef.current = audio;
    void audio.play();
  }

  async function handleVoiceTurn(blob: Blob) {
    if (!profileId) {
      setError("Select a voice profile first");
      return;
    }
    setError(null);
    setStatus("Listening...");
    try {
      setStatus("Thinking and speaking...");
      const result = await voiceChat({ audio: blob, profile_id: profileId, engine, ...delivery });
      setMessages((prev) => [
        ...prev,
        { role: "user", text: result.user_message },
        { role: "assistant", text: result.assistant_message, audioSrc: result.audioSrc },
      ]);
      playAudio(result.audioSrc);
      setStatus(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Voice chat failed");
      setStatus(null);
    }
  }

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
      await handleVoiceTurn(blob);
    };
    mediaRecorderRef.current = recorder;
    recorder.start();
    setRecording(true);
    setStatus("Recording... click Stop when done speaking");
  }

  function stopRecording() {
    mediaRecorderRef.current?.stop();
    setRecording(false);
  }

  async function handleTextSend() {
    if (!textInput.trim() || !profileId) return;
    setError(null);
    setStatus("Thinking and speaking...");
    try {
      const result = await chatWithVoice({
        message: textInput.trim(),
        profile_id: profileId,
        engine,
        ...delivery,
      });
      setMessages((prev) => [
        ...prev,
        { role: "user", text: result.user_message },
        { role: "assistant", text: result.assistant_message, audioSrc: result.audioSrc },
      ]);
      setTextInput("");
      playAudio(result.audioSrc);
      setStatus(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Chat failed");
      setStatus(null);
    }
  }

  return (
    <div className="card space-y-4">
      <div>
        <h2 className="text-xl font-semibold">Talk</h2>
        <p className="mt-1 text-sm text-zinc-400">
          Speak or type to your cloned voice. Ultron transcribes you, replies with Ollama, and
          speaks back in the voice you cloned.
        </p>
      </div>

      <VoiceProfilePicker
        value={profileId}
        onChange={setProfileId}
        engine={engine}
        onEngineChange={setEngine}
        showEngine
        id="talk-voice"
      />

      <VoiceCharacterControls value={delivery} onChange={setDelivery} id="talk-voice-character" />

      <div className="flex flex-wrap gap-3">
        {!recording ? (
          <button className="btn-primary" type="button" onClick={startRecording} disabled={!profileId}>
            Hold to Talk
          </button>
        ) : (
          <button className="btn-secondary" type="button" onClick={stopRecording}>
            Stop & Send
          </button>
        )}
      </div>

      <div className="flex gap-3">
        <input
          className="input"
          placeholder="Or type a message..."
          value={textInput}
          onChange={(event) => setTextInput(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter") void handleTextSend();
          }}
        />
        <button className="btn-primary shrink-0" type="button" onClick={handleTextSend} disabled={!profileId}>
          Send
        </button>
      </div>

      {status && <p className="text-sm text-violet-300">{status}</p>}
      {error && <p className="text-sm text-red-400">{error}</p>}

      <div className="max-h-80 space-y-3 overflow-y-auto rounded-xl border border-zinc-800 bg-zinc-950 p-4">
        {messages.length === 0 && (
          <p className="text-sm text-zinc-500">Your conversation will appear here.</p>
        )}
        {messages.map((message, index) => (
          <div
            key={`${message.role}-${index}`}
            className={message.role === "user" ? "text-zinc-300" : "text-violet-200"}
          >
            <p className="text-xs uppercase tracking-wide text-zinc-500">{message.role}</p>
            <p className="mt-1 text-sm">{message.text}</p>
            {message.audioSrc && message.role === "assistant" && (
              <button
                className="btn-secondary mt-2 text-xs"
                type="button"
                onClick={() => playAudio(message.audioSrc!)}
              >
                Replay
              </button>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
