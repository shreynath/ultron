"use client";

import { useEffect, useRef, useState } from "react";
import { chatWithVoice, voiceChat, type CognitiveMetadata } from "@/lib/api";
import { VoiceProfilePicker } from "@/components/VoiceProfilePicker";
import { DEFAULT_VOICE_DELIVERY, VoiceCharacterControls, type VoiceDelivery } from "@/components/VoiceCharacterControls";

export type ConversationState = "idle" | "listening" | "transcribing" | "thinking" | "speaking" | "error";

type ConversationMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  createdAt: string;
  status?: "sending" | "complete" | "error";
  metadata?: CognitiveMetadata;
};

const MAX_VISIBLE_HISTORY = 24;
const MAX_BRAIN_HISTORY = 8;
export function TalkMode() {
  const [profileId, setProfileId] = useState("");
  const [engine, setEngine] = useState("qwen");
  const [voiceEnabled, setVoiceEnabled] = useState(true);
  const [delivery, setDelivery] = useState<VoiceDelivery>(DEFAULT_VOICE_DELIVERY);
  const [conversationState, setConversationState] = useState<ConversationState>("idle");
  const [error, setError] = useState<string | null>(null);
  const [textInput, setTextInput] = useState("");
  const [messages, setMessages] = useState<ConversationMessage[]>([]);
  const [audioLevel, setAudioLevel] = useState(0);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const turnRef = useRef(0);
  const audioContextRef = useRef<AudioContext | null>(null);
  const animationFrameRef = useRef<number | null>(null);

  useEffect(() => () => {
    mediaRecorderRef.current?.stream.getTracks().forEach((track) => track.stop());
    audioRef.current?.pause();
    stopMonitoring();
  }, []);

  function stopMonitoring() {
    if (animationFrameRef.current !== null) cancelAnimationFrame(animationFrameRef.current);
    animationFrameRef.current = null;
    const context = audioContextRef.current;
    audioContextRef.current = null;
    if (context?.state !== "closed") void context?.close();
    setAudioLevel(0);
  }

  function monitor(source: MediaStream | HTMLAudioElement, outputAudio: boolean) {
    stopMonitoring();
    try {
      const context = new AudioContext();
      const analyser = context.createAnalyser();
      analyser.fftSize = 64;
      const node = source instanceof MediaStream ? context.createMediaStreamSource(source) : context.createMediaElementSource(source);
      node.connect(analyser);
      if (outputAudio) analyser.connect(context.destination);
      audioContextRef.current = context;
      const values = new Uint8Array(analyser.frequencyBinCount);
      const sample = () => {
        analyser.getByteFrequencyData(values);
        const average = values.reduce((total, value) => total + value, 0) / values.length / 255;
        setAudioLevel(Math.min(1, average * 2.4));
        animationFrameRef.current = requestAnimationFrame(sample);
      };
      void context.resume();
      sample();
    } catch {
      setAudioLevel(0);
    }
  }

  function addMessage(message: ConversationMessage) {
    setMessages((previous) => [...previous, message].slice(-MAX_VISIBLE_HISTORY));
  }

  function historyForBrain() {
    return messages.slice(-MAX_BRAIN_HISTORY).map(({ role, content }) => ({ role, content }));
  }

  function playAudio(src: string, turn: number) {
    if (turn !== turnRef.current) return;
    audioRef.current?.pause();
    const audio = new Audio(src);
    audioRef.current = audio;
    monitor(audio, true);
    audio.onended = () => {
      if (turn === turnRef.current && audioRef.current === audio) stopMonitoring();
    };
    void audio.play().catch(() => {
      if (turn === turnRef.current && audioRef.current === audio) stopMonitoring();
    });
  }

  async function completeTurn(result: { user_message: string; assistant_message: string; metadata: CognitiveMetadata; voice_error?: string | null; audioSrc?: string }, turn: number) {
    if (turn !== turnRef.current) return;
    const assistantId = crypto.randomUUID();
    addMessage({ id: crypto.randomUUID(), role: "user", content: result.user_message, createdAt: new Date().toISOString(), status: "complete" });
    addMessage({ id: assistantId, role: "assistant", content: result.assistant_message, createdAt: new Date().toISOString(), status: "complete", metadata: result.metadata });
    setTextInput("");
    if (result.audioSrc) playAudio(result.audioSrc, turn);
    if (turn === turnRef.current) setConversationState("idle");
  }

  async function sendText() {
    const message = textInput.trim();
    if (!message || (conversationState !== "idle" && conversationState !== "error")) return;
    const turn = ++turnRef.current;
    audioRef.current?.pause();
    setError(null);
    setConversationState("thinking");
    try {
      const result = await chatWithVoice({ message, history: historyForBrain(), profile_id: profileId || undefined, voice_enabled: voiceEnabled && Boolean(profileId), engine, ...delivery });
      await completeTurn(result, turn);
    } catch (err) {
      if (turn === turnRef.current) {
        setError(err instanceof Error ? err.message : "I couldn't process that request. Please try again.");
        setConversationState("error");
      }
    }
  }

  async function startRecording() {
    if (conversationState !== "idle" && conversationState !== "error") return;
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      monitor(stream, false);
      const recorder = new MediaRecorder(stream);
      chunksRef.current = [];
      recorder.ondataavailable = (event) => { if (event.data.size > 0) chunksRef.current.push(event.data); };
      recorder.onstop = async () => {
        stream.getTracks().forEach((track) => track.stop());
        stopMonitoring();
        const blob = new Blob(chunksRef.current, { type: "audio/webm" });
        if (!blob.size) {
          setError("I couldn't understand the recording. Please try again.");
          setConversationState("error");
          return;
        }
        const turn = ++turnRef.current;
        setConversationState("transcribing");
        try {
          const result = await voiceChat({ audio: blob, history: historyForBrain(), profile_id: profileId || undefined, voice_enabled: voiceEnabled && Boolean(profileId), engine, ...delivery });
          setConversationState("thinking");
          await completeTurn(result, turn);
        } catch (err) {
          if (turn === turnRef.current) {
            setError(err instanceof Error ? err.message : "I couldn't understand the recording. Please try again.");
            setConversationState("error");
          }
        }
      };
      mediaRecorderRef.current = recorder;
      recorder.start();
      setConversationState("listening");
    } catch {
      setError("Microphone access is unavailable. You can type your message instead.");
      setConversationState("error");
    }
  }

  function stopRecording() { mediaRecorderRef.current?.stop(); }

  const busy = conversationState !== "idle" && conversationState !== "error";
  return (
    <section className="conversation-shell glass-panel liquid-target" aria-label="Ultron conversation">
      <header className="conversation-header conversation-tools">
        <div className="flex items-center gap-3">
          <details className="voice-settings">
            <summary>⚙ Voice</summary>
            <div className="voice-settings-panel">
              <label className="flex items-center gap-2 text-sm text-zinc-200"><input type="checkbox" checked={voiceEnabled} onChange={(event) => setVoiceEnabled(event.target.checked)} />Speak responses automatically</label>
              <VoiceProfilePicker value={profileId} onChange={setProfileId} engine={engine} onEngineChange={setEngine} showEngine id="conversation-voice" />
              <VoiceCharacterControls value={delivery} onChange={setDelivery} id="conversation-delivery" />
            </div>
          </details>
        </div>
      </header>

      <div className="conversation-history" aria-live="polite">
        {messages.length === 0 ? <div className="conversation-empty"><div className="core-glyph" aria-hidden="true" /><p>Speak, and I&apos;ll listen.</p><span>Type a message or use the microphone.</span></div> : messages.map((message) => (
          <article key={message.id} className={`message message-${message.role}`} aria-label={message.role === "user" ? "Your message" : "Ultron response"}>
            <p className="message-author">{message.role === "user" ? "You" : "Ultron"}</p>
            <p className="message-content">{message.content}</p>
          </article>
        ))}
        {(conversationState === "thinking" || conversationState === "transcribing") && <div className="conversation-processing" role="status"><span aria-hidden="true" /><span aria-hidden="true" /><span aria-hidden="true" />Ultron is thinking…</div>}
      </div>

      <div className="conversation-composer">
        {error && <p className="conversation-error" role="alert">{error}</p>}
        <div className="composer-row">
          <input className="conversation-input" aria-label="Ask Ultron anything" placeholder="Ask Ultron anything…" value={textInput} onChange={(event) => setTextInput(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter") void sendText(); }} disabled={busy} />
          <button className={`composer-icon ${conversationState === "listening" ? "is-listening" : ""}`} type="button" onClick={conversationState === "listening" ? stopRecording : startRecording} disabled={busy && conversationState !== "listening"} aria-label={conversationState === "listening" ? "Stop recording" : "Start voice input"}>{conversationState === "listening" ? <span className="mic-bars" aria-hidden="true"><i style={{ height: `${8 + audioLevel * 20}px` }} /><i style={{ height: `${12 + audioLevel * 24}px` }} /><i style={{ height: `${8 + audioLevel * 20}px` }} /></span> : "🎙"}</button>
          <button className="composer-send" type="button" onClick={() => void sendText()} disabled={busy || !textInput.trim()} aria-label="Send message">↑</button>
        </div>
      </div>
    </section>
  );
}
