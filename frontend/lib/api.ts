const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

export type VoiceDelivery = {
  voice_style: "ultron_cinematic" | "natural";
  speed: number;
  bass_boost: number;
};

export type HealthResponse = {
  status: string;
  voicebox: { ok: boolean; detail: unknown };
  tools: { ffmpeg: boolean; yt_dlp: boolean };
  ollama_configured: boolean;
};

export type VoiceProfile = {
  id: string;
  name: string;
  engine?: string;
  language?: string;
  description?: string;
  sample_count?: number;
};

export function friendlyError(message: string): string {
  const lower = message.toLowerCase();
  if (lower.includes("connection attempts failed") || lower.includes("voicebox unavailable")) {
    return "Voicebox is not running. Open the Voicebox app from Applications, then refresh this page.";
  }
  if (lower.includes("no samples found")) {
    return "This voice has no reference audio. Delete it on the Voices page and clone again.";
  }
  if (lower.includes("failed to fetch") || lower.includes("networkerror")) {
    return "Cannot reach the Ultron API. Start it with: ./scripts/dev.sh";
  }
  return message;
}

async function parseError(response: Response, fallback: string): Promise<never> {
  try {
    const data = await response.json();
    const detail = data.detail;
    if (typeof detail === "string") throw new Error(friendlyError(detail));
    if (Array.isArray(detail)) {
      const message = detail.map((item) => item.msg).filter(Boolean).join("; ");
      throw new Error(friendlyError(message || fallback));
    }
    throw new Error(friendlyError(fallback));
  } catch (err) {
    if (err instanceof Error && err.message !== fallback) throw err;
    throw new Error(friendlyError(fallback));
  }
}

export async function getHealth(): Promise<HealthResponse> {
  const response = await fetch(`${API_URL}/api/health`, { cache: "no-store" });
  if (!response.ok) throw new Error("Health check failed");
  return response.json();
}

export async function listVoices(): Promise<VoiceProfile[]> {
  const response = await fetch(`${API_URL}/api/voices`, { cache: "no-store" });
  if (!response.ok) await parseError(response, "Failed to load voices");
  const data = await response.json();
  return data.profiles ?? [];
}

export async function cloneFromUrl(input: {
  url: string;
  name: string;
  engine?: string;
  language?: string;
}) {
  const response = await fetch(`${API_URL}/api/voices/clone`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  });
  if (!response.ok) await parseError(response, "Clone failed");
  return response.json();
}

export async function cloneFromFile(input: {
  file: File;
  name: string;
  engine?: string;
  language?: string;
}) {
  const form = new FormData();
  form.append("file", input.file);
  form.append("name", input.name);
  form.append("engine", input.engine || "qwen");
  form.append("language", input.language || "en");

  const response = await fetch(`${API_URL}/api/voices/clone/upload`, {
    method: "POST",
    body: form,
  });
  if (!response.ok) await parseError(response, "Upload clone failed");
  return response.json();
}

export async function deleteVoice(profileId: string) {
  const response = await fetch(`${API_URL}/api/voices/${profileId}`, { method: "DELETE" });
  if (!response.ok) await parseError(response, "Delete failed");
}

export async function transcribeAudio(file: Blob) {
  const form = new FormData();
  form.append("file", file, "recording.webm");
  const response = await fetch(`${API_URL}/api/transcribe`, { method: "POST", body: form });
  if (!response.ok) await parseError(response, "Transcription failed");
  const data = await response.json();
  return data.text as string;
}

export async function refineTranscript(text: string) {
  const response = await fetch(`${API_URL}/api/transcribe/refine`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  if (!response.ok) await parseError(response, "Refinement failed");
  const data = await response.json();
  return data.text as string;
}

export async function speakText(input: {
  text: string;
  profile_id: string;
  language?: string;
  engine?: string;
  voice_style?: VoiceDelivery["voice_style"];
  speed?: number;
  bass_boost?: number;
}) {
  const response = await fetch(`${API_URL}/api/speak`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      ...input,
      engine: input.engine || "qwen",
      voice_style: input.voice_style || "ultron_cinematic",
      speed: input.speed ?? 0.86,
      bass_boost: input.bass_boost ?? 6,
    }),
  });
  if (!response.ok) await parseError(response, "Speech generation failed");
  const data = await response.json();
  return `data:audio/wav;base64,${data.audio_base64}` as string;
}

export type ChatResult = {
  user_message: string;
  assistant_message: string;
  audio_base64: string;
};

export async function chatWithVoice(input: {
  message: string;
  profile_id: string;
  engine?: string;
  voice_style?: VoiceDelivery["voice_style"];
  speed?: number;
  bass_boost?: number;
}): Promise<ChatResult & { audioSrc: string }> {
  const response = await fetch(`${API_URL}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      ...input,
      engine: input.engine || "qwen",
      voice_style: input.voice_style || "ultron_cinematic",
      speed: input.speed ?? 0.86,
      bass_boost: input.bass_boost ?? 6,
    }),
  });
  if (!response.ok) await parseError(response, "Chat failed");
  const data = await response.json();
  return {
    ...data,
    audioSrc: `data:audio/wav;base64,${data.audio_base64}`,
  };
}

export async function voiceChat(input: {
  audio: Blob;
  profile_id: string;
  engine?: string;
  voice_style?: VoiceDelivery["voice_style"];
  speed?: number;
  bass_boost?: number;
}): Promise<ChatResult & { audioSrc: string }> {
  const form = new FormData();
  form.append("file", input.audio, "recording.webm");
  form.append("profile_id", input.profile_id);
  form.append("engine", input.engine || "qwen");
  form.append("voice_style", input.voice_style || "ultron_cinematic");
  form.append("speed", String(input.speed ?? 0.86));
  form.append("bass_boost", String(input.bass_boost ?? 6));

  const response = await fetch(`${API_URL}/api/chat/voice`, { method: "POST", body: form });
  if (!response.ok) await parseError(response, "Voice chat failed");
  const data = await response.json();
  return {
    ...data,
    audioSrc: `data:audio/wav;base64,${data.audio_base64}`,
  };
}
