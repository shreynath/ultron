const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

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
};

export async function getHealth(): Promise<HealthResponse> {
  const response = await fetch(`${API_URL}/api/health`, { cache: "no-store" });
  if (!response.ok) throw new Error("Health check failed");
  return response.json();
}

export async function listVoices(): Promise<VoiceProfile[]> {
  const response = await fetch(`${API_URL}/api/voices`, { cache: "no-store" });
  if (!response.ok) throw new Error("Failed to load voices");
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
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || "Clone failed");
  return data;
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
  form.append("engine", input.engine || "qwen3-tts");
  form.append("language", input.language || "en");

  const response = await fetch(`${API_URL}/api/voices/clone/upload`, {
    method: "POST",
    body: form,
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || "Upload clone failed");
  return data;
}

export async function deleteVoice(profileId: string) {
  const response = await fetch(`${API_URL}/api/voices/${profileId}`, { method: "DELETE" });
  if (!response.ok) {
    const data = await response.json();
    throw new Error(data.detail || "Delete failed");
  }
}

export async function transcribeAudio(file: Blob) {
  const form = new FormData();
  form.append("file", file, "recording.webm");
  const response = await fetch(`${API_URL}/api/transcribe`, { method: "POST", body: form });
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || "Transcription failed");
  return data.text as string;
}

export async function refineTranscript(text: string) {
  const response = await fetch(`${API_URL}/api/transcribe/refine`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || "Refinement failed");
  return data.text as string;
}

export async function speakText(input: {
  text: string;
  profile_id: string;
  language?: string;
  engine?: string;
}) {
  const response = await fetch(`${API_URL}/api/speak`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || "Speech generation failed");
  return `data:audio/wav;base64,${data.audio_base64}` as string;
}
