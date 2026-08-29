"use client";

import { useEffect, useState } from "react";
import { listVoices, type VoiceProfile } from "@/lib/api";

const ENGINE_OPTIONS = [
  { value: "qwen", label: "Qwen3-TTS (best cloning)" },
  { value: "chatterbox", label: "Chatterbox Multilingual" },
  { value: "chatterbox_turbo", label: "Chatterbox Turbo" },
  { value: "luxtts", label: "LuxTTS (fast)" },
  { value: "tada", label: "TADA" },
];

type VoiceProfilePickerProps = {
  value: string;
  onChange: (profileId: string) => void;
  engine?: string;
  onEngineChange?: (engine: string) => void;
  showEngine?: boolean;
  id?: string;
};

export function VoiceProfilePicker({
  value,
  onChange,
  engine = "qwen",
  onEngineChange,
  showEngine = false,
  id = "voice-select",
}: VoiceProfilePickerProps) {
  const [profiles, setProfiles] = useState<VoiceProfile[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    listVoices()
      .then((data) => {
        setProfiles(data);
        const ready = data.find((profile) => (profile.sample_count ?? 1) > 0) ?? data[0];
        if (ready && !value) onChange(ready.id);
      })
      .catch(() => setError("Could not load voice profiles"));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="space-y-4">
      <div>
        <label className="label" htmlFor={id}>
          Voice profile
        </label>
        <select
          id={id}
          className="input"
          value={value}
          onChange={(event) => onChange(event.target.value)}
        >
          {profiles.length === 0 && <option value="">No voices yet</option>}
          {profiles.map((profile) => (
            <option key={profile.id} value={profile.id} disabled={profile.sample_count === 0}>
              {profile.name}
              {profile.sample_count === 0 ? " (needs re-clone)" : ""}
            </option>
          ))}
        </select>
        {error && <p className="mt-2 text-sm text-red-400">{error}</p>}
      </div>

      {showEngine && onEngineChange && (
        <div>
          <label className="label" htmlFor={`${id}-engine`}>
            TTS engine
          </label>
          <select
            id={`${id}-engine`}
            className="input"
            value={engine}
            onChange={(event) => onEngineChange(event.target.value)}
          >
            {ENGINE_OPTIONS.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </div>
      )}
    </div>
  );
}

export { ENGINE_OPTIONS };
