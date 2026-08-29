"use client";

export type VoiceDelivery = {
  voice_style: "ultron_cinematic" | "natural";
  speed: number;
  bass_boost: number;
};

export const DEFAULT_VOICE_DELIVERY: VoiceDelivery = {
  voice_style: "ultron_cinematic",
  speed: 0.86,
  bass_boost: 6,
};

type VoiceCharacterControlsProps = {
  value: VoiceDelivery;
  onChange: (value: VoiceDelivery) => void;
  id?: string;
};

export function VoiceCharacterControls({
  value,
  onChange,
  id = "voice-character",
}: VoiceCharacterControlsProps) {
  const cinematic = value.voice_style === "ultron_cinematic";

  return (
    <div className="space-y-4 rounded-xl border border-zinc-800 bg-zinc-950/60 p-4">
      <div>
        <h3 className="text-sm font-medium text-zinc-200">Voice character</h3>
        <p className="mt-1 text-xs text-zinc-500">
          Adds depth, reverb, vibrato, and slower pacing. Switch to Natural for a plain cloned voice.
        </p>
      </div>

      <div>
        <label className="label" htmlFor={`${id}-style`}>
          Style
        </label>
        <select
          id={`${id}-style`}
          className="input"
          value={value.voice_style}
          onChange={(event) =>
            onChange({
              ...value,
              voice_style: event.target.value as VoiceDelivery["voice_style"],
            })
          }
        >
          <option value="ultron_cinematic">Cinematic (deep, reverb, slower)</option>
          <option value="natural">Natural (plain clone)</option>
        </select>
      </div>

      {cinematic && (
        <>
          <div>
            <label className="label" htmlFor={`${id}-speed`}>
              Pace — {Math.round(value.speed * 100)}% (lower = slower, more deliberate)
            </label>
            <input
              id={`${id}-speed`}
              className="w-full accent-violet-500"
              type="range"
              min={70}
              max={100}
              step={1}
              value={Math.round(value.speed * 100)}
              onChange={(event) =>
                onChange({ ...value, speed: Number(event.target.value) / 100 })
              }
            />
          </div>

          <div>
            <label className="label" htmlFor={`${id}-bass`}>
              Bass presence — {value.bass_boost.toFixed(0)} dB
            </label>
            <input
              id={`${id}-bass`}
              className="w-full accent-violet-500"
              type="range"
              min={0}
              max={12}
              step={1}
              value={value.bass_boost}
              onChange={(event) =>
                onChange({ ...value, bass_boost: Number(event.target.value) })
              }
            />
          </div>
        </>
      )}
    </div>
  );
}
