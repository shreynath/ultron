"use client";

export type VoiceDelivery = {
  voice_style: "theater_auto" | "ultron_cinematic" | "reference" | "natural";
};

export const DEFAULT_VOICE_DELIVERY: VoiceDelivery = {
  voice_style: "theater_auto",
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
  return (
    <div className="space-y-4 rounded-xl border border-zinc-800 bg-zinc-950/60 p-4">
      <div>
        <h3 className="text-sm font-medium text-zinc-200">Theater delivery</h3>
        <p className="mt-1 text-xs text-zinc-500">
          Analyzes your text and applies depth, reverb, pace, and emphasis in one smooth pass.
        </p>
      </div>

      <div>
        <label className="label" htmlFor={`${id}-style`}>
          Mode
        </label>
        <select
          id={`${id}-style`}
          className="input"
          value={value.voice_style}
          onChange={(event) =>
            onChange({
              voice_style: event.target.value as VoiceDelivery["voice_style"],
            })
          }
        >
          <option value="theater_auto">Theater (auto — recommended)</option>
          <option value="reference">Reference (flat clone)</option>
          <option value="ultron_cinematic">Cinematic (heavy FX)</option>
          <option value="natural">Natural (no FX)</option>
        </select>
      </div>
    </div>
  );
}
