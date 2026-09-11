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
  const theaterEnabled = value.voice_style !== "natural";
  return (
    <div className="card space-y-4 !rounded-[14px] !p-4">
      <div className="flex items-start justify-between gap-4">
        <div>
        <h3 className="text-sm font-medium text-[color:var(--steel-200)]">Theater delivery</h3>
        <p className="mt-1 text-xs text-[color:var(--text-tertiary)]">
          Analyzes your text and applies depth, reverb, pace, and emphasis in one smooth pass.
        </p>
        </div>
        <label className="flex cursor-pointer items-center gap-2 text-xs text-[color:var(--text-secondary)]"><input type="checkbox" checked={theaterEnabled} onChange={(event) => onChange({ voice_style: event.target.checked ? "theater_auto" : "natural" })} />On</label>
      </div>

      {theaterEnabled && <div>
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
      </div>}
    </div>
  );
}
